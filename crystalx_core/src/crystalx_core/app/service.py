"""The application service: one API for every frontend.

The Gradio app and the ``crystalx solve`` command both talk to :class:`RunService`; neither knows
how a run is executed. The service offers exactly what a frontend needs and nothing more:

* :meth:`RunService.capabilities` -- code readiness and static tool availability;
* :meth:`RunService.preview` -- diagnosis, preconditions and the plan, *without* starting a run;
* :meth:`RunService.submit` -- start a run; returns a durable run id at once;
* :meth:`RunService.status`, :meth:`RunService.summary`, :meth:`RunService.events`,
  :meth:`RunService.log_tail` -- read a run's state, results, event log and live program output;
* :meth:`RunService.cancel` -- request cancellation (the program's process group is terminated);
* :meth:`RunService.artefacts`, :meth:`RunService.artefact_path`, :meth:`RunService.export_bundle`
  -- downloads, restricted to the run's own recorded artefacts.

**Durability.** Every run is a directory under ``runs_root`` with its state, summary and event log
on disk, written as the run progresses. Status is always read from disk, so a restarted service
still knows about every run. A run that was in progress when its service stopped is reported as
``interrupted``; its files are left exactly as they were.

**Isolation.** Each run has its own directory; every program runs in a subdirectory of it. Runs are
executed on worker threads, at most ``max_concurrent`` at a time (one by default -- SHELXT is
already multi-threaded).

No database, no accounts, no job queue process: a directory per run is the whole persistence model.
"""

from __future__ import annotations

import datetime as _dt
import functools
import json
import os
import socket
import threading
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from crystalx_core.app.capabilities import capability_report
from crystalx_core.app.demo import is_demo_input
from crystalx_core.app.mock_toolchain import MockToolchain
from crystalx_core.app.models import (
    TERMINAL_PHASES,
    RunConfig,
    RunMode,
    RunPhase,
    SolveRequest,
    ValidationIssue,
)
from crystalx_core.app.pipeline import (
    REQUEST_FILE,
    STATE_FILE,
    SUMMARY_FILE,
    SolvePipeline,
    build_registry,
    diagnosis_display,
    plan_display,
)
from crystalx_core.app.staging import safe_stem
from crystalx_core.app.toolchain import RealToolchain, Toolchain
from crystalx_core.domain.data import DataDiagnosis
from crystalx_core.domain.ids import RUN_ID_PATTERN, make_stage_id
from crystalx_core.domain.onion import StageKind
from crystalx_core.domain.radiation import RadiationAssumption
from crystalx_core.domain.solving import BranchBudget
from crystalx_core.errors import CrystalXError
from crystalx_core.io.shelx_reader import read_ins_header, scan_hkl
from crystalx_core.solvers.context import SolverContext, StagedCase
from crystalx_core.solvers.selector import SelectionPolicy, StrategySelector
from crystalx_core.tools.runner import CancelToken
from crystalx_core.workflow.preconditions import check_preconditions
from crystalx_core.workspace.events import read_events
from crystalx_core.workspace.store import RunWorkspace

__all__ = ["RunService", "RunHandle", "RequestInvalid", "UnknownRun", "default_toolchain"]

#: Files every run exposes for download in addition to its recorded artefacts.
_STANDARD_FILES = (SUMMARY_FILE, STATE_FILE, REQUEST_FILE, "result.json", "events.ndjson", "run.json")
_LOG_TAIL_BYTES = 64_000


class RequestInvalid(ValueError):
    """The request cannot start. ``issues`` carries one stable code per problem."""

    def __init__(self, issues: list[ValidationIssue]) -> None:
        super().__init__("; ".join(f"{i.field}: {i.message}" for i in issues))
        self.issues = issues


class UnknownRun(LookupError):
    """No run with that id exists under this service's root (or the id is malformed)."""


@dataclass(frozen=True)
class RunHandle:
    run_id: str
    run_dir: Path


@dataclass
class _Active:
    thread: threading.Thread
    cancel: CancelToken
    pipeline: SolvePipeline
    done: threading.Event = field(default_factory=threading.Event)


def default_toolchain(request: SolveRequest, candidate_check=None) -> Toolchain:
    """Mock stand-ins for the demo, the user's own programs otherwise.

    ``candidate_check`` is the injected deeper check (see ``ResolverConfig.candidate_check``); the
    service passes the one it was given, so a run selects what the tool panel and the capability
    report selected.
    """
    if request.mode is RunMode.MOCK_DEMO:
        return MockToolchain(step_seconds=request.config.mock_step_seconds)
    return RealToolchain.for_config(request.config, candidate_check=candidate_check)


def _now() -> _dt.datetime:
    return _dt.datetime.now(_dt.timezone.utc)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:  # pragma: no cover - platform specific
        return False
    return True


class RunService:
    """Starts, tracks and reads solve runs. Thread-safe."""

    def reconstruct(self, *, repository: Path, **options) -> dict:
        """Synchronous iterative application API, using the shared OnionWorkflow controller.

        Synthetic mode accepts only the built-in demo. Real reconstruction returns prerequisites
        without dispatch. Frontends can expose its retained history in a later UI milestone.
        """
        from crystalx_core.app.reconstruction import ReconstructionService
        reconstruction = ReconstructionService(repository, self.runs_root / "reconstruction")
        if "initial_handoff" in options:
            return reconstruction.run_from_initial_handoff(options.pop("initial_handoff"), **options)
        return reconstruction.run(**options)

    def __init__(
        self,
        runs_root: Path | str,
        *,
        max_concurrent: int = 1,
        toolchain_factory: Callable[[SolveRequest], Toolchain] = default_toolchain,
        pipeline_factory: Callable[..., SolvePipeline] = SolvePipeline,
        bundles_dir: Path | str | None = None,
        clock: Callable[[], _dt.datetime] = _now,
        candidate_check: Callable[..., Any] | None = None,
    ) -> None:
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be at least 1")
        self.runs_root = Path(runs_root)
        self.runs_root.mkdir(parents=True, exist_ok=True)
        self.bundles_dir = Path(bundles_dir) if bundles_dir else self.runs_root / "_bundles"
        self._candidate_check = candidate_check
        if toolchain_factory is default_toolchain and candidate_check is not None:
            toolchain_factory = functools.partial(default_toolchain, candidate_check=candidate_check)
        self._toolchain_factory = toolchain_factory
        self._pipeline_factory = pipeline_factory
        self._clock = clock
        self._slots = threading.BoundedSemaphore(max_concurrent)
        self._lock = threading.Lock()
        self._active: dict[str, _Active] = {}

    # ------------------------------------------------------------------ no run needed
    def capabilities(
        self,
        mode: RunMode = RunMode.REAL,
        toolbox_dir: Path | None = None,
        *,
        config: RunConfig | None = None,
    ) -> dict:
        """Static tool report for a mode. ``config`` (when given) supplies the same locations and
        strict-native policy a run with that configuration would use."""
        if mode is RunMode.MOCK_DEMO:
            return capability_report(MockToolchain())
        if config is not None:
            return capability_report(RealToolchain.for_config(config, candidate_check=self._candidate_check))
        return capability_report(RealToolchain(toolbox_dir=toolbox_dir, candidate_check=self._candidate_check))

    def preview(self, request: SolveRequest) -> dict[str, Any]:
        """Diagnosis, preconditions and plan for a request, reading the files without copying them.

        Nothing is executed and nothing is written. Raises :class:`RequestInvalid` for bad input.
        """
        issues = request.validate()
        if issues:
            raise RequestInvalid(issues)
        try:
            header = read_ins_header(request.ins_path)
            reflections = scan_hkl(request.hkl_path, cell=header.cell, hklf_hint=header.hklf_format)
        except CrystalXError as exc:
            raise RequestInvalid([ValidationIssue("unreadable_input", "input", str(exc))]) from exc
        diagnosis = DataDiagnosis(reflections=reflections, ins=header)
        prior = request.prior.to_prior()
        report = check_preconditions(diagnosis, prior, space_group_from_solver=True)
        toolchain = self._toolchain_factory(request)
        availability = toolchain.availability()
        config = request.config
        stage_id = make_stage_id(1, StageKind.INITIAL_SOLUTION.value)
        trace = StrategySelector(
            SelectionPolicy(default_branch_budget=BranchBudget(max_wall_clock_s=config.shelxt_timeout_s))
        ).select(diagnosis, prior, stage_id=stage_id, availability=availability)
        context = SolverContext(
            case=StagedCase(stem=safe_stem(Path(request.ins_path).name)),
            diagnosis=diagnosis,
            prior=prior,
            radiation=RadiationAssumption.from_wavelength(diagnosis.wavelength_a),
        )
        return {
            "mode": request.mode.value,
            "diagnosis": diagnosis_display(diagnosis, request.prior),
            "preconditions": {
                "may_start": report.may_start,
                "summary": report.summary(),
                "items": [
                    {
                        "kind": p.kind.value,
                        "blocking": p.blocking,
                        "summary": p.summary,
                        "evidence": p.evidence,
                        "remedy": p.remedy,
                        "override_hint": p.override_hint,
                    }
                    for p in report.preconditions
                ],
            },
            "plan": plan_display(trace, build_registry(context, config), config),
            "capabilities": capability_report(toolchain),
        }

    # ------------------------------------------------------------------ runs
    def submit(self, request: SolveRequest) -> RunHandle:
        """Validate, create the run directory, and start the run on a worker thread."""
        issues = request.validate()
        if request.mode is RunMode.MOCK_DEMO and not issues and not is_demo_input(
            request.ins_path, request.hkl_path
        ):
            issues.append(
                ValidationIssue(
                    "mock_requires_demo",
                    "mode",
                    "The demo toolchain only runs on the built-in synthetic demo case. Real data "
                    "needs real programs.",
                )
            )
        if issues:
            raise RequestInvalid(issues)

        workspace = RunWorkspace.create(
            self.runs_root,
            label=request.label,
            clock=self._clock,
            metadata={"app": "crystalx", "mode": request.mode.value},
        )
        cancel = CancelToken()
        pipeline = self._pipeline_factory(
            workspace=workspace,
            request=request,
            toolchain=self._toolchain_factory(request),
            cancel=cancel,
            clock=self._clock,
        )
        workspace.write_json(REQUEST_FILE, pipeline.request_record())
        pipeline.write_initial_state()
        run_id = str(workspace.run_id)
        thread = threading.Thread(
            target=self._worker, args=(run_id,), name=f"crystalx-{run_id}", daemon=True
        )
        active = _Active(thread=thread, cancel=cancel, pipeline=pipeline)
        with self._lock:
            self._active[run_id] = active
        thread.start()
        return RunHandle(run_id=run_id, run_dir=workspace.layout.root)

    def _worker(self, run_id: str) -> None:
        with self._lock:
            active = self._active[run_id]
        acquired = False
        try:
            while not acquired:
                acquired = self._slots.acquire(timeout=0.2)
                if not acquired and active.cancel.cancelled:
                    break
            active.pipeline.run()
        finally:
            if acquired:
                self._slots.release()
            active.done.set()
            with self._lock:
                self._active.pop(run_id, None)

    def cancel(self, run_id: str, reason: str = "cancelled by user") -> bool:
        """Request cancellation. Returns ``False`` when the run is not active in this service."""
        self._run_dir(run_id)
        with self._lock:
            active = self._active.get(run_id)
        if active is None:
            return False
        active.cancel.cancel(reason)
        return True

    def wait(self, run_id: str, timeout: float | None = None) -> dict[str, Any]:
        with self._lock:
            active = self._active.get(run_id)
        if active is not None:
            active.done.wait(timeout)
        return self.status(run_id)

    def shutdown(self, *, timeout: float = 15.0) -> None:
        """Cancel every active run and wait for its programs to be stopped."""
        with self._lock:
            active = list(self._active.values())
        for item in active:
            item.cancel.cancel("the service is shutting down")
        for item in active:
            item.done.wait(timeout)

    def is_active(self, run_id: str) -> bool:
        with self._lock:
            return run_id in self._active

    # ------------------------------------------------------------------ reading
    def status(self, run_id: str) -> dict[str, Any]:
        run_dir = self._run_dir(run_id)
        state = self._read_json(run_dir / STATE_FILE) or {
            "run_id": run_id,
            "phase": "unknown",
            "message": "this run directory has no state file",
        }
        phase = state.get("phase")
        if phase not in {p.value for p in TERMINAL_PHASES} and not self.is_active(run_id):
            service = state.get("service") or {}
            pid, host = service.get("pid"), service.get("host")
            elsewhere = (
                isinstance(pid, int)
                and pid != os.getpid()
                and host == socket.gethostname()
                and _pid_alive(pid)
            )
            if elsewhere:
                state["note"] = "this run is being executed by another process"
            else:
                state["recorded_phase"] = phase
                state["phase"] = RunPhase.INTERRUPTED.value
                state["message"] = (
                    "the service stopped while this run was in progress; its record is kept "
                    "exactly as it was left"
                )
        state["active_in_this_service"] = self.is_active(run_id)
        return state

    def summary(self, run_id: str) -> dict[str, Any] | None:
        return self._read_json(self._run_dir(run_id) / SUMMARY_FILE)

    def events(self, run_id: str, *, after: int = 0, limit: int = 1000) -> list[dict[str, Any]]:
        events = list(read_events(self._run_dir(run_id) / "events.ndjson"))
        return [dict(event, index=i) for i, event in enumerate(events)][after : after + limit]

    def log_tail(self, run_id: str, *, max_bytes: int = _LOG_TAIL_BYTES) -> tuple[str, str]:
        """``(relative path, text)`` of the program output currently being written, if any."""
        run_dir = self._run_dir(run_id)
        state = self._read_json(run_dir / STATE_FILE) or {}
        relative = state.get("current_log")
        if not relative:
            return "", ""
        path = self._inside(run_dir, relative)
        if path is None or not path.is_file():
            return relative, ""
        size = path.stat().st_size
        with open(path, "rb") as handle:
            if size > max_bytes:
                handle.seek(size - max_bytes)
            return relative, handle.read().decode("utf-8", errors="replace")

    def viewer_payload(self, run_id: str, name: str) -> dict[str, Any] | None:
        if name not in ("solution", "refined"):
            raise ValueError("viewer name must be 'solution' or 'refined'")
        return self._read_json(self._run_dir(run_id) / "viewer" / f"{name}.json")

    def list_runs(self, *, limit: int = 50) -> list[dict[str, Any]]:
        rows = []
        for path in sorted(self.runs_root.iterdir(), reverse=True):
            if not path.is_dir() or path.is_symlink():
                continue
            run_id = path.name.split("_", 1)[0]
            if not RUN_ID_PATTERN.match(run_id):
                continue
            try:
                state = self.status(run_id)
            except UnknownRun:
                continue
            rows.append(
                {
                    "run_id": run_id,
                    "label": state.get("label", ""),
                    "mode": state.get("mode", ""),
                    "phase": state.get("phase", ""),
                    "message": state.get("message", ""),
                    "created_at": state.get("created_at", ""),
                }
            )
            if len(rows) >= limit:
                break
        return rows

    # ------------------------------------------------------------------ downloads
    def artefacts(self, run_id: str) -> list[str]:
        """Relative paths a frontend may offer for download: this run's recorded files only."""
        run_dir = self._run_dir(run_id)
        summary = self._read_json(run_dir / SUMMARY_FILE) or {}
        names = [n for n in _STANDARD_FILES if (run_dir / n).is_file()]
        names += [f"viewer/{n}.json" for n in ("solution", "refined") if (run_dir / "viewer" / f"{n}.json").is_file()]
        for relative in summary.get("artefacts", []):
            if isinstance(relative, str) and relative not in names and self._inside(run_dir, relative):
                names.append(relative)
        return names

    def artefact_path(self, run_id: str, relative: str) -> Path:
        """Resolve one download. Only names returned by :meth:`artefacts` are accepted."""
        run_dir = self._run_dir(run_id)
        if relative not in self.artefacts(run_id):
            raise PermissionError(f"{relative!r} is not a downloadable artefact of run {run_id}")
        path = self._inside(run_dir, relative)
        if path is None or not path.is_file() or path.is_symlink():
            raise FileNotFoundError(relative)
        return path

    def export_bundle(self, run_id: str) -> Path:
        """A zip of the whole run directory, written outside it. Only for finished runs."""
        run_dir = self._run_dir(run_id)
        state = self.status(run_id)
        if state.get("phase") not in {p.value for p in TERMINAL_PHASES}:
            raise RuntimeError("the run has not finished; export it when it has")
        self.bundles_dir.mkdir(parents=True, exist_ok=True)
        target = self.bundles_dir / f"{run_dir.name}.zip"
        tmp = target.with_suffix(".zip.tmp")
        with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
            for path in sorted(run_dir.rglob("*")):
                if path.is_symlink() or not path.is_file():
                    continue
                bundle.write(path, arcname=f"{run_dir.name}/{path.relative_to(run_dir).as_posix()}")
        os.replace(tmp, target)
        return target

    # ------------------------------------------------------------------ internals
    def _run_dir(self, run_id: str) -> Path:
        if not isinstance(run_id, str) or not RUN_ID_PATTERN.match(run_id):
            raise UnknownRun(f"not a run id: {run_id!r}")
        matches = [p for p in self.runs_root.glob(f"{run_id}_*") if p.is_dir() and not p.is_symlink()]
        if len(matches) != 1:
            raise UnknownRun(f"no run {run_id} under {self.runs_root}")
        run_dir = matches[0]
        if self.runs_root.resolve() not in run_dir.resolve().parents:
            raise UnknownRun(f"run {run_id} is outside {self.runs_root}")
        return run_dir

    @staticmethod
    def _inside(run_dir: Path, relative: str) -> Path | None:
        if not relative or relative.startswith(("/", "\\")) or "\\" in relative:
            return None
        if any(part in ("..", "") for part in relative.split("/")):
            return None
        candidate = run_dir / relative
        try:
            resolved = candidate.resolve()
        except OSError:
            return None
        root = run_dir.resolve()
        if root not in resolved.parents:
            return None
        return candidate

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any] | None:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
