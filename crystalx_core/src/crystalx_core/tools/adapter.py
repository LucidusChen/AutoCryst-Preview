"""Generic adapter for any external crystallographic program.

This is the single place where a resolved executable, a safe invocation and a complete provenance
record come together. Every solver backend goes through it, so no backend can invent its own way of
running a program or its own idea of what to record.

What one invocation records:

* the executable's identity -- path, ``sha256``, architecture, where it was resolved from, and
  whether it needed Rosetta translation;
* the exact ``argv`` array, working directory, environment delta and any scripted stdin;
* a digest of every declared input **before** the program ran;
* a digest of every file the program created or modified, discovered by comparing a snapshot of the
  working directory taken before and after -- not only the files we thought to ask for;
* timing, exit code, and the classified diagnostic.

The point of the before/after snapshot is that SHELXT and SHELXE write files whose names depend on
what they found. Recording only the expected ones would quietly lose the alternative space-group
solutions and the second-hand density-modification output.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from pathlib import Path
import stat
from typing import Callable, Iterable, Mapping, Sequence

from crystalx_core.domain.tools import (
    ResolutionSource,
    ToolDiagnostic,
    ToolDiagnosticKind,
    ToolIdentity,
    ToolKind,
    ToolResolution,
)
from crystalx_core.errors import ToolResolutionError
from crystalx_core.tools.registry import spec_for
from crystalx_core.tools.runner import CancelToken, ResourceLimits, ExecutionOutcome, ExecutionPlan, ProcessRunner
from crystalx_core.workspace.hashing import sha256_file, sha256_text

__all__ = [
    "FileRecord",
    "ToolInvocationRecord",
    "ExternalToolAdapter",
    "snapshot_directory",
]


def _utc_now() -> _dt.datetime:
    return _dt.datetime.now(_dt.timezone.utc)


@dataclass(frozen=True)
class FileRecord:
    """One file's identity at a point in time."""

    relative_path: str
    sha256: str
    size_bytes: int
    mtime_epoch: float
    role: str = ""
    """``input``, ``output``, ``modified``, ``log``."""

    @classmethod
    def of(cls, path: Path, *, base: Path, role: str = "") -> "FileRecord":
        stat = path.stat()
        try:
            relative = str(path.relative_to(base))
        except ValueError:
            relative = str(path)
        return cls(
            relative_path=relative,
            sha256=sha256_file(path),
            size_bytes=stat.st_size,
            mtime_epoch=stat.st_mtime,
            role=role,
        )


#: Files larger than this are compared by size and modification time only (hashing them before and
#: after every invocation would cost more than the record is worth). Recorded as a limitation.
SNAPSHOT_HASH_LIMIT_BYTES = 256 * 1024 * 1024


def snapshot_directory(directory: Path, *, strict_regular_files: bool = False) -> dict[str, tuple[int, int, str]]:
    """``{relative path: (size, mtime_ns, sha256)}``, used to detect what a program wrote or changed.

    The digest makes modification detection independent of the filesystem's timestamp resolution
    (a coarse-mtime mount can otherwise hide an in-place rewrite of the same size). Files above
    :data:`SNAPSHOT_HASH_LIMIT_BYTES` get an empty digest and fall back to size and time.
    """
    snapshot: dict[str, tuple[int, int, str]] = {}
    if not directory.is_dir():
        return snapshot
    for path in directory.rglob("*"):
        if strict_regular_files:
            mode = path.lstat()
            if (stat.S_ISLNK(mode.st_mode) or not (stat.S_ISDIR(mode.st_mode) or stat.S_ISREG(mode.st_mode))
                    or (stat.S_ISREG(mode.st_mode) and mode.st_nlink != 1)):
                raise ValueError("Strict tool snapshot refuses symlinks, aliases and nonregular files")
        if path.is_file():
            info = path.stat()
            digest = sha256_file(path) if info.st_size <= SNAPSHOT_HASH_LIMIT_BYTES else ""
            snapshot[str(path.relative_to(directory))] = (info.st_size, info.st_mtime_ns, digest)
    return snapshot


@dataclass(frozen=True)
class ToolInvocationRecord:
    """Everything needed to audit or reproduce one external-program run."""

    tool: ToolKind
    label: str
    executable_path: str
    executable_sha256: str
    executable_arch: str
    executable_source: str
    executable_version: str
    required_translation: bool
    """True when the binary ran through Rosetta 2. Recorded because it changes timing, and because
    a run that only works with translation is a fact about the machine, not about the data."""

    argv: tuple[str, ...]
    cwd: str
    env_overrides: dict[str, str]
    env_unset: tuple[str, ...]
    stdin_sha256: str
    stdin_script: str
    timeout_s: float
    started_at: str
    ended_at: str
    duration_s: float
    exit_code: int | None
    diagnostic: str
    diagnostic_message: str
    succeeded: bool
    inputs: tuple[FileRecord, ...] = ()
    outputs: tuple[FileRecord, ...] = ()
    expected_outputs: tuple[str, ...] = ()
    missing_outputs: tuple[str, ...] = ()
    stdout_path: str = ""
    stderr_path: str = ""
    remedy: str = ""
    limits: dict | None = None
    """What the program was allowed to consume. ``None`` when no limits were requested."""

    cancelled: bool = False
    group_cleanup: str = "none"
    """What had to be stopped in the program's process group after it ended, was cancelled or
    timed out: ``none``, ``terminated`` (SIGTERM sufficed) or ``killed`` (a member ignored it)."""

    def output_named(self, name: str) -> FileRecord | None:
        for record in self.outputs:
            if record.relative_path == name:
                return record
        return None

    def outputs_with_suffix(self, suffix: str) -> tuple[FileRecord, ...]:
        lowered = suffix.lower()
        return tuple(r for r in self.outputs if r.relative_path.lower().endswith(lowered))

    @property
    def input_hashes(self) -> dict[str, str]:
        return {r.relative_path: r.sha256 for r in self.inputs}

    @property
    def output_hashes(self) -> dict[str, str]:
        return {r.relative_path: r.sha256 for r in self.outputs}


class ExternalToolAdapter:
    """Runs one external program and produces a :class:`ToolInvocationRecord`.

    Construct it from a :class:`ToolResolution` so that a tool which is present but unusable -- an
    x86_64 build on Apple Silicon without Rosetta, say -- fails here with the resolver's diagnosis
    rather than as a confusing error from deep inside a backend.
    """

    def __init__(
        self,
        resolution: ToolResolution,
        *,
        runner: ProcessRunner | None = None,
        clock: Callable[[], _dt.datetime] = _utc_now,
        strict_regular_files: bool = False,
    ) -> None:
        if resolution.identity is None or not resolution.is_usable:
            raise ToolResolutionError(
                f"Cannot build an adapter for {resolution.kind.value}: "
                f"{resolution.diagnostic.message}",
                diagnostic=resolution.diagnostic.kind.value,
                context={"remedy": resolution.diagnostic.remedy},
            )
        self.resolution = resolution
        self.identity: ToolIdentity = resolution.identity
        self.runner = runner or ProcessRunner()
        self._clock = clock
        self._strict_regular_files = strict_regular_files

    @property
    def kind(self) -> ToolKind:
        return self.identity.kind

    @property
    def required_translation(self) -> bool:
        return self.resolution.diagnostic.kind is ToolDiagnosticKind.OK_VIA_TRANSLATION

    # ------------------------------------------------------------------ planning
    def build_plan(
        self,
        *,
        args: Sequence[str],
        cwd: Path,
        timeout_s: float = 600.0,
        stdin_script: str | None = None,
        expected_outputs: Sequence[str] = (),
        declared_inputs: Sequence[str] = (),
        env_overrides: Mapping[str, str] | None = None,
        env_unset: Sequence[str] = (),
        label: str = "",
        headless: bool = True,
        limits: ResourceLimits | None = None,
    ) -> ExecutionPlan:
        """Build the plan without running it. This is what ``--dry-run`` prints.

        ``headless`` removes ``DISPLAY`` for programs that would otherwise try to open an X11
        window; the PLATON documentation is explicit that the graphical build does this by default.
        """
        unset = list(env_unset)
        spec = spec_for(self.kind)
        if headless and spec.needs_x11 and "DISPLAY" not in unset:
            unset.append("DISPLAY")
        return ExecutionPlan(
            argv=(str(self.identity.path), *args),
            cwd=cwd,
            timeout_s=timeout_s,
            stdin_script=stdin_script,
            env_overrides=dict(env_overrides or {}),
            env_unset=tuple(unset),
            expected_outputs=tuple(expected_outputs),
            declared_inputs=tuple(declared_inputs),
            label=label or self.kind.value,
            tool_sha256=self.identity.sha256,
            tool_kind=self.kind,
            limits=limits,
        )

    # ------------------------------------------------------------------ execution
    def invoke(
        self,
        plan: ExecutionPlan,
        *,
        declared_inputs: Iterable[str] = (),
        log_dir: Path | None = None,
        cancel: CancelToken | None = None,
    ) -> tuple[ExecutionOutcome, ToolInvocationRecord]:
        """Run a plan and return both the raw outcome and the provenance record.

        ``declared_inputs`` are paths relative to the plan's working directory that the program is
        expected to read; when omitted the plan's own ``declared_inputs`` are used. They are hashed
        before the run so that a later reader can prove what went in, not only what came out.
        """
        cwd = plan.cwd
        names = tuple(declared_inputs) or plan.declared_inputs
        input_records = tuple(
            FileRecord.of(cwd / name, base=cwd, role="input")
            for name in names
            if (cwd / name).is_file()
        )
        before = snapshot_directory(cwd, strict_regular_files=self._strict_regular_files)

        outcome = self._refuse_if_changed(plan) or self.runner.run(plan, log_dir=log_dir, cancel=cancel)

        after = snapshot_directory(cwd, strict_regular_files=self._strict_regular_files)
        log_names = {
            str(p.relative_to(cwd))
            for p in (outcome.stdout_path, outcome.stderr_path)
            if p is not None and _is_within(p, cwd)
        }
        output_records: list[FileRecord] = []
        for relative, stats in sorted(after.items()):
            previous = before.get(relative)
            if previous == stats:
                continue
            role = "log" if relative in log_names else ("modified" if previous else "output")
            output_records.append(FileRecord.of(cwd / relative, base=cwd, role=role))

        record = ToolInvocationRecord(
            tool=self.kind,
            label=plan.label,
            executable_path=str(self.identity.path),
            executable_sha256=self.identity.sha256,
            executable_arch=self.identity.arch.value,
            executable_source=self.identity.source.value,
            executable_version=self.identity.version or "",
            required_translation=self.required_translation,
            argv=plan.argv,
            cwd=str(cwd),
            env_overrides=dict(plan.env_overrides),
            env_unset=tuple(plan.env_unset),
            stdin_sha256=sha256_text(plan.stdin_script) if plan.stdin_script else "",
            stdin_script=plan.stdin_script or "",
            timeout_s=plan.timeout_s,
            started_at=outcome.started_at,
            ended_at=outcome.ended_at,
            duration_s=outcome.duration_s,
            exit_code=outcome.exit_code,
            diagnostic=outcome.diagnostic.kind.value,
            diagnostic_message=outcome.diagnostic.message,
            succeeded=outcome.succeeded,
            inputs=input_records,
            outputs=tuple(output_records),
            expected_outputs=plan.expected_outputs,
            missing_outputs=outcome.missing_outputs,
            stdout_path=str(outcome.stdout_path) if outcome.stdout_path else "",
            stderr_path=str(outcome.stderr_path) if outcome.stderr_path else "",
            remedy=outcome.diagnostic.remedy,
            limits=plan.limits.to_record() if plan.limits else None,
            cancelled=outcome.cancelled,
            group_cleanup=outcome.group_cleanup,
        )
        return outcome, record

    def _refuse_if_changed(self, plan: ExecutionPlan) -> ExecutionOutcome | None:
        """Refuse to start an executable whose digest differs from the one resolved and recorded."""
        if self.identity.source is ResolutionSource.SYNTHETIC_DEMO:
            return None
        executable = Path(plan.argv[0])
        try:
            current = sha256_file(executable)
        except OSError as exc:
            current = f"unreadable: {exc}"
        if current == self.identity.sha256:
            return None
        now = self._clock().isoformat()
        return ExecutionOutcome(
            plan=plan,
            diagnostic=ToolDiagnostic(
                kind=ToolDiagnosticKind.EXECUTABLE_CHANGED,
                message=(
                    f"{executable} does not match the executable resolved for this run "
                    f"(recorded sha256 {self.identity.sha256[:16]}, now {current[:16]}); not started"
                ),
                path=executable,
                remedy="Resolve the tools again; do not replace programs while a run is prepared.",
            ),
            exit_code=None,
            started_at=now,
            ended_at=now,
            duration_s=0.0,
            missing_outputs=tuple(plan.expected_outputs),
        )

    def run(
        self,
        *,
        args: Sequence[str],
        cwd: Path,
        timeout_s: float = 600.0,
        stdin_script: str | None = None,
        expected_outputs: Sequence[str] = (),
        declared_inputs: Iterable[str] = (),
        env_overrides: Mapping[str, str] | None = None,
        env_unset: Sequence[str] = (),
        label: str = "",
        log_dir: Path | None = None,
        headless: bool = True,
    ) -> tuple[ExecutionOutcome, ToolInvocationRecord]:
        """Convenience: build a plan and invoke it in one call."""
        plan = self.build_plan(
            args=args,
            cwd=cwd,
            timeout_s=timeout_s,
            stdin_script=stdin_script,
            expected_outputs=expected_outputs,
            env_overrides=env_overrides,
            env_unset=env_unset,
            label=label,
            headless=headless,
        )
        return self.invoke(plan, declared_inputs=declared_inputs, log_dir=log_dir)

    def describe(self) -> ToolDiagnostic:
        """The resolver's verdict on this executable, for display."""
        return self.resolution.diagnostic


def _is_within(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
    except ValueError:
        return False
    return True
