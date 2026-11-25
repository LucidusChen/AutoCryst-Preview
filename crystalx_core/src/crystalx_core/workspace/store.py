"""The immutable run workspace.

Rules enforced here, not by convention:

1. **The user's originals are never touched.** Inputs are copied in and the copies are made
   read-only. The source files are hashed before and after so a violation is detectable.
2. **Nothing is ever overwritten.** Writing to an existing artefact path raises. A retry creates a
   *new* iteration; it does not replace the old one.
3. **A crash leaves a readable record.** The event log is append-only and flushed per line, and the
   candidate index is rewritten atomically after every change.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import secrets
import shutil
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from crystalx_core._serde import dumps, to_jsonable
from crystalx_core.domain.candidate import Candidate, Lineage
from crystalx_core.domain.ids import (
    BranchId,
    CandidateId,
    RunId,
    StageId,
    make_run_id,
    slugify,
)
from crystalx_core.domain.onion import OnionStage
from crystalx_core.domain.result import RunResult
from crystalx_core.domain.solving import InitialSolutionBranch, BranchOutcome, SelectionTrace
from crystalx_core.errors import WorkspaceError
from crystalx_core.version import CORE_VERSION, RECORD_SCHEMA_VERSION
from crystalx_core.workspace.events import EventKind, EventLog
from crystalx_core.workspace.hashing import sha256_file
from crystalx_core.workspace.layout import RunLayout

__all__ = ["RunWorkspace", "StagedInput", "Clock", "EntropySource"]

#: Injected so tests are deterministic.
Clock = Callable[[], _dt.datetime]
EntropySource = Callable[[], str]

_READ_ONLY = stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH  # 0o444


def _utc_now() -> _dt.datetime:
    return _dt.datetime.now(_dt.timezone.utc)


def _hex8() -> str:
    return secrets.token_hex(4)


@dataclass(frozen=True)
class StagedInput:
    """One input file copied into the workspace."""

    role: str
    original_path: Path
    staged_path: Path
    sha256: str
    size_bytes: int
    original_mtime_epoch: float


@dataclass
class RunWorkspace:
    """A single, immutable run directory."""

    layout: RunLayout
    run_id: RunId
    created_at: _dt.datetime
    events: EventLog
    lineage: Lineage = field(default_factory=Lineage)
    inputs: dict[str, StagedInput] = field(default_factory=dict)
    _finalized: bool = False

    # ------------------------------------------------------------------ construction
    @classmethod
    def create(
        cls,
        workspace_root: Path | str,
        *,
        label: str = "run",
        clock: Clock = _utc_now,
        entropy: EntropySource = _hex8,
        metadata: Mapping[str, Any] | None = None,
    ) -> "RunWorkspace":
        """Create a fresh run directory. Fails rather than reusing an existing one."""
        created_at = clock()
        run_id = make_run_id(created_at, entropy())
        layout = RunLayout.for_run(workspace_root, run_id, slugify(label))
        if layout.root.exists():
            raise WorkspaceError(
                f"Run directory already exists: {layout.root}",
                run_id=str(run_id),
                evidence_paths=[layout.root],
            )
        for directory in (
            layout.root,
            layout.inputs_dir,
            layout.stages_dir,
            layout.candidates_dir,
        ):
            directory.mkdir(parents=True, exist_ok=False)

        events = EventLog(layout.events, clock=clock)
        workspace = cls(layout=layout, run_id=run_id, created_at=created_at, events=events)
        workspace._write_json(
            layout.run_json,
            {
                "schema_version": RECORD_SCHEMA_VERSION,
                "core_version": CORE_VERSION,
                "run_id": str(run_id),
                "created_at": created_at.isoformat(),
                "label": label,
                "metadata": dict(metadata or {}),
            },
        )
        events.emit(EventKind.RUN_STARTED, f"run {run_id} created", run_id=str(run_id))
        return workspace

    # ------------------------------------------------------------------ inputs
    def stage_input(self, role: str, source: Path | str) -> StagedInput:
        """Copy an input in, make the copy read-only, and record its digest.

        ``role`` is the logical name (``hkl``, ``ins``, ``reference_cif``), used as the key in the
        manifest and in candidate provenance.
        """
        self._assert_open()
        src = Path(source)
        if not src.is_file():
            raise WorkspaceError(
                f"Input for role {role!r} is not a file: {src}",
                run_id=str(self.run_id),
                evidence_paths=[src],
            )
        if role in self.inputs:
            raise WorkspaceError(
                f"Input role {role!r} already staged as {self.inputs[role].staged_path}",
                run_id=str(self.run_id),
            )

        digest_before = sha256_file(src)
        stat_before = src.stat()
        destination = self.layout.inputs_dir / src.name
        if destination.exists():
            destination = self.layout.inputs_dir / f"{role}__{src.name}"
        if destination.exists():
            raise WorkspaceError(
                f"Refusing to overwrite staged input {destination}", run_id=str(self.run_id)
            )

        shutil.copy2(src, destination)
        os.chmod(destination, _READ_ONLY)

        digest_after = sha256_file(src)
        if digest_after != digest_before:
            raise WorkspaceError(
                f"Source file changed while staging: {src}",
                run_id=str(self.run_id),
                evidence_paths=[src],
                context={"sha256_before": digest_before, "sha256_after": digest_after},
            )

        staged = StagedInput(
            role=role,
            original_path=src,
            staged_path=destination,
            sha256=digest_before,
            size_bytes=stat_before.st_size,
            original_mtime_epoch=stat_before.st_mtime,
        )
        self.inputs[role] = staged
        self._write_input_manifest()
        self.events.emit(
            EventKind.INPUT_STAGED,
            f"staged {role}",
            role=role,
            source=str(src),
            staged=str(destination),
            sha256=staged.sha256,
        )
        return staged

    def input_hashes(self) -> dict[str, str]:
        return {role: staged.sha256 for role, staged in sorted(self.inputs.items())}

    def _write_input_manifest(self) -> None:
        payload = {
            "schema_version": RECORD_SCHEMA_VERSION,
            "entries": [
                {
                    "role": staged.role,
                    "original_path": str(staged.original_path),
                    "staged_name": staged.staged_path.name,
                    "sha256": staged.sha256,
                    "size_bytes": staged.size_bytes,
                    "original_mtime_epoch": staged.original_mtime_epoch,
                }
                for _, staged in sorted(self.inputs.items())
            ],
        }
        self._write_json(self.layout.input_manifest, payload, allow_replace=True)

    # ------------------------------------------------------------------ stages
    def open_stage(self, stage: OnionStage) -> Path:
        """Create the directory for a stage and write its descriptor."""
        self._assert_open()
        directory = self.layout.stage_dir(stage.stage_id)
        if directory.exists():
            raise WorkspaceError(
                f"Stage directory already exists: {directory}",
                run_id=str(self.run_id),
                stage_id=str(stage.stage_id),
            )
        directory.mkdir(parents=True, exist_ok=False)
        self._write_json(self.layout.stage_json(stage.stage_id), to_jsonable(stage))
        self.events.emit(
            EventKind.STAGE_STARTED,
            f"stage {stage.stage_id} started",
            stage_id=str(stage.stage_id),
            stage_kind=stage.kind.value,
        )
        return directory

    def update_stage(self, stage: OnionStage) -> None:
        """Rewrite a stage descriptor in place (state changes only; history is in the event log)."""
        self._assert_open()
        path = self.layout.stage_json(stage.stage_id)
        if not path.exists():
            raise WorkspaceError(
                f"Stage {stage.stage_id} was never opened", stage_id=str(stage.stage_id)
            )
        self._write_json(path, to_jsonable(stage), allow_replace=True)
        self.events.emit(
            EventKind.STAGE_FINISHED,
            f"stage {stage.stage_id} -> {stage.state.value}",
            stage_id=str(stage.stage_id),
            state=stage.state.value,
        )

    def write_selection_trace(self, stage_id: StageId, trace: SelectionTrace) -> Path:
        path = self.layout.selection_trace(stage_id)
        self._write_json(path, to_jsonable(trace))
        return path

    # ------------------------------------------------------------------ branches
    def open_branch(self, branch: InitialSolutionBranch) -> Path:
        """Create the directory for one solver branch and record its plan."""
        self._assert_open()
        directory = self.layout.branch_dir(branch.stage_id, branch.branch_id)
        if directory.exists():
            raise WorkspaceError(
                f"Branch directory already exists: {directory}",
                run_id=str(self.run_id),
                branch_id=str(branch.branch_id),
            )
        (directory / "outputs").mkdir(parents=True, exist_ok=False)
        self._write_json(
            self.layout.branch_json(branch.stage_id, branch.branch_id), to_jsonable(branch)
        )
        self.events.emit(
            EventKind.BRANCH_NOMINATED,
            f"branch {branch.branch_id} ({branch.kind.value}) planned",
            stage_id=str(branch.stage_id),
            branch_id=str(branch.branch_id),
            solver=branch.kind.value,
            priority=branch.priority,
        )
        return directory

    def record_branch_state(self, branch: InitialSolutionBranch) -> None:
        self._assert_open()
        path = self.layout.branch_json(branch.stage_id, branch.branch_id)
        self._write_json(path, to_jsonable(branch), allow_replace=True)
        self.events.emit(
            EventKind.BRANCH_STATE_CHANGED,
            f"branch {branch.branch_id} -> {branch.state.value}",
            branch_id=str(branch.branch_id),
            state=branch.state.value,
        )

    def record_branch_outcome(
        self, stage_id: StageId, branch_id: BranchId, outcome: BranchOutcome
    ) -> Path:
        """Persist a branch result, including a failure. Failed branches are evidence, not noise."""
        self._assert_open()
        path = self.layout.branch_outcome_json(stage_id, branch_id)
        self._write_json(path, to_jsonable(outcome))
        return path

    # ------------------------------------------------------------------ candidates
    def add_candidate(self, candidate: Candidate) -> Path:
        """Register a candidate, create its directory, and rewrite the lineage index."""
        self._assert_open()
        self.lineage.add(candidate)
        directory = self.layout.candidate_dir(candidate.candidate_id)
        directory.mkdir(parents=True, exist_ok=False)
        self._write_json(
            self.layout.candidate_json(candidate.candidate_id), to_jsonable(candidate)
        )
        self._write_candidate_index()
        self.events.emit(
            EventKind.CANDIDATE_CREATED,
            f"candidate {candidate.candidate_id} created",
            candidate_id=str(candidate.candidate_id),
            parent_id=str(candidate.parent_id) if candidate.parent_id else None,
            mutations=[m.kind.value for m in candidate.mutations],
        )
        return directory

    def update_candidate(self, candidate: Candidate) -> None:
        self._assert_open()
        previous = self.lineage.get(candidate.candidate_id)
        self.lineage.replace(candidate)
        self._write_json(
            self.layout.candidate_json(candidate.candidate_id),
            to_jsonable(candidate),
            allow_replace=True,
        )
        self._write_candidate_index()
        if previous.status is not candidate.status:
            self.events.emit(
                EventKind.CANDIDATE_STATUS_CHANGED,
                f"candidate {candidate.candidate_id}: "
                f"{previous.status.value} -> {candidate.status.value}",
                candidate_id=str(candidate.candidate_id),
                previous=previous.status.value,
                current=candidate.status.value,
                reason=candidate.rejection_reason or None,
            )

    def next_candidate_index(self) -> int:
        return self.lineage.next_index()

    def _write_candidate_index(self) -> None:
        payload = {
            "schema_version": RECORD_SCHEMA_VERSION,
            "count": len(self.lineage),
            "candidates": [to_jsonable(c) for c in self.lineage.iter_all()],
        }
        self._write_json(self.layout.candidate_index, payload, allow_replace=True)

    # ------------------------------------------------------------------ artefacts
    def artefact_path(self, *parts: str) -> Path:
        """Resolve a path inside the run, refusing anything that escapes the workspace."""
        candidate = self.layout.root.joinpath(*parts)
        root = self.layout.root.resolve()
        try:
            resolved = candidate.resolve()
        except OSError as exc:  # pragma: no cover - platform dependent
            raise WorkspaceError(f"Cannot resolve artefact path {candidate}") from exc
        if root != resolved and root not in resolved.parents:
            raise WorkspaceError(
                f"Refusing to write outside the run workspace: {candidate}",
                run_id=str(self.run_id),
            )
        return candidate

    def write_text(self, relative: str, text: str, *, allow_replace: bool = False) -> Path:
        self._assert_open()
        path = self.artefact_path(relative)
        self._guard_overwrite(path, allow_replace)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def write_json(self, relative: str, value: Any, *, allow_replace: bool = False) -> Path:
        self._assert_open()
        path = self.artefact_path(relative)
        self._write_json(path, to_jsonable(value), allow_replace=allow_replace)
        return path

    # ------------------------------------------------------------------ finalise
    def finalize(self, result: RunResult) -> Path:
        """Write ``result.json`` and close the run. Idempotence is not offered on purpose."""
        self._assert_open()
        path = self.layout.result_json
        self._write_json(path, to_jsonable(result))
        self.events.emit(
            EventKind.RUN_FINISHED,
            f"run {self.run_id} finished: {result.stop_reason.kind.value}",
            run_id=str(self.run_id),
            stop_reason=result.stop_reason.kind.value,
            ranked=len(result.ranked),
        )
        object.__setattr__(self, "_finalized", True)
        return path

    @property
    def is_finalized(self) -> bool:
        return self._finalized

    # ------------------------------------------------------------------ internals
    def _assert_open(self) -> None:
        if self._finalized:
            raise WorkspaceError(
                f"Run {self.run_id} is finalized and cannot be modified", run_id=str(self.run_id)
            )

    def _guard_overwrite(self, path: Path, allow_replace: bool) -> None:
        if path.exists() and not allow_replace:
            raise WorkspaceError(
                f"Refusing to overwrite existing artefact: {path}. "
                "Create a new iteration instead.",
                run_id=str(self.run_id),
                evidence_paths=[path],
            )

    def _write_json(self, path: Path, payload: Any, *, allow_replace: bool = False) -> None:
        self._guard_overwrite(path, allow_replace)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        os.replace(tmp, path)

    def describe(self) -> str:  # pragma: no cover - human aid
        return dumps(
            {
                "run_id": str(self.run_id),
                "root": str(self.layout.root),
                "inputs": sorted(self.inputs),
                "candidates": len(self.lineage),
                "finalized": self._finalized,
            }
        )
