"""On-disk layout of a run workspace.

Directory names are meant to be read by a human in a file browser. A crystallographer looking at
``stages/s4-shell_expansion-02/branches/b01-shelxs_patt/`` should know where they are without
consulting documentation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from crystalx_core.domain.ids import BranchId, CandidateId, RunId, StageId

__all__ = ["RunLayout", "RUN_JSON", "RESULT_JSON", "EVENTS_NDJSON", "INPUT_MANIFEST"]

RUN_JSON = "run.json"
RESULT_JSON = "result.json"
EVENTS_NDJSON = "events.ndjson"
INPUT_MANIFEST = "manifest.json"
CANDIDATE_INDEX = "index.json"
SELECTION_TRACE = "selection_trace.json"
OUTPUT_MANIFEST = "manifest.json"


@dataclass(frozen=True)
class RunLayout:
    """Path arithmetic for one run. Pure: creates nothing."""

    root: Path

    @classmethod
    def for_run(cls, workspace_root: Path | str, run_id: RunId, slug: str) -> "RunLayout":
        """``<workspace_root>/<run_id>_<slug>``.

        The run id leads so that lexicographic sorting of the runs directory is chronological.
        """
        return cls(Path(workspace_root) / f"{run_id}_{slug}")

    # -- top level ------------------------------------------------------------------
    @property
    def run_json(self) -> Path:
        return self.root / RUN_JSON

    @property
    def result_json(self) -> Path:
        return self.root / RESULT_JSON

    @property
    def events(self) -> Path:
        return self.root / EVENTS_NDJSON

    @property
    def inputs_dir(self) -> Path:
        return self.root / "inputs"

    @property
    def input_manifest(self) -> Path:
        return self.inputs_dir / INPUT_MANIFEST

    @property
    def stages_dir(self) -> Path:
        return self.root / "stages"

    @property
    def candidates_dir(self) -> Path:
        return self.root / "candidates"

    @property
    def candidate_index(self) -> Path:
        return self.candidates_dir / CANDIDATE_INDEX

    # -- stages ---------------------------------------------------------------------
    def stage_dir(self, stage_id: StageId) -> Path:
        return self.stages_dir / str(stage_id)

    def stage_json(self, stage_id: StageId) -> Path:
        return self.stage_dir(stage_id) / "stage.json"

    def selection_trace(self, stage_id: StageId) -> Path:
        return self.stage_dir(stage_id) / SELECTION_TRACE

    # -- branches -------------------------------------------------------------------
    def branches_dir(self, stage_id: StageId) -> Path:
        return self.stage_dir(stage_id) / "branches"

    def branch_dir(self, stage_id: StageId, branch_id: BranchId) -> Path:
        return self.branches_dir(stage_id) / str(branch_id)

    def branch_json(self, stage_id: StageId, branch_id: BranchId) -> Path:
        return self.branch_dir(stage_id, branch_id) / "branch.json"

    def branch_outcome_json(self, stage_id: StageId, branch_id: BranchId) -> Path:
        return self.branch_dir(stage_id, branch_id) / "outcome.json"

    def branch_outputs_dir(self, stage_id: StageId, branch_id: BranchId) -> Path:
        return self.branch_dir(stage_id, branch_id) / "outputs"

    def branch_output_manifest(self, stage_id: StageId, branch_id: BranchId) -> Path:
        return self.branch_outputs_dir(stage_id, branch_id) / OUTPUT_MANIFEST

    # -- candidates -----------------------------------------------------------------
    def candidate_dir(self, candidate_id: CandidateId) -> Path:
        return self.candidates_dir / str(candidate_id)

    def candidate_json(self, candidate_id: CandidateId) -> Path:
        return self.candidate_dir(candidate_id) / "candidate.json"
