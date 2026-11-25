"""Run outcomes.

"No trustworthy candidate" is a first-class, expected result -- not a crash and not a silent pick of
the least-bad model. It carries the reason and the required next action so the workflow can either
run another onion stage or hand over to a human crystallographer.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

from crystalx_core.domain.gates import EligibilityReport
from crystalx_core.domain.ids import CandidateId, RunId, StageId

__all__ = [
    "StopReasonKind",
    "StopReason",
    "ReviewCheckpointKind",
    "ReviewCheckpoint",
    "RankedCandidate",
    "NoTrustworthyCandidate",
    "RunResult",
]


class StopReasonKind(str, enum.Enum):
    CONVERGED = "converged"
    """A candidate passed every gate and the workflow is satisfied."""

    REVIEW_CHECKPOINT = "review_checkpoint"
    """Progress was made but a human must decide before continuing."""

    NO_TRUSTWORTHY_CANDIDATE = "no_trustworthy_candidate"
    """Everything tried so far failed a hard gate. Another onion stage may still succeed."""

    ALL_BRANCHES_EXHAUSTED = "all_branches_exhausted"
    BUDGET_EXCEEDED = "budget_exceeded"
    TOOL_UNAVAILABLE = "tool_unavailable"
    INPUT_INVALID = "input_invalid"
    DATA_QUALITY_INSUFFICIENT = "data_quality_insufficient"
    USER_ABORTED = "user_aborted"
    INTERNAL_ERROR = "internal_error"

    @property
    def is_success(self) -> bool:
        return self is StopReasonKind.CONVERGED

    @property
    def is_resumable(self) -> bool:
        """True when running another stage could plausibly change the answer."""
        return self in (
            StopReasonKind.REVIEW_CHECKPOINT,
            StopReasonKind.NO_TRUSTWORTHY_CANDIDATE,
            StopReasonKind.BUDGET_EXCEEDED,
        )


@dataclass(frozen=True)
class StopReason:
    kind: StopReasonKind
    message: str
    stage_id: StageId | None = None
    evidence_paths: tuple[str, ...] = ()
    suggested_next_action: str = ""


class ReviewCheckpointKind(str, enum.Enum):
    SPACE_GROUP_AMBIGUOUS = "space_group_ambiguous"
    COMPETING_HYPOTHESES = "competing_hypotheses"
    """Two or more heavy-atom hypotheses are equally supported. A human should choose."""

    SEVERE_VALIDATION_ALERT = "severe_validation_alert"
    SUSPECTED_DISORDER = "suspected_disorder"
    """Recorded, not solved. This project does not claim automatic disorder resolution."""

    SUSPECTED_TWINNING = "suspected_twinning"
    RESIDUAL_DENSITY_UNEXPLAINED = "residual_density_unexplained"
    CHARGE_IMBALANCE = "charge_imbalance"
    COORDINATION_IMPLAUSIBLE = "coordination_implausible"
    DATA_QUALITY = "data_quality"


@dataclass(frozen=True)
class ReviewCheckpoint:
    """A point where the workflow deliberately stops and asks a human."""

    kind: ReviewCheckpointKind
    question: str
    stage_id: StageId | None = None
    candidates: tuple[CandidateId, ...] = ()
    evidence_paths: tuple[str, ...] = ()
    options: tuple[str, ...] = ()
    recommendation: str = ""
    blocking: bool = True


@dataclass(frozen=True)
class RankedCandidate:
    """An eligible candidate with its ordering key spelled out.

    ``sort_key`` is stored so a reader can reproduce the ordering by hand and see exactly which
    number decided it.
    """

    candidate_id: CandidateId
    rank: int
    sort_key: tuple[float, ...]
    key_names: tuple[str, ...]
    eligibility: EligibilityReport
    summary: str = ""


@dataclass(frozen=True)
class NoTrustworthyCandidate:
    """Why nothing is acceptable yet, and what to do about it."""

    reason: str
    ineligible: tuple[EligibilityReport, ...] = ()
    suggested_next_stage: str = ""
    requires_human_expert: bool = False


@dataclass(frozen=True)
class RunResult:
    """The end state of a run."""

    run_id: RunId
    stop_reason: StopReason
    ranked: tuple[RankedCandidate, ...] = ()
    ineligible: tuple[EligibilityReport, ...] = ()
    review_checkpoints: tuple[ReviewCheckpoint, ...] = ()
    no_trustworthy_candidate: NoTrustworthyCandidate | None = None
    stages_completed: tuple[StageId, ...] = ()
    notes: tuple[str, ...] = ()
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def best(self) -> RankedCandidate | None:
        return self.ranked[0] if self.ranked else None

    @property
    def succeeded(self) -> bool:
        return self.stop_reason.kind.is_success and bool(self.ranked)
