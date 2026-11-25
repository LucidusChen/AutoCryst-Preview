"""The peeling-onion stage model.

Stages are bounded, ordered, and reversible. A stage that cannot succeed rolls back to a named
earlier stage on a different branch; the abandoned path is preserved rather than deleted, because
"this route failed" is evidence about the data.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field

from crystalx_core.domain.ids import BranchId, CandidateId, StageId

__all__ = [
    "StageKind",
    "StageState",
    "StageBudget",
    "StagePolicy",
    "OnionStage",
    "RollbackDecision",
    "STAGE_ORDER",
    "next_stage_kind",
]


class StageKind(str, enum.Enum):
    """The five stages of the peeling-onion workflow."""

    DIAGNOSE = "diagnose"
    """S0. Characterise data and chemistry before touching a solver."""

    INITIAL_SOLUTION = "initial_solution"
    """S1. Run the selected solver portfolio. The goal is an unambiguous heavy-atom substructure,
    not a finished structure."""

    ANCHOR = "anchor"
    """S2. Validate the metal nodes and their immediate coordination. Keep alternatives."""

    FIRST_SHELL = "first_shell"
    """S3. Donor atoms and the directly bonded organic fragments."""

    SHELL_EXPANSION = "shell_expansion"
    """S4. Repeats, indexed by shell number, until the skeleton is complete or the budget runs out."""

    FINALIZE = "finalize"
    """S5. Full refinement, residual-density review, validation, candidate comparison."""


#: Canonical forward order. ``SHELL_EXPANSION`` may repeat with an increasing shell index.
STAGE_ORDER: tuple[StageKind, ...] = (
    StageKind.DIAGNOSE,
    StageKind.INITIAL_SOLUTION,
    StageKind.ANCHOR,
    StageKind.FIRST_SHELL,
    StageKind.SHELL_EXPANSION,
    StageKind.FINALIZE,
)


def next_stage_kind(current: StageKind) -> StageKind | None:
    """The stage that normally follows ``current``, or ``None`` after ``FINALIZE``."""
    index = STAGE_ORDER.index(current)
    if index + 1 >= len(STAGE_ORDER):
        return None
    return STAGE_ORDER[index + 1]


class StageState(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    ROLLED_BACK = "rolled_back"
    ABANDONED = "abandoned"
    BUDGET_EXCEEDED = "budget_exceeded"
    REVIEW_REQUIRED = "review_required"


@dataclass(frozen=True)
class StageBudget:
    """Hard bounds. A stage that hits one stops; it does not quietly keep going.

    ``max_wall_clock_s`` of ``None`` means unbounded, which is only appropriate for stages that do
    no external work.
    """

    max_attempts: int = 3
    max_candidates: int = 8
    max_wall_clock_s: float | None = 1800.0
    max_new_atoms: int | None = None
    """Cap on how many atoms one shell may add. Prevents a single round from swallowing the
    structure and skipping the intermediate evidence."""

    max_tool_calls: int | None = None
    max_failures: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.max_attempts, int) or isinstance(self.max_attempts, bool) or self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if not isinstance(self.max_candidates, int) or isinstance(self.max_candidates, bool) or self.max_candidates < 1:
            raise ValueError("max_candidates must be >= 1")
        if self.max_wall_clock_s is not None and (
            not math.isfinite(self.max_wall_clock_s) or self.max_wall_clock_s <= 0
        ):
            raise ValueError("max_wall_clock_s must be positive or None")
        for name in ("max_new_atoms", "max_tool_calls", "max_failures"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 0):
                raise ValueError(f"{name} must be a nonnegative integer or None")


@dataclass(frozen=True)
class StagePolicy:
    """What this stage is allowed to do."""

    allowed_elements: tuple[str, ...] = ()
    """Empty means "anything in the declared SFAC"."""

    max_distance_from_model_a: float | None = None
    """New atoms must lie within this distance of the existing model. Keeps a shell a shell."""

    min_distance_between_atoms_a: float = 0.8
    require_experimental_evidence: bool = True
    """When true, a new atom needs support beyond model inference before it is accepted."""

    allow_shelxe_autotracing: bool = False
    """Off unless ``ChemicalPrior.is_peptide_like`` is set. SHELXE main-chain tracing assumes
    polypeptide geometry and misleads badly on metal-organic small molecules."""

    notes: str = ""


@dataclass(frozen=True)
class OnionStage:
    """One stage instance in a run."""

    stage_id: StageId
    kind: StageKind
    index: int
    shell: int | None = None
    state: StageState = StageState.PENDING
    budget: StageBudget = field(default_factory=StageBudget)
    policy: StagePolicy = field(default_factory=StagePolicy)
    parent_stage_id: StageId | None = None
    input_candidate: CandidateId | None = None
    produced_candidates: tuple[CandidateId, ...] = ()
    branches: tuple[BranchId, ...] = ()
    summary: str = ""

    def with_state(self, state: StageState, *, summary: str = "") -> "OnionStage":
        """Return a copy in a new state. Stages are immutable; history is kept by replacement."""
        return OnionStage(
            stage_id=self.stage_id,
            kind=self.kind,
            index=self.index,
            shell=self.shell,
            state=state,
            budget=self.budget,
            policy=self.policy,
            parent_stage_id=self.parent_stage_id,
            input_candidate=self.input_candidate,
            produced_candidates=self.produced_candidates,
            branches=self.branches,
            summary=summary or self.summary,
        )


@dataclass(frozen=True)
class RollbackDecision:
    """A recorded retreat to an earlier stage.

    The abandoned branches and candidates are named and kept. Rollback is a normal part of solving a
    difficult structure, not an error, and the record must make the path visible afterwards.
    """

    from_stage: StageId
    to_stage: StageId
    reason: str
    abandoned_branches: tuple[BranchId, ...] = ()
    abandoned_candidates: tuple[CandidateId, ...] = ()
    alternative_selected: BranchId | None = None
    requires_human_review: bool = False

    def __post_init__(self) -> None:
        if not self.reason.strip():
            raise ValueError("A rollback must state a reason")
        if self.from_stage == self.to_stage:
            raise ValueError("A rollback must move to a different stage")
