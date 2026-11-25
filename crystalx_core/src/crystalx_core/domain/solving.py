"""The initial-solution solver portfolio.

Initial phasing is a portfolio of branches, not a hard-coded SHELXT step. Each backend is described
by *what evidence it produces* and *when it is scientifically appropriate*, so the selector can
justify its choice from measurable properties of the user's own files.

The scientific roles are deliberately separated. In particular SHELXC, SHELXD and SHELXE are three
tools with three jobs, not one black-box command: SHELXC prepares the data and measures the
anomalous signal, SHELXD finds the substructure, SHELXE improves phases and determines the hand.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

from crystalx_core.domain.ids import BranchId, StageId

__all__ = [
    "SolverKind",
    "SolverRole",
    "BackendReadiness",
    "Applicability",
    "ApplicabilityVerdict",
    "SolverCapability",
    "BranchState",
    "BranchBudget",
    "InitialSolutionBranch",
    "BranchOutcome",
    "SelectionRuleVerdict",
    "SelectionRuleResult",
    "SelectionTrace",
    "SOLVER_CAPABILITIES",
    "capability_for",
    "VALID_BRANCH_TRANSITIONS",
]


class SolverKind(str, enum.Enum):
    """A distinct scientific route to an initial solution."""

    SHELXT = "shelxt"
    SHELXS_PATT = "shelxs_patt"
    SHELXS_TREF = "shelxs_tref"
    SHELXD_ABINITIO = "shelxd_abinitio"
    SHELXC_SHELXD = "shelxc_shelxd"
    SHELXE_DENSITY_MOD = "shelxe_density_mod"
    ANODE_VERIFY = "anode_verify"


class SolverRole(str, enum.Enum):
    """What a branch contributes to the workflow."""

    FULL_STRUCTURE = "full_structure"
    """Attempts a complete trial structure."""

    SUBSTRUCTURE = "substructure"
    """Finds heavy atoms only. Exactly what the anchor stage needs."""

    PHASE_IMPROVEMENT = "phase_improvement"
    """Improves an existing phase set and can discriminate the hand."""

    VERIFICATION = "verification"
    """Tests an existing hypothesis against an independent observable."""


class ApplicabilityVerdict(str, enum.Enum):
    APPLICABLE = "applicable"
    NOT_APPLICABLE = "not_applicable"
    """Ruled out by a hard prerequisite. Always accompanied by a stated reason."""

    INSUFFICIENT_INFORMATION = "insufficient_information"
    """We cannot tell yet. Reported honestly rather than guessed."""


@dataclass(frozen=True)
class Applicability:
    """Whether a backend can be used on this data, and how strongly it is preferred."""

    verdict: ApplicabilityVerdict
    priority: int = 0
    """Higher runs first. Only meaningful when ``verdict`` is ``APPLICABLE``."""

    reasons: tuple[str, ...] = ()
    missing_information: tuple[str, ...] = ()

    @property
    def is_applicable(self) -> bool:
        return self.verdict is ApplicabilityVerdict.APPLICABLE


@dataclass(frozen=True)
class SolverCapability:
    """Static description of a backend: its role, prerequisites and figures of merit."""

    kind: SolverKind
    role: SolverRole
    summary: str
    required_tools: tuple[str, ...]
    requires_space_group: bool = False
    requires_unmerged_friedel_pairs: bool = False
    requires_atomic_resolution: bool = False
    atomic_resolution_d_min_a: float | None = None
    benefits_from_heavy_atom: bool = False
    determines_space_group: bool = False
    figures_of_merit: tuple[str, ...] = ()
    caveat: str = ""


#: The portfolio. Every entry states the scientific role explicitly so that no branch is treated as
#: a legacy fallback merely because the upstream demo script happened to call SHELXT.
SOLVER_CAPABILITIES: dict[SolverKind, SolverCapability] = {
    SolverKind.SHELXT: SolverCapability(
        kind=SolverKind.SHELXT,
        role=SolverRole.FULL_STRUCTURE,
        summary=(
            "Dual-space recycling with automatic space-group determination within the Laue group. "
            "The right default for good-quality routine data when the space group is not confidently "
            "known."
        ),
        required_tools=("shelxt",),
        requires_space_group=False,
        determines_space_group=True,
        figures_of_merit=("R1", "CFOM", "space_group_rank"),
    ),
    SolverKind.SHELXS_PATT: SolverCapability(
        kind=SolverKind.SHELXS_PATT,
        role=SolverRole.SUBSTRUCTURE,
        summary=(
            "Patterson interpretation. The correct classical route to a dominant heavy atom when the "
            "space group is known and there is no usable anomalous signal. Not a legacy option."
        ),
        required_tools=("shelxs",),
        requires_space_group=True,
        benefits_from_heavy_atom=True,
        figures_of_merit=("patterson_peak_height",),
    ),
    SolverKind.SHELXS_TREF: SolverCapability(
        kind=SolverKind.SHELXS_TREF,
        role=SolverRole.FULL_STRUCTURE,
        summary=(
            "Classical direct methods. A diagnostic cross-check against SHELXT and a fallback when "
            "the space group is already decided."
        ),
        required_tools=("shelxs",),
        requires_space_group=True,
        figures_of_merit=("CFOM",),
    ),
    SolverKind.SHELXD_ABINITIO: SolverCapability(
        kind=SolverKind.SHELXD_ABINITIO,
        role=SolverRole.FULL_STRUCTURE,
        summary="Dual-space direct methods for ab initio solution at true atomic resolution.",
        required_tools=("shelxd",),
        requires_space_group=True,
        requires_atomic_resolution=True,
        atomic_resolution_d_min_a=1.2,
        figures_of_merit=("CC",),
    ),
    SolverKind.SHELXC_SHELXD: SolverCapability(
        kind=SolverKind.SHELXC_SHELXD,
        role=SolverRole.SUBSTRUCTURE,
        summary=(
            "SHELXC prepares the data and measures the anomalous signal against resolution; SHELXD "
            "then solves the anomalous substructure. High priority when unmerged Friedel pairs and a "
            "real anomalous scatterer are both present."
        ),
        required_tools=("shelxc", "shelxd"),
        requires_space_group=True,
        requires_unmerged_friedel_pairs=True,
        benefits_from_heavy_atom=True,
        figures_of_merit=("CC", "CCweak", "site_occupancy_falloff", "mean_d_over_sig"),
    ),
    SolverKind.SHELXE_DENSITY_MOD: SolverCapability(
        kind=SolverKind.SHELXE_DENSITY_MOD,
        role=SolverRole.PHASE_IMPROVEMENT,
        summary=(
            "Density modification and phase extension, run for both hands. The original-versus-"
            "inverted comparison is one of the strongest available tests that a substructure is real."
        ),
        required_tools=("shelxe",),
        requires_space_group=True,
        figures_of_merit=("contrast", "connectivity", "pseudo_free_cc", "hand_ratio"),
        caveat=(
            "Main-chain autotracing assumes polypeptide geometry and must stay disabled unless the "
            "user declared a genuine peptide. Density modification and hand determination are safe "
            "in both cases."
        ),
    ),
    SolverKind.ANODE_VERIFY: SolverCapability(
        kind=SolverKind.ANODE_VERIFY,
        role=SolverRole.VERIFICATION,
        summary=(
            "Anomalous difference density at each proposed site. Answers 'is that atom really the "
            "metal we think it is' with a measured observable rather than a model score."
        ),
        required_tools=("anode",),
        requires_unmerged_friedel_pairs=True,
        benefits_from_heavy_atom=True,
        figures_of_merit=("anomalous_peak_sigma",),
    ),
}


def capability_for(kind: SolverKind) -> SolverCapability:
    return SOLVER_CAPABILITIES[kind]


class BackendReadiness(str, enum.Enum):
    """How far a backend has been taken.

    A backend can be useful before it is finished: showing a crystallographer the exact command that
    *would* run is valuable even when the output parser does not exist yet. This enum makes the
    difference explicit instead of letting a half-built backend look production-ready.
    """

    PRODUCTION = "production"
    """Command shape and output parser are both implemented and tested."""

    PLAN_ONLY = "plan_only"
    """The command can be built and displayed, but the output parser is not written. The branch
    machine refuses to execute these unless explicitly allowed, so nothing can run unverified."""

    EXPERIMENTAL = "experimental"
    """Runs, but the interpretation should not be trusted without review."""


class BranchState(str, enum.Enum):
    """Lifecycle of one solver branch."""

    PLANNED = "planned"
    SKIPPED = "skipped"
    """Deliberately not run this time, e.g. lower priority than the budget allowed."""

    NOT_APPLICABLE = "not_applicable"
    """Ruled out by a hard prerequisite. Retained with its reason; never silently dropped."""

    BLOCKED_TOOL_UNAVAILABLE = "blocked_tool_unavailable"
    BLOCKED_BACKEND_PLAN_ONLY = "blocked_backend_plan_only"
    """The backend can build the command but cannot yet interpret the result."""

    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    BUDGET_EXCEEDED = "budget_exceeded"
    ABORTED = "aborted"

    @property
    def is_terminal(self) -> bool:
        return self not in (BranchState.PLANNED, BranchState.RUNNING)


#: Allowed state transitions. Anything else is a programming error and raises.
VALID_BRANCH_TRANSITIONS: dict[BranchState, frozenset[BranchState]] = {
    BranchState.PLANNED: frozenset(
        {
            BranchState.RUNNING,
            BranchState.SKIPPED,
            BranchState.NOT_APPLICABLE,
            BranchState.BLOCKED_TOOL_UNAVAILABLE,
            BranchState.BLOCKED_BACKEND_PLAN_ONLY,
            BranchState.ABORTED,
        }
    ),
    BranchState.RUNNING: frozenset(
        {
            BranchState.SUCCEEDED,
            BranchState.FAILED,
            BranchState.BUDGET_EXCEEDED,
            BranchState.ABORTED,
        }
    ),
    BranchState.SKIPPED: frozenset(),
    BranchState.NOT_APPLICABLE: frozenset(),
    BranchState.BLOCKED_TOOL_UNAVAILABLE: frozenset(),
    BranchState.BLOCKED_BACKEND_PLAN_ONLY: frozenset(),
    BranchState.SUCCEEDED: frozenset(),
    BranchState.FAILED: frozenset(),
    BranchState.BUDGET_EXCEEDED: frozenset(),
    BranchState.ABORTED: frozenset(),
}


@dataclass(frozen=True)
class BranchBudget:
    """Per-branch bounds, so one slow route cannot starve the portfolio."""

    max_wall_clock_s: float = 900.0
    max_attempts: int = 1
    solver_trials: int | None = None
    """Backend-specific trial count, e.g. SHELXD ``NTRY``."""

    def __post_init__(self) -> None:
        if self.max_wall_clock_s <= 0:
            raise ValueError("max_wall_clock_s must be positive")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")


@dataclass(frozen=True)
class InitialSolutionBranch:
    """One nominated route, with everything needed to reproduce and judge it."""

    branch_id: BranchId
    kind: SolverKind
    stage_id: StageId
    state: BranchState = BranchState.PLANNED
    priority: int = 0
    applicability: Applicability | None = None
    budget: BranchBudget = field(default_factory=BranchBudget)
    space_group_assumption: str = ""
    heavy_atom_assumption: tuple[str, ...] = ()
    expected_sites: int | None = None
    resolution_cut_a: float | None = None
    use_anomalous: bool = False
    enable_autotracing: bool = False
    """Only ever true when the user declared a peptide-like system."""

    parameters: dict[str, str] = field(default_factory=dict)
    nominated_because: tuple[str, ...] = ()

    def transition(self, new_state: BranchState) -> "InitialSolutionBranch":
        """Return a copy in ``new_state``, rejecting an illegal transition."""
        allowed = VALID_BRANCH_TRANSITIONS[self.state]
        if new_state not in allowed:
            raise ValueError(
                f"Illegal branch transition {self.state.value} -> {new_state.value} "
                f"for {self.branch_id}; allowed: {sorted(s.value for s in allowed) or 'none'}"
            )
        return InitialSolutionBranch(
            branch_id=self.branch_id,
            kind=self.kind,
            stage_id=self.stage_id,
            state=new_state,
            priority=self.priority,
            applicability=self.applicability,
            budget=self.budget,
            space_group_assumption=self.space_group_assumption,
            heavy_atom_assumption=self.heavy_atom_assumption,
            expected_sites=self.expected_sites,
            resolution_cut_a=self.resolution_cut_a,
            use_anomalous=self.use_anomalous,
            enable_autotracing=self.enable_autotracing,
            parameters=dict(self.parameters),
            nominated_because=self.nominated_because,
        )


@dataclass(frozen=True)
class BranchOutcome:
    """What a branch produced, including when it produced nothing.

    A failed branch is evidence about the data, not noise. It is kept.
    """

    branch_id: BranchId
    state: BranchState
    summary: str = ""
    figures_of_merit: dict[str, float] = field(default_factory=dict)
    produced_paths: tuple[str, ...] = ()
    hypothesis_ids: tuple[str, ...] = ()
    failure_reason: str = ""
    diagnostic_code: str = ""
    wall_clock_s: float | None = None


class SelectionRuleVerdict(str, enum.Enum):
    PASS = "pass"
    FAIL = "fail"
    INSUFFICIENT_INFORMATION = "insufficient_information"
    NOT_EVALUATED = "not_evaluated"


@dataclass(frozen=True)
class SelectionRuleResult:
    """One named rule, its inputs, and what it concluded."""

    rule_id: str
    description: str
    verdict: SelectionRuleVerdict
    detail: str = ""
    inputs: dict[str, str] = field(default_factory=dict)
    affects: tuple[SolverKind, ...] = ()


@dataclass(frozen=True)
class SelectionTrace:
    """The full, auditable record of why these branches were chosen.

    Written to ``stages/s1_initial_solution/selection_trace.json``. Nothing about the choice of
    solver is implicit, and the crystallographer is never asked to make it.
    """

    rules: tuple[SelectionRuleResult, ...]
    nominated: tuple[InitialSolutionBranch, ...]
    excluded: tuple[tuple[SolverKind, str], ...] = ()
    notes: tuple[str, ...] = ()

    def rule(self, rule_id: str) -> SelectionRuleResult | None:
        for result in self.rules:
            if result.rule_id == rule_id:
                return result
        return None

    @property
    def nominated_kinds(self) -> tuple[SolverKind, ...]:
        return tuple(branch.kind for branch in self.nominated)
