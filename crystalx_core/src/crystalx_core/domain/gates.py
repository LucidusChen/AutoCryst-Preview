"""Scientific eligibility gates.

Gates come before ranking and are not part of it. A candidate that fails a gate is ``INELIGIBLE``
and cannot be ordered against anything, however good its R factor looks. This is the mechanism that
stops a numerically attractive but scientifically wrong model from winning.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

from crystalx_core.domain.ids import CandidateId

__all__ = [
    "GateKind",
    "GateOutcome",
    "GateResult",
    "EligibilityReport",
    "RankingPolicy",
    "DEFAULT_RANKING_POLICY",
]


class GateKind(str, enum.Enum):
    """The hard checks, in evaluation order."""

    G1_PARSE_OK = "g1_parse_ok"
    """Every required output was parsed. An unparsed refinement is not a result."""

    G2_REFINEMENT_CONVERGED = "g2_refinement_converged"
    G3_SAME_REFLECTION_BASIS = "g3_same_reflection_basis"
    """Comparability is a gate, not a footnote: different reflection file, resolution cut, HKLF
    format or twin law means the candidates are simply not comparable."""

    G4_CHEMICAL_VALIDITY = "g4_chemical_validity"
    G5_COORDINATION_PLAUSIBLE = "g5_coordination_plausible"
    G6_DATA_PARAMETER_RATIO = "g6_data_parameter_ratio"
    G7_NO_UNRESOLVED_SEVERE_ALERT = "g7_no_unresolved_severe_alert"
    """Unresolved PLATON A/B alerts route the run to human review; they are never averaged away."""


class GateOutcome(str, enum.Enum):
    PASS = "pass"
    FAIL = "fail"
    NOT_EVALUATED = "not_evaluated"
    """The information needed was absent. Treated as blocking unless the policy waives the gate."""

    WAIVED = "waived"
    """Explicitly waived by the user, with a recorded reason."""

    @property
    def blocks_ranking(self) -> bool:
        return self in (GateOutcome.FAIL, GateOutcome.NOT_EVALUATED)


@dataclass(frozen=True)
class GateResult:
    kind: GateKind
    outcome: GateOutcome
    message: str
    observed: str = ""
    expected: str = ""
    waiver_reason: str = ""
    routes_to_review: bool = False
    """True when failing this gate means "a human should look", rather than "this is wrong"."""


@dataclass(frozen=True)
class EligibilityReport:
    """The gate verdict for one candidate."""

    candidate_id: CandidateId
    results: tuple[GateResult, ...] = ()

    @property
    def failures(self) -> tuple[GateResult, ...]:
        return tuple(r for r in self.results if r.outcome.blocks_ranking)

    @property
    def is_eligible(self) -> bool:
        return not self.failures

    @property
    def requires_review(self) -> bool:
        return any(r.routes_to_review for r in self.failures)

    def result(self, kind: GateKind) -> GateResult | None:
        for result in self.results:
            if result.kind is kind:
                return result
        return None

    def summary(self) -> str:
        if self.is_eligible:
            return "eligible"
        return "ineligible: " + "; ".join(f"{r.kind.value}={r.message}" for r in self.failures)


@dataclass(frozen=True)
class RankingPolicy:
    """How eligible candidates are ordered, and what is merely recorded.

    ``ordering_keys`` is the whole ordering. There is no weighted sum, and model confidence is
    deliberately absent: it may order the *search* but never the *result*.
    """

    ordering_keys: tuple[str, ...] = (
        "r1",
        "wr2",
        "goof_deviation",
        "max_residual_away_from_heavy",
    )
    min_data_to_parameter_ratio: float = 10.0
    max_shift_over_esd: float = 0.02
    residual_peak_threshold_e_per_a3: float = 1.0
    severe_alert_routes_to_review: bool = True
    charge_imbalance_is_fatal: bool = True
    """Owner-configurable. Default is a hard gate because for a non-expert an unbalanced formula is
    almost always a real omission."""

    require_same_reflection_basis: bool = True
    recorded_but_not_ranked: tuple[str, ...] = (
        "n_restraints",
        "n_constraints",
        "data_to_parameter_ratio",
        "formula",
        "crystalx_confidence",
    )
    waived_gates: dict[str, str] = field(default_factory=dict)
    """``gate kind value -> reason``. A waiver is always recorded with its reason."""


DEFAULT_RANKING_POLICY = RankingPolicy()
