"""Evaluating the hard scientific gates.

Gates decide *eligibility*. They are not scores and they are not weighted. A candidate that fails one
cannot be ranked at all, however good its R factor is, and a severe validation alert routes the run
to human review instead of being averaged away.

Missing information counts as blocking (``NOT_EVALUATED``), not as a pass. A refinement whose output
could not be parsed is not a result.
"""

from __future__ import annotations

from crystalx_core.domain.candidate import Candidate
from crystalx_core.domain.gates import (
    EligibilityReport,
    GateKind,
    GateOutcome,
    GateResult,
    RankingPolicy,
)
from crystalx_core.domain.metrics import GeometryIssueKind

__all__ = ["evaluate_gates", "evaluate_all"]


def evaluate_gates(
    candidate: Candidate,
    policy: RankingPolicy,
    *,
    reference_basis_key: str | None = None,
    heavy_elements: frozenset[str] = frozenset(),
) -> EligibilityReport:
    """Run every gate against one candidate.

    ``reference_basis_key`` is the reflection basis the candidate is being compared *against*. When
    it is ``None`` the comparability gate is skipped, which is correct for a single candidate that is
    not yet being compared with anything.
    """
    results: list[GateResult] = [
        _g1_parse(candidate, policy),
        _g2_converged(candidate, policy),
        _g3_basis(candidate, policy, reference_basis_key),
        _g4_chemistry(candidate, policy),
        _g5_coordination(candidate, policy),
        _g6_ratio(candidate, policy),
        _g7_alerts(candidate, policy),
    ]
    return EligibilityReport(candidate_id=candidate.candidate_id, results=tuple(_apply_waivers(results, policy)))


def evaluate_all(
    candidates: tuple[Candidate, ...],
    policy: RankingPolicy,
    *,
    heavy_elements: frozenset[str] = frozenset(),
) -> tuple[EligibilityReport, ...]:
    """Evaluate a set of candidates against a common reflection basis.

    The basis is taken from the first candidate that has one; every other candidate is then checked
    against it, so a model refined on a different resolution cut is caught rather than compared.
    """
    reference_key: str | None = None
    for candidate in candidates:
        key = _basis_key(candidate)
        if key:
            reference_key = key
            break
    return tuple(
        evaluate_gates(
            candidate,
            policy,
            reference_basis_key=reference_key,
            heavy_elements=heavy_elements,
        )
        for candidate in candidates
    )


def _apply_waivers(results: list[GateResult], policy: RankingPolicy) -> list[GateResult]:
    if not policy.waived_gates:
        return results
    adjusted: list[GateResult] = []
    for result in results:
        reason = policy.waived_gates.get(result.kind.value)
        if reason and result.outcome.blocks_ranking:
            adjusted.append(
                GateResult(
                    kind=result.kind,
                    outcome=GateOutcome.WAIVED,
                    message=f"waived: {reason} (original: {result.message})",
                    observed=result.observed,
                    expected=result.expected,
                    waiver_reason=reason,
                )
            )
        else:
            adjusted.append(result)
    return adjusted


def _basis_key(candidate: Candidate) -> str:
    if candidate.metrics and candidate.metrics.reflection_basis_key:
        return candidate.metrics.reflection_basis_key
    return candidate.provenance.reflection_basis_key()


def _g1_parse(candidate: Candidate, _policy: RankingPolicy) -> GateResult:
    metrics = candidate.metrics
    if metrics is None:
        return GateResult(
            kind=GateKind.G1_PARSE_OK,
            outcome=GateOutcome.NOT_EVALUATED,
            message="no refinement metrics are attached to this candidate",
        )
    if not metrics.parse_ok:
        return GateResult(
            kind=GateKind.G1_PARSE_OK,
            outcome=GateOutcome.FAIL,
            message=metrics.parse_failure_reason or "refinement output could not be parsed",
        )
    return GateResult(
        kind=GateKind.G1_PARSE_OK,
        outcome=GateOutcome.PASS,
        message="refinement output parsed",
    )


def _g2_converged(candidate: Candidate, policy: RankingPolicy) -> GateResult:
    metrics = candidate.metrics
    if metrics is None:
        return GateResult(
            kind=GateKind.G2_REFINEMENT_CONVERGED,
            outcome=GateOutcome.NOT_EVALUATED,
            message="no refinement has been run for this candidate",
        )
    if metrics.converged is False:
        return GateResult(
            kind=GateKind.G2_REFINEMENT_CONVERGED,
            outcome=GateOutcome.FAIL,
            message="the refinement did not converge",
        )
    shift = metrics.max_shift_over_esd
    if shift is not None and shift > policy.max_shift_over_esd:
        return GateResult(
            kind=GateKind.G2_REFINEMENT_CONVERGED,
            outcome=GateOutcome.FAIL,
            message=f"maximum shift/esd {shift:.3f} exceeds {policy.max_shift_over_esd:.3f}",
            observed=f"{shift:.3f}",
            expected=f"<= {policy.max_shift_over_esd:.3f}",
        )
    if metrics.converged is None and shift is None:
        return GateResult(
            kind=GateKind.G2_REFINEMENT_CONVERGED,
            outcome=GateOutcome.NOT_EVALUATED,
            message="convergence could not be determined from the refinement output",
        )
    return GateResult(
        kind=GateKind.G2_REFINEMENT_CONVERGED,
        outcome=GateOutcome.PASS,
        message="refinement converged",
    )


def _g3_basis(
    candidate: Candidate, policy: RankingPolicy, reference_basis_key: str | None
) -> GateResult:
    if not policy.require_same_reflection_basis or reference_basis_key is None:
        return GateResult(
            kind=GateKind.G3_SAME_REFLECTION_BASIS,
            outcome=GateOutcome.PASS,
            message="no comparison basis required",
        )
    own = _basis_key(candidate)
    if own == reference_basis_key:
        return GateResult(
            kind=GateKind.G3_SAME_REFLECTION_BASIS,
            outcome=GateOutcome.PASS,
            message="same reflection file, resolution cut, HKLF format and twin law",
            observed=own,
        )
    return GateResult(
        kind=GateKind.G3_SAME_REFLECTION_BASIS,
        outcome=GateOutcome.FAIL,
        message=(
            "this candidate was refined against a different reflection basis, so its R factors "
            "are not comparable with the others"
        ),
        observed=own,
        expected=reference_basis_key,
        routes_to_review=True,
    )


def _g4_chemistry(candidate: Candidate, policy: RankingPolicy) -> GateResult:
    geometry = candidate.geometry
    if geometry is None:
        return GateResult(
            kind=GateKind.G4_CHEMICAL_VALIDITY,
            outcome=GateOutcome.NOT_EVALUATED,
            message="no geometry report is attached to this candidate",
        )
    if not geometry.evaluated:
        return GateResult(
            kind=GateKind.G4_CHEMICAL_VALIDITY,
            outcome=GateOutcome.NOT_EVALUATED,
            message=geometry.not_evaluated_reason or "geometry was not evaluated",
        )
    fatal = [
        issue
        for issue in geometry.fatal_issues
        if issue.kind is not GeometryIssueKind.CHARGE_IMBALANCE
        or policy.charge_imbalance_is_fatal
    ]
    if fatal:
        return GateResult(
            kind=GateKind.G4_CHEMICAL_VALIDITY,
            outcome=GateOutcome.FAIL,
            message="; ".join(issue.message for issue in fatal[:4]),
            observed=", ".join(issue.kind.value for issue in fatal),
        )
    return GateResult(
        kind=GateKind.G4_CHEMICAL_VALIDITY,
        outcome=GateOutcome.PASS,
        message="no fatal chemical or geometric problems",
    )


def _g5_coordination(candidate: Candidate, _policy: RankingPolicy) -> GateResult:
    geometry = candidate.geometry
    if geometry is None or not geometry.evaluated:
        return GateResult(
            kind=GateKind.G5_COORDINATION_PLAUSIBLE,
            outcome=GateOutcome.NOT_EVALUATED,
            message="coordination was not assessed",
        )
    offenders = [
        issue
        for issue in geometry.issues
        if issue.kind
        in (
            GeometryIssueKind.IMPLAUSIBLE_COORDINATION_NUMBER,
            GeometryIssueKind.METAL_DONOR_DISTANCE_OUT_OF_RANGE,
        )
        and issue.is_fatal
    ]
    if offenders:
        return GateResult(
            kind=GateKind.G5_COORDINATION_PLAUSIBLE,
            outcome=GateOutcome.FAIL,
            message="; ".join(issue.message for issue in offenders[:4]),
            routes_to_review=True,
        )
    return GateResult(
        kind=GateKind.G5_COORDINATION_PLAUSIBLE,
        outcome=GateOutcome.PASS,
        message="metal coordination is within the declared expectations",
    )


def _g6_ratio(candidate: Candidate, policy: RankingPolicy) -> GateResult:
    metrics = candidate.metrics
    ratio = metrics.data_to_parameter_ratio if metrics else None
    if ratio is None:
        return GateResult(
            kind=GateKind.G6_DATA_PARAMETER_RATIO,
            outcome=GateOutcome.NOT_EVALUATED,
            message="reflection and parameter counts are not both available",
        )
    if ratio < policy.min_data_to_parameter_ratio:
        return GateResult(
            kind=GateKind.G6_DATA_PARAMETER_RATIO,
            outcome=GateOutcome.FAIL,
            message=(
                f"data/parameter ratio {ratio:.1f} is below the declared minimum "
                f"{policy.min_data_to_parameter_ratio:.1f}; the model is over-parameterised"
            ),
            observed=f"{ratio:.2f}",
            expected=f">= {policy.min_data_to_parameter_ratio:.1f}",
        )
    return GateResult(
        kind=GateKind.G6_DATA_PARAMETER_RATIO,
        outcome=GateOutcome.PASS,
        message=f"data/parameter ratio {ratio:.1f}",
        observed=f"{ratio:.2f}",
    )


def _g7_alerts(candidate: Candidate, policy: RankingPolicy) -> GateResult:
    validation = candidate.validation
    if validation is None:
        return GateResult(
            kind=GateKind.G7_NO_UNRESOLVED_SEVERE_ALERT,
            outcome=GateOutcome.NOT_EVALUATED,
            message="no validation has been run for this candidate",
        )
    if not validation.completed:
        return GateResult(
            kind=GateKind.G7_NO_UNRESOLVED_SEVERE_ALERT,
            outcome=GateOutcome.NOT_EVALUATED,
            message=validation.failure_reason or "validation did not complete",
        )
    severe = validation.unresolved_severe
    if severe:
        listed = ", ".join(f"{a.severity.value}:{a.code}" for a in severe[:5])
        return GateResult(
            kind=GateKind.G7_NO_UNRESOLVED_SEVERE_ALERT,
            outcome=GateOutcome.FAIL,
            message=(
                f"{len(severe)} unresolved level A/B alert(s): {listed}. "
                "A severe alert is a review gate, not a score term."
            ),
            observed=listed,
            routes_to_review=policy.severe_alert_routes_to_review,
        )
    return GateResult(
        kind=GateKind.G7_NO_UNRESOLVED_SEVERE_ALERT,
        outcome=GateOutcome.PASS,
        message="no unresolved level A/B alerts",
    )
