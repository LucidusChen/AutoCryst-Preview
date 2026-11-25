"""Eligibility gates and candidate ordering.

The scientific commitment being tested: a lower R factor never wins on its own. A candidate must
first pass hard gates, comparability is one of those gates, a severe validation alert routes to human
review rather than being averaged away, and "no candidate is trustworthy yet" is a valid answer.
"""

from __future__ import annotations

import pytest

from crystalx_core.domain.candidate import Candidate, CandidateProvenance
from crystalx_core.domain.gates import GateKind, GateOutcome, RankingPolicy
from crystalx_core.domain.ids import make_candidate_id, make_stage_id
from crystalx_core.domain.metrics import (
    Alert,
    AlertSeverity,
    AtomConfidence,
    ConfidenceReport,
    GeometryIssue,
    GeometryIssueKind,
    GeometryReport,
    RefinementMetrics,
    ResidualDensity,
    ResidualDensityPeak,
    ValidationReport,
)
from crystalx_core.domain.onion import StageKind
from crystalx_core.workflow.gates import evaluate_all, evaluate_gates
from crystalx_core.workflow.ranking import ORDERING_KEY_NAMES, describe_ordering, rank_candidates

POLICY = RankingPolicy()
HEAVY = frozenset({"Ag"})
STAGE = make_stage_id(5, StageKind.FINALIZE.value)


def make_candidate(
    index: int,
    *,
    r1: float | None = 0.04,
    wr2: float | None = 0.10,
    goof: float | None = 1.02,
    converged: bool | None = True,
    shift: float | None = 0.001,
    n_reflections: int | None = 4000,
    n_parameters: int | None = 300,
    parse_ok: bool = True,
    alerts: tuple[Alert, ...] = (),
    validation: ValidationReport | None = None,
    geometry_issues: tuple[GeometryIssue, ...] = (),
    geometry_evaluated: bool = True,
    residual: ResidualDensity | None = None,
    basis: str = "hklsha|dmin=None|hklf=4|twin=none",
    confidence: ConfidenceReport | None = None,
) -> Candidate:
    provenance = CandidateProvenance(
        created_at="2026-08-18T06:45:00+00:00",
        stage_id=STAGE,
        reflection_file_sha256="hklsha",
        hklf_format=4,
    )
    return Candidate(
        candidate_id=make_candidate_id(index),
        provenance=provenance,
        metrics=RefinementMetrics(
            r1=r1,
            wr2=wr2,
            goof=goof,
            converged=converged,
            max_shift_over_esd=shift,
            n_reflections_observed=n_reflections,
            n_parameters=n_parameters,
            parse_ok=parse_ok,
            parse_failure_reason="" if parse_ok else "could not read the .lst",
            residual_density=residual,
            reflection_basis_key=basis,
        ),
        geometry=GeometryReport(
            issues=geometry_issues,
            evaluated=geometry_evaluated,
            not_evaluated_reason="" if geometry_evaluated else "geometry engine not available",
            charge_balance_net=0,
            charge_balance_checked=True,
        ),
        validation=validation if validation is not None else ValidationReport("platon_special", alerts=alerts),
        confidence=confidence,
    )


def outcome_of(candidate: Candidate, kind: GateKind, policy: RankingPolicy = POLICY):
    report = evaluate_gates(candidate, policy, reference_basis_key=None)
    result = report.result(kind)
    assert result is not None
    return result


# --------------------------------------------------------------------------------------
# individual gates
# --------------------------------------------------------------------------------------


def test_a_clean_candidate_passes_every_gate():
    report = evaluate_gates(make_candidate(0), POLICY)
    assert report.is_eligible
    assert report.summary() == "eligible"


def test_unparsed_refinement_is_not_a_result():
    result = outcome_of(make_candidate(0, parse_ok=False), GateKind.G1_PARSE_OK)
    assert result.outcome is GateOutcome.FAIL
    assert "could not read" in result.message


def test_missing_metrics_block_rather_than_pass():
    candidate = Candidate(
        candidate_id=make_candidate_id(0),
        provenance=CandidateProvenance(created_at="now", stage_id=STAGE),
    )
    report = evaluate_gates(candidate, POLICY)
    assert not report.is_eligible
    parse = report.result(GateKind.G1_PARSE_OK)
    assert parse is not None and parse.outcome is GateOutcome.NOT_EVALUATED


def test_non_convergence_fails():
    assert outcome_of(make_candidate(0, converged=False), GateKind.G2_REFINEMENT_CONVERGED).outcome is (
        GateOutcome.FAIL
    )


def test_large_shift_over_esd_fails():
    result = outcome_of(make_candidate(0, shift=0.9), GateKind.G2_REFINEMENT_CONVERGED)
    assert result.outcome is GateOutcome.FAIL
    assert "shift/esd" in result.message


def test_fatal_geometry_fails_chemical_validity():
    issue = GeometryIssue(
        GeometryIssueKind.ATOM_CLASH, "C4 and O2 are 0.6 A apart", ("C4", "O2"), 0.6
    )
    result = outcome_of(make_candidate(0, geometry_issues=(issue,)), GateKind.G4_CHEMICAL_VALIDITY)
    assert result.outcome is GateOutcome.FAIL
    assert "0.6 A apart" in result.message


def test_non_fatal_geometry_does_not_fail():
    issue = GeometryIssue(
        GeometryIssueKind.BOND_TOO_LONG, "slightly long C-C", ("C1", "C2"), is_fatal=False
    )
    assert outcome_of(make_candidate(0, geometry_issues=(issue,)), GateKind.G4_CHEMICAL_VALIDITY).outcome is (
        GateOutcome.PASS
    )


def test_charge_imbalance_is_fatal_by_default_and_configurable():
    issue = GeometryIssue(
        GeometryIssueKind.CHARGE_IMBALANCE, "2 Ag(+) against 1 triflate: net +1"
    )
    strict = outcome_of(make_candidate(0, geometry_issues=(issue,)), GateKind.G4_CHEMICAL_VALIDITY)
    assert strict.outcome is GateOutcome.FAIL

    lenient = RankingPolicy(charge_imbalance_is_fatal=False)
    relaxed = outcome_of(
        make_candidate(0, geometry_issues=(issue,)), GateKind.G4_CHEMICAL_VALIDITY, lenient
    )
    assert relaxed.outcome is GateOutcome.PASS


def test_implausible_coordination_routes_to_review():
    issue = GeometryIssue(
        GeometryIssueKind.IMPLAUSIBLE_COORDINATION_NUMBER,
        "Ag1 has 7 donors, declared maximum is 4",
        ("Ag1",),
    )
    result = outcome_of(
        make_candidate(0, geometry_issues=(issue,)), GateKind.G5_COORDINATION_PLAUSIBLE
    )
    assert result.outcome is GateOutcome.FAIL
    assert result.routes_to_review


def test_unevaluated_geometry_blocks():
    result = outcome_of(make_candidate(0, geometry_evaluated=False), GateKind.G4_CHEMICAL_VALIDITY)
    assert result.outcome is GateOutcome.NOT_EVALUATED


def test_poor_data_parameter_ratio_fails():
    result = outcome_of(
        make_candidate(0, n_reflections=900, n_parameters=300), GateKind.G6_DATA_PARAMETER_RATIO
    )
    assert result.outcome is GateOutcome.FAIL
    assert "over-parameterised" in result.message


def test_severe_alert_fails_and_routes_to_review():
    alert = Alert("PLAT201", AlertSeverity.A, "Isotropic non-H atoms in main residue", alert_type=2)
    result = outcome_of(make_candidate(0, alerts=(alert,)), GateKind.G7_NO_UNRESOLVED_SEVERE_ALERT)
    assert result.outcome is GateOutcome.FAIL
    assert result.routes_to_review
    assert "review gate, not a score term" in result.message


def test_acknowledged_alert_no_longer_blocks():
    alert = Alert(
        "PLAT201",
        AlertSeverity.A,
        "Isotropic non-H atoms",
        acknowledged=True,
        acknowledgement_reason="expected for this disordered anion; documented",
    )
    assert outcome_of(make_candidate(0, alerts=(alert,)), GateKind.G7_NO_UNRESOLVED_SEVERE_ALERT).outcome is (
        GateOutcome.PASS
    )


def test_level_c_alerts_do_not_block():
    alert = Alert("PLAT097", AlertSeverity.C, "Large reported max shift", alert_type=3)
    assert outcome_of(make_candidate(0, alerts=(alert,)), GateKind.G7_NO_UNRESOLVED_SEVERE_ALERT).outcome is (
        GateOutcome.PASS
    )


def test_incomplete_validation_blocks_rather_than_passes():
    validation = ValidationReport(
        "platon_special", completed=False, failure_reason="check.def version mismatch"
    )
    result = outcome_of(make_candidate(0, validation=validation), GateKind.G7_NO_UNRESOLVED_SEVERE_ALERT)
    assert result.outcome is GateOutcome.NOT_EVALUATED
    assert "check.def" in result.message


def test_a_waiver_is_recorded_with_its_reason():
    policy = RankingPolicy(
        waived_gates={GateKind.G6_DATA_PARAMETER_RATIO.value: "small-molecule case, accepted by owner"}
    )
    report = evaluate_gates(make_candidate(0, n_reflections=900, n_parameters=300), policy)
    result = report.result(GateKind.G6_DATA_PARAMETER_RATIO)
    assert result is not None
    assert result.outcome is GateOutcome.WAIVED
    assert "accepted by owner" in result.message
    assert report.is_eligible


# --------------------------------------------------------------------------------------
# comparability
# --------------------------------------------------------------------------------------


def test_different_reflection_basis_is_refused_comparison():
    same = make_candidate(0)
    different = make_candidate(1, basis="OTHER|dmin=0.9|hklf=4|twin=none")
    reports = evaluate_all((same, different), POLICY)

    assert reports[0].is_eligible
    assert not reports[1].is_eligible
    failing = reports[1].result(GateKind.G3_SAME_REFLECTION_BASIS)
    assert failing is not None
    assert failing.outcome is GateOutcome.FAIL
    assert "not comparable" in failing.message


def test_comparability_can_be_switched_off_for_a_single_candidate():
    candidate = make_candidate(0, basis="anything")
    report = evaluate_gates(candidate, POLICY, reference_basis_key=None)
    result = report.result(GateKind.G3_SAME_REFLECTION_BASIS)
    assert result is not None and result.outcome is GateOutcome.PASS


# --------------------------------------------------------------------------------------
# ranking
# --------------------------------------------------------------------------------------


def test_lower_r1_wins_among_eligible_candidates():
    better = make_candidate(0, r1=0.031)
    worse = make_candidate(1, r1=0.058)
    candidates = (worse, better)
    outcome = rank_candidates(candidates, evaluate_all(candidates, POLICY), POLICY)

    assert outcome.has_winner
    assert outcome.best is not None
    assert str(outcome.best.candidate_id) == "c000"
    assert outcome.best.key_names == ORDERING_KEY_NAMES


def test_a_failing_gate_beats_a_better_r_factor():
    """The central rule: eligibility first, numbers second."""
    attractive_but_wrong = make_candidate(
        0,
        r1=0.019,
        alerts=(Alert("PLAT213", AlertSeverity.A, "Atom suspiciously close to a symmetry element"),),
    )
    honest = make_candidate(1, r1=0.055)
    candidates = (attractive_but_wrong, honest)
    outcome = rank_candidates(candidates, evaluate_all(candidates, POLICY), POLICY)

    assert outcome.best is not None
    assert str(outcome.best.candidate_id) == "c001"
    assert [str(r.candidate_id) for r in outcome.ineligible] == ["c000"]
    assert outcome.review_checkpoints


def test_tie_on_r1_is_broken_by_wr2_then_goof():
    a = make_candidate(0, r1=0.040, wr2=0.120, goof=1.30)
    b = make_candidate(1, r1=0.040, wr2=0.100, goof=1.30)
    outcome = rank_candidates((a, b), evaluate_all((a, b), POLICY), POLICY)
    assert str(outcome.ranked[0].candidate_id) == "c001"

    c = make_candidate(0, r1=0.040, wr2=0.100, goof=1.40)
    d = make_candidate(1, r1=0.040, wr2=0.100, goof=1.01)
    outcome = rank_candidates((c, d), evaluate_all((c, d), POLICY), POLICY)
    assert str(outcome.ranked[0].candidate_id) == "c001"


def test_residual_density_beside_a_heavy_atom_is_not_penalised():
    """A big peak next to silver is usually absorption; the same peak in a void is a missing atom."""
    near_metal = make_candidate(
        0,
        r1=0.040,
        wr2=0.100,
        goof=1.00,
        residual=ResidualDensity(
            max_peak_e_per_a3=2.4,
            peaks=(
                ResidualDensityPeak(2.4, 0.1, 0.1, 0.1, nearest_atom="Ag1", nearest_atom_element="Ag"),
            ),
        ),
    )
    in_a_void = make_candidate(
        1,
        r1=0.040,
        wr2=0.100,
        goof=1.00,
        residual=ResidualDensity(
            max_peak_e_per_a3=2.4,
            peaks=(
                ResidualDensityPeak(2.4, 0.5, 0.5, 0.5, nearest_atom="C7", nearest_atom_element="C"),
            ),
        ),
    )
    outcome = rank_candidates(
        (in_a_void, near_metal),
        evaluate_all((in_a_void, near_metal), POLICY),
        POLICY,
        heavy_elements=HEAVY,
    )
    assert str(outcome.best.candidate_id) == "c000"  # the one whose peak sits on the metal


def test_missing_r1_sorts_last_rather_than_winning():
    unknown = make_candidate(0, r1=None)
    known = make_candidate(1, r1=0.20)
    outcome = rank_candidates((unknown, known), evaluate_all((unknown, known), POLICY), POLICY)
    assert str(outcome.ranked[0].candidate_id) == "c001"


def test_no_eligible_candidate_reports_no_trustworthy_candidate():
    bad = make_candidate(0, converged=False)
    outcome = rank_candidates((bad,), evaluate_all((bad,), POLICY), POLICY)

    assert not outcome.has_winner
    assert outcome.no_trustworthy_candidate is not None
    assert "none passed the eligibility gates" in outcome.no_trustworthy_candidate.reason
    assert outcome.no_trustworthy_candidate.suggested_next_stage == "shell_expansion"


def test_empty_candidate_set_is_handled():
    outcome = rank_candidates((), (), POLICY)
    assert not outcome.has_winner
    assert outcome.no_trustworthy_candidate is not None
    assert "No candidate has been produced yet" in outcome.no_trustworthy_candidate.reason


def test_model_confidence_is_not_part_of_the_ordering_key():
    """A very confident model must not outrank a better refinement."""
    confident_but_worse = make_candidate(
        0,
        r1=0.070,
        confidence=ConfidenceReport(
            model_name="crystalx-heavy",
            atoms=(AtomConfidence("C1", "C", (("C", 0.999), ("N", 0.001))),),
        ),
    )
    unconfident_but_better = make_candidate(
        1,
        r1=0.035,
        confidence=ConfidenceReport(
            model_name="crystalx-heavy",
            atoms=(AtomConfidence("C1", "C", (("C", 0.41), ("N", 0.39))),),
        ),
    )
    candidates = (confident_but_worse, unconfident_but_better)
    outcome = rank_candidates(candidates, evaluate_all(candidates, POLICY), POLICY)

    assert str(outcome.best.candidate_id) == "c001"
    assert "crystalx" not in " ".join(ORDERING_KEY_NAMES)
    assert "confidence is not part of the key" in describe_ordering()


def test_ranked_summary_exposes_the_deciding_numbers():
    candidate = make_candidate(0, r1=0.0412, wr2=0.1103, goof=1.04)
    outcome = rank_candidates((candidate,), evaluate_all((candidate,), POLICY), POLICY)
    summary = outcome.ranked[0].summary
    assert "r1=0.0412" in summary
    assert "wr2=0.1103" in summary
    assert "data/param=" in summary


def test_confidence_report_ranks_the_most_ambiguous_atom_first():
    """This is what drives margin-based candidate generation in a later slice."""
    report = ConfidenceReport(
        atoms=(
            AtomConfidence("C1", "C", (("C", 0.95), ("N", 0.03))),
            AtomConfidence("N2", "N", (("N", 0.44), ("O", 0.42))),
            AtomConfidence("O3", "O", (("O", 0.80), ("N", 0.15))),
        )
    )
    assert [a.atom_id for a in report.most_ambiguous(2)] == ["N2", "O3"]
    assert report.atoms[1].margin == pytest.approx(0.02)
