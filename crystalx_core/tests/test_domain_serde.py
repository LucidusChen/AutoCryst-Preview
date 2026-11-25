"""Every domain type must survive a JSON round trip.

The run record is the product's memory. If a type cannot be written and read back exactly, the
record silently loses evidence.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
from pathlib import Path

import pytest

from crystalx_core._serde import canonical_json, dumps, from_jsonable, loads, to_jsonable
from crystalx_core.domain import (
    Alert,
    AlertSeverity,
    AnchorSet,
    Applicability,
    ApplicabilityVerdict,
    AtomConfidence,
    BinaryArch,
    BinaryFormat,
    BinaryProbe,
    BranchBudget,
    BranchOutcome,
    BranchState,
    Candidate,
    CandidateProvenance,
    CandidateStatus,
    ChargeModel,
    ChemicalPrior,
    ConfidenceReport,
    CoordinationLayer,
    CounterAnionKind,
    CounterAnionSpec,
    DonorAssignment,
    DonorAtomSpec,
    DonorGroupKind,
    DonorGroupSpec,
    EligibilityReport,
    EvidenceKind,
    GateKind,
    GateOutcome,
    GateResult,
    GeometryIssue,
    GeometryIssueKind,
    GeometryReport,
    HeavyAtomHypothesis,
    HeavyAtomSite,
    HypothesisEvidence,
    InitialSolutionBranch,
    LockDirective,
    LockMode,
    MetalPrior,
    Mutation,
    MutationKind,
    OnionStage,
    RankingPolicy,
    ReferenceStructure,
    RefinementMetrics,
    ResidualDensity,
    ResidualDensityPeak,
    ResolutionPolicy,
    RollbackDecision,
    SelectionRuleResult,
    SelectionRuleVerdict,
    SelectionTrace,
    SolverKind,
    StageBudget,
    StageKind,
    StagePolicy,
    StageState,
    StopReason,
    StopReasonKind,
    UnitCell,
    ValidationReport,
    make_branch_id,
    make_candidate_id,
    make_stage_id,
)


def _round_trip(instance):
    """Serialise, deserialise, and assert equality."""
    payload = to_jsonable(instance)
    restored = from_jsonable(type(instance), payload)
    assert restored == instance, f"round trip changed {type(instance).__name__}"
    return restored


def test_unit_cell_round_trip():
    # Hand-authored asymmetric cell; no private measurement values.
    _round_trip(UnitCell(8.5, 10.25, 13.75, 72.0, 81.0, 96.0))


def test_chemistry_round_trip():
    # Invented chemistry exercises nested dataclasses and enums without encoding a local case.
    prior = ChemicalPrior(
        metals=(MetalPrior("Cu", oxidation_state=2, coordination_number_min=4, coordination_number_max=6),),
        donor_groups=(
            DonorGroupSpec(
                kind=DonorGroupKind.AMINE,
                donors=(DonorAtomSpec("N", 2.0, 2.8, note="synthetic donor window"),),
                label="synthetic donor group",
                expected_count=2,
            ),
        ),
        counter_anions=(CounterAnionSpec(CounterAnionKind.NITRATE, charge=-1),),
        reference_structures=(ReferenceStructure(path=Path("/tmp/synthetic-reference.cif"), label="synthetic reference"),),
        charge_model=ChargeModel(metal_charge_total=4, anion_charge_total=-4, declared_by_user=True),
        is_peptide_like=False,
        declared_elements=("Cu", "C", "N", "O"),
    )
    _round_trip(prior)


def test_hypothesis_round_trip():
    hypothesis = HeavyAtomHypothesis(
        hypothesis_id="h0",
        sites=(HeavyAtomSite(atom_id="Ag1", element="Ag", x=0.1, y=0.2, z=0.3, peak_height=99.0),),
        origin_branch=make_branch_id(0, "shelxd"),
        evidence=(
            HypothesisEvidence(EvidenceKind.SHELXD_SUBSTRUCTURE, "CC 42", value=42.0),
            HypothesisEvidence(EvidenceKind.ANODE_ANOMALOUS_PEAK, "18 sigma", value=18.0, units="sigma"),
        ),
        confidence=0.9,
    )
    anchor = AnchorSet(
        anchor_id="a0",
        hypothesis=hypothesis,
        layers=(
            CoordinationLayer(
                metal_atom_id="Ag1",
                metal_element="Ag",
                donors=(DonorAssignment("N1", "N", "Ag1", 2.13, within_expected_window=True),),
                geometry_label="linear",
                geometry_is_plausible=True,
            ),
        ),
        lock_directives=(
            LockDirective(
                mode=LockMode.HARD_LOCK,
                atom_ids=("Ag1",),
                reason="anchor validated by two independent experimental sources",
                instruction_lines=("Ag1   4   10.10000  10.20000  10.30000  11.00000  0.02000",),
            ),
        ),
        validated=True,
    )
    _round_trip(anchor)


def test_branch_and_trace_round_trip():
    branch = InitialSolutionBranch(
        branch_id=make_branch_id(1, SolverKind.SHELXC_SHELXD.value),
        kind=SolverKind.SHELXC_SHELXD,
        stage_id=make_stage_id(1, StageKind.INITIAL_SOLUTION.value),
        state=BranchState.PLANNED,
        priority=90,
        applicability=Applicability(ApplicabilityVerdict.APPLICABLE, 90, ("anomalous data",)),
        budget=BranchBudget(max_wall_clock_s=120.0, solver_trials=1000),
        heavy_atom_assumption=("Ag",),
        use_anomalous=True,
    )
    trace = SelectionTrace(
        rules=(
            SelectionRuleResult(
                rule_id="R01_FRIEDEL_PAIRS_PRESENT",
                description="d",
                verdict=SelectionRuleVerdict.PASS,
                detail="ok",
                inputs={"n": "12"},
                affects=(SolverKind.SHELXC_SHELXD,),
            ),
        ),
        nominated=(branch,),
        excluded=((SolverKind.SHELXD_ABINITIO, "resolution too low"),),
        notes=("note",),
    )
    _round_trip(trace)
    _round_trip(BranchOutcome(branch.branch_id, BranchState.SUCCEEDED, figures_of_merit={"CC": 41.2}))


def test_candidate_round_trip():
    provenance = CandidateProvenance(
        created_at="2026-08-18T06:45:00+00:00",
        stage_id=make_stage_id(2, StageKind.ANCHOR.value),
        branch_id=make_branch_id(0, "shelxt"),
        parent_id=None,
        tool_identities={"shelxl": "abc123"},
        commands=(("shelxl", "case"),),
        reflection_file_sha256="deadbeef",
        hklf_format=4,
        reference_structure_influenced=True,
        reference_structure_labels=("parent",),
    )
    candidate = Candidate(
        candidate_id=make_candidate_id(3),
        provenance=provenance,
        status=CandidateStatus.REFINED,
        mutations=(
            Mutation(
                kind=MutationKind.ELEMENT_FLIP,
                description="N3 -> O3",
                atoms=("N3",),
                from_value="N",
                to_value="O",
                driver="crystalx_margin",
                margin=0.03,
            ),
        ),
        metrics=RefinementMetrics(
            r1=0.0412,
            wr2=0.1103,
            goof=1.04,
            n_reflections_observed=4200,
            n_parameters=310,
            n_restraints=6,
            converged=True,
            residual_density=ResidualDensity(
                max_peak_e_per_a3=1.2,
                min_hole_e_per_a3=-0.8,
                peaks=(
                    ResidualDensityPeak(1.2, 0.1, 0.1, 0.1, nearest_atom="Ag1", nearest_atom_element="Ag", distance_to_nearest_a=0.9),
                ),
            ),
            reflection_basis_key="deadbeef|dmin=None|hklf=4|twin=none",
        ),
        geometry=GeometryReport(
            issues=(
                GeometryIssue(GeometryIssueKind.BOND_TOO_LONG, "C1-C2 too long", ("C1", "C2"), 1.9, 1.3, 1.6),
            ),
            coordination_numbers={"Ag1": 2},
            charge_balance_net=0,
            charge_balance_checked=True,
        ),
        validation=ValidationReport(
            tool="platon_special",
            alerts=(Alert("PLAT934", AlertSeverity.B, "Number of (Iobs-Icalc)/Sigma > 10 ...", alert_type=3),),
            check_def_version="240101",
        ),
        confidence=ConfidenceReport(
            model_name="crystalx-heavy",
            atoms=(AtomConfidence("C1", "C", (("C", 0.91), ("N", 0.07))),),
        ),
    )
    restored = _round_trip(candidate)
    assert restored.metrics is not None
    assert restored.metrics.data_to_parameter_ratio == pytest.approx(4200 / 310)


def test_onion_and_result_round_trip():
    stage = OnionStage(
        stage_id=make_stage_id(4, StageKind.SHELL_EXPANSION.value, shell=2),
        kind=StageKind.SHELL_EXPANSION,
        index=4,
        shell=2,
        state=StageState.RUNNING,
        budget=StageBudget(max_attempts=2, max_candidates=4, max_new_atoms=12),
        policy=StagePolicy(allowed_elements=("C", "N", "O"), max_distance_from_model_a=2.2),
    )
    _round_trip(stage)
    _round_trip(
        RollbackDecision(
            from_stage=stage.stage_id,
            to_stage=make_stage_id(3, StageKind.FIRST_SHELL.value),
            reason="shell 2 would not refine",
            abandoned_branches=(make_branch_id(0, "shelxt"),),
        )
    )
    _round_trip(StopReason(StopReasonKind.NO_TRUSTWORTHY_CANDIDATE, "nothing passed the gates"))


def test_gate_report_round_trip():
    report = EligibilityReport(
        candidate_id=make_candidate_id(0),
        results=(
            GateResult(GateKind.G1_PARSE_OK, GateOutcome.PASS, "parsed"),
            GateResult(
                GateKind.G7_NO_UNRESOLVED_SEVERE_ALERT,
                GateOutcome.FAIL,
                "1 level A alert",
                routes_to_review=True,
            ),
        ),
    )
    restored = _round_trip(report)
    assert not restored.is_eligible
    assert restored.requires_review


def test_tools_round_trip():
    _round_trip(BinaryProbe(BinaryFormat.MACH_O_FAT, BinaryArch.UNIVERSAL, (BinaryArch.X86_64, BinaryArch.ARM64)))


def test_ranking_policy_round_trip():
    _round_trip(RankingPolicy(waived_gates={"g6_data_parameter_ratio": "small-molecule, accepted"}))


def test_resolution_policy_round_trip():
    _round_trip(ResolutionPolicy(d_min_a=0.84, d_max_a=25.0, applied_by="user"))


def test_dumps_loads_helpers():
    cell = UnitCell(5.0, 5.0, 5.0, 90.0, 90.0, 90.0)
    assert loads(UnitCell, dumps(cell)) == cell


def test_canonical_json_is_order_independent():
    a = {"b": 1, "a": {"y": 2, "x": 3}}
    b = {"a": {"x": 3, "y": 2}, "b": 1}
    assert canonical_json(a) == canonical_json(b)


def test_datetime_and_path_round_trip():
    @dataclasses.dataclass(frozen=True)
    class Sample:
        when: _dt.datetime
        where: Path

    original = Sample(_dt.datetime(2026, 8, 18, tzinfo=_dt.timezone.utc), Path("/tmp/x"))
    payload = to_jsonable(original)
    assert isinstance(payload["when"], str)
    assert from_jsonable(Sample, payload) == original


def test_missing_required_field_raises():
    with pytest.raises(KeyError):
        from_jsonable(UnitCell, {"a": 1.0, "b": 2.0})


def test_unserialisable_type_raises():
    with pytest.raises(TypeError):
        to_jsonable(object())
