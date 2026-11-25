"""The peeling-onion workflow as an executable state machine.

Each numbered requirement from the CTO brief has at least one test here:

1. heavy-atom / metal node search               -> ``test_the_five_stages_run_in_order``
2. metal coordination-layer validation          -> ``test_the_anchor_carries_forward``
3. first coordination sphere and donor atoms    -> ``test_the_first_shell_builds_on_the_anchor``
4. organic skeleton, shell by shell             -> ``test_shells_repeat_until_nothing_new_appears``
5. inference / refinement / density / validation-> ``test_a_shell_executor_sees_the_previous_model``
6. candidate comparison, rollback or continue   -> ``test_a_rollback_is_recorded_and_keeps_the_work``

Plus the two rules the driver enforces regardless of executor: locks are never silent, and a low R
factor cannot buy acceptance.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from conftest import write_hkl, write_ins
from crystalx_core.domain.candidate import (
    Candidate,
    CandidateProvenance,
    CandidateStatus,
    Mutation,
    MutationKind,
)
from crystalx_core.domain.chemistry import ChemicalPrior, MetalPrior
from crystalx_core.domain.data import DataDiagnosis
from crystalx_core.domain.gates import RankingPolicy
from crystalx_core.domain.hypothesis import (
    AnchorSet,
    CoordinationLayer,
    DonorAssignment,
    EvidenceKind,
    HeavyAtomHypothesis,
    HeavyAtomSite,
    HypothesisEvidence,
    LockDirective,
    LockMode,
    format_fixed_shelxl_value,
)
from crystalx_core.domain.ids import make_candidate_id
from crystalx_core.domain.metrics import (
    Alert,
    AlertSeverity,
    GeometryIssue,
    GeometryIssueKind,
    GeometryReport,
    RefinementMetrics,
    ValidationReport,
)
from crystalx_core.domain.onion import StageKind, StageState
from crystalx_core.domain.radiation import RadiationAssumption
from crystalx_core.domain.result import ReviewCheckpoint, ReviewCheckpointKind, StopReasonKind
from crystalx_core.io.shelx_reader import read_ins_header, scan_hkl
from crystalx_core.workflow.onion import (
    LockLedger,
    OnionPolicy,
    OnionWorkflow,
    StageRequest,
    StageResult,
)
from crystalx_core.workspace.events import EventKind, read_events
from crystalx_core.workspace.store import RunWorkspace

BASIS = "hklsha|dmin=None|hklf=4|twin=none"


# --------------------------------------------------------------------------------------
# fixtures and helpers
# --------------------------------------------------------------------------------------


@pytest.fixture
def workspace(tmp_path: Path, frozen_clock, fixed_entropy) -> RunWorkspace:
    return RunWorkspace.create(
        tmp_path / "runs", label="onion", clock=frozen_clock, entropy=fixed_entropy
    )


@pytest.fixture
def diagnosis(tmp_path: Path) -> DataDiagnosis:
    ins = write_ins(tmp_path / "case.ins")
    hkl = write_hkl(tmp_path / "case.hkl", include_friedel=True)
    header = read_ins_header(ins)
    return DataDiagnosis(
        reflections=scan_hkl(hkl, cell=header.cell, hklf_hint=header.hklf_format), ins=header
    )


PRIOR = ChemicalPrior(metals=(MetalPrior("Ag", oxidation_state=1, expected_sites=2),))
RADIATION = RadiationAssumption.from_wavelength(1.54184)


def good_metrics(r1: float = 0.040) -> RefinementMetrics:
    return RefinementMetrics(
        r1=r1,
        wr2=r1 * 2.7,
        goof=1.03,
        converged=True,
        max_shift_over_esd=0.001,
        n_reflections_observed=4000,
        n_parameters=280,
        reflection_basis_key=BASIS,
    )


def candidate(
    index: int,
    stage_id,
    *,
    parent=None,
    r1: float = 0.040,
    alerts: tuple[Alert, ...] = (),
    geometry_issues: tuple[GeometryIssue, ...] = (),
    label: str = "",
) -> Candidate:
    return Candidate(
        candidate_id=make_candidate_id(index),
        provenance=CandidateProvenance(
            created_at="2026-08-18T06:45:00+00:00",
            stage_id=stage_id,
            parent_id=parent,
            reflection_file_sha256="hklsha",
            hklf_format=4,
        ),
        label=label,
        metrics=good_metrics(r1),
        geometry=GeometryReport(
            issues=geometry_issues, charge_balance_net=0, charge_balance_checked=True
        ),
        validation=ValidationReport("platon_special", alerts=alerts),
    )


def anchor_set(validated: bool = True) -> AnchorSet:
    hypothesis = HeavyAtomHypothesis(
        hypothesis_id="h0",
        sites=(HeavyAtomSite("Ag1", "Ag", 0.25, 0.5, 0.75),),
        evidence=(
            HypothesisEvidence(EvidenceKind.SHELXD_SUBSTRUCTURE, "CC 41", value=41.0),
            HypothesisEvidence(EvidenceKind.ANODE_ANOMALOUS_PEAK, "17 sigma", value=17.0),
        ),
    )
    return AnchorSet(
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
        validated=validated,
    )


@dataclass
class RecordingExecutor:
    """A stage executor that returns a scripted result and remembers what it was asked."""

    result_factory: object = None
    seen: list[StageRequest] = field(default_factory=list)

    def execute(self, request: StageRequest) -> StageResult:
        self.seen.append(request)
        if self.result_factory is None:
            return StageResult()
        return self.result_factory(request)  # type: ignore[operator]


def executors_for(**by_kind) -> dict:
    return {StageKind[k.upper()]: v for k, v in by_kind.items()}


def run_workflow(workspace, diagnosis, executors, policy=None):
    workflow = OnionWorkflow(workspace, executors, policy=policy or OnionPolicy())
    result = workflow.run(diagnosis=diagnosis, prior=PRIOR, radiation=RADIATION)
    return workflow, result


# --------------------------------------------------------------------------------------
# lock ledger
# --------------------------------------------------------------------------------------


def test_the_ledger_gives_every_lock_a_stable_addressable_id():
    ledger = LockLedger()
    first = ledger.add(LockDirective(LockMode.HARD_LOCK, ("Ag1",), "confirmed"))
    second = ledger.add(LockDirective(LockMode.SOFT_RESTRAINT, ("N1",), "hold near the donor"))
    assert (first, second) == ("lock-000", "lock-001")
    assert ledger.get("lock-000").mode is LockMode.HARD_LOCK
    assert ledger.active_ids() == ("lock-000", "lock-001")
    assert len(ledger) == 2


def test_the_ledger_can_list_by_mode():
    ledger = LockLedger()
    ledger.add(LockDirective(LockMode.HARD_LOCK, ("Ag1",), "a"))
    ledger.add(LockDirective(LockMode.EXPLORATORY, ("Ag2",), "b"))
    ledger.add(LockDirective(LockMode.HARD_LOCK, ("Ag3",), "c"))
    assert [i for i, _ in ledger.by_mode(LockMode.HARD_LOCK)] == ["lock-000", "lock-002"]


# --------------------------------------------------------------------------------------
# stage sequencing
# --------------------------------------------------------------------------------------


def test_the_five_stages_run_in_order(workspace, diagnosis):
    calls: list[str] = []

    def record(name):
        def factory(request: StageRequest) -> StageResult:
            calls.append(f"{name}:{request.stage.stage_id}")
            if name == "anchor":
                return StageResult(anchor=anchor_set())
            if name == "first_shell":
                return StageResult(candidates=(candidate(0, request.stage.stage_id),))
            if name == "shell_expansion":
                return StageResult(converged=True)  # explicit synthetic evidence contract
            return StageResult()

        return RecordingExecutor(factory)

    executors = executors_for(
        initial_solution=record("initial_solution"),
        anchor=record("anchor"),
        first_shell=record("first_shell"),
        shell_expansion=record("shell_expansion"),
        finalize=record("finalize"),
    )
    workflow, _ = run_workflow(workspace, diagnosis, executors)

    kinds = [c.split(":")[0] for c in calls]
    assert kinds[:3] == ["initial_solution", "anchor", "first_shell"]
    assert "shell_expansion" in kinds
    assert kinds[-1] == "finalize"
    assert [s.kind.value for s in workflow.trace.stages][0] == "initial_solution"


def test_a_missing_executor_is_skipped_with_a_note_not_a_crash(workspace, diagnosis):
    """A build without a given stage implemented degrades gracefully rather than raising."""
    workflow, result = run_workflow(
        workspace, diagnosis, {}, OnionPolicy(require_anchor_before_expansion=False)
    )
    assert any("no executor registered" in note for note in workflow.trace.notes)
    assert result.stop_reason.kind is StopReasonKind.NO_TRUSTWORTHY_CANDIDATE


def test_with_no_executors_the_anchor_gate_still_fires_first(workspace, diagnosis):
    """Even an empty build must not walk past the "no validated anchor" guard."""
    _, result = run_workflow(workspace, diagnosis, {})
    assert result.stop_reason.kind is StopReasonKind.REVIEW_CHECKPOINT
    assert "No metal core has been validated" in result.stop_reason.message


def test_stage_directories_are_created_and_named_for_the_shell(workspace, diagnosis):
    executors = executors_for(
        initial_solution=RecordingExecutor(lambda r: StageResult()),
        anchor=RecordingExecutor(lambda r: StageResult(anchor=anchor_set())),
        shell_expansion=RecordingExecutor(lambda r: StageResult()),
    )
    run_workflow(workspace, diagnosis, executors)
    names = sorted(p.name for p in workspace.layout.stages_dir.iterdir())
    assert "s1-initial_solution" in names
    assert "s2-anchor" in names
    assert any(n.startswith("s4-shell_expansion-01") for n in names)


# --------------------------------------------------------------------------------------
# the anchor gate
# --------------------------------------------------------------------------------------


def test_shell_expansion_is_refused_before_the_anchor_is_validated(workspace, diagnosis):
    """Growing outward from an unconfirmed core is how a plausible wrong structure gets built."""
    executors = executors_for(
        initial_solution=RecordingExecutor(lambda r: StageResult()),
        anchor=RecordingExecutor(lambda r: StageResult(anchor=anchor_set(validated=False))),
        first_shell=RecordingExecutor(lambda r: StageResult()),
    )
    _, result = run_workflow(workspace, diagnosis, executors)

    assert result.stop_reason.kind is StopReasonKind.REVIEW_CHECKPOINT
    assert "unconfirmed" in result.stop_reason.message
    assert result.review_checkpoints
    assert result.review_checkpoints[0].kind is ReviewCheckpointKind.COMPETING_HYPOTHESES


def test_the_anchor_gate_can_be_relaxed_deliberately(workspace, diagnosis):
    executors = executors_for(
        initial_solution=RecordingExecutor(lambda r: StageResult()),
        first_shell=RecordingExecutor(lambda r: StageResult()),
        finalize=RecordingExecutor(lambda r: StageResult()),
    )
    _, result = run_workflow(
        workspace, diagnosis, executors, OnionPolicy(require_anchor_before_expansion=False)
    )
    assert result.stop_reason.kind is not StopReasonKind.REVIEW_CHECKPOINT


def test_the_anchor_carries_forward_to_later_stages(workspace, diagnosis):
    first_shell = RecordingExecutor(lambda r: StageResult())
    executors = executors_for(
        initial_solution=RecordingExecutor(lambda r: StageResult()),
        anchor=RecordingExecutor(lambda r: StageResult(anchor=anchor_set())),
        first_shell=first_shell,
    )
    run_workflow(workspace, diagnosis, executors)

    assert first_shell.seen
    seen_anchor = first_shell.seen[0].anchor
    assert seen_anchor is not None
    assert seen_anchor.validated
    assert seen_anchor.metal_atom_ids == ("Ag1",)
    assert seen_anchor.layers[0].coordination_number == 1


def test_the_first_shell_builds_on_the_anchor_and_its_donors(workspace, diagnosis):
    captured: dict[str, object] = {}

    def first_shell(request: StageRequest) -> StageResult:
        captured["donors"] = request.anchor.layers[0].donors if request.anchor else ()
        return StageResult(candidates=(candidate(0, request.stage.stage_id),))

    executors = executors_for(
        initial_solution=RecordingExecutor(lambda r: StageResult()),
        anchor=RecordingExecutor(lambda r: StageResult(anchor=anchor_set())),
        first_shell=RecordingExecutor(first_shell),
    )
    run_workflow(workspace, diagnosis, executors)

    donors = captured["donors"]
    assert donors and donors[0].donor_element == "N"  # type: ignore[index]
    assert donors[0].distance_a == pytest.approx(2.13)  # type: ignore[index]


# --------------------------------------------------------------------------------------
# locks
# --------------------------------------------------------------------------------------


def test_a_lock_is_recorded_logged_and_stamped_into_provenance(workspace, diagnosis):
    directive = LockDirective(
        mode=LockMode.HARD_LOCK,
        atom_ids=("Ag1",),
        reason="SHELXD CC and an ANODE anomalous peak both place this silver",
        instruction_lines=(
            f"Ag1 4 {format_fixed_shelxl_value(0.25)} {format_fixed_shelxl_value(0.5)} "
            f"{format_fixed_shelxl_value(0.75)} 11.00000 0.02000",
        ),
    )

    def anchor_stage(request: StageRequest) -> StageResult:
        return StageResult(
            anchor=anchor_set(),
            lock_directives=(directive,),
            candidates=(candidate(0, request.stage.stage_id),),
        )

    executors = executors_for(
        initial_solution=RecordingExecutor(lambda r: StageResult()),
        anchor=RecordingExecutor(anchor_stage),
    )
    workflow, _ = run_workflow(workspace, diagnosis, executors)

    # 1. written to the stage directory, with the generated SHELXL line
    stage_dir = workspace.layout.stage_dir(workflow.trace.stages[1].stage_id)
    payload = json.loads((stage_dir / "lock_directives.json").read_text())
    assert payload[0]["lock_id"] == "lock-000"
    assert payload[0]["directive"]["mode"] == "hard_lock"
    assert "10.25000" in payload[0]["directive"]["instruction_lines"][0]

    # 2. logged as an event, with the reason
    events = [e for e in read_events(workspace.layout.events) if e["kind"] == EventKind.LOCK_APPLIED]
    assert events and "ANODE anomalous peak" in events[0]["message"]
    assert events[0]["lock_id"] == "lock-000"

    # 3. referenced from the candidate's provenance
    stored = workspace.lineage.get(make_candidate_id(0))
    assert stored.provenance.lock_directive_ids == ("lock-000",)


def test_all_three_lock_modes_survive_into_the_record(workspace, diagnosis):
    directives = (
        LockDirective(LockMode.HARD_LOCK, ("Ag1",), "confirmed by two independent sources"),
        LockDirective(
            LockMode.SOFT_RESTRAINT,
            ("Ag1", "N1"),
            "hold the Ag-N distance in the declared window",
            instruction_lines=("DFIX 2.15 0.02 Ag1 N1",),
        ),
        LockDirective(LockMode.EXPLORATORY, ("Ag2",), "second site unconfirmed; let the data vote"),
    )

    def anchor_stage(request: StageRequest) -> StageResult:
        return StageResult(anchor=anchor_set(), lock_directives=directives)

    executors = executors_for(
        initial_solution=RecordingExecutor(lambda r: StageResult()),
        anchor=RecordingExecutor(anchor_stage),
    )
    workflow, _ = run_workflow(workspace, diagnosis, executors)

    modes = {d.mode for d in workflow.ledger.directives.values()}
    assert modes == {LockMode.HARD_LOCK, LockMode.SOFT_RESTRAINT, LockMode.EXPLORATORY}
    stage_dir = workspace.layout.stage_dir(workflow.trace.stages[1].stage_id)
    payload = json.loads((stage_dir / "lock_directives.json").read_text())
    assert [p["directive"]["mode"] for p in payload] == [
        "hard_lock",
        "soft_restraint",
        "exploratory",
    ]
    assert payload[1]["directive"]["instruction_lines"] == ["DFIX 2.15 0.02 Ag1 N1"]


def test_later_candidates_inherit_earlier_locks(workspace, diagnosis):
    def anchor_stage(request: StageRequest) -> StageResult:
        return StageResult(
            anchor=anchor_set(),
            candidates=(candidate(0, request.stage.stage_id),),
            lock_directives=(LockDirective(LockMode.HARD_LOCK, ("Ag1",), "confirmed"),),
        )

    def first_shell(request: StageRequest) -> StageResult:
        return StageResult(candidates=(candidate(1, request.stage.stage_id, parent=make_candidate_id(0)),))

    executors = executors_for(
        initial_solution=RecordingExecutor(lambda r: StageResult()),
        anchor=RecordingExecutor(anchor_stage),
        first_shell=RecordingExecutor(first_shell),
    )
    run_workflow(workspace, diagnosis, executors)

    stored = workspace.lineage.get(make_candidate_id(1))
    assert "lock-000" in stored.provenance.lock_directive_ids


def test_an_executor_is_told_which_locks_are_already_in_force(workspace, diagnosis):
    first_shell = RecordingExecutor(lambda r: StageResult())

    executors = executors_for(
        initial_solution=RecordingExecutor(lambda r: StageResult()),
        anchor=RecordingExecutor(
            lambda r: StageResult(
                anchor=anchor_set(),
                candidates=(candidate(0, r.stage.stage_id),),
                lock_directives=(LockDirective(LockMode.HARD_LOCK, ("Ag1",), "confirmed"),),
            )
        ),
        first_shell=first_shell,
    )
    run_workflow(workspace, diagnosis, executors)
    assert first_shell.seen[0].active_lock_ids == ("lock-000",)


# --------------------------------------------------------------------------------------
# candidates and the eligibility rule
# --------------------------------------------------------------------------------------


def test_a_chemically_implausible_candidate_cannot_carry_the_run_forward(workspace, diagnosis):
    """A low R factor must not buy acceptance, and must not become the parent of the next shell."""
    clash = GeometryIssue(
        GeometryIssueKind.ATOM_CLASH, "C4 and O2 are 0.6 A apart", ("C4", "O2"), 0.6
    )

    def first_shell(request: StageRequest) -> StageResult:
        return StageResult(
            candidates=(
                candidate(0, request.stage.stage_id, r1=0.012, geometry_issues=(clash,)),
            )
        )

    seen_parent: dict[str, object] = {}

    def shell(request: StageRequest) -> StageResult:
        seen_parent["input"] = request.input_candidate
        return StageResult()

    executors = executors_for(
        initial_solution=RecordingExecutor(lambda r: StageResult()),
        anchor=RecordingExecutor(lambda r: StageResult(anchor=anchor_set())),
        first_shell=RecordingExecutor(first_shell),
        shell_expansion=RecordingExecutor(shell),
    )
    workflow, result = run_workflow(workspace, diagnosis, executors)

    assert seen_parent["input"] is None, "an ineligible model must not become the next parent"
    assert any("passed the eligibility gates" in n for n in workflow.trace.notes)
    assert workspace.lineage.get(make_candidate_id(0)).status is CandidateStatus.REJECTED
    assert result.ranked == ()


def test_an_eligible_candidate_becomes_the_parent_of_the_next_stage(workspace, diagnosis):
    def first_shell(request: StageRequest) -> StageResult:
        return StageResult(
            candidates=(
                candidate(0, request.stage.stage_id, r1=0.070, label="worse"),
                candidate(1, request.stage.stage_id, r1=0.031, label="better"),
            )
        )

    seen: dict[str, object] = {}

    def shell(request: StageRequest) -> StageResult:
        seen["input"] = request.input_candidate
        return StageResult()

    executors = executors_for(
        initial_solution=RecordingExecutor(lambda r: StageResult()),
        anchor=RecordingExecutor(lambda r: StageResult(anchor=anchor_set())),
        first_shell=RecordingExecutor(first_shell),
        shell_expansion=RecordingExecutor(shell),
    )
    run_workflow(workspace, diagnosis, executors)

    parent = seen["input"]
    assert parent is not None
    assert parent.label == "better"  # type: ignore[union-attr]


def test_a_severe_alert_routes_the_run_to_review(workspace, diagnosis):
    alert = Alert("PLAT213", AlertSeverity.A, "Atom too close to a symmetry element")

    def first_shell(request: StageRequest) -> StageResult:
        return StageResult(
            candidates=(candidate(0, request.stage.stage_id, r1=0.020, alerts=(alert,)),)
        )

    executors = executors_for(
        initial_solution=RecordingExecutor(lambda r: StageResult()),
        anchor=RecordingExecutor(lambda r: StageResult(anchor=anchor_set())),
        first_shell=RecordingExecutor(first_shell),
    )
    _, result = run_workflow(workspace, diagnosis, executors)

    assert result.stop_reason.kind is StopReasonKind.REVIEW_CHECKPOINT
    assert result.ranked == ()
    assert result.ineligible


def test_candidate_lineage_is_preserved_across_stages(workspace, diagnosis):
    def first_shell(request: StageRequest) -> StageResult:
        return StageResult(candidates=(candidate(0, request.stage.stage_id, label="root"),))

    def shell(request: StageRequest) -> StageResult:
        if request.shell != 1:
            return StageResult()
        parent = request.input_candidate
        child = candidate(
            1,
            request.stage.stage_id,
            parent=parent.candidate_id if parent else None,
            r1=0.035,
            label="shell 1",
        )
        return StageResult(
            candidates=(
                type(child)(
                    **{
                        **{f: getattr(child, f) for f in child.__dataclass_fields__},
                        "mutations": (
                            Mutation(
                                MutationKind.ATOM_ADDED,
                                "added the pyridine ring",
                                driver="difference_fourier",
                            ),
                        ),
                    }
                ),
            )
        )

    executors = executors_for(
        initial_solution=RecordingExecutor(lambda r: StageResult()),
        anchor=RecordingExecutor(lambda r: StageResult(anchor=anchor_set())),
        first_shell=RecordingExecutor(first_shell),
        shell_expansion=RecordingExecutor(shell),
    )
    run_workflow(workspace, diagnosis, executors)

    ancestry = workspace.lineage.ancestry(make_candidate_id(1))
    assert [c.label for c in ancestry] == ["root", "shell 1"]
    assert ancestry[1].mutations[0].kind is MutationKind.ATOM_ADDED


# --------------------------------------------------------------------------------------
# shells
# --------------------------------------------------------------------------------------


def test_shells_repeat_until_no_accepted_progress_requires_review(workspace, diagnosis):
    """Migration: no new lineage entry is stagnation, never proof of a completed skeleton."""
    produced: list[int] = []

    def shell(request: StageRequest) -> StageResult:
        n = request.shell or 0
        if n <= 2:
            produced.append(n)
            return StageResult(
                candidates=(candidate(len(workspace.lineage), request.stage.stage_id),)
            )
        return StageResult()  # nothing more to add

    executors = executors_for(
        initial_solution=RecordingExecutor(lambda r: StageResult()),
        anchor=RecordingExecutor(lambda r: StageResult(anchor=anchor_set())),
        shell_expansion=RecordingExecutor(shell),
    )
    workflow, result = run_workflow(workspace, diagnosis, executors)

    assert produced == [1, 2]
    assert workflow.trace.shells_run == 3
    assert any("does not establish convergence" in n for n in workflow.trace.notes)
    assert result.stop_reason.kind is StopReasonKind.REVIEW_CHECKPOINT
    assert result.metadata["shells_run"] == "3"


def test_the_shell_budget_is_a_hard_stop_that_asks_for_a_human(workspace, diagnosis):
    def shell(request: StageRequest) -> StageResult:
        return StageResult(candidates=(candidate(len(workspace.lineage), request.stage.stage_id),))

    executors = executors_for(
        initial_solution=RecordingExecutor(lambda r: StageResult()),
        anchor=RecordingExecutor(lambda r: StageResult(anchor=anchor_set())),
        shell_expansion=RecordingExecutor(shell),
    )
    workflow, _ = run_workflow(workspace, diagnosis, executors, OnionPolicy(max_shells=3))

    assert workflow.trace.shells_run == 3
    assert any("needs a crystallographer" in n for n in workflow.trace.notes)


def test_a_shell_executor_sees_the_previous_model_and_the_data(workspace, diagnosis):
    """S4 alternates inference, refinement, density inspection and validation -- it needs all of it."""
    seen: list[StageRequest] = []

    def first_shell(request: StageRequest) -> StageResult:
        return StageResult(candidates=(candidate(0, request.stage.stage_id, label="core"),))

    def shell(request: StageRequest) -> StageResult:
        seen.append(request)
        return StageResult()

    executors = executors_for(
        initial_solution=RecordingExecutor(lambda r: StageResult()),
        anchor=RecordingExecutor(lambda r: StageResult(anchor=anchor_set())),
        first_shell=RecordingExecutor(first_shell),
        shell_expansion=RecordingExecutor(shell),
    )
    run_workflow(workspace, diagnosis, executors)

    request = seen[0]
    assert request.input_candidate is not None and request.input_candidate.label == "core"
    assert request.anchor is not None and request.anchor.validated
    assert request.diagnosis.reflections.n_reflections > 0
    assert request.prior.metal_elements == ("Ag",)
    assert request.radiation.source.value == "Cu_Ka"
    assert request.ledger is not None
    assert request.shell == 1


# --------------------------------------------------------------------------------------
# rollback
# --------------------------------------------------------------------------------------


def test_a_rollback_is_recorded_and_keeps_the_work(workspace, diagnosis):
    def anchor_stage(request: StageRequest) -> StageResult:
        return StageResult(
            candidates=(candidate(0, request.stage.stage_id, label="attempt"),),
            should_rollback=True,
            rollback_reason="no metal node survived the coordination check",
        )

    executors = executors_for(
        initial_solution=RecordingExecutor(lambda r: StageResult()),
        anchor=RecordingExecutor(anchor_stage),
    )
    workflow, _ = run_workflow(workspace, diagnosis, executors)

    assert workflow.trace.rollbacks
    decision = workflow.trace.rollbacks[0]
    assert "no metal node survived" in decision.reason
    assert decision.abandoned_candidates == (make_candidate_id(0),)

    stage_dir = workspace.layout.stage_dir(decision.from_stage)
    assert json.loads((stage_dir / "rollback.json").read_text())["reason"] == decision.reason
    # the abandoned candidate is kept, not deleted
    assert workspace.lineage.get(make_candidate_id(0)).label == "attempt"
    kinds = [e["kind"] for e in read_events(workspace.layout.events)]
    assert EventKind.STAGE_ROLLED_BACK in kinds


def test_a_rollback_that_needs_a_human_stops_the_run(workspace, diagnosis):
    def anchor_stage(request: StageRequest) -> StageResult:
        return StageResult(
            should_rollback=True,
            rollback_reason="two hypotheses are equally supported",
            requires_human_review=True,
        )

    executors = executors_for(
        initial_solution=RecordingExecutor(lambda r: StageResult()),
        anchor=RecordingExecutor(anchor_stage),
        first_shell=RecordingExecutor(lambda r: StageResult()),
    )
    workflow, result = run_workflow(workspace, diagnosis, executors)

    assert result.stop_reason.kind is StopReasonKind.REVIEW_CHECKPOINT
    assert "equally supported" in result.stop_reason.message
    first_shell = workflow.executors[StageKind.FIRST_SHELL]
    assert first_shell.seen == [], "the run stopped before the next stage"  # type: ignore[union-attr]


def test_a_recoverable_rollback_lets_the_run_continue(workspace, diagnosis):
    def anchor_stage(request: StageRequest) -> StageResult:
        return StageResult(should_rollback=True, rollback_reason="this branch did not work")

    finalize = RecordingExecutor(lambda r: StageResult())
    executors = executors_for(
        initial_solution=RecordingExecutor(lambda r: StageResult()),
        anchor=RecordingExecutor(anchor_stage),
        finalize=finalize,
    )
    run_workflow(workspace, diagnosis, executors, OnionPolicy(require_anchor_before_expansion=False))
    assert finalize.seen, "a recoverable rollback does not end the run"


def test_the_rolled_back_stage_is_marked_and_not_counted_as_completed(workspace, diagnosis):
    executors = executors_for(
        initial_solution=RecordingExecutor(lambda r: StageResult()),
        anchor=RecordingExecutor(
            lambda r: StageResult(should_rollback=True, rollback_reason="nothing usable")
        ),
    )
    workflow, result = run_workflow(workspace, diagnosis, executors)
    states = {s.kind: s.state for s in workflow.trace.stages}
    assert states[StageKind.ANCHOR] is StageState.ROLLED_BACK
    assert all("anchor" not in str(sid) for sid in result.stages_completed)


# --------------------------------------------------------------------------------------
# result
# --------------------------------------------------------------------------------------


def test_a_clean_run_converges_and_records_the_radiation_assumption(workspace, diagnosis):
    def first_shell(request: StageRequest) -> StageResult:
        return StageResult(candidates=(candidate(0, request.stage.stage_id, r1=0.032),))

    executors = executors_for(
        initial_solution=RecordingExecutor(lambda r: StageResult()),
        anchor=RecordingExecutor(lambda r: StageResult(anchor=anchor_set())),
        first_shell=RecordingExecutor(first_shell),
        finalize=RecordingExecutor(lambda r: StageResult(converged=True)),
    )
    _, result = run_workflow(workspace, diagnosis, executors)

    assert result.stop_reason.kind is StopReasonKind.CONVERGED
    assert result.best is not None
    assert result.metadata["radiation"] == "Cu_Ka"
    assert result.metadata["radiation_assumed"] == "False"


def test_a_run_with_no_candidate_reports_no_trustworthy_candidate(workspace, diagnosis):
    executors = executors_for(
        initial_solution=RecordingExecutor(lambda r: StageResult()),
        anchor=RecordingExecutor(lambda r: StageResult(anchor=anchor_set())),
        finalize=RecordingExecutor(lambda r: StageResult()),
    )
    _, result = run_workflow(workspace, diagnosis, executors)
    assert result.stop_reason.kind is StopReasonKind.NO_TRUSTWORTHY_CANDIDATE
    assert result.no_trustworthy_candidate is not None


def test_an_executor_checkpoint_reaches_the_result(workspace, diagnosis):
    checkpoint = ReviewCheckpoint(
        kind=ReviewCheckpointKind.SUSPECTED_DISORDER,
        question="the triflate looks disordered; model it in two parts?",
    )

    def shell(request: StageRequest) -> StageResult:
        return StageResult(review_checkpoints=(checkpoint,))

    executors = executors_for(
        initial_solution=RecordingExecutor(lambda r: StageResult()),
        anchor=RecordingExecutor(lambda r: StageResult(anchor=anchor_set())),
        shell_expansion=RecordingExecutor(shell),
    )
    _, result = run_workflow(workspace, diagnosis, executors)

    assert any(c.kind is ReviewCheckpointKind.SUSPECTED_DISORDER for c in result.review_checkpoints)
    assert result.stop_reason.kind is StopReasonKind.REVIEW_CHECKPOINT


def test_the_result_can_be_finalised_into_the_workspace(workspace, diagnosis):
    def first_shell(request: StageRequest) -> StageResult:
        return StageResult(candidates=(candidate(0, request.stage.stage_id),))

    executors = executors_for(
        initial_solution=RecordingExecutor(lambda r: StageResult()),
        anchor=RecordingExecutor(lambda r: StageResult(anchor=anchor_set())),
        first_shell=RecordingExecutor(first_shell),
    )
    _, result = run_workflow(workspace, diagnosis, executors)
    workspace.finalize(result)

    payload = json.loads(workspace.layout.result_json.read_text())
    assert payload["run_id"] == str(workspace.run_id)
    assert payload["metadata"]["locks_issued"] == "0"
