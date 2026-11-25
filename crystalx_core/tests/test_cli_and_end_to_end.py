"""The CLI, and one end-to-end walk through a whole mini-run.

The end-to-end test is the one that proves the pieces fit: stage inputs, diagnose, select a
portfolio, run branches through a scripted executor, create candidates with lineage, gate them, rank
them, and finalise. No crystallographic program is executed at any point.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from conftest import CPU_ARM64, CPU_X86_64, macho_thin, write_executable, write_hkl, write_ins
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
    EvidenceKind,
    HeavyAtomHypothesis,
    HeavyAtomSite,
    HypothesisEvidence,
    LockDirective,
    LockMode,
    format_fixed_shelxl_value,
)
from crystalx_core.domain.ids import make_candidate_id, make_stage_id
from crystalx_core.domain.metrics import (
    Alert,
    AlertSeverity,
    GeometryReport,
    RefinementMetrics,
    ValidationReport,
)
from crystalx_core.domain.onion import OnionStage, StageKind, StageState
from crystalx_core.domain.result import RunResult, StopReason, StopReasonKind
from crystalx_core.domain.solving import BranchState, SolverKind
from crystalx_core.io.shelx_reader import read_ins_header, scan_hkl
from crystalx_core.solvers.base import BackendRegistry
from crystalx_core.solvers.machine import InitialSolutionMachine
from crystalx_core.solvers.mock import MockBackend, ScriptedExecutor, ScriptedResponse
from crystalx_core.solvers.selector import StrategySelector
from crystalx_core.tools.runner import ProcessRunner
from crystalx_core.workflow.gates import evaluate_all
from crystalx_core.workflow.ranking import rank_candidates
from crystalx_core.workspace.events import EventKind, read_events
from crystalx_core.workspace.store import RunWorkspace
from crystalx_core.cli.main import main


# --------------------------------------------------------------------------------------
# SHELXL lock formatting
# --------------------------------------------------------------------------------------


def test_fixed_coordinate_uses_the_shelxl_free_variable_convention():
    """SHELXL reads ``10.xxxxx`` as 'fix at 0.xxxxx'. Getting this wrong silently frees an anchor."""
    assert format_fixed_shelxl_value(0.25) == "10.25000"
    assert format_fixed_shelxl_value(0.0) == "10.00000"
    assert format_fixed_shelxl_value(-0.125) == "9.87500"


def test_out_of_range_coordinate_is_refused_rather_than_emitted():
    with pytest.raises(ValueError) as excinfo:
        format_fixed_shelxl_value(1.5)
    assert "Wrap the coordinate" in str(excinfo.value)


def test_a_lock_directive_must_state_a_reason():
    with pytest.raises(ValueError):
        LockDirective(mode=LockMode.HARD_LOCK, atom_ids=("Ag1",), reason="   ")


def test_free_mode_emits_no_instructions():
    with pytest.raises(ValueError):
        LockDirective(
            mode=LockMode.FREE, atom_ids=("Ag1",), reason="free", instruction_lines=("AFIX 0",)
        )


def test_hard_lock_carries_the_generated_lines_for_inspection():
    directive = LockDirective(
        mode=LockMode.HARD_LOCK,
        atom_ids=("Ag1",),
        reason="two independent experimental sources place this silver",
        instruction_lines=(
            f"Ag1   4  {format_fixed_shelxl_value(0.25)}  "
            f"{format_fixed_shelxl_value(0.5)}  {format_fixed_shelxl_value(0.75)}  "
            "11.00000  0.02000",
        ),
    )
    assert "10.25000" in directive.instruction_lines[0]
    assert directive.reversible


# --------------------------------------------------------------------------------------
# hypothesis evidence policy
# --------------------------------------------------------------------------------------


def test_model_confidence_alone_is_not_independent_experimental_support():
    hypothesis = HeavyAtomHypothesis(
        hypothesis_id="h0",
        sites=(HeavyAtomSite("Ag1", "Ag", 0.1, 0.2, 0.3),),
        evidence=(
            HypothesisEvidence(EvidenceKind.CRYSTALX_INFERENCE, "p=0.99", value=0.99),
            HypothesisEvidence(EvidenceKind.USER_PRIOR, "user says Ag"),
            HypothesisEvidence(EvidenceKind.REFERENCE_STRUCTURE, "parent scaffold has Ag"),
        ),
        confidence=0.99,
    )
    assert hypothesis.experimental_evidence() == ()
    assert not hypothesis.has_independent_experimental_support()


def test_two_distinct_experimental_sources_count_as_independent():
    hypothesis = HeavyAtomHypothesis(
        hypothesis_id="h0",
        sites=(HeavyAtomSite("Ag1", "Ag", 0.1, 0.2, 0.3),),
        evidence=(
            HypothesisEvidence(EvidenceKind.SHELXD_SUBSTRUCTURE, "CC 42", value=42.0),
            HypothesisEvidence(EvidenceKind.ANODE_ANOMALOUS_PEAK, "18 sigma", value=18.0),
        ),
    )
    assert hypothesis.has_independent_experimental_support()


def test_repeating_one_source_does_not_count_twice():
    hypothesis = HeavyAtomHypothesis(
        hypothesis_id="h0",
        sites=(HeavyAtomSite("Ag1", "Ag", 0.1, 0.2, 0.3),),
        evidence=(
            HypothesisEvidence(EvidenceKind.SHELXD_SUBSTRUCTURE, "CC 42", value=42.0),
            HypothesisEvidence(EvidenceKind.SHELXD_SUBSTRUCTURE, "CC 41", value=41.0),
        ),
    )
    assert not hypothesis.has_independent_experimental_support()


def test_contradicting_evidence_is_retained():
    hypothesis = HeavyAtomHypothesis(
        hypothesis_id="h0",
        sites=(HeavyAtomSite("Ag1", "Ag", 0.1, 0.2, 0.3),),
        evidence=(
            HypothesisEvidence(EvidenceKind.SHELXD_SUBSTRUCTURE, "CC 42", value=42.0),
            HypothesisEvidence(
                EvidenceKind.ANODE_ANOMALOUS_PEAK, "only 1.2 sigma", value=1.2, supports=False
            ),
        ),
    )
    assert len(hypothesis.contradicting_evidence()) == 1


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def test_cli_doctor_reports_the_mixed_architecture_machine(toolbox, capsys, monkeypatch):
    for name in ("shelxt", "shelxl"):
        write_executable(toolbox / name, macho_thin(CPU_X86_64))
    write_executable(toolbox / "platon_special", macho_thin(CPU_ARM64))

    monkeypatch.setattr("platform.system", lambda: "Darwin")
    monkeypatch.setattr("platform.machine", lambda: "arm64")
    monkeypatch.setattr("crystalx_core.tools.resolver.rosetta_is_installed", lambda **_: False)

    assert main(["doctor", "--toolbox", str(toolbox), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["host"]["arch"] == "arm64"
    assert payload["host"]["rosetta_available"] is False
    assert payload["tools"]["shelxt"]["diagnostic"] == "rosetta_missing"
    assert "install-rosetta" in payload["tools"]["shelxt"]["remedy"]
    assert payload["tools"]["platon_special"]["usable"] is True


def test_cli_diagnose_prints_the_facts(tmp_path: Path, capsys):
    ins = write_ins(tmp_path / "case.ins")
    hkl = write_hkl(tmp_path / "case.hkl", include_friedel=True)

    assert main(["diagnose", "--hkl", str(hkl), "--ins", str(ins)]) == 0
    out = capsys.readouterr().out

    assert "friedel_pairs_present" in out
    assert "not computed" in out  # completeness is honest about being unavailable
    assert "0.71073" in out


def test_cli_plan_explains_its_choice(tmp_path: Path, capsys):
    ins = write_ins(tmp_path / "case.ins")
    hkl = write_hkl(tmp_path / "case.hkl", include_friedel=True)

    assert main(["plan", "--hkl", str(hkl), "--ins", str(ins), "--metal", "Ag"]) == 0
    out = capsys.readouterr().out

    assert "R01_FRIEDEL_PAIRS_PRESENT: pass" in out
    assert "nominate b00-shelxc_shelxd" in out
    assert "because" in out


def test_cli_plan_reports_when_nothing_applies(tmp_path: Path, capsys):
    """Exit code 1 and a plain statement, rather than a traceback."""
    ins = write_ins(tmp_path / "bare.ins", latt=None, symm=(), sfac=("C",), unit=(4,))
    hkl = write_hkl(tmp_path / "bare.hkl")

    assert (
        main(
            [
                "plan",
                "--hkl",
                str(hkl),
                "--ins",
                str(ins),
                "--exclude-solver",
                SolverKind.SHELXT.value,
            ]
        )
        == 1
    )
    assert "not a crash" in capsys.readouterr().out


def test_cli_force_solver_is_recorded_as_an_override(tmp_path: Path, capsys):
    ins = write_ins(tmp_path / "case.ins")
    hkl = write_hkl(tmp_path / "case.hkl")

    assert (
        main(
            [
                "plan",
                "--hkl",
                str(hkl),
                "--ins",
                str(ins),
                "--force-solver",
                SolverKind.SHELXD_ABINITIO.value,
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "R09_USER_OVERRIDE: pass" in out
    assert "shelxd_abinitio" in out
    assert "override" in out


def test_cli_reports_a_parse_error_without_a_traceback(tmp_path: Path, capsys):
    bad = tmp_path / "bad.ins"
    bad.write_text("TITL x\nCELL 1 2\n", encoding="utf-8")
    hkl = write_hkl(tmp_path / "c.hkl")

    assert main(["diagnose", "--hkl", str(hkl), "--ins", str(bad)]) == 2
    err = capsys.readouterr().err
    assert "error: [PARSE_ERROR]" in err


def test_cli_policy_states_the_ordering(capsys):
    assert main(["policy"]) == 0
    out = capsys.readouterr().out
    assert "r1 then wr2" in out
    assert "no weighted score" in out


# --------------------------------------------------------------------------------------
# merge guard through the CLI
# --------------------------------------------------------------------------------------


def _merge_case(tmp_path: Path, name: str, wavelength: float) -> tuple[str, str]:
    ins = write_ins(tmp_path / f"{name}.ins", wavelength=wavelength)
    hkl = write_hkl(tmp_path / f"{name}.hkl")
    return str(ins), str(hkl)


def test_merge_check_allows_matching_copper_datasets(tmp_path, capsys):
    a = _merge_case(tmp_path, "cu1", 1.54184)
    b = _merge_case(tmp_path, "cu2", 1.54178)
    assert main(["merge-check", "--dataset", *a, "--dataset", *b]) == 0
    assert "MERGE ALLOWED" in capsys.readouterr().out


def test_merge_check_refuses_a_molybdenum_dataset_among_copper_ones(tmp_path, capsys):
    """Exit code 1 and a plain explanation, so a script can stop the merge."""
    a = _merge_case(tmp_path, "cu1", 1.54184)
    b = _merge_case(tmp_path, "mo1", 0.71073)
    assert main(["merge-check", "--dataset", *a, "--dataset", *b]) == 1
    out = capsys.readouterr().out
    assert "MERGE REFUSED" in out
    assert "radiation_mismatch" in out
    assert "different X-ray sources cannot be merged" in out


def test_merge_check_json_is_machine_readable(tmp_path, capsys):
    a = _merge_case(tmp_path, "cu1", 1.54184)
    b = _merge_case(tmp_path, "mo1", 0.71073)
    assert main(["merge-check", "--dataset", *a, "--dataset", *b, "--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["issues"][0]["kind"] == "radiation_mismatch"
    assert [f["radiation"] for f in payload["fingerprints"]] == ["Cu_Ka", "Mo_Ka"]


# --------------------------------------------------------------------------------------
# end to end
# --------------------------------------------------------------------------------------


def test_full_mini_run_produces_a_complete_record(tmp_path: Path, frozen_clock, fixed_entropy):
    user_data = tmp_path / "user_data"
    user_data.mkdir()
    ins = write_ins(user_data / "agpep.ins")
    hkl = write_hkl(user_data / "agpep.hkl", include_friedel=True)
    original_ins = ins.read_bytes()
    original_hkl = hkl.read_bytes()

    workspace = RunWorkspace.create(
        tmp_path / "runs", label="Ag peptide", clock=frozen_clock, entropy=fixed_entropy
    )
    workspace.stage_input("ins", ins)
    workspace.stage_input("hkl", hkl)

    # -- S0 diagnose ---------------------------------------------------------------
    header = read_ins_header(ins)
    reflections = scan_hkl(hkl, cell=header.cell, hklf_hint=header.hklf_format)
    diagnosis = DataDiagnosis(reflections=reflections, ins=header)
    prior = ChemicalPrior(metals=(MetalPrior("Ag", oxidation_state=1, expected_sites=2),))

    s0 = OnionStage(
        stage_id=make_stage_id(0, StageKind.DIAGNOSE.value), kind=StageKind.DIAGNOSE, index=0
    )
    workspace.open_stage(s0)
    workspace.write_json(f"stages/{s0.stage_id}/diagnosis.json", diagnosis)
    workspace.update_stage(s0.with_state(StageState.COMPLETED))

    # -- S1 initial solution -------------------------------------------------------
    s1 = OnionStage(
        stage_id=make_stage_id(1, StageKind.INITIAL_SOLUTION.value),
        kind=StageKind.INITIAL_SOLUTION,
        index=1,
    )
    workspace.open_stage(s1)
    trace = StrategySelector().select(diagnosis, prior, stage_id=s1.stage_id)
    workspace.write_selection_trace(s1.stage_id, trace)

    assert trace.nominated_kinds[0] is SolverKind.SHELXC_SHELXD

    registry = BackendRegistry()
    for nominated in trace.nominated:
        registry.register(
            MockBackend(
                nominated.kind,
                argv=(nominated.kind.value, "agpep"),
                expected_outputs=("agpep_fa.res",),
                hypothesis_ids=(f"h-{nominated.kind.value}",),
                figures_of_merit={"CC": 38.0},
            )
        )
    executor = ScriptedExecutor(
        [ScriptedResponse(creates=["agpep_fa.res"]) for _ in trace.nominated]
    )
    machine = InitialSolutionMachine(
        registry, workspace, runner=ProcessRunner(executor, base_env={})
    )
    portfolio = machine.run_portfolio(s1, trace)
    workspace.update_stage(s1.with_state(StageState.COMPLETED))

    assert portfolio.any_success
    assert all(r.branch.state is BranchState.SUCCEEDED for r in portfolio.reports)

    # -- S2 anchor: two competing candidates ---------------------------------------
    s2 = OnionStage(
        stage_id=make_stage_id(2, StageKind.ANCHOR.value), kind=StageKind.ANCHOR, index=2
    )
    workspace.open_stage(s2)

    def provenance(parent=None):
        return CandidateProvenance(
            created_at="2026-08-18T06:45:00+00:00",
            stage_id=s2.stage_id,
            branch_id=portfolio.reports[0].branch_id,
            parent_id=parent,
            reflection_file_sha256=reflections.sha256,
            hklf_format=int(reflections.hklf_format),
            tool_identities={"shelxl": "sha-of-shelxl"},
            commands=(("shelxl", "agpep_AI"),),
        )

    def metrics(r1: float) -> RefinementMetrics:
        return RefinementMetrics(
            r1=r1,
            wr2=r1 * 2.6,
            goof=1.03,
            converged=True,
            max_shift_over_esd=0.001,
            n_reflections_observed=4000,
            n_parameters=280,
            reflection_basis_key=(
                f"{reflections.sha256}|dmin=None|hklf={int(reflections.hklf_format)}|twin=none"
            ),
        )

    root = Candidate(
        candidate_id=make_candidate_id(0),
        provenance=provenance(),
        label="SHELXC/D substructure, Ag2",
        metrics=metrics(0.062),
        geometry=GeometryReport(charge_balance_net=0, charge_balance_checked=True),
        validation=ValidationReport("platon_special"),
    )
    workspace.add_candidate(root)

    flip = Candidate(
        candidate_id=make_candidate_id(1),
        provenance=provenance(parent=root.candidate_id),
        label="margin flip: N7 -> O7",
        mutations=(
            Mutation(
                MutationKind.ELEMENT_FLIP,
                "N7 -> O7 (smallest top-1/top-2 margin)",
                atoms=("N7",),
                from_value="N",
                to_value="O",
                driver="crystalx_margin",
                margin=0.02,
            ),
        ),
        metrics=metrics(0.041),
        geometry=GeometryReport(charge_balance_net=0, charge_balance_checked=True),
        validation=ValidationReport("platon_special"),
    )
    workspace.add_candidate(flip)

    tempting = Candidate(
        candidate_id=make_candidate_id(2),
        provenance=provenance(parent=root.candidate_id),
        label="lowest R, but a level A alert",
        metrics=metrics(0.028),
        geometry=GeometryReport(charge_balance_net=0, charge_balance_checked=True),
        validation=ValidationReport(
            "platon_special",
            alerts=(Alert("PLAT213", AlertSeverity.A, "Atom too close to a symmetry element"),),
        ),
    )
    workspace.add_candidate(tempting)

    anchor_lock = LockDirective(
        mode=LockMode.HARD_LOCK,
        atom_ids=("Ag1", "Ag2"),
        reason="substructure confirmed by SHELXD CC and an ANODE anomalous peak",
        instruction_lines=(
            f"Ag1   4  {format_fixed_shelxl_value(0.25)}  {format_fixed_shelxl_value(0.25)}  "
            f"{format_fixed_shelxl_value(0.25)}  11.00000  0.02000",
        ),
        applied_at_stage=str(s2.stage_id),
    )
    workspace.write_json(f"stages/{s2.stage_id}/lock_directives.json", [anchor_lock])
    workspace.events.emit(
        EventKind.LOCK_APPLIED, anchor_lock.reason, atoms=list(anchor_lock.atom_ids)
    )
    workspace.update_stage(s2.with_state(StageState.COMPLETED))

    # -- S5 gates and ranking -------------------------------------------------------
    candidates = workspace.lineage.iter_all()
    reports = evaluate_all(candidates, RankingPolicy())
    outcome = rank_candidates(candidates, reports, RankingPolicy(), heavy_elements=frozenset({"Ag"}))

    # The lowest R factor carries a level A alert, so it must not win.
    assert str(outcome.best.candidate_id) == "c001"
    assert [str(r.candidate_id) for r in outcome.ineligible] == ["c002"]
    assert outcome.review_checkpoints

    for report in reports:
        candidate = workspace.lineage.get(report.candidate_id)
        workspace.update_candidate(
            candidate.with_status(
                CandidateStatus.VALIDATED if report.is_eligible else CandidateStatus.REJECTED,
                reason="" if report.is_eligible else report.summary(),
            )
        )

    result = RunResult(
        run_id=workspace.run_id,
        stop_reason=StopReason(
            StopReasonKind.REVIEW_CHECKPOINT,
            "a level A alert on the lowest-R candidate needs a human decision",
            stage_id=s2.stage_id,
            suggested_next_action="inspect PLAT213 on c002 before accepting or rejecting it",
        ),
        ranked=outcome.ranked,
        ineligible=outcome.ineligible,
        review_checkpoints=outcome.review_checkpoints,
        stages_completed=(s0.stage_id, s1.stage_id, s2.stage_id),
    )
    workspace.finalize(result)

    # -- what the record must contain ----------------------------------------------
    root_dir = workspace.layout.root
    assert (root_dir / "run.json").exists()
    assert (root_dir / "result.json").exists()
    assert (root_dir / "inputs" / "manifest.json").exists()
    assert (root_dir / "stages" / str(s1.stage_id) / "selection_trace.json").exists()
    assert (root_dir / "candidates" / "index.json").exists()

    saved_trace = json.loads(
        (root_dir / "stages" / str(s1.stage_id) / "selection_trace.json").read_text()
    )
    assert [r["rule_id"] for r in saved_trace["rules"]][0] == "R01_FRIEDEL_PAIRS_PRESENT"
    assert saved_trace["excluded"]

    index = json.loads((root_dir / "candidates" / "index.json").read_text())
    assert index["count"] == 3
    rejected = [c for c in index["candidates"] if c["status"] == "rejected"]
    assert len(rejected) == 1
    assert "g7_no_unresolved_severe_alert" in rejected[0]["rejection_reason"]

    saved_result = json.loads((root_dir / "result.json").read_text())
    assert saved_result["stop_reason"]["kind"] == "review_checkpoint"
    assert saved_result["ranked"][0]["candidate_id"] == "c001"

    # the user's originals are byte-identical
    assert ins.read_bytes() == original_ins
    assert hkl.read_bytes() == original_hkl

    kinds = {e["kind"] for e in read_events(root_dir / "events.ndjson")}
    assert {
        EventKind.RUN_STARTED,
        EventKind.INPUT_STAGED,
        EventKind.STAGE_STARTED,
        EventKind.BRANCH_NOMINATED,
        EventKind.COMMAND_STARTED,
        EventKind.CANDIDATE_CREATED,
        EventKind.LOCK_APPLIED,
        EventKind.RUN_FINISHED,
    } <= kinds


def test_no_torch_or_heavy_dependency_is_imported():
    """crystalx_core must install and run with the standard library alone.

    ``gemmi`` is on the banned list even though it is an approved optional extra: the point is that
    importing the core must not pull it in. Only ``crystalx_core.providers.gemmi_provider`` may,
    and that module is deliberately absent from the import sweep below.
    """
    import subprocess
    import sys

    code = (
        "import sys, crystalx_core, crystalx_core.domain, crystalx_core.tools, "
        "crystalx_core.solvers, crystalx_core.workflow, crystalx_core.io, "
        "crystalx_core.workspace, crystalx_core.cli.main; "
        "banned = {'torch', 'torch_geometric', 'torch_scatter', 'torch_cluster', "
        "'numpy', 'scipy', 'rdkit', 'cctbx', 'iotbx', 'ase', 'gemmi'}; "
        "loaded = banned & set(sys.modules); "
        "print(sorted(loaded))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert completed.stdout.strip() == "[]"


# --------------------------------------------------------------------------------------
# packaging metadata
# --------------------------------------------------------------------------------------


def _read_pyproject() -> dict:
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - Python 3.10
        pytest.skip("tomllib needs Python 3.11+; the assertions below are metadata-only")
    root = Path(__file__).resolve().parents[1] / "pyproject.toml"
    if not root.exists():  # pragma: no cover - installed-wheel run
        pytest.skip("running against an installed distribution, not the source tree")
    return tomllib.loads(root.read_text(encoding="utf-8"))


def test_the_declared_version_comes_from_one_place():
    """A hard-coded version in ``pyproject.toml`` is how the two silently diverge.

    P2.0 flagged the risk; the fix is ``dynamic = ["version"]`` pointing at
    ``crystalx_core.version.CORE_VERSION``. This test fails if anyone puts the literal back.
    """
    from crystalx_core.version import CORE_VERSION

    project = _read_pyproject()["project"]
    assert "version" not in project, "version must stay dynamic, not be pinned in pyproject.toml"
    assert "version" in project["dynamic"]

    attr = _read_pyproject()["tool"]["setuptools"]["dynamic"]["version"]["attr"]
    assert attr == "crystalx_core.version.CORE_VERSION"
    assert CORE_VERSION.count(".") == 2, "expected MAJOR.MINOR.PATCH"


def test_the_core_declares_no_runtime_dependencies():
    """Asserted rather than trusted: the core must install on a machine with nothing else on it."""
    project = _read_pyproject()["project"]
    assert project["dependencies"] == [], "the core must install with nothing else on the machine"
    extras = project["optional-dependencies"]
    # Extras are allowed; hard dependencies are not. Each extra must be justified here so that
    # adding one is a deliberate act rather than a drift.
    assert set(extras) == {"dev", "geo"}
    assert all(spec.startswith("pytest") for spec in extras["dev"])
    assert all(spec.startswith("gemmi") for spec in extras["geo"])


def test_the_package_is_runnable_without_being_installed():
    """``python -m crystalx_core`` must work from a bare source tree, and warn about nothing.

    ``python -m crystalx_core.cli.main`` emits a ``RuntimeWarning`` from ``runpy`` because the
    ``cli`` package has already imported that module. ``__main__.py`` exists to give a clean route.
    """
    import subprocess
    import sys

    src = Path(__file__).resolve().parents[1] / "src"
    if not src.is_dir():  # pragma: no cover - installed-wheel run
        pytest.skip("running against an installed distribution, not the source tree")

    completed = subprocess.run(
        [sys.executable, "-W", "error::RuntimeWarning", "-m", "crystalx_core", "--version"],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(src), "PATH": os.environ.get("PATH", "")},
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip().startswith("crystalx-core ")
    assert "RuntimeWarning" not in completed.stderr


def test_every_pytest_marker_used_in_the_suite_is_registered():
    """``--strict-markers`` catches this at run time; this catches it in review."""
    registered = {
        entry.split(":", 1)[0]
        for entry in _read_pyproject()["tool"]["pytest"]["ini_options"]["markers"]
    }
    assert {"real_tools", "platon_data"} <= registered
