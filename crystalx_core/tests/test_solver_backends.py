"""The SHELXT vertical slice, and the heavy-atom route contracts.

Two things are being protected.

**SHELXT is a backend, not the architecture.** Its module is the only place in the package that
knows the string ``shelxt``; the selector, the branch model and the state machine stay
solver-agnostic. A test asserts that directly.

**The heavy-atom routes are first class, not "later".** SHELXC/SHELXD, SHELXE, SHELXS Patterson and
ANODE all have real applicability logic and real command shapes today. What they do not yet have is
verified output parsers, so they are marked plan-only and the machine refuses to execute them --
visibly, with a reason -- rather than producing an uninterpretable result.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import elf, write_executable, write_hkl, write_ins
from crystalx_core.domain.chemistry import ChemicalPrior, MetalPrior
from crystalx_core.domain.data import DataDiagnosis
from crystalx_core.domain.hypothesis import EvidenceKind
from crystalx_core.domain.ids import make_branch_id, make_stage_id
from crystalx_core.domain.onion import OnionStage, StageKind
from crystalx_core.domain.radiation import RadiationAssumption
from crystalx_core.domain.solving import (
    ApplicabilityVerdict,
    BackendReadiness,
    BranchBudget,
    BranchState,
    InitialSolutionBranch,
    SelectionTrace,
    SolverKind,
)
from crystalx_core.domain.tools import BinaryArch, HostPlatform, ToolKind
from crystalx_core.io.shelx_reader import read_ins_header, scan_hkl
from crystalx_core.solvers.base import BackendRegistry, backend_readiness
from crystalx_core.solvers.context import SolverContext, StagedCase
from crystalx_core.solvers.heavy_atom import (
    AnodeVerifyBackend,
    ShelxcShelxdBackend,
    ShelxeDensityModBackend,
    ShelxsPattersonBackend,
    build_shelxc_instructions,
    heavy_atom_backends,
)
from crystalx_core.solvers.machine import InitialSolutionMachine, make_plan_preview
from crystalx_core.solvers.mock import ScriptedExecutor, ScriptedResponse
from crystalx_core.solvers.shelxt import ShelxtBackend
from crystalx_core.tools.adapter import ExternalToolAdapter
from crystalx_core.tools.resolver import ResolverConfig, ToolResolver
from crystalx_core.tools.runner import ProcessRunner
from crystalx_core.workspace.store import RunWorkspace

STAGE_ID = make_stage_id(1, StageKind.INITIAL_SOLUTION.value)
LINUX = HostPlatform("Linux", BinaryArch.X86_64)

SHELXT_SOLUTION = """TITL agpep in P2(1)/c
CELL 1.54184 10.0000 11.0000 12.0000 90.000 100.000 90.000
ZERR 2.00 0.0010 0.0010 0.0010 0.000 0.000 0.000
LATT 1
SYMM -X, 0.5+Y, 0.5-Z
SFAC C N O Ag
UNIT 24 4 8 2
FVAR 0.5
AG1   4  0.25000  0.50000  0.75000  11.00000  0.02000
AG2   4  0.75000  0.50000  0.25000  11.00000  0.02100
N1    2  0.31000  0.42000  0.66000  11.00000  0.03000
C1    1  0.40000  0.35000  0.60000  11.00000  0.04000
C2    1  0.45000  0.30000  0.55000  11.00000  0.04100
HKLF 4
END
"""

SHELXT_ALTERNATIVE = SHELXT_SOLUTION.replace("P2(1)/c", "Pc").replace("LATT 1", "LATT -1")


def diagnose(case: dict[str, Path]) -> DataDiagnosis:
    header = read_ins_header(case["ins"])
    return DataDiagnosis(
        reflections=scan_hkl(case["hkl"], cell=header.cell, hklf_hint=header.hklf_format),
        ins=header,
    )


def make_context(case: dict[str, Path], *, adapters=None, prior=None) -> SolverContext:
    header = read_ins_header(case["ins"])
    return SolverContext(
        case=StagedCase(stem="agpep"),
        diagnosis=diagnose(case),
        prior=prior or ChemicalPrior(metals=(MetalPrior("Ag", expected_sites=2),)),
        radiation=RadiationAssumption.from_wavelength(header.wavelength_a),
        adapters=adapters or {},
    )


def branch(kind: SolverKind, index: int = 0, **kwargs) -> InitialSolutionBranch:
    kwargs.setdefault("budget", BranchBudget(max_wall_clock_s=60.0))
    return InitialSolutionBranch(
        branch_id=make_branch_id(index, kind.value), kind=kind, stage_id=STAGE_ID, **kwargs
    )


# --------------------------------------------------------------------------------------
# StagedCase
# --------------------------------------------------------------------------------------


def test_a_staged_case_derives_the_shelx_file_names():
    case = StagedCase(stem="agpep")
    assert case.ins_name == "agpep.ins"
    assert case.hkl_name == "agpep.hkl"
    assert case.declared_inputs == ("agpep.ins", "agpep.hkl")


def test_a_stem_with_a_space_or_slash_is_refused():
    """SHELX takes a bare stem; a path or a space silently changes what it reads."""
    from crystalx_core.errors import ConfigError

    for bad in ("has space", "dir/stem", ""):
        with pytest.raises(ConfigError):
            StagedCase(stem=bad)


def test_planning_works_without_a_resolved_tool(merged_case):
    """--dry-run must work on a machine where the program is not installed."""
    context = make_context(merged_case)
    assert context.executable_path(ToolKind.SHELXT) == "shelxt"


# --------------------------------------------------------------------------------------
# SHELXT: planning
# --------------------------------------------------------------------------------------


def test_shelxt_plans_a_single_invocation_on_the_stem(merged_case, tmp_path: Path):
    backend = ShelxtBackend(make_context(merged_case))
    plans = backend.plan(branch(SolverKind.SHELXT), work_dir=tmp_path)

    assert len(plans) == 1
    plan = plans[0]
    assert plan.argv == ("shelxt", "agpep")
    assert plan.cwd == tmp_path
    assert plan.expected_outputs == ("agpep_a.res",)
    assert plan.declared_inputs == ("agpep.ins", "agpep.hkl")
    assert plan.tool_kind is ToolKind.SHELXT


def test_shelxt_uses_the_resolved_executable_when_one_exists(
    merged_case, toolbox: Path, tmp_path: Path
):
    executable = write_executable(toolbox / "shelxt", elf(0x3E))
    resolution = ToolResolver(
        ResolverConfig(
            toolbox_dir=toolbox, search_path=[], include_known_locations=False, include_olex2=False
        ),
        host=LINUX,
        environ={"PATH": ""},
    ).resolve(ToolKind.SHELXT)
    adapter = ExternalToolAdapter(resolution, runner=ProcessRunner(ScriptedExecutor(), base_env={}))
    backend = ShelxtBackend(make_context(merged_case, adapters={ToolKind.SHELXT: adapter}))

    plan = backend.plan(branch(SolverKind.SHELXT), work_dir=tmp_path)[0]
    assert plan.argv[0] == str(executable)


def test_shelxt_respects_the_branch_time_budget(merged_case, tmp_path: Path):
    backend = ShelxtBackend(make_context(merged_case))
    plan = backend.plan(
        branch(SolverKind.SHELXT, budget=BranchBudget(max_wall_clock_s=45.0)), work_dir=tmp_path
    )[0]
    assert plan.timeout_s == pytest.approx(45.0)


def test_shelxt_is_applicable_but_downgraded_on_twinned_data(merged_case, tmp_path: Path):
    from crystalx_core.domain.data import HklfFormat

    context = make_context(merged_case)
    backend = ShelxtBackend(context)
    normal = backend.applicability(context.diagnosis, context.prior)
    assert normal.verdict is ApplicabilityVerdict.APPLICABLE

    twinned_header = read_ins_header(merged_case["ins"])
    twinned = DataDiagnosis(
        reflections=context.diagnosis.reflections,
        ins=type(twinned_header)(
            **{
                **{f: getattr(twinned_header, f) for f in twinned_header.__dataclass_fields__},
                "hklf_format": HklfFormat.HKLF5,
            }
        ),
    )
    downgraded = backend.applicability(twinned, context.prior)
    assert downgraded.priority < normal.priority
    assert any("twinned" in r for r in downgraded.reasons)


# --------------------------------------------------------------------------------------
# SHELXT: parsing
# --------------------------------------------------------------------------------------


def test_shelxt_reads_every_space_group_solution(merged_case, tmp_path: Path):
    (tmp_path / "agpep_a.res").write_text(SHELXT_SOLUTION, encoding="utf-8")
    (tmp_path / "agpep_b.res").write_text(SHELXT_ALTERNATIVE, encoding="utf-8")
    backend = ShelxtBackend(make_context(merged_case))

    solutions, warnings = backend.collect_solutions(tmp_path)

    assert [s.letter for s in solutions] == ["a", "b"]
    assert solutions[0].is_primary
    assert warnings == []
    assert solutions[0].n_atoms == 5
    assert "Ag" in solutions[0].elements


def test_an_unreadable_alternative_does_not_lose_the_others(merged_case, tmp_path: Path):
    (tmp_path / "agpep_a.res").write_text(SHELXT_SOLUTION, encoding="utf-8")
    (tmp_path / "agpep_b.res").write_text("TITL broken\nCELL 1 2\n", encoding="utf-8")
    backend = ShelxtBackend(make_context(merged_case))

    solutions, warnings = backend.collect_solutions(tmp_path)
    assert [s.letter for s in solutions] == ["a"]
    assert warnings and "agpep_b.res" in warnings[0]


def test_a_file_from_another_case_is_not_picked_up(merged_case, tmp_path: Path):
    (tmp_path / "agpep_a.res").write_text(SHELXT_SOLUTION, encoding="utf-8")
    (tmp_path / "other_a.res").write_text(SHELXT_SOLUTION, encoding="utf-8")
    solutions, _ = ShelxtBackend(make_context(merged_case)).collect_solutions(tmp_path)
    assert [s.res_path.name for s in solutions] == ["agpep_a.res"]


def test_the_hypothesis_keeps_only_heavy_atoms_and_flags_the_element_as_assumed(
    merged_case, tmp_path: Path
):
    """SHELXT assigns elements from peak heights -- the step CrystalX exists to replace."""
    (tmp_path / "agpep_a.res").write_text(SHELXT_SOLUTION, encoding="utf-8")
    backend = ShelxtBackend(make_context(merged_case))
    solutions, _ = backend.collect_solutions(tmp_path)

    hypothesis = backend.build_hypothesis(branch(SolverKind.SHELXT), solutions[0])
    assert [s.atom_id for s in hypothesis.sites] == ["AG1", "AG2"]
    assert all(site.element == "Ag" for site in hypothesis.sites)
    assert all(site.element_is_assumed for site in hypothesis.sites)
    assert hypothesis.confidence is None
    assert "starting guess" in hypothesis.notes
    assert hypothesis.evidence[0].kind is EvidenceKind.SHELXT_SOLUTION


def test_a_single_solution_is_not_independent_experimental_support(merged_case, tmp_path: Path):
    """One SHELXT solution is one source. Locking an anchor needs more than that."""
    (tmp_path / "agpep_a.res").write_text(SHELXT_SOLUTION, encoding="utf-8")
    backend = ShelxtBackend(make_context(merged_case))
    solutions, _ = backend.collect_solutions(tmp_path)
    hypothesis = backend.build_hypothesis(branch(SolverKind.SHELXT), solutions[0])
    assert not hypothesis.has_independent_experimental_support()


def test_a_label_element_disagreement_becomes_contradicting_evidence(merged_case, tmp_path: Path):
    broken = SHELXT_SOLUTION.replace("N1    2 ", "O9    2 ")
    (tmp_path / "agpep_a.res").write_text(broken, encoding="utf-8")
    backend = ShelxtBackend(make_context(merged_case))
    solutions, _ = backend.collect_solutions(tmp_path)
    hypothesis = backend.build_hypothesis(branch(SolverKind.SHELXT), solutions[0])
    assert hypothesis.contradicting_evidence()
    assert "disagree" in hypothesis.contradicting_evidence()[0].summary


def test_parse_reports_success_with_computed_figures_of_merit(merged_case, tmp_path: Path):
    backend = ShelxtBackend(make_context(merged_case))
    executor = ScriptedExecutor(
        [
            ScriptedResponse(
                creates=["agpep_a.res"], file_contents={"agpep_a.res": SHELXT_SOLUTION}
            )
        ]
    )
    runner = ProcessRunner(executor, base_env={})
    b = branch(SolverKind.SHELXT)
    outcomes = tuple(runner.run(p) for p in backend.plan(b, work_dir=tmp_path))

    outcome = backend.parse(b, outcomes)
    assert outcome.state is BranchState.SUCCEEDED
    assert outcome.figures_of_merit["n_solutions"] == 1.0
    assert outcome.figures_of_merit["n_atoms_primary"] == 5.0
    assert outcome.figures_of_merit["n_heavy_sites_primary"] == 2.0
    assert outcome.hypothesis_ids == ("b00-shelxt-a",)


def test_a_clean_exit_with_no_solution_is_a_failure_not_a_success(merged_case, tmp_path: Path):
    """Exit code zero and no solution file is caught by the expected-output check, not accepted."""
    backend = ShelxtBackend(make_context(merged_case))
    runner = ProcessRunner(ScriptedExecutor([ScriptedResponse(exit_code=0)]), base_env={})
    b = branch(SolverKind.SHELXT)
    outcomes = tuple(runner.run(p) for p in backend.plan(b, work_dir=tmp_path))

    outcome = backend.parse(b, outcomes)
    assert outcome.state is BranchState.FAILED
    assert outcome.diagnostic_code == "expected_output_missing"
    assert "did not produce" in outcome.summary


def test_a_solution_file_that_cannot_be_read_is_also_a_failure(merged_case, tmp_path: Path):
    """The program claimed success and wrote the file, but nothing in it is usable."""
    backend = ShelxtBackend(make_context(merged_case))
    executor = ScriptedExecutor(
        [ScriptedResponse(creates=["agpep_a.res"], file_contents={"agpep_a.res": "TITL x\nCELL 1 2\n"})]
    )
    runner = ProcessRunner(executor, base_env={})
    b = branch(SolverKind.SHELXT)
    outcomes = tuple(runner.run(p) for p in backend.plan(b, work_dir=tmp_path))

    outcome = backend.parse(b, outcomes)
    assert outcome.state is BranchState.FAILED
    assert outcome.diagnostic_code == "no_solution_parsed"
    assert "silent no-op is not a solution" in outcome.summary


def test_a_nonzero_exit_is_reported_with_the_program_message(merged_case, tmp_path: Path):
    backend = ShelxtBackend(make_context(merged_case))
    runner = ProcessRunner(
        ScriptedExecutor([ScriptedResponse(exit_code=2, stderr="cell too large\n")]), base_env={}
    )
    b = branch(SolverKind.SHELXT)
    outcomes = tuple(runner.run(p) for p in backend.plan(b, work_dir=tmp_path))
    outcome = backend.parse(b, outcomes)
    assert outcome.state is BranchState.FAILED
    assert "cell too large" in outcome.failure_reason


def test_parse_with_no_invocation_is_a_failure(merged_case):
    backend = ShelxtBackend(make_context(merged_case))
    outcome = backend.parse(branch(SolverKind.SHELXT), ())
    assert outcome.state is BranchState.FAILED
    assert "never invoked" in outcome.summary


def test_shelxt_is_production_ready(merged_case):
    assert backend_readiness(ShelxtBackend(make_context(merged_case))) is BackendReadiness.PRODUCTION


# --------------------------------------------------------------------------------------
# the core must not know about SHELXT
# --------------------------------------------------------------------------------------


def test_no_core_module_mentions_shelxt():
    """The selector, branch model and state machine must stay solver-agnostic.

    ``domain.solving`` legitimately names every route in its capability table; the modules checked
    here are the ones that must never special-case one program.
    """
    import crystalx_core.solvers.machine as machine
    import crystalx_core.workflow.onion as onion
    import crystalx_core.workflow.ranking as ranking

    for module in (machine, onion, ranking):
        source = Path(module.__file__).read_text(encoding="utf-8").lower()
        assert "shelxt" not in source, f"{module.__name__} mentions shelxt"


def test_the_selector_never_imports_a_backend():
    import crystalx_core.solvers.selector as selector

    source = Path(selector.__file__).read_text(encoding="utf-8")
    assert "from crystalx_core.solvers.shelxt" not in source
    assert "from crystalx_core.solvers.heavy_atom" not in source


# --------------------------------------------------------------------------------------
# heavy-atom routes
# --------------------------------------------------------------------------------------


def test_every_heavy_atom_backend_is_plan_only_for_now(merged_case):
    context = make_context(merged_case)
    for backend in heavy_atom_backends(context):
        assert backend_readiness(backend) is BackendReadiness.PLAN_ONLY


def test_shelxc_shelxd_plans_two_steps_in_the_right_order(anomalous_case, tmp_path: Path):
    """SHELXC measures the anomalous signal; SHELXD then solves. Two questions, two invocations."""
    backend = ShelxcShelxdBackend(make_context(anomalous_case))
    plans = backend.plan(
        branch(SolverKind.SHELXC_SHELXD, expected_sites=2, space_group_assumption="P2(1)/c"),
        work_dir=tmp_path,
    )

    assert [p.label for p in plans] == ["shelxc", "shelxd"]
    assert plans[0].argv == ("shelxc", "agpep")
    assert plans[0].stdin_script is not None
    assert "SAD agpep.hkl" in plans[0].stdin_script
    assert "FIND 2" in plans[0].stdin_script
    assert "SFAC Ag" in plans[0].stdin_script
    assert "SPAG P2(1)/c" in plans[0].stdin_script
    assert plans[0].expected_outputs == ("agpep_fa.ins", "agpep_fa.hkl")

    assert plans[1].argv == ("shelxd", "agpep_fa")
    assert plans[1].declared_inputs == ("agpep_fa.ins", "agpep_fa.hkl")
    assert plans[1].expected_outputs == ("agpep_fa.res",)


def test_shelxc_instructions_are_built_from_the_data_not_hard_coded():
    script = build_shelxc_instructions(
        stem="case",
        hkl_name="case.hkl",
        cell=(10.0, 11.0, 12.0, 90.0, 100.0, 90.0),
        space_group="P2(1)2(1)2(1)",
        find_sites=4,
        heavy_element="Ag",
    )
    assert script.splitlines() == [
        "CELL 10.0000 11.0000 12.0000 90.0000 100.0000 90.0000",
        "SPAG P2(1)2(1)2(1)",
        "SAD case.hkl",
        "FIND 4",
        "SFAC Ag",
        "MAXM 2",
    ]


def test_shelxc_shelxd_is_not_applicable_on_merged_data_and_says_why(merged_case):
    context = make_context(merged_case)
    verdict = ShelxcShelxdBackend(context).applicability(context.diagnosis, context.prior)
    assert verdict.verdict is ApplicabilityVerdict.NOT_APPLICABLE
    assert any("Friedel" in r for r in verdict.reasons)


def test_shelxc_shelxd_leads_when_anomalous_data_and_a_metal_are_present(anomalous_case):
    context = make_context(anomalous_case)
    verdict = ShelxcShelxdBackend(context).applicability(context.diagnosis, context.prior)
    assert verdict.verdict is ApplicabilityVerdict.APPLICABLE
    assert verdict.priority == 90


def test_shelxe_plans_both_hands(anomalous_case, tmp_path: Path):
    """Comparing the original and inverted hands is how a substructure is confirmed."""
    backend = ShelxeDensityModBackend(make_context(anomalous_case))
    plans = backend.plan(branch(SolverKind.SHELXE_DENSITY_MOD), work_dir=tmp_path)

    assert [p.label for p in plans] == ["shelxe_original_hand", "shelxe_inverted_hand"]
    assert "-i" not in plans[0].argv
    assert plans[1].argv[-1] == "-i"
    assert plans[0].expected_outputs == ("agpep.phs",)
    assert plans[1].expected_outputs == ("agpep_i.phs",)


def test_shelxe_autotracing_is_off_by_default(anomalous_case, tmp_path: Path):
    backend = ShelxeDensityModBackend(make_context(anomalous_case))
    plans = backend.plan(branch(SolverKind.SHELXE_DENSITY_MOD), work_dir=tmp_path)
    assert all("-a" not in p.argv for p in plans)


def test_shelxe_autotracing_is_refused_without_a_peptide_declaration(anomalous_case, tmp_path: Path):
    """Main-chain tracing assumes polypeptide geometry; on a metal-organic it invents nonsense."""
    backend = ShelxeDensityModBackend(make_context(anomalous_case))
    with pytest.raises(ValueError) as excinfo:
        backend.plan(
            branch(SolverKind.SHELXE_DENSITY_MOD, enable_autotracing=True), work_dir=tmp_path
        )
    assert "polypeptide geometry" in str(excinfo.value)


def test_shelxe_autotracing_is_allowed_for_a_declared_peptide(anomalous_case, tmp_path: Path):
    prior = ChemicalPrior(metals=(MetalPrior("Ag"),), is_peptide_like=True)
    backend = ShelxeDensityModBackend(make_context(anomalous_case, prior=prior))
    plans = backend.plan(
        branch(SolverKind.SHELXE_DENSITY_MOD, enable_autotracing=True), work_dir=tmp_path
    )
    assert all("-a" in p.argv for p in plans)


def test_shelxs_patterson_plans_its_own_instruction_file(merged_case, tmp_path: Path):
    backend = ShelxsPattersonBackend(make_context(merged_case))
    plan = backend.plan(branch(SolverKind.SHELXS_PATT), work_dir=tmp_path)[0]
    assert plan.argv == ("shelxs", "agpep_patt")
    assert plan.declared_inputs == ("agpep_patt.ins", "agpep_patt.hkl")


def test_shelxs_patterson_needs_a_heavy_atom_and_a_space_group(light_unknown_sg_case):
    context = make_context(light_unknown_sg_case, prior=ChemicalPrior())
    verdict = ShelxsPattersonBackend(context).applicability(context.diagnosis, context.prior)
    assert verdict.verdict is ApplicabilityVerdict.NOT_APPLICABLE
    assert any("space group" in r for r in verdict.reasons)


def test_patterson_is_described_as_the_classical_route_not_a_fallback(merged_case):
    context = make_context(merged_case)
    verdict = ShelxsPattersonBackend(context).applicability(context.diagnosis, context.prior)
    assert verdict.verdict is ApplicabilityVerdict.APPLICABLE
    assert any("classical route" in r for r in verdict.reasons)


def test_anode_needs_anomalous_data(merged_case, anomalous_case):
    merged_context = make_context(merged_case)
    assert AnodeVerifyBackend(merged_context).applicability(
        merged_context.diagnosis, merged_context.prior
    ).verdict is ApplicabilityVerdict.NOT_APPLICABLE

    anom_context = make_context(anomalous_case)
    assert AnodeVerifyBackend(anom_context).applicability(
        anom_context.diagnosis, anom_context.prior
    ).verdict is ApplicabilityVerdict.APPLICABLE


def test_a_plan_only_parse_never_invents_a_result(anomalous_case, tmp_path: Path):
    backend = ShelxcShelxdBackend(make_context(anomalous_case))
    b = branch(SolverKind.SHELXC_SHELXD)
    runner = ProcessRunner(ScriptedExecutor([ScriptedResponse(), ScriptedResponse()]), base_env={})
    outcomes = tuple(runner.run(p) for p in backend.plan(b, work_dir=tmp_path))

    outcome = backend.parse(b, outcomes)
    assert outcome.state is BranchState.FAILED
    assert outcome.diagnostic_code == "parser_not_implemented"
    assert "no scientific conclusion" in outcome.summary
    assert outcome.figures_of_merit == {}
    assert outcome.hypothesis_ids == ()


# --------------------------------------------------------------------------------------
# the machine refuses to run a plan-only backend
# --------------------------------------------------------------------------------------


@pytest.fixture
def workspace(tmp_path: Path, frozen_clock, fixed_entropy) -> RunWorkspace:
    return RunWorkspace.create(
        tmp_path / "runs", label="backend test", clock=frozen_clock, entropy=fixed_entropy
    )


def test_the_machine_blocks_a_plan_only_backend_with_a_stated_reason(workspace, anomalous_case):
    stage = OnionStage(stage_id=STAGE_ID, kind=StageKind.INITIAL_SOLUTION, index=1)
    workspace.open_stage(stage)
    registry = BackendRegistry()
    registry.register(ShelxcShelxdBackend(make_context(anomalous_case)))
    executor = ScriptedExecutor()
    machine = InitialSolutionMachine(
        registry, workspace, runner=ProcessRunner(executor, base_env={})
    )

    trace = SelectionTrace(rules=(), nominated=(branch(SolverKind.SHELXC_SHELXD),))
    result = machine.run_portfolio(stage, trace)

    report = result.reports[0]
    assert report.branch.state is BranchState.BLOCKED_BACKEND_PLAN_ONLY
    assert "plan-only backend" in report.outcome.failure_reason or (
        "plan-only backend" in report.outcome.summary
    )
    assert executor.calls == [], "a plan-only backend must not be executed"


def test_plan_only_can_be_opted_into_explicitly(workspace, anomalous_case):
    stage = OnionStage(stage_id=STAGE_ID, kind=StageKind.INITIAL_SOLUTION, index=1)
    workspace.open_stage(stage)
    registry = BackendRegistry()
    registry.register(ShelxcShelxdBackend(make_context(anomalous_case)))
    executor = ScriptedExecutor([ScriptedResponse(), ScriptedResponse()])
    machine = InitialSolutionMachine(
        registry,
        workspace,
        runner=ProcessRunner(executor, base_env={}),
        allow_plan_only=True,
    )

    result = machine.run_portfolio(
        stage, SelectionTrace(rules=(), nominated=(branch(SolverKind.SHELXC_SHELXD),))
    )
    assert executor.calls, "with the opt-in the commands do run"
    assert result.reports[0].outcome.diagnostic_code == "parser_not_implemented"


def test_dry_run_shows_the_heavy_atom_commands_without_executing_them(anomalous_case, tmp_path: Path):
    """The whole point of plan-only: the crystallographer can see what would run."""
    context = make_context(anomalous_case)
    registry = BackendRegistry()
    for backend in heavy_atom_backends(context):
        registry.register(backend)

    work_dir = tmp_path / "would_run_here"
    work_dir.mkdir()
    preview = make_plan_preview(
        registry,
        [branch(SolverKind.SHELXC_SHELXD, 0), branch(SolverKind.SHELXE_DENSITY_MOD, 1)],
        work_dir=work_dir,
    )
    text = json.dumps(preview)
    assert "shelxc" in text and "shelxd" in text and "shelxe" in text
    assert list(work_dir.iterdir()) == [], "planning must not touch the filesystem"


# --------------------------------------------------------------------------------------
# provenance flows through the machine for a real adapter
# --------------------------------------------------------------------------------------


def test_a_real_adapter_writes_a_provenance_file_per_invocation(
    workspace, merged_case, toolbox: Path
):
    write_executable(toolbox / "shelxt", elf(0x3E))
    resolution = ToolResolver(
        ResolverConfig(
            toolbox_dir=toolbox, search_path=[], include_known_locations=False, include_olex2=False
        ),
        host=LINUX,
        environ={"PATH": ""},
    ).resolve(ToolKind.SHELXT)
    executor = ScriptedExecutor(
        [ScriptedResponse(creates=["agpep_a.res"], file_contents={"agpep_a.res": SHELXT_SOLUTION})]
    )
    adapter = ExternalToolAdapter(resolution, runner=ProcessRunner(executor, base_env={}))

    stage = OnionStage(stage_id=STAGE_ID, kind=StageKind.INITIAL_SOLUTION, index=1)
    workspace.open_stage(stage)
    registry = BackendRegistry()
    registry.register(
        ShelxtBackend(make_context(merged_case, adapters={ToolKind.SHELXT: adapter}))
    )
    machine = InitialSolutionMachine(
        registry, workspace, adapters={ToolKind.SHELXT: adapter}
    )

    result = machine.run_portfolio(
        stage, SelectionTrace(rules=(), nominated=(branch(SolverKind.SHELXT),))
    )
    assert result.any_success

    provenance = (
        workspace.layout.branch_dir(STAGE_ID, make_branch_id(0, "shelxt"))
        / "provenance_00.json"
    )
    assert provenance.exists()
    payload = json.loads(provenance.read_text())
    assert payload["tool"] == "shelxt"
    assert payload["executable_sha256"]
    assert payload["executable_arch"] == "x86_64"
    assert "agpep_a.res" in [o["relative_path"] for o in payload["outputs"]]
