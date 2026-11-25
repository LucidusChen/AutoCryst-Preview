"""The branch state machine.

What is being protected: every branch is *kept* -- succeeded, failed, blocked, skipped, or never
applicable -- and each one carries the reason for its state. A branch that failed is evidence about
the data, so losing it loses information. The machine also never picks a winner; that happens later,
after the scientific gates.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from crystalx_core.domain.ids import make_branch_id, make_stage_id
from crystalx_core.domain.onion import OnionStage, RollbackDecision, StageKind, StageState
from crystalx_core.domain.solving import (
    BranchBudget,
    BranchState,
    InitialSolutionBranch,
    SelectionTrace,
    SolverKind,
)
from crystalx_core.domain.tools import BinaryArch, HostPlatform, ToolKind
from crystalx_core.solvers.base import BackendRegistry, tool_requirements_met
from crystalx_core.solvers.machine import (
    InitialSolutionMachine,
    apply_rollback,
    budget_summary,
    decide_rollback,
    make_plan_preview,
)
from crystalx_core.solvers.mock import MockBackend, ScriptedExecutor, ScriptedResponse
from crystalx_core.tools.resolver import ResolverConfig, ToolResolver
from crystalx_core.tools.runner import ProcessRunner
from crystalx_core.workspace.events import EventKind, read_events
from crystalx_core.workspace.store import RunWorkspace

STAGE_ID = make_stage_id(1, StageKind.INITIAL_SOLUTION.value)


@pytest.fixture
def workspace(tmp_path: Path, frozen_clock, fixed_entropy) -> RunWorkspace:
    return RunWorkspace.create(
        tmp_path / "runs", label="branch test", clock=frozen_clock, entropy=fixed_entropy
    )


@pytest.fixture
def stage(workspace: RunWorkspace) -> OnionStage:
    stage = OnionStage(stage_id=STAGE_ID, kind=StageKind.INITIAL_SOLUTION, index=1)
    workspace.open_stage(stage)
    return stage


def branch(kind: SolverKind, index: int = 0, **kwargs) -> InitialSolutionBranch:
    return InitialSolutionBranch(
        branch_id=make_branch_id(index, kind.value), kind=kind, stage_id=STAGE_ID, **kwargs
    )


# --------------------------------------------------------------------------------------
# transitions
# --------------------------------------------------------------------------------------


def test_legal_transition_sequence():
    b = branch(SolverKind.SHELXT)
    assert b.state is BranchState.PLANNED
    running = b.transition(BranchState.RUNNING)
    assert running.state is BranchState.RUNNING
    done = running.transition(BranchState.SUCCEEDED)
    assert done.state is BranchState.SUCCEEDED
    assert done.state.is_terminal


def test_illegal_transition_is_refused_with_the_allowed_set():
    b = branch(SolverKind.SHELXT)
    with pytest.raises(ValueError) as excinfo:
        b.transition(BranchState.SUCCEEDED)
    assert "Illegal branch transition planned -> succeeded" in str(excinfo.value)
    assert "running" in str(excinfo.value)


def test_terminal_states_cannot_be_left():
    done = branch(SolverKind.SHELXT).transition(BranchState.RUNNING).transition(BranchState.FAILED)
    with pytest.raises(ValueError):
        done.transition(BranchState.RUNNING)


def test_transition_preserves_the_branch_plan():
    original = branch(
        SolverKind.SHELXC_SHELXD,
        heavy_atom_assumption=("Ag",),
        use_anomalous=True,
        expected_sites=3,
        budget=BranchBudget(max_wall_clock_s=42.0, solver_trials=500),
    )
    moved = original.transition(BranchState.RUNNING)
    assert moved.heavy_atom_assumption == ("Ag",)
    assert moved.use_anomalous
    assert moved.expected_sites == 3
    assert moved.budget.solver_trials == 500


def test_budget_validation():
    with pytest.raises(ValueError):
        BranchBudget(max_wall_clock_s=0)
    with pytest.raises(ValueError):
        BranchBudget(max_attempts=0)


# --------------------------------------------------------------------------------------
# running the portfolio
# --------------------------------------------------------------------------------------


def _machine(workspace, registry, executor, availability=None) -> InitialSolutionMachine:
    return InitialSolutionMachine(
        registry,
        workspace,
        runner=ProcessRunner(executor, base_env={}),
        availability=availability,
    )


def test_successful_branch_is_recorded_with_its_outputs(workspace, stage):
    registry = BackendRegistry()
    registry.register(
        MockBackend(
            SolverKind.SHELXT,
            argv=("shelxt", "case"),
            expected_outputs=("case_a.res",),
            hypothesis_ids=("h0",),
            figures_of_merit={"CFOM": 0.31},
        )
    )
    executor = ScriptedExecutor([ScriptedResponse(creates=["case_a.res"])])
    trace = SelectionTrace(rules=(), nominated=(branch(SolverKind.SHELXT),))

    result = _machine(workspace, registry, executor).run_portfolio(stage, trace)

    assert result.any_success
    report = result.reports[0]
    assert report.branch.state is BranchState.SUCCEEDED
    assert report.outcome.figures_of_merit == {"CFOM": 0.31}
    assert result.hypothesis_ids() == ("h0",)

    outcome_json = json.loads(
        workspace.layout.branch_outcome_json(STAGE_ID, report.branch_id).read_text()
    )
    assert outcome_json["state"] == "succeeded"


def test_failed_branch_is_kept_with_its_reason(workspace, stage):
    registry = BackendRegistry()
    registry.register(MockBackend(SolverKind.SHELXT, argv=("shelxt", "case")))
    executor = ScriptedExecutor([ScriptedResponse(exit_code=1, stderr="no solution found")])
    trace = SelectionTrace(rules=(), nominated=(branch(SolverKind.SHELXT),))

    result = _machine(workspace, registry, executor).run_portfolio(stage, trace)

    assert not result.any_success
    report = result.failed[0]
    assert report.branch.state is BranchState.FAILED
    assert "no solution found" in report.outcome.failure_reason
    assert workspace.layout.branch_outcome_json(STAGE_ID, report.branch_id).exists()


def test_all_branches_run_by_default_so_routes_can_be_compared(workspace, stage):
    """Two independent routes are more informative than the first one that returns something."""
    registry = BackendRegistry()
    registry.register(MockBackend(SolverKind.SHELXT, argv=("shelxt", "case")))
    registry.register(MockBackend(SolverKind.SHELXS_PATT, argv=("shelxs", "case")))
    executor = ScriptedExecutor([ScriptedResponse(), ScriptedResponse()])
    trace = SelectionTrace(
        rules=(),
        nominated=(branch(SolverKind.SHELXT, 0), branch(SolverKind.SHELXS_PATT, 1)),
    )

    result = _machine(workspace, registry, executor).run_portfolio(stage, trace)
    assert len(result.succeeded) == 2


def test_stop_after_first_success_marks_the_rest_skipped(workspace, stage):
    registry = BackendRegistry()
    registry.register(MockBackend(SolverKind.SHELXT, argv=("shelxt", "case")))
    registry.register(MockBackend(SolverKind.SHELXS_PATT, argv=("shelxs", "case")))
    executor = ScriptedExecutor([ScriptedResponse()])
    trace = SelectionTrace(
        rules=(),
        nominated=(branch(SolverKind.SHELXT, 0), branch(SolverKind.SHELXS_PATT, 1)),
    )

    result = _machine(workspace, registry, executor).run_portfolio(
        stage, trace, stop_after_first_success=True
    )
    assert result.reports[1].branch.state is BranchState.SKIPPED
    assert "skipped" in result.reports[1].outcome.summary


def test_multi_step_branch_stops_at_the_first_failure(workspace, stage):
    """SHELXC -> SHELXD -> SHELXE: if SHELXC produces nothing, SHELXD must not run."""
    registry = BackendRegistry()
    registry.register(
        MockBackend(SolverKind.SHELXC_SHELXD, argv=("shelxc", "case"), n_plans=3)
    )
    executor = ScriptedExecutor(
        [ScriptedResponse(), ScriptedResponse(exit_code=2, stderr="no anomalous signal")]
    )
    trace = SelectionTrace(rules=(), nominated=(branch(SolverKind.SHELXC_SHELXD),))

    result = _machine(workspace, registry, executor).run_portfolio(stage, trace)

    assert len(executor.calls) == 2
    assert result.reports[0].branch.state is BranchState.FAILED
    assert "no anomalous signal" in result.reports[0].outcome.failure_reason


def test_branch_blocked_when_its_program_is_unusable(workspace, stage, toolbox):
    """The Rosetta case again, this time seen from the branch machine."""
    from conftest import CPU_X86_64, macho_thin, write_executable

    write_executable(toolbox / "shelxt", macho_thin(CPU_X86_64))
    availability = ToolResolver(
        ResolverConfig(
            toolbox_dir=toolbox, search_path=[], include_known_locations=False, include_olex2=False
        ),
        host=HostPlatform("Darwin", BinaryArch.ARM64, rosetta_available=False),
        environ={"PATH": ""},
    ).resolve_all([ToolKind.SHELXT])

    registry = BackendRegistry()
    registry.register(MockBackend(SolverKind.SHELXT, argv=("shelxt", "case")))
    executor = ScriptedExecutor()
    trace = SelectionTrace(rules=(), nominated=(branch(SolverKind.SHELXT),))

    result = _machine(workspace, registry, executor, availability).run_portfolio(stage, trace)

    report = result.reports[0]
    assert report.branch.state is BranchState.BLOCKED_TOOL_UNAVAILABLE
    assert "Rosetta" in report.outcome.failure_reason
    assert executor.calls == []


def test_missing_backend_is_skipped_not_crashed(workspace, stage):
    registry = BackendRegistry()  # empty
    trace = SelectionTrace(rules=(), nominated=(branch(SolverKind.SHELXT),))
    result = _machine(workspace, registry, ScriptedExecutor()).run_portfolio(stage, trace)
    assert result.reports[0].branch.state is BranchState.SKIPPED
    assert "no backend is registered" in result.reports[0].outcome.summary


def test_excluded_solvers_are_logged_with_their_reason(workspace, stage):
    trace = SelectionTrace(
        rules=(),
        nominated=(),
        excluded=((SolverKind.SHELXC_SHELXD, "Friedel mates are merged"),),
    )
    _machine(workspace, BackendRegistry(), ScriptedExecutor()).run_portfolio(stage, trace)
    events = [
        e
        for e in read_events(workspace.layout.events)
        if e["kind"] == EventKind.BRANCH_NOT_APPLICABLE
    ]
    assert events and "Friedel" in events[0]["reason"]


def test_commands_are_recorded_before_they_run(workspace, stage):
    registry = BackendRegistry()
    registry.register(MockBackend(SolverKind.SHELXT, argv=("shelxt", "case")))
    executor = ScriptedExecutor([ScriptedResponse()])
    trace = SelectionTrace(rules=(), nominated=(branch(SolverKind.SHELXT),))

    _machine(workspace, registry, executor).run_portfolio(stage, trace)

    command_file = (
        workspace.layout.branch_dir(STAGE_ID, make_branch_id(0, "shelxt")) / "command_00.json"
    )
    assert json.loads(command_file.read_text())["argv"] == ["shelxt", "case"]


def test_duplicate_backend_registration_is_refused():
    registry = BackendRegistry()
    registry.register(MockBackend(SolverKind.SHELXT))
    with pytest.raises(Exception):
        registry.register(MockBackend(SolverKind.SHELXT))


# --------------------------------------------------------------------------------------
# rollback
# --------------------------------------------------------------------------------------


def test_no_rollback_when_a_branch_succeeded(workspace, stage):
    registry = BackendRegistry()
    registry.register(MockBackend(SolverKind.SHELXT, argv=("shelxt", "case")))
    trace = SelectionTrace(rules=(), nominated=(branch(SolverKind.SHELXT),))
    result = _machine(workspace, registry, ScriptedExecutor([ScriptedResponse()])).run_portfolio(
        stage, trace
    )
    assert decide_rollback(stage=stage, result=result, fallback_stage=make_stage_id(0, "diagnose")) is None


def test_rollback_names_the_abandoned_work_and_asks_for_a_human_when_nothing_is_left(
    workspace, stage
):
    registry = BackendRegistry()
    registry.register(MockBackend(SolverKind.SHELXT, argv=("shelxt", "case")))
    trace = SelectionTrace(rules=(), nominated=(branch(SolverKind.SHELXT),))
    result = _machine(
        workspace, registry, ScriptedExecutor([ScriptedResponse(exit_code=1)])
    ).run_portfolio(stage, trace)

    decision = decide_rollback(
        stage=stage, result=result, fallback_stage=make_stage_id(0, StageKind.DIAGNOSE.value)
    )
    assert decision is not None
    assert decision.abandoned_branches == (make_branch_id(0, "shelxt"),)
    assert decision.requires_human_review is True


def test_rollback_selects_an_alternative_when_one_remains(workspace, stage):
    registry = BackendRegistry()
    registry.register(MockBackend(SolverKind.SHELXT, argv=("shelxt", "case")))
    trace = SelectionTrace(rules=(), nominated=(branch(SolverKind.SHELXT),))
    result = _machine(
        workspace, registry, ScriptedExecutor([ScriptedResponse(exit_code=1)])
    ).run_portfolio(stage, trace)

    alternative = make_branch_id(1, "shelxs_patt")
    decision = decide_rollback(
        stage=stage,
        result=result,
        fallback_stage=make_stage_id(0, StageKind.DIAGNOSE.value),
        remaining_alternatives=[alternative],
    )
    assert decision is not None
    assert decision.alternative_selected == alternative
    assert decision.requires_human_review is False


def test_applying_a_rollback_records_it_and_keeps_everything(workspace, stage):
    decision = RollbackDecision(
        from_stage=stage.stage_id,
        to_stage=make_stage_id(0, StageKind.DIAGNOSE.value),
        reason="every route failed on this data",
        abandoned_branches=(make_branch_id(0, "shelxt"),),
    )
    rolled = apply_rollback(workspace, stage, decision)

    assert rolled.state is StageState.ROLLED_BACK
    assert (workspace.layout.stage_dir(stage.stage_id) / "rollback.json").exists()
    kinds = [e["kind"] for e in read_events(workspace.layout.events)]
    assert EventKind.STAGE_ROLLED_BACK in kinds


def test_rollback_requires_a_reason_and_a_different_stage():
    with pytest.raises(ValueError):
        RollbackDecision(from_stage=STAGE_ID, to_stage=make_stage_id(0, "diagnose"), reason="  ")
    with pytest.raises(ValueError):
        RollbackDecision(from_stage=STAGE_ID, to_stage=STAGE_ID, reason="loop")


# --------------------------------------------------------------------------------------
# dry run
# --------------------------------------------------------------------------------------


def test_plan_preview_shows_the_commands_without_running_them(tmp_path: Path):
    """Backs ``--dry-run``: the user sees exactly what would touch their installation."""
    registry = BackendRegistry()
    registry.register(
        MockBackend(SolverKind.SHELXT, argv=("shelxt", "case"), expected_outputs=("case_a.res",))
    )
    preview = make_plan_preview(registry, [branch(SolverKind.SHELXT)], work_dir=tmp_path)
    assert preview[0]["plans"][0]["argv"] == ["shelxt", "case"]
    assert preview[0]["plans"][0]["expected_outputs"] == ["case_a.res"]
    assert list(tmp_path.iterdir()) == []


def test_plan_preview_reports_a_missing_backend(tmp_path: Path):
    preview = make_plan_preview(BackendRegistry(), [branch(SolverKind.SHELXT)], work_dir=tmp_path)
    assert preview[0]["note"] == "no backend registered"


def test_budget_summary_totals_the_portfolio():
    summary = budget_summary(
        [
            branch(SolverKind.SHELXT, 0, budget=BranchBudget(max_wall_clock_s=100.0)),
            branch(SolverKind.SHELXS_PATT, 1, budget=BranchBudget(max_wall_clock_s=200.0)),
        ]
    )
    assert summary["total_s"] == 300.0


def test_tool_requirements_reports_each_missing_program(toolbox):
    availability = ToolResolver(
        ResolverConfig(
            toolbox_dir=toolbox, search_path=[], include_known_locations=False, include_olex2=False
        ),
        host=HostPlatform("Linux", BinaryArch.X86_64),
        environ={"PATH": ""},
    ).resolve_all([ToolKind.SHELXC, ToolKind.SHELXD])

    ok, missing = tool_requirements_met(SolverKind.SHELXC_SHELXD, availability)
    assert not ok
    assert len(missing) == 2
    assert any("shelxc" in m for m in missing)
    assert any("shelxd" in m for m in missing)
