"""Synthetic safety contracts for the existing iterative workflow; no external program runs."""

from dataclasses import replace

import pytest

from test_onion_workflow import (
    PRIOR, RADIATION, RecordingExecutor, anchor_set, candidate, diagnosis,
    executors_for, run_workflow, workspace,
)
from crystalx_core.domain.candidate import CandidateStatus
from crystalx_core.domain.hypothesis import LockDirective, LockMode
from crystalx_core.domain.ids import make_candidate_id
from crystalx_core.domain.onion import StageBudget, StageKind, StageState
from crystalx_core.domain.result import ReviewCheckpoint, ReviewCheckpointKind, StopReasonKind
from crystalx_core.tools.runner import CancelToken
from crystalx_core.workflow.onion import (
    LockLedger, OnionPolicy, OnionWorkflow, StageControlStop, StageResourceMeter, StageResult,
)


def _base(**extra):
    entries = dict(initial_solution=RecordingExecutor(lambda r: StageResult()),
                   anchor=RecordingExecutor(lambda r: StageResult(anchor=anchor_set())),
                   first_shell=RecordingExecutor(lambda r: StageResult(candidates=(candidate(0, r.stage.stage_id),))),
                   finalize=RecordingExecutor(lambda r: StageResult()))
    entries.update(extra)
    return executors_for(**entries)


def test_ranking_and_low_residual_alone_never_establish_convergence(workspace, diagnosis):
    workflow, result = run_workflow(workspace, diagnosis, _base())
    assert result.best is not None
    assert result.stop_reason.kind is StopReasonKind.REVIEW_CHECKPOINT
    assert not result.succeeded
    assert not workflow._explicit_convergence


def test_explicit_shell_convergence_is_consumed_with_eligible_model(workspace, diagnosis):
    shell = RecordingExecutor(lambda r: StageResult(converged=True))
    finalize = RecordingExecutor(lambda r: StageResult())
    workflow, result = run_workflow(workspace, diagnosis, _base(shell_expansion=shell, finalize=finalize))
    assert len(shell.seen) == 1 and finalize.seen
    assert workflow.trace.shells_run == 1
    assert result.stop_reason.kind is StopReasonKind.CONVERGED


def test_convergence_without_an_accepted_model_is_not_consumed(workspace, diagnosis):
    executors = _base(first_shell=RecordingExecutor(lambda r: StageResult()),
                      shell_expansion=RecordingExecutor(lambda r: StageResult(converged=True)))
    _, result = run_workflow(workspace, diagnosis, executors)
    assert not result.succeeded
    assert result.stop_reason.kind is StopReasonKind.REVIEW_CHECKPOINT


@pytest.mark.parametrize("rollback", [False, True])
def test_standalone_human_requirement_always_stops_sequence(workspace, diagnosis, rollback):
    next_stage = RecordingExecutor(lambda r: StageResult())
    executors = _base(anchor=RecordingExecutor(lambda r: StageResult(anchor=anchor_set(),
                      requires_human_review=True, should_rollback=rollback)), first_shell=next_stage)
    _, result = run_workflow(workspace, diagnosis, executors)
    assert result.stop_reason.kind is StopReasonKind.REVIEW_CHECKPOINT
    assert not next_stage.seen


def test_a_blocking_checkpoint_stops_before_the_next_stage(workspace, diagnosis):
    next_stage = RecordingExecutor(lambda r: StageResult())
    point = ReviewCheckpoint(ReviewCheckpointKind.SUSPECTED_DISORDER, "synthetic decision", blocking=True)
    executors = _base(anchor=RecordingExecutor(lambda r: StageResult(anchor=anchor_set(), review_checkpoints=(point,))),
                      first_shell=next_stage)
    _, result = run_workflow(workspace, diagnosis, executors)
    assert result.stop_reason.kind is StopReasonKind.REVIEW_CHECKPOINT
    assert not next_stage.seen


def test_nonblocking_observation_does_not_prevent_explicit_convergence(workspace, diagnosis):
    point = ReviewCheckpoint(ReviewCheckpointKind.DATA_QUALITY, "recorded observation", blocking=False)
    executors = _base(anchor=RecordingExecutor(lambda r: StageResult(anchor=anchor_set(), review_checkpoints=(point,))),
                      shell_expansion=RecordingExecutor(lambda r: StageResult(converged=True)))
    _, result = run_workflow(workspace, diagnosis, executors)
    assert result.succeeded


def test_shell_count_exhaustion_stops_before_finalize(workspace, diagnosis):
    finalize = RecordingExecutor(lambda r: StageResult(converged=True))
    def shell(r):
        return StageResult(candidates=(candidate(len(workspace.lineage), r.stage.stage_id,
                                                parent=r.input_candidate.candidate_id),))
    workflow, result = run_workflow(workspace, diagnosis,
        _base(shell_expansion=RecordingExecutor(shell), finalize=finalize), OnionPolicy(max_shells=2))
    assert workflow.trace.shells_run == 2
    assert result.stop_reason.kind is StopReasonKind.BUDGET_EXCEEDED
    assert not finalize.seen


@pytest.mark.parametrize("field,amount", [("attempts_used", 2), ("tool_calls_used", 2), ("failures_used", 2)])
def test_post_return_usage_over_budget_is_rejected(workspace, diagnosis, field, amount):
    result = StageResult(candidates=(), **{field: amount})
    executors = _base(initial_solution=RecordingExecutor(lambda r: result))
    budgets = {StageKind.INITIAL_SOLUTION.value: StageBudget(max_attempts=1, max_tool_calls=1, max_failures=1)}
    workflow, outcome = run_workflow(workspace, diagnosis, executors, OnionPolicy(stage_budgets=budgets))
    assert outcome.stop_reason.kind is StopReasonKind.BUDGET_EXCEEDED
    assert workflow.trace.stages[-1].state is StageState.BUDGET_EXCEEDED


def test_excess_candidates_are_retained_but_never_promoted(workspace, diagnosis):
    def initial(r):
        return StageResult(candidates=tuple(candidate(n, r.stage.stage_id) for n in range(3)),
                           lock_directives=(LockDirective(LockMode.EXPLORATORY, ("M1",), "trial only"),))
    policy = OnionPolicy(stage_budgets={StageKind.INITIAL_SOLUTION.value: StageBudget(max_candidates=2)})
    workflow, outcome = run_workflow(workspace, diagnosis, _base(initial_solution=RecordingExecutor(initial)), policy)
    assert outcome.stop_reason.kind is StopReasonKind.BUDGET_EXCEEDED
    assert len(workspace.lineage) == 3 and not outcome.ranked
    assert all(c.status is CandidateStatus.ABANDONED for c in workspace.lineage.iter_all())
    assert workflow.ledger.active_ids() == () and len(workflow.ledger) == 1


def test_added_atom_cap_is_checked_against_observed_candidate_count(workspace, diagnosis):
    def initial(r):
        return StageResult(candidates=(replace(candidate(0, r.stage.stage_id), n_non_hydrogen_atoms=3),))
    policy = OnionPolicy(stage_budgets={StageKind.INITIAL_SOLUTION.value: StageBudget(max_new_atoms=2)})
    _, outcome = run_workflow(workspace, diagnosis, _base(initial_solution=RecordingExecutor(initial)), policy)
    assert outcome.stop_reason.kind is StopReasonKind.BUDGET_EXCEEDED


def test_unknown_added_atom_usage_under_a_cap_blocks(workspace, diagnosis):
    executors = _base(initial_solution=RecordingExecutor(lambda r: StageResult(candidates=(candidate(0, r.stage.stage_id),))))
    policy = OnionPolicy(stage_budgets={StageKind.INITIAL_SOLUTION.value: StageBudget(max_new_atoms=2)})
    _, outcome = run_workflow(workspace, diagnosis, executors, policy)
    assert outcome.stop_reason.kind is StopReasonKind.BUDGET_EXCEEDED


def test_reservation_prevents_second_proposal_and_call(workspace, diagnosis):
    calls = []
    def initial(r):
        for _ in range(2):
            r.meter.reserve(attempts=1, new_atoms=1)
            r.before_call()
            calls.append("synthetic provider call")
        return StageResult()
    policy = OnionPolicy(stage_budgets={StageKind.INITIAL_SOLUTION.value: StageBudget(max_attempts=1)})
    _, outcome = run_workflow(workspace, diagnosis, _base(initial_solution=RecordingExecutor(initial)), policy)
    assert calls == ["synthetic provider call"]
    assert outcome.stop_reason.kind is StopReasonKind.BUDGET_EXCEEDED


def test_reservation_and_observed_usage_are_not_double_charged(workspace, diagnosis):
    def initial(r):
        r.meter.reserve(attempts=1, new_atoms=1)
        r.before_call()
        return StageResult(attempts_used=1, tool_calls_used=1, new_atoms_added=1)
    policy = OnionPolicy(stage_budgets={StageKind.INITIAL_SOLUTION.value:
                                      StageBudget(max_attempts=1, max_tool_calls=1, max_new_atoms=1)})
    workflow, outcome = run_workflow(workspace, diagnosis, _base(initial_solution=RecordingExecutor(initial)), policy)
    assert workflow.trace.stages[0].state is StageState.COMPLETED
    assert outcome.stop_reason.kind is not StopReasonKind.BUDGET_EXCEEDED


def test_reservations_are_atomic():
    meter = StageResourceMeter(StageBudget(max_attempts=1, max_new_atoms=0))
    with pytest.raises(StageControlStop):
        meter.reserve(attempts=1, new_atoms=1)
    assert meter.used["attempts"] == 0


def test_cancellation_before_execute_prevents_any_stage_call(workspace, diagnosis):
    cancel = CancelToken()
    cancel.cancel()
    initial = RecordingExecutor(lambda r: StageResult())
    workflow = OnionWorkflow(workspace, _base(initial_solution=initial), cancel_token=cancel)
    result = workflow.run(diagnosis=diagnosis, prior=PRIOR, radiation=RADIATION)
    assert not initial.seen and result.stop_reason.kind is StopReasonKind.USER_ABORTED


def test_cooperative_cancellation_prevents_external_call(workspace, diagnosis):
    cancel = CancelToken()
    calls = []
    def initial(r):
        cancel.cancel("synthetic cancellation")
        r.before_call()
        calls.append("must never run")
        return StageResult()
    workflow = OnionWorkflow(workspace, _base(initial_solution=RecordingExecutor(initial)), cancel_token=cancel)
    outcome = workflow.run(diagnosis=diagnosis, prior=PRIOR, radiation=RADIATION)
    assert not calls and outcome.stop_reason.kind is StopReasonKind.USER_ABORTED


def test_wall_budget_is_checked_after_noncooperative_executor_returns(workspace, diagnosis):
    clock = [0.0]
    def initial(r):
        clock[0] = 2.0
        return StageResult(candidates=(candidate(0, r.stage.stage_id),))
    policy = OnionPolicy(stage_budgets={StageKind.INITIAL_SOLUTION.value: StageBudget(max_wall_clock_s=1)})
    workflow = OnionWorkflow(workspace, _base(initial_solution=RecordingExecutor(initial)), policy=policy,
                             clock=lambda: clock[0])
    outcome = workflow.run(diagnosis=diagnosis, prior=PRIOR, radiation=RADIATION)
    assert outcome.stop_reason.kind is StopReasonKind.BUDGET_EXCEEDED
    assert workspace.lineage.get(make_candidate_id(0)).status is CandidateStatus.ABANDONED


def test_only_selected_candidate_lock_set_becomes_active_and_ranked(workspace, diagnosis):
    def initial(r):
        return StageResult(candidates=(candidate(0, r.stage.stage_id, r1=0.01),
                                       candidate(1, r.stage.stage_id, r1=0.08)),
            accepted_candidate_id=make_candidate_id(1), candidate_lock_directives={
                "c000": (LockDirective(LockMode.EXPLORATORY, ("M0",), "not selected"),),
                "c001": (LockDirective(LockMode.SOFT_RESTRAINT, ("M1",), "accepted trial"),),
            })
    executors = _base(initial_solution=RecordingExecutor(initial), first_shell=RecordingExecutor(lambda r: StageResult()),
                      finalize=RecordingExecutor(lambda r: StageResult(converged=True)))
    workflow, outcome = run_workflow(workspace, diagnosis, executors)
    assert workflow.ledger.active_ids() == ("lock-001",)
    assert len(workflow.ledger) == 2
    assert outcome.best.candidate_id == make_candidate_id(1)
    assert workspace.lineage.get("c000").status is CandidateStatus.SUPERSEDED


def test_rejected_shell_restores_previous_locks_and_continues_next_shell(workspace, diagnosis):
    seen = []
    def initial(r):
        return StageResult(candidates=(candidate(0, r.stage.stage_id),),
                           lock_directives=(LockDirective(LockMode.SOFT_RESTRAINT, ("M0",), "accepted base"),))
    def shell(r):
        seen.append((r.shell, r.input_candidate.candidate_id, r.active_lock_ids))
        trial = candidate(r.shell, r.stage.stage_id, parent=r.input_candidate.candidate_id,
                          r1=0.001 if r.shell == 1 else 0.05)
        return StageResult(candidates=(trial,), should_rollback=r.shell == 1,
                           rollback_reason="synthetic rejected trial" if r.shell == 1 else "",
                           converged=r.shell == 2,
                           lock_directives=(LockDirective(LockMode.EXPLORATORY, (f"M{r.shell}",), "trial"),))
    executors = _base(initial_solution=RecordingExecutor(initial), first_shell=RecordingExecutor(lambda r: StageResult()),
                      shell_expansion=RecordingExecutor(shell), finalize=RecordingExecutor(lambda r: StageResult(converged=True)))
    workflow, outcome = run_workflow(workspace, diagnosis, executors)
    assert seen == [(1, "c000", ("lock-000",)), (2, "c000", ("lock-000",))]
    assert workflow.ledger.active_ids() == ("lock-000", "lock-002")
    assert len(workflow.ledger) == 3
    assert workspace.lineage.get("c001").status is CandidateStatus.ABANDONED
    assert outcome.best.candidate_id == "c002"
    assert all(r.candidate_id != "c001" for r in outcome.ranked)


def test_forged_candidate_lock_ids_do_not_escape_the_branch(workspace, diagnosis):
    def initial(r):
        trial = candidate(0, r.stage.stage_id)
        trial = replace(trial, provenance=replace(trial.provenance, lock_directive_ids=("foreign-lock",)))
        return StageResult(candidates=(trial,))
    workflow, _ = run_workflow(workspace, diagnosis, _base(initial_solution=RecordingExecutor(initial),
                                first_shell=RecordingExecutor(lambda r: StageResult())))
    assert workspace.lineage.get("c000").provenance.lock_directive_ids == ()
    assert workflow.ledger.active_ids() == ()


def test_sparse_lock_snapshot_restores_active_subset_and_next_identity():
    directive = LockDirective(LockMode.SOFT_RESTRAINT, ("M1",), "synthetic retained lock")
    ledger = LockLedger.from_snapshot({"lock-002": directive, "lock-010": directive},
                                      order=("lock-002", "lock-010"), active_ids=("lock-002",))
    assert ledger.next_id() == "lock-011"
    assert ledger.add(directive, activate=False) == "lock-011"
    assert ledger.active_ids() == ("lock-002",)
    ledger.restore_active(("lock-010",))
    assert ledger.active_ids() == ("lock-010",) and len(ledger) == 3
    with pytest.raises(ValueError):
        ledger.restore_active(("missing",))
    assert ledger.active_ids() == ("lock-010",)


class TransactionExecutor:
    def __init__(self, factory, *, commit_raises=False, abort_raises=False):
        self.factory = factory
        self.commit_raises = commit_raises
        self.abort_raises = abort_raises
        self.calls = []
        self.state = "previous accepted model"

    def execute(self, request):
        self.calls.append("execute")
        self.pending = "trial model"
        return self.factory(request)

    def commit_stage(self, request, chosen_candidate, active_lock_ids):
        self.calls.append("commit")
        assert chosen_candidate is not None
        assert chosen_candidate.provenance.lock_directive_ids == active_lock_ids
        self.state = self.pending
        if self.commit_raises:
            raise RuntimeError("synthetic sensitive exception text")

    def abort_stage(self, request, reason):
        self.calls.append("abort")
        self.state = "previous accepted model"
        if self.abort_raises:
            raise RuntimeError("synthetic abort detail")


def test_executor_state_commits_only_after_controller_acceptance(workspace, diagnosis):
    executor = TransactionExecutor(lambda r: StageResult(candidates=(candidate(0, r.stage.stage_id),)))
    run_workflow(workspace, diagnosis, _base(initial_solution=executor))
    assert executor.calls == ["execute", "commit"]
    assert executor.state == "trial model"


@pytest.mark.parametrize("failure", ["budget", "review", "rollback", "gate"])
def test_executor_pending_state_aborts_on_every_rejection_path(workspace, diagnosis, failure):
    def initial(r):
        trial = candidate(0, r.stage.stage_id)
        if failure == "gate":
            from crystalx_core.domain.metrics import GeometryReport
            trial = replace(trial, geometry=GeometryReport(evaluated=False))
        return StageResult(candidates=(trial,), attempts_used=2 if failure == "budget" else 1,
                           requires_human_review=failure == "review", should_rollback=failure == "rollback")
    executor = TransactionExecutor(initial)
    policy = OnionPolicy(stage_budgets={StageKind.INITIAL_SOLUTION.value: StageBudget(max_attempts=1)})
    run_workflow(workspace, diagnosis, _base(initial_solution=executor), policy)
    assert executor.calls == ["execute", "abort"]
    assert executor.state == "previous accepted model"


def test_commit_exception_restores_controller_and_provider_states(workspace, diagnosis):
    executor = TransactionExecutor(lambda r: StageResult(candidates=(candidate(0, r.stage.stage_id),),
        lock_directives=(LockDirective(LockMode.EXPLORATORY, ("M1",), "trial only"),)), commit_raises=True)
    workflow, result = run_workflow(workspace, diagnosis, _base(initial_solution=executor))
    assert executor.calls == ["execute", "commit", "abort"]
    assert executor.state == "previous accepted model"
    assert result.stop_reason.kind is StopReasonKind.INTERNAL_ERROR
    assert workflow.ledger.active_ids() == () and len(workflow.ledger) == 1
    assert workspace.lineage.get("c000").status is CandidateStatus.ABANDONED
    assert not result.ranked
    assert "sensitive exception text" not in repr(result)


def test_provider_exception_is_recorded_and_aborted_without_leaking_message(workspace, diagnosis):
    def provider_failure(r):
        raise RuntimeError("synthetic private provider detail")
    executor = TransactionExecutor(provider_failure)
    _, result = run_workflow(workspace, diagnosis, _base(initial_solution=executor))
    assert executor.calls == ["execute", "abort"]
    assert result.stop_reason.kind is StopReasonKind.INTERNAL_ERROR
    assert "private provider detail" not in repr(result)


def test_abort_hook_failure_is_not_called_a_second_time(workspace, diagnosis):
    executor = TransactionExecutor(lambda r: StageResult(requires_human_review=True), abort_raises=True)
    _, result = run_workflow(workspace, diagnosis, _base(initial_solution=executor))
    assert executor.calls == ["execute", "abort"]
    assert result.stop_reason.kind is StopReasonKind.INTERNAL_ERROR
