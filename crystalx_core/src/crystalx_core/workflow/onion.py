"""The peeling-onion workflow, as an executable state machine.

The stages are not documentation. This module drives them:

* **S1** find the heavy atoms / metal nodes, through the solver portfolio and the branch machine;
* **S2** validate those nodes and their coordination, keeping competing hypotheses alive;
* **S3** grow the first coordination sphere -- donor atoms and the fragments hanging off them;
* **S4** expand the organic skeleton shell by shell, each round free to alternate inference,
  refinement, difference-density inspection and validation;
* **S5** finalise, compare candidates, and either converge, ask a human, or stop safely.

Two rules are enforced here rather than left to each executor.

**A lock is never silent.** Every :class:`LockDirective` an executor returns is given an id, written
to the stage directory and logged as an event. Its intended trial candidates reference it in their
provenance; only the selected eligible candidate's set becomes active. Rejected directives stay
in history without contaminating the accepted branch.

**A low R factor cannot buy acceptance.** After every stage the candidates go through the eligibility
gates before anything is ranked, so a chemically implausible model is ineligible no matter how well
it fits.

The scientific work itself lives behind :class:`StageExecutor`, so this driver stays independent of
which solver, which refinement program, and which inference engine are in use.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace
from typing import Callable, Mapping, Protocol, Sequence

from crystalx_core.domain.candidate import Candidate, CandidateStatus, Lineage
from crystalx_core.domain.chemistry import ChemicalPrior
from crystalx_core.domain.data import DataDiagnosis
from crystalx_core.domain.gates import EligibilityReport, GateOutcome, RankingPolicy
from crystalx_core.domain.hypothesis import AnchorSet, LockDirective, LockMode
from crystalx_core.domain.ids import CandidateId, StageId, make_stage_id
from crystalx_core.domain.onion import (
    OnionStage,
    RollbackDecision,
    StageBudget,
    StageKind,
    StagePolicy,
    StageState,
)
from crystalx_core.domain.radiation import RadiationAssumption
from crystalx_core.domain.result import (
    ReviewCheckpoint,
    ReviewCheckpointKind,
    RunResult,
    StopReason,
    StopReasonKind,
)
from crystalx_core.errors import WorkspaceError
from crystalx_core.tools.runner import CancelToken
from crystalx_core.workflow.gates import evaluate_all
from crystalx_core.workflow.ranking import RankingOutcome, rank_candidates
from crystalx_core.workspace.events import EventKind
from crystalx_core.workspace.store import RunWorkspace

__all__ = [
    "StageRequest",
    "StageResult",
    "StageExecutor",
    "OnionPolicy",
    "LockLedger",
    "OnionWorkflow",
    "WorkflowTrace",
    "StageResourceMeter",
    "StageControlStop",
]


@dataclass
class LockLedger:
    """Assigns ids to lock directives and keeps them addressable.

    Provenance stores ids rather than whole directives so that one directive applied across many
    candidates is recorded once and referenced many times, and so a reader can ask "what was
    ``lock-002`` and why" and get a single answer.
    """

    directives: dict[str, LockDirective] = field(default_factory=dict)
    order: list[str] = field(default_factory=list)
    active: list[str] = field(default_factory=list)

    def next_id(self) -> str:
        suffixes = [int(key[5:]) for key in self.directives if key.startswith("lock-") and key[5:].isdigit()]
        index = max(suffixes, default=-1) + 1
        while f"lock-{index:03d}" in self.directives:
            index += 1
        return f"lock-{index:03d}"

    def add(self, directive: LockDirective, *, activate: bool = True) -> str:
        lock_id = self.next_id()
        self.directives[lock_id] = directive
        self.order.append(lock_id)
        if activate:
            self.active.append(lock_id)
        return lock_id

    def add_all(self, directives: Sequence[LockDirective], *, activate: bool = True) -> tuple[str, ...]:
        return tuple(self.add(d, activate=activate) for d in directives)

    def get(self, lock_id: str) -> LockDirective:
        return self.directives[lock_id]

    def active_ids(self) -> tuple[str, ...]:
        """The accepted branch's active set, distinct from append-only issued history."""
        return tuple(self.active)

    def restore_active(self, ids: Sequence[str]) -> None:
        restored = tuple(ids)
        if len(set(restored)) != len(restored) or any(key not in self.directives for key in restored):
            raise ValueError("active lock ids must be unique and present in the history")
        self.active[:] = restored

    @classmethod
    def from_snapshot(cls, directives: Mapping[str, LockDirective], *, order: Sequence[str] | None = None,
                      active_ids: Sequence[str] = ()) -> "LockLedger":
        history = list(directives) if order is None else list(order)
        if len(set(history)) != len(history) or set(history) != set(directives):
            raise ValueError("snapshot order must name each historical lock exactly once")
        if any(not isinstance(value, LockDirective) for value in directives.values()):
            raise ValueError("snapshot directives must be parsed LockDirective objects")
        ledger = cls(dict(directives), history)
        ledger.restore_active(active_ids)
        return ledger

    def by_mode(self, mode: LockMode) -> tuple[tuple[str, LockDirective], ...]:
        return tuple(
            (lock_id, self.directives[lock_id])
            for lock_id in self.order
            if self.directives[lock_id].mode is mode
        )

    def __len__(self) -> int:
        return len(self.order)


class StageControlStop(RuntimeError):
    """Cooperative pre-work cancellation or resource bound; caught by the driver."""

    def __init__(self, kind: StopReasonKind, message: str) -> None:
        super().__init__(message)
        self.kind = kind


@dataclass
class StageResourceMeter:
    """Finite reservations before expensive work, reconciled with returned evidence afterward.

    The driver checks on both sides of execute(). A Python executor must call checkpoint/reserve
    at its internal work boundaries; this cannot preempt an arbitrary blocking Python function.
    External process deadlines and cancellation remain the ProcessRunner's responsibility.
    """

    budget: StageBudget
    clock: Callable[[], float] = time.monotonic
    cancel_token: CancelToken | None = None
    used: dict[str, int] = field(default_factory=lambda: {
        "attempts": 0, "candidates": 0, "new_atoms": 0, "tool_calls": 0, "failures": 0,
    })
    started: float = field(init=False)

    def __post_init__(self) -> None:
        self.started = self.clock()

    def checkpoint(self) -> None:
        if self.cancel_token is not None and self.cancel_token.cancelled:
            raise StageControlStop(StopReasonKind.USER_ABORTED,
                                   self.cancel_token.reason or "cancelled by user")
        elapsed = self.clock() - self.started
        if not math.isfinite(elapsed) or elapsed < 0:
            raise StageControlStop(StopReasonKind.BUDGET_EXCEEDED, "stage resource clock is invalid")
        if self.budget.max_wall_clock_s is not None and elapsed >= self.budget.max_wall_clock_s:
            raise StageControlStop(StopReasonKind.BUDGET_EXCEEDED, "stage wall-clock budget exhausted")

    def reserve(self, *, attempts: int = 0, candidates: int = 0, new_atoms: int = 0,
                tool_calls: int = 0, failures: int = 0) -> None:
        self.checkpoint()
        requested = {"attempts": attempts, "candidates": candidates, "new_atoms": new_atoms,
                     "tool_calls": tool_calls, "failures": failures}
        limits = {"attempts": self.budget.max_attempts, "candidates": self.budget.max_candidates,
                  "new_atoms": self.budget.max_new_atoms, "tool_calls": self.budget.max_tool_calls,
                  "failures": self.budget.max_failures}
        for key, count in requested.items():
            if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                raise ValueError("resource reservations must be nonnegative integers")
            if limits[key] is not None and self.used[key] + count > limits[key]:
                raise StageControlStop(StopReasonKind.BUDGET_EXCEEDED, f"stage {key} budget exhausted")
        for key, count in requested.items():
            self.used[key] += count

    def reconcile(self, **observed: int) -> None:
        """Ensure at least observed usage, without charging a prior reservation twice."""
        for key, value in observed.items():
            if key not in self.used or not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError("observed resource usage must contain known nonnegative integer counts")
        # Unlike a reservation, observed work has already happened. Keep it even on overrun.
        self.used.update({key: max(value, self.used[key]) for key, value in observed.items()})
        self.reserve()

    def before_call(self) -> None:
        self.reserve(tool_calls=1)


@dataclass
class StageRequest:
    """Everything a stage executor is given."""

    stage: OnionStage
    workspace: RunWorkspace
    diagnosis: DataDiagnosis
    prior: ChemicalPrior
    radiation: RadiationAssumption
    lineage: Lineage
    anchor: AnchorSet | None = None
    input_candidate: Candidate | None = None
    active_lock_ids: tuple[str, ...] = ()
    ledger: LockLedger | None = None
    meter: StageResourceMeter | None = None

    def checkpoint(self) -> None:
        if self.meter is not None:
            self.meter.checkpoint()

    def before_call(self) -> None:
        if self.meter is not None:
            self.meter.before_call()

    @property
    def shell(self) -> int | None:
        return self.stage.shell


@dataclass
class StageResult:
    """Everything a stage executor gives back.

    An executor that achieved nothing says so with ``should_rollback``; it never invents a candidate
    to keep the workflow moving.
    """

    candidates: tuple[Candidate, ...] = ()
    anchor: AnchorSet | None = None
    lock_directives: tuple[LockDirective, ...] = ()
    review_checkpoints: tuple[ReviewCheckpoint, ...] = ()
    should_rollback: bool = False
    rollback_reason: str = ""
    requires_human_review: bool = False
    converged: bool = False
    """For shell expansion: nothing further can be added from the current evidence."""

    notes: tuple[str, ...] = ()
    summary: str = ""
    accepted_candidate_id: CandidateId | None = None
    candidate_lock_directives: Mapping[str, tuple[LockDirective, ...]] = field(default_factory=dict)
    attempts_used: int = 1
    tool_calls_used: int = 0
    failures_used: int = 0
    new_atoms_added: int | None = None


class StageExecutor(Protocol):
    """Performs the scientific work of one onion stage."""

    def execute(self, request: StageRequest) -> StageResult: ...


@dataclass(frozen=True)
class OnionPolicy:
    """Bounds and behaviour for a whole run."""

    max_shells: int = 6
    """How many times S4 may repeat. A structure that still is not complete after this needs a
    human, not another automatic round."""

    stop_on_first_rollback: bool = False
    ranking_policy: RankingPolicy = field(default_factory=RankingPolicy)
    stage_budgets: Mapping[str, StageBudget] = field(default_factory=dict)
    stage_policies: Mapping[str, StagePolicy] = field(default_factory=dict)
    heavy_elements: frozenset[str] = frozenset()
    require_anchor_before_expansion: bool = True
    """Refuse to grow shells before a metal core has been validated. Growing outward from an
    unvalidated anchor is how a plausible-looking wrong structure gets built."""

    def __post_init__(self) -> None:
        if not isinstance(self.max_shells, int) or isinstance(self.max_shells, bool) or self.max_shells < 1:
            raise ValueError("max_shells must be a positive integer")

    def budget_for(self, kind: StageKind) -> StageBudget:
        return self.stage_budgets.get(kind.value, StageBudget())

    def policy_for(self, kind: StageKind) -> StagePolicy:
        return self.stage_policies.get(kind.value, StagePolicy())


@dataclass
class WorkflowTrace:
    """The path a run actually took, stage by stage."""

    stages: list[OnionStage] = field(default_factory=list)
    rollbacks: list[RollbackDecision] = field(default_factory=list)
    checkpoints: list[ReviewCheckpoint] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    shells_run: int = 0

    @property
    def completed_stage_ids(self) -> tuple[StageId, ...]:
        return tuple(s.stage_id for s in self.stages if s.state is StageState.COMPLETED)


class OnionWorkflow:
    """Drives the stages, applies the locks, and refuses to accept an ineligible candidate."""

    #: Stage order, and the index each one occupies in the run directory.
    SEQUENCE: tuple[tuple[int, StageKind], ...] = (
        (1, StageKind.INITIAL_SOLUTION),
        (2, StageKind.ANCHOR),
        (3, StageKind.FIRST_SHELL),
        (4, StageKind.SHELL_EXPANSION),
        (5, StageKind.FINALIZE),
    )

    def __init__(
        self,
        workspace: RunWorkspace,
        executors: Mapping[StageKind, StageExecutor],
        *,
        policy: OnionPolicy | None = None,
        clock: Callable[[], float] = time.monotonic,
        cancel_token: CancelToken | None = None,
    ) -> None:
        self.workspace = workspace
        self.executors = dict(executors)
        self.policy = policy or OnionPolicy()
        self.ledger = LockLedger()
        self.trace = WorkflowTrace()
        self.clock = clock
        self.cancel_token = cancel_token
        self._last_stage_converged = False
        self._explicit_convergence = False
        self._excluded_candidates: set[str] = set()
        self._stage_lock_ids: dict[str, list[str]] = {}

    # ------------------------------------------------------------------ driving
    def run(
        self,
        *,
        diagnosis: DataDiagnosis,
        prior: ChemicalPrior,
        radiation: RadiationAssumption,
    ) -> RunResult:
        """Execute the workflow and return a result, never a bare exception for a science failure."""
        anchor: AnchorSet | None = None
        current: Candidate | None = None
        stop: StopReason | None = None

        for index, kind in self.SEQUENCE:
            if kind is StageKind.SHELL_EXPANSION:
                anchor, current, stop = self._run_shells(
                    index=index,
                    diagnosis=diagnosis,
                    prior=prior,
                    radiation=radiation,
                    anchor=anchor,
                    current=current,
                )
            else:
                anchor, current, stop = self._run_single_stage(
                    index=index,
                    kind=kind,
                    diagnosis=diagnosis,
                    prior=prior,
                    radiation=radiation,
                    anchor=anchor,
                    current=current,
                )
            if stop is not None:
                break

        candidates = tuple(c for c in self.workspace.lineage.iter_all()
                           if str(c.candidate_id) not in self._excluded_candidates
                           and c.status not in (CandidateStatus.ABANDONED, CandidateStatus.SUPERSEDED))
        reports = evaluate_all(candidates, self.policy.ranking_policy)
        outcome = rank_candidates(
            candidates,
            reports,
            self.policy.ranking_policy,
            heavy_elements=self.policy.heavy_elements,
        )
        self._record_statuses(reports)

        if stop is None:
            stop = self._stop_from_ranking(outcome)

        checkpoints = tuple(self.trace.checkpoints) + outcome.review_checkpoints
        return RunResult(
            run_id=self.workspace.run_id,
            stop_reason=stop,
            ranked=outcome.ranked,
            ineligible=outcome.ineligible,
            review_checkpoints=checkpoints,
            no_trustworthy_candidate=outcome.no_trustworthy_candidate,
            stages_completed=self.trace.completed_stage_ids,
            notes=tuple(self.trace.notes),
            metadata={
                "shells_run": str(self.trace.shells_run),
                "locks_issued": str(len(self.ledger)),
                "active_lock_ids": ",".join(self.ledger.active_ids()),
                "accepted_candidate_id": str(current.candidate_id) if current is not None else "",
                "radiation": radiation.source.value,
                "radiation_assumed": str(radiation.is_assumed),
            },
        )

    # ------------------------------------------------------------------ stages
    def _run_single_stage(
        self, **arguments,
    ) -> tuple[AnchorSet | None, Candidate | None, StopReason | None]:
        """Commit provider state only after the controller's acceptance decision.

        Optional hooks are local transactions, not execution hooks: commit_stage(request,
        chosen_candidate_or_none, active_lock_ids) and abort_stage(request, reason).
        """
        prior_anchor, prior_current = arguments["anchor"], arguments["current"]
        prior_active = self.ledger.active_ids()
        prior_candidates = dict(self.workspace.lineage.candidates)
        prior_excluded = set(self._excluded_candidates)
        self._transaction_request: StageRequest | None = None
        self._transaction_executor = self.executors.get(arguments["kind"])
        self._transaction_chosen: Candidate | None = None
        self._abort_notified = False
        try:
            outcome = self._execute_single_stage(**arguments)
            request = self._transaction_request
            last = self.trace.stages[-1] if self.trace.stages else None
            accepted = (outcome[2] is None and last is not None and last.state is StageState.COMPLETED
                        and (not last.produced_candidates or self._transaction_chosen is not None))
            if accepted and request is not None:
                commit = getattr(self._transaction_executor, "commit_stage", None)
                if commit is not None:
                    commit(request, self._transaction_chosen, self.ledger.active_ids())
            elif request is not None:
                self._abort_transaction(request, outcome[2].message if outcome[2] else
                                        last.summary if last else "stage not accepted")
            return outcome
        except Exception as exc:
            # Unexpected provider/hook failures must not export an unaccepted pending model.
            self.ledger.restore_active(prior_active)
            self._excluded_candidates = prior_excluded
            for previous in prior_candidates.values():
                self.workspace.update_candidate(previous)
            fresh = tuple(c for key, c in self.workspace.lineage.candidates.items() if key not in prior_candidates)
            self._abandon(fresh, "stage transaction did not commit")
            request = self._transaction_request
            if request is not None:
                try:
                    self._abort_transaction(request, "stage_transaction_failed:" + type(exc).__name__)
                except Exception as abort_error:
                    self.trace.notes.append("abort hook failed: " + type(abort_error).__name__)
                stage = request.stage.with_state(StageState.ABANDONED,
                                                summary="stage transaction failed: " + type(exc).__name__)
                self.workspace.update_stage(stage)
                if self.trace.stages and self.trace.stages[-1].stage_id == stage.stage_id:
                    self.trace.stages[-1] = stage
                else:
                    self.trace.stages.append(stage)
                self.workspace.write_json(f"stages/{stage.stage_id}/transaction_error.json",
                                          {"exception_type": type(exc).__name__, "accepted": False})
                if request.meter is not None:
                    self.workspace.write_json(f"stages/{stage.stage_id}/resource_usage.json",
                                              dict(request.meter.used), allow_replace=True)
            self._explicit_convergence = False
            return prior_anchor, prior_current, StopReason(
                StopReasonKind.INTERNAL_ERROR, "stage transaction failed: " + type(exc).__name__,
                stage_id=request.stage.stage_id if request is not None else None,
            )

    def _abort_transaction(self, request: StageRequest, reason: str) -> None:
        if self._abort_notified:
            return
        self._abort_notified = True
        abort = getattr(self._transaction_executor, "abort_stage", None)
        if abort is not None:
            abort(request, reason)

    def _execute_single_stage(
        self, *, index: int, kind: StageKind, diagnosis: DataDiagnosis,
        prior: ChemicalPrior, radiation: RadiationAssumption, anchor: AnchorSet | None,
        current: Candidate | None, shell: int | None = None,
    ) -> tuple[AnchorSet | None, Candidate | None, StopReason | None]:
        self._last_stage_converged = False
        stage = OnionStage(
            stage_id=make_stage_id(index, kind.value, shell=shell), kind=kind, index=index,
            shell=shell, budget=self.policy.budget_for(kind), policy=self.policy.policy_for(kind),
            input_candidate=current.candidate_id if current else None,
        )
        self.workspace.open_stage(stage)
        meter = StageResourceMeter(stage.budget, self.clock, self.cancel_token)
        prior_locks = self.ledger.active_ids()
        request = StageRequest(stage=stage, workspace=self.workspace, diagnosis=diagnosis,
                               prior=prior, radiation=radiation, lineage=self.workspace.lineage,
                               anchor=anchor, input_candidate=current, active_lock_ids=prior_locks,
                               ledger=self.ledger, meter=meter)
        self._transaction_request = request
        try:
            meter.checkpoint()
        except StageControlStop as exc:
            return anchor, current, self._control_stop(stage, meter, exc)
        if (self.policy.require_anchor_before_expansion
                and kind in (StageKind.FIRST_SHELL, StageKind.SHELL_EXPANSION)
                and (anchor is None or not anchor.validated)):
            question = ("No metal core has been validated, so shell expansion would build outward from "
                        "an unconfirmed anchor. Confirm the anchor, or choose a different branch.")
            return anchor, current, self._review_stop(stage, question,
                                                     checkpoint_kind=ReviewCheckpointKind.COMPETING_HYPOTHESES)
        executor = self.executors.get(kind)
        if executor is None:
            skipped = stage.with_state(StageState.ABANDONED, summary=f"no executor registered for {kind.value}")
            self.workspace.update_stage(skipped)
            self.trace.stages.append(skipped)
            self.trace.notes.append(f"stage {kind.value} skipped: no executor registered")
            self._explicit_convergence = False
            return anchor, current, None
        result: StageResult | None = None
        control: StageControlStop | None = None
        try:
            raw_result = executor.execute(request)
            if not isinstance(raw_result, StageResult):
                raise StageControlStop(StopReasonKind.INTERNAL_ERROR, "executor did not return a StageResult")
            result = raw_result
            self._reconcile_usage(request, result)
        except StageControlStop as exc:
            control = exc
        finally:
            # An executor cannot activate proposed/history locks by mutating the shared ledger.
            self.ledger.restore_active(prior_locks)
        if result is None:
            assert control is not None
            return anchor, current, self._control_stop(stage, meter, control)
        common = self._apply_locks(stage, result.lock_directives)
        candidate_locks = {}
        for candidate in result.candidates:
            specific = self._apply_locks(stage, result.candidate_lock_directives.get(str(candidate.candidate_id), ()))
            candidate_locks[str(candidate.candidate_id)] = tuple(dict.fromkeys((*prior_locks, *common, *specific)))
        produced = self._register_candidates(stage, result.candidates, common, candidate_locks)
        stage = replace(stage, produced_candidates=tuple(c.candidate_id for c in produced))
        self._record_checkpoints(stage, result.review_checkpoints)
        self.trace.notes.extend(result.notes)
        if control is not None:
            self._abandon(produced, str(control))
            return anchor, current, self._control_stop(stage, meter, control)
        self.workspace.write_json(f"stages/{stage.stage_id}/resource_usage.json", dict(meter.used))
        if result.should_rollback:
            self._explicit_convergence = False
            self._abandon(produced, result.rollback_reason or "trial rejected")
            decision = RollbackDecision(
                from_stage=stage.stage_id, to_stage=self._previous_stage_id(index),
                reason=result.rollback_reason or f"{kind.value} produced nothing usable",
                abandoned_candidates=tuple(c.candidate_id for c in produced),
                requires_human_review=result.requires_human_review,
            )
            self._record_rollback(stage, decision)
            if result.requires_human_review or any(c.blocking for c in result.review_checkpoints):
                return anchor, current, self._review_stop(stage, decision.reason, append_stage=False)
            if self.policy.stop_on_first_rollback:
                return anchor, current, StopReason(StopReasonKind.ALL_BRANCHES_EXHAUSTED, decision.reason,
                                                  stage_id=stage.stage_id)
            return anchor, current, None
        chosen = self._pick_forward_candidate(produced, result.accepted_candidate_id)
        self._transaction_chosen = chosen
        self._record_statuses(evaluate_all(tuple(produced), self.policy.ranking_policy))
        blocking = result.requires_human_review or any(c.blocking for c in self.trace.checkpoints)
        if blocking:
            # Preserve the review proposals, but leave the previously accepted model/locks active.
            self._excluded_candidates.update(str(c.candidate_id) for c in produced
                                             if c.status not in (CandidateStatus.REJECTED, CandidateStatus.ABANDONED)
                                             and evaluate_all((c,), self.policy.ranking_policy)[0].is_eligible)
            question = next((c.question for c in reversed(self.trace.checkpoints) if c.blocking),
                            result.summary or "the executor requires a crystallographer's review")
            return anchor, current, self._review_stop(stage, question)
        if result.accepted_candidate_id is not None and chosen is None:
            return anchor, current, self._review_stop(stage, "the requested accepted candidate failed eligibility or was not returned")
        if chosen is not None:
            self.ledger.restore_active(chosen.provenance.lock_directive_ids)
            self._explicit_convergence = False
            if current is not None and current.candidate_id != chosen.candidate_id:
                self.workspace.update_candidate(current.with_status(CandidateStatus.SUPERSEDED,
                                                                      reason="replaced by the accepted successor"))
                self._excluded_candidates.add(str(current.candidate_id))
            for other in produced:
                if other.candidate_id != chosen.candidate_id:
                    report = evaluate_all((other,), self.policy.ranking_policy)[0]
                    if report.is_eligible:
                        self.workspace.update_candidate(other.with_status(CandidateStatus.SUPERSEDED,
                                                                          reason="not selected for the accepted branch"))
                        self._excluded_candidates.add(str(other.candidate_id))
        next_candidate = chosen or current
        next_anchor = result.anchor if result.anchor is not None and (not produced or chosen is not None) else anchor
        self._last_stage_converged = bool(result.converged and next_candidate is not None
                                           and not self.policy.ranking_policy.waived_gates
                                           and evaluate_all((next_candidate,), self.policy.ranking_policy)[0].is_eligible)
        if self._last_stage_converged:
            self._explicit_convergence = True
        elif result.converged:
            self.trace.notes.append("executor convergence claim has no eligible accepted model; not consumed as convergence")
        completed = stage.with_state(StageState.COMPLETED,
                                     summary=result.summary or f"{kind.value} produced {len(produced)} candidate(s)")
        self.workspace.update_stage(completed)
        self.trace.stages.append(completed)
        return next_anchor, next_candidate, None

    def _reconcile_usage(self, request: StageRequest, result: StageResult) -> None:
        assert request.meter is not None
        quantities = {"attempts": result.attempts_used, "tool_calls": result.tool_calls_used,
                      "failures": result.failures_used, "candidates": len(result.candidates)}
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in quantities.values()):
            raise StageControlStop(StopReasonKind.BUDGET_EXCEEDED, "executor resource usage is invalid")
        additions = result.new_atoms_added
        if additions is None and result.candidates:
            parent_count = request.input_candidate.n_non_hydrogen_atoms if request.input_candidate else 0
            counts = [c.n_non_hydrogen_atoms for c in result.candidates]
            if parent_count is not None and all(n is not None for n in counts):
                additions = sum(max(0, n - parent_count) for n in counts)
        if additions is not None:
            if not isinstance(additions, int) or isinstance(additions, bool) or additions < 0:
                raise StageControlStop(StopReasonKind.BUDGET_EXCEEDED, "executor added-atom usage is invalid")
            quantities["new_atoms"] = additions
        elif result.candidates and request.stage.budget.max_new_atoms is not None and not request.meter.used["new_atoms"]:
            raise StageControlStop(StopReasonKind.BUDGET_EXCEEDED, "added-atom budget cannot be verified")
        request.meter.reconcile(**quantities)

    def _record_checkpoints(self, stage: OnionStage, checkpoints: Sequence[ReviewCheckpoint]) -> None:
        for checkpoint in checkpoints:
            self.trace.checkpoints.append(checkpoint)
            self.workspace.events.emit(EventKind.REVIEW_CHECKPOINT, checkpoint.question,
                                       stage_id=str(stage.stage_id), checkpoint=checkpoint.kind.value)

    def _control_stop(self, stage: OnionStage, meter: StageResourceMeter, exc: StageControlStop) -> StopReason:
        state = StageState.BUDGET_EXCEEDED if exc.kind is StopReasonKind.BUDGET_EXCEEDED else StageState.ABANDONED
        stopped = stage.with_state(state, summary=str(exc))
        self.workspace.update_stage(stopped)
        self.trace.stages.append(stopped)
        self.workspace.write_json(f"stages/{stage.stage_id}/resource_usage.json", dict(meter.used))
        return StopReason(exc.kind, str(exc), stage_id=stage.stage_id)

    def _review_stop(self, stage: OnionStage, question: str, *, append_stage: bool = True,
                     checkpoint_kind: ReviewCheckpointKind = ReviewCheckpointKind.RESIDUAL_DENSITY_UNEXPLAINED) -> StopReason:
        checkpoint = ReviewCheckpoint(checkpoint_kind, question,
                                      stage_id=stage.stage_id, blocking=True)
        self._record_checkpoints(stage, (checkpoint,))
        reviewed = stage.with_state(StageState.REVIEW_REQUIRED, summary=question)
        self.workspace.update_stage(reviewed)
        if append_stage:
            self.trace.stages.append(reviewed)
        elif self.trace.stages and self.trace.stages[-1].stage_id == stage.stage_id:
            self.trace.stages[-1] = reviewed
        return StopReason(StopReasonKind.REVIEW_CHECKPOINT, question, stage_id=stage.stage_id,
                          suggested_next_action="review retained evidence before continuing")

    def _abandon(self, candidates: Sequence[Candidate], reason: str) -> None:
        for candidate in candidates:
            self.workspace.update_candidate(candidate.with_status(CandidateStatus.ABANDONED, reason=reason))
            self._excluded_candidates.add(str(candidate.candidate_id))

    def _run_shells(
        self,
        *,
        index: int,
        diagnosis: DataDiagnosis,
        prior: ChemicalPrior,
        radiation: RadiationAssumption,
        anchor: AnchorSet | None,
        current: Candidate | None,
    ) -> tuple[AnchorSet | None, Candidate | None, StopReason | None]:
        """Repeat S4 until it converges, the shell budget runs out, or it asks for help."""
        for shell in range(1, self.policy.max_shells + 1):
            before = current.candidate_id if current is not None else None
            anchor, current, stop = self._run_single_stage(
                index=index,
                kind=StageKind.SHELL_EXPANSION,
                diagnosis=diagnosis,
                prior=prior,
                radiation=radiation,
                anchor=anchor,
                current=current,
                shell=shell,
            )
            self.trace.shells_run = shell
            if stop is not None:
                return anchor, current, stop
            last = self.trace.stages[-1] if self.trace.stages else None
            if last is not None and last.state is StageState.ROLLED_BACK:
                self.trace.notes.append(f"shell {shell} rejected; restored the previous accepted model and active locks")
                continue
            if last is not None and last.state is StageState.ABANDONED:
                return anchor, current, None
            if self._last_stage_converged:
                self.trace.notes.append(f"shell {shell} reported explicit convergence with an eligible accepted model")
                return anchor, current, None
            if (current.candidate_id if current is not None else None) == before:
                assert last is not None
                question = f"shell {shell} made no accepted model progress; this does not establish convergence"
                self.trace.notes.append(question)
                self.trace.stages.pop()
                return anchor, current, self._review_stop(last, question)
        self.trace.notes.append(
            f"reached the {self.policy.max_shells}-shell limit; a structure still incomplete after "
            "this needs a crystallographer rather than another automatic round"
        )
        return anchor, current, StopReason(
            StopReasonKind.BUDGET_EXCEEDED, f"maximum of {self.policy.max_shells} shell iterations exhausted",
            suggested_next_action="review the retained attempts before extending the bounded plan",
        )

    # ------------------------------------------------------------------ helpers
    def _apply_locks(
        self, stage: OnionStage, directives: Sequence[LockDirective]
    ) -> tuple[str, ...]:
        """Record every proposed lock without activating it on the accepted branch."""
        if not directives:
            return ()
        lock_ids = self.ledger.add_all(directives, activate=False)
        self._stage_lock_ids.setdefault(str(stage.stage_id), []).extend(lock_ids)
        payload = [
            {"lock_id": lock_id, "directive": self.ledger.get(lock_id)}
            for lock_id in self._stage_lock_ids[str(stage.stage_id)]
        ]
        self.workspace.write_json(f"stages/{stage.stage_id}/lock_directives.json", payload, allow_replace=True)
        for lock_id, directive in zip(lock_ids, directives):
            self.workspace.events.emit(
                EventKind.LOCK_APPLIED,
                f"proposed {directive.mode.value} on {', '.join(str(a) for a in directive.atom_ids)}: "
                f"{directive.reason}",
                stage_id=str(stage.stage_id),
                lock_id=lock_id,
                mode=directive.mode.value,
                atoms=[str(a) for a in directive.atom_ids],
                instruction_lines=list(directive.instruction_lines),
                active_on_accepted_branch=False,
            )
        return lock_ids

    def _register_candidates(
        self,
        stage: OnionStage,
        candidates: Sequence[Candidate],
        new_lock_ids: tuple[str, ...],
        candidate_lock_ids: Mapping[str, tuple[str, ...]] | None = None,
    ) -> tuple[Candidate, ...]:
        """Add candidates to the lineage, stamping the locks in force onto their provenance."""
        active = tuple(dict.fromkeys((*self.ledger.active_ids(), *new_lock_ids)))
        registered: list[Candidate] = []
        for candidate in candidates:
            stamped = _with_lock_ids(candidate, (candidate_lock_ids or {}).get(str(candidate.candidate_id), active))
            try:
                self.workspace.add_candidate(stamped)
            except (ValueError, WorkspaceError) as exc:
                self.trace.notes.append(
                    f"candidate {candidate.candidate_id} rejected by the lineage: {exc}"
                )
                continue
            registered.append(stamped)
        if new_lock_ids and registered:
            self.trace.notes.append(
                f"{len(new_lock_ids)} lock(s) issued at {stage.stage_id} apply to "
                f"{len(registered)} new candidate(s)"
            )
        return tuple(registered)

    def _record_rollback(self, stage: OnionStage, decision: RollbackDecision) -> None:
        self.workspace.write_json(f"stages/{stage.stage_id}/rollback.json", decision)
        self.workspace.events.emit(
            EventKind.STAGE_ROLLED_BACK,
            decision.reason,
            stage_id=str(decision.from_stage),
            to_stage=str(decision.to_stage),
            requires_human_review=decision.requires_human_review,
        )
        rolled = stage.with_state(StageState.ROLLED_BACK, summary=decision.reason)
        self.workspace.update_stage(rolled)
        self.trace.stages.append(rolled)
        self.trace.rollbacks.append(decision)

    def _previous_stage_id(self, index: int) -> StageId:
        for stage in reversed(self.trace.stages):
            if stage.index <= index and stage.state is StageState.COMPLETED:
                return stage.stage_id
        return make_stage_id(0, StageKind.DIAGNOSE.value)

    def _pick_forward_candidate(self, produced: Sequence[Candidate],
                                accepted_candidate_id: CandidateId | None = None) -> Candidate | None:
        """Choose which candidate the next stage builds on.

        Only *eligible* candidates may carry the run forward. This is where "a lower R factor does
        not buy acceptance" is enforced inside the workflow, not merely at the end: a chemically
        implausible model cannot become the parent of the next shell.
        """
        if not produced:
            return None
        reports = evaluate_all(tuple(produced), self.policy.ranking_policy)
        outcome = rank_candidates(
            tuple(produced),
            reports,
            self.policy.ranking_policy,
            heavy_elements=self.policy.heavy_elements,
        )
        self.trace.checkpoints.extend(outcome.review_checkpoints)
        if not outcome.ranked:
            self.trace.notes.append(
                "no candidate from this stage passed the eligibility gates, so the next stage "
                "continues from the previous model rather than an unvalidated one"
            )
            return None
        eligible_ids = {candidate.candidate_id for candidate in outcome.ranked}
        best_id: CandidateId = accepted_candidate_id or outcome.ranked[0].candidate_id
        if best_id not in eligible_ids:
            return None
        for candidate in produced:
            if candidate.candidate_id == best_id:
                return candidate
        return None

    def _record_statuses(self, reports: Sequence[EligibilityReport]) -> None:
        for report in reports:
            try:
                candidate = self.workspace.lineage.get(report.candidate_id)
            except KeyError:  # pragma: no cover - lineage is the source of the reports
                continue
            if candidate.status in (CandidateStatus.ABANDONED, CandidateStatus.SUPERSEDED):
                continue
            status = (
                (CandidateStatus.REFINED if any(r.outcome is GateOutcome.WAIVED for r in report.results)
                 else CandidateStatus.VALIDATED) if report.is_eligible else CandidateStatus.REJECTED
            )
            if candidate.status is status:
                continue
            self.workspace.update_candidate(
                candidate.with_status(
                    status, reason="" if report.is_eligible else report.summary()
                )
            )

    def _stop_from_ranking(self, outcome: RankingOutcome) -> StopReason:
        if any(c.blocking for c in (*outcome.review_checkpoints, *self.trace.checkpoints)):
            return StopReason(
                kind=StopReasonKind.REVIEW_CHECKPOINT,
                message="a decision is needed from a crystallographer before continuing",
                suggested_next_action="inspect the review checkpoints in result.json",
            )
        if not outcome.ranked:
            return StopReason(
                kind=StopReasonKind.NO_TRUSTWORTHY_CANDIDATE,
                message=(
                    outcome.no_trustworthy_candidate.reason
                    if outcome.no_trustworthy_candidate
                    else "no candidate passed the eligibility gates"
                ),
                suggested_next_action="run another shell-expansion stage or a different branch",
            )
        if self._explicit_convergence:
            return StopReason(StopReasonKind.CONVERGED,
                              "an executor explicitly established convergence and the accepted model passed the gates")
        self.trace.checkpoints.append(ReviewCheckpoint(
            ReviewCheckpointKind.RESIDUAL_DENSITY_UNEXPLAINED,
            "eligible provisional models exist, but scientific convergence was not established",
            blocking=True,
        ))
        return StopReason(StopReasonKind.REVIEW_CHECKPOINT,
                          "eligible provisional models exist; ranking alone is not convergence")


def _with_lock_ids(candidate: Candidate, lock_ids: tuple[str, ...]) -> Candidate:
    provenance = candidate.provenance
    if provenance.lock_directive_ids == lock_ids:
        return candidate
    updated = type(provenance)(
        **{
            **{f: getattr(provenance, f) for f in provenance.__dataclass_fields__},
            "lock_directive_ids": lock_ids,
        }
    )
    return type(candidate)(
        **{
            **{f: getattr(candidate, f) for f in candidate.__dataclass_fields__},
            "provenance": updated,
        }
    )
