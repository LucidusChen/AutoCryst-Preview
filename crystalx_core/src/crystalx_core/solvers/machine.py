"""Bounded execution of the solver portfolio, with rollback.

Branches are run in priority order under a per-branch budget. Every branch is kept, including the
ones that fail and the ones that were never applicable, because "this route did not work on this
data" is itself information a crystallographer wants.

The machine never picks a winner. It gathers outcomes; ranking happens later, after the scientific
gates.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

from crystalx_core.domain.ids import BranchId, StageId
from crystalx_core.domain.onion import OnionStage, RollbackDecision, StageState
from crystalx_core.domain.solving import (
    BackendReadiness,
    BranchOutcome,
    BranchState,
    InitialSolutionBranch,
    SelectionTrace,
)
from crystalx_core.domain.tools import ToolAvailabilityReport, ToolKind
from crystalx_core.errors import SolverError
from crystalx_core.solvers.base import BackendRegistry, backend_readiness, tool_requirements_met
from crystalx_core.tools.adapter import ExternalToolAdapter, ToolInvocationRecord
from crystalx_core.tools.runner import CancelToken, ExecutionOutcome, ProcessRunner, ResourceLimits
from crystalx_core.workspace.events import EventKind
from crystalx_core.workspace.store import RunWorkspace

__all__ = ["BranchExecutionReport", "InitialSolutionMachine", "PortfolioResult", "BranchPreparer"]

#: Called with a branch that is about to run and its freshly created working directory, so the
#: caller can stage the case files there (SHELX programs read ``<stem>.ins``/``<stem>.hkl`` from the
#: current directory). Raising refuses the branch with the error recorded as the reason.
BranchPreparer = Callable[[InitialSolutionBranch, Path], None]


@dataclass(frozen=True)
class BranchExecutionReport:
    """One branch's final state and everything it produced."""

    branch: InitialSolutionBranch
    outcome: BranchOutcome
    executions: tuple[ExecutionOutcome, ...] = ()

    @property
    def branch_id(self) -> BranchId:
        return self.branch.branch_id

    @property
    def succeeded(self) -> bool:
        return self.outcome.state is BranchState.SUCCEEDED


@dataclass
class PortfolioResult:
    """Result of running one stage's worth of branches."""

    stage_id: StageId
    trace: SelectionTrace
    reports: list[BranchExecutionReport] = field(default_factory=list)
    rollbacks: list[RollbackDecision] = field(default_factory=list)

    @property
    def succeeded(self) -> tuple[BranchExecutionReport, ...]:
        return tuple(r for r in self.reports if r.succeeded)

    @property
    def failed(self) -> tuple[BranchExecutionReport, ...]:
        return tuple(r for r in self.reports if not r.succeeded)

    @property
    def any_success(self) -> bool:
        return bool(self.succeeded)

    def hypothesis_ids(self) -> tuple[str, ...]:
        ids: list[str] = []
        for report in self.succeeded:
            ids.extend(report.outcome.hypothesis_ids)
        return tuple(ids)


class InitialSolutionMachine:
    """Runs nominated branches under their budgets and records everything."""

    def __init__(
        self,
        registry: BackendRegistry,
        workspace: RunWorkspace,
        *,
        runner: ProcessRunner | None = None,
        availability: ToolAvailabilityReport | None = None,
        adapters: Mapping[ToolKind, ExternalToolAdapter] | None = None,
        allow_plan_only: bool = False,
        cancel: CancelToken | None = None,
        limits: ResourceLimits | None = None,
        prepare: BranchPreparer | None = None,
    ) -> None:
        self.registry = registry
        self.workspace = workspace
        self.runner = runner
        self.availability = availability
        self.adapters: dict[ToolKind, ExternalToolAdapter] = dict(adapters or {})
        self.allow_plan_only = allow_plan_only
        """Plan-only backends are refused by default. Their command shape is defined but their
        output parser is not, so running one could only produce an uninterpretable result."""

        self.cancel = cancel
        """Shared with the caller. Once set, no further branch starts, and a running program is
        stopped through its process group; the branch is then recorded as ``ABORTED``."""

        self.limits = limits
        """Operating-system limits applied to every plan that does not already carry its own."""

        self.prepare = prepare

    def run_portfolio(
        self,
        stage: OnionStage,
        trace: SelectionTrace,
        *,
        stop_after_first_success: bool = False,
    ) -> PortfolioResult:
        """Execute the nominated branches in priority order.

        ``stop_after_first_success`` is off by default on purpose: comparing two independent routes
        is more informative than taking the first that returns something.
        """
        result = PortfolioResult(stage_id=stage.stage_id, trace=trace)

        for branch in trace.nominated:
            if self.cancel is not None and self.cancel.cancelled:
                self.workspace.open_branch(branch)
                aborted = branch.transition(BranchState.ABORTED)
                outcome = self._outcome(
                    aborted,
                    f"not started: {self.cancel.reason or 'cancelled'}",
                    diagnostic_code="cancelled",
                )
                self._persist(aborted, outcome)
                result.reports.append(BranchExecutionReport(aborted, outcome))
                continue
            if stop_after_first_success and result.any_success:
                skipped = branch.transition(BranchState.SKIPPED)
                self._persist(skipped, self._outcome(skipped, "skipped after an earlier success"))
                result.reports.append(
                    BranchExecutionReport(
                        skipped, self._outcome(skipped, "skipped after an earlier success")
                    )
                )
                continue
            result.reports.append(self._run_branch(branch))

        # Branches the selector explicitly ruled out are still recorded, with their reason.
        for kind, reason in trace.excluded:
            self.workspace.events.emit(
                EventKind.BRANCH_NOT_APPLICABLE,
                f"{kind.value} not used: {reason}",
                stage_id=str(stage.stage_id),
                solver=kind.value,
                reason=reason,
            )

        return result

    def _run_branch(self, branch: InitialSolutionBranch) -> BranchExecutionReport:
        work_dir = self.workspace.open_branch(branch)

        if self.availability is not None:
            ok, missing = tool_requirements_met(branch.kind, self.availability)
            if not ok:
                blocked = branch.transition(BranchState.BLOCKED_TOOL_UNAVAILABLE)
                outcome = self._outcome(
                    blocked,
                    "required program unavailable: " + "; ".join(missing),
                    diagnostic_code="tool_unavailable",
                )
                self._persist(blocked, outcome)
                return BranchExecutionReport(blocked, outcome)

        if not self.registry.has(branch.kind):
            planned = branch.transition(BranchState.SKIPPED)
            outcome = self._outcome(
                planned,
                f"no backend is registered for {branch.kind.value} in this build",
                diagnostic_code="backend_missing",
            )
            self._persist(planned, outcome)
            return BranchExecutionReport(planned, outcome)

        backend = self.registry.get(branch.kind)

        readiness = backend_readiness(backend)
        if readiness is BackendReadiness.PLAN_ONLY and not self.allow_plan_only:
            blocked = branch.transition(BranchState.BLOCKED_BACKEND_PLAN_ONLY)
            outcome = self._outcome(
                blocked,
                (
                    f"{branch.kind.value} is a plan-only backend in this build: the command shape "
                    "is defined but its output parser is not implemented, so running it could "
                    "only produce an uninterpretable result. Use --dry-run to see the command."
                ),
                diagnostic_code="backend_plan_only",
            )
            self._persist(blocked, outcome)
            return BranchExecutionReport(blocked, outcome)

        running = branch.transition(BranchState.RUNNING)
        self.workspace.record_branch_state(running)

        if self.prepare is not None:
            try:
                self.prepare(running, work_dir)
            except Exception as exc:  # noqa: BLE001 - recorded as the branch's failure reason
                failed = running.transition(BranchState.FAILED)
                outcome = self._outcome(
                    failed,
                    f"the case could not be staged into the branch directory: {exc}",
                    diagnostic_code="staging_error",
                )
                self._persist(failed, outcome)
                return BranchExecutionReport(failed, outcome)

        try:
            plans = backend.plan(running, work_dir=work_dir)
        except Exception as exc:  # noqa: BLE001 - a backend defect must not lose the run
            failed = running.transition(BranchState.FAILED)
            outcome = self._outcome(
                failed, f"backend could not build an execution plan: {exc}", diagnostic_code="plan_error"
            )
            self._persist(failed, outcome)
            return BranchExecutionReport(failed, outcome)

        if self.limits is not None:
            plans = tuple(
                plan if plan.limits is not None else dataclasses.replace(plan, limits=self.limits)
                for plan in plans
            )

        executions: list[ExecutionOutcome] = []
        elapsed = 0.0
        budget_exceeded = False
        cancelled = False

        for index, plan in enumerate(plans):
            self.workspace.write_json(
                f"stages/{running.stage_id}/branches/{running.branch_id}/command_{index:02d}.json",
                plan.to_record(),
            )
            self.workspace.events.emit(
                EventKind.COMMAND_STARTED,
                " ".join(plan.argv),
                branch_id=str(running.branch_id),
                argv=list(plan.argv),
                cwd=str(plan.cwd),
            )
            adapter = self.adapters.get(plan.tool_kind) if plan.tool_kind else None
            if adapter is not None:
                execution, provenance = adapter.invoke(plan, log_dir=work_dir, cancel=self.cancel)
                self._write_provenance(running, index, provenance)
            else:
                if self.runner is None:
                    raise SolverError(
                        "A ProcessRunner is required to execute solver branches",
                        branch_id=str(running.branch_id),
                    )
                execution = self.runner.run(plan, log_dir=work_dir, cancel=self.cancel)
            executions.append(execution)
            elapsed += execution.duration_s
            self.workspace.write_json(
                f"stages/{running.stage_id}/branches/{running.branch_id}/result_{index:02d}.json",
                execution.to_record(),
            )
            self.workspace.events.emit(
                EventKind.COMMAND_FINISHED,
                execution.diagnostic.message,
                branch_id=str(running.branch_id),
                diagnostic=execution.diagnostic.kind.value,
                exit_code=execution.exit_code,
                duration_s=round(execution.duration_s, 3),
            )
            if execution.cancelled:
                cancelled = True
                break
            if not execution.succeeded:
                break
            if elapsed > running.budget.max_wall_clock_s:
                budget_exceeded = True
                break

        outcome = backend.parse(running, tuple(executions))
        if cancelled:
            reason = (self.cancel.reason if self.cancel else "") or "cancelled"
            outcome = BranchOutcome(
                branch_id=outcome.branch_id,
                state=BranchState.ABORTED,
                summary=f"stopped on request: {reason}",
                produced_paths=outcome.produced_paths,
                failure_reason=f"stopped on request: {reason}",
                diagnostic_code="cancelled",
                wall_clock_s=elapsed,
            )
            aborted = running.transition(BranchState.ABORTED)
            self._persist(aborted, outcome)
            return BranchExecutionReport(aborted, outcome, tuple(executions))
        if budget_exceeded:
            final_state = BranchState.BUDGET_EXCEEDED
            outcome = BranchOutcome(
                branch_id=outcome.branch_id,
                state=final_state,
                summary=outcome.summary,
                figures_of_merit=outcome.figures_of_merit,
                produced_paths=outcome.produced_paths,
                hypothesis_ids=outcome.hypothesis_ids,
                failure_reason=(
                    f"branch exceeded its {running.budget.max_wall_clock_s:g}s budget "
                    f"after {elapsed:.1f}s"
                ),
                diagnostic_code="budget_exceeded",
                wall_clock_s=elapsed,
            )
        else:
            final_state = outcome.state

        if final_state not in (BranchState.SUCCEEDED, BranchState.FAILED, BranchState.BUDGET_EXCEEDED):
            final_state = BranchState.FAILED

        finished = running.transition(final_state)
        self._persist(finished, outcome)
        return BranchExecutionReport(finished, outcome, tuple(executions))

    def _write_provenance(
        self, branch: InitialSolutionBranch, index: int, record: ToolInvocationRecord
    ) -> None:
        """Store the full invocation record beside the command it describes."""
        self.workspace.write_json(
            f"stages/{branch.stage_id}/branches/{branch.branch_id}/provenance_{index:02d}.json",
            record,
        )

    def _persist(self, branch: InitialSolutionBranch, outcome: BranchOutcome) -> None:
        self.workspace.record_branch_state(branch)
        self.workspace.record_branch_outcome(branch.stage_id, branch.branch_id, outcome)

    @staticmethod
    def _outcome(
        branch: InitialSolutionBranch, reason: str, *, diagnostic_code: str = ""
    ) -> BranchOutcome:
        return BranchOutcome(
            branch_id=branch.branch_id,
            state=branch.state,
            summary=reason,
            failure_reason=reason if branch.state is not BranchState.SUCCEEDED else "",
            diagnostic_code=diagnostic_code,
        )


def decide_rollback(
    *,
    stage: OnionStage,
    result: PortfolioResult,
    fallback_stage: StageId,
    remaining_alternatives: Sequence[BranchId] = (),
) -> RollbackDecision | None:
    """Decide whether a stage with no usable outcome should retreat.

    Returns ``None`` when the stage succeeded. Otherwise a recorded retreat naming the abandoned
    work, and whether a human is needed because nothing is left to try.
    """
    if result.any_success:
        return None
    abandoned = tuple(report.branch_id for report in result.reports)
    return RollbackDecision(
        from_stage=stage.stage_id,
        to_stage=fallback_stage,
        reason=(
            f"No branch at {stage.stage_id} produced a usable result "
            f"({len(abandoned)} attempted)."
        ),
        abandoned_branches=abandoned,
        alternative_selected=remaining_alternatives[0] if remaining_alternatives else None,
        requires_human_review=not remaining_alternatives,
    )


def apply_rollback(
    workspace: RunWorkspace, stage: OnionStage, decision: RollbackDecision
) -> OnionStage:
    """Record a rollback and mark the stage. Nothing is deleted."""
    workspace.events.emit(
        EventKind.STAGE_ROLLED_BACK,
        decision.reason,
        stage_id=str(decision.from_stage),
        to_stage=str(decision.to_stage),
        abandoned_branches=[str(b) for b in decision.abandoned_branches],
        requires_human_review=decision.requires_human_review,
    )
    workspace.write_json(
        f"stages/{stage.stage_id}/rollback.json",
        decision,
    )
    rolled = stage.with_state(StageState.ROLLED_BACK, summary=decision.reason)
    workspace.update_stage(rolled)
    return rolled


def iter_branch_ids(branches: Iterable[InitialSolutionBranch]) -> tuple[BranchId, ...]:
    return tuple(branch.branch_id for branch in branches)


def budget_summary(branches: Iterable[InitialSolutionBranch]) -> Mapping[str, float]:
    """Total wall-clock the portfolio is allowed to consume. Shown before anything runs."""
    total = 0.0
    per_branch: dict[str, float] = {}
    for branch in branches:
        per_branch[str(branch.branch_id)] = branch.budget.max_wall_clock_s
        total += branch.budget.max_wall_clock_s
    return {"total_s": total, **per_branch}


def make_plan_preview(
    registry: BackendRegistry,
    branches: Sequence[InitialSolutionBranch],
    *,
    work_dir: Path,
) -> list[dict[str, object]]:
    """Build every execution plan without running anything.

    This backs ``crystalx solve --dry-run``: the user can see the exact argument arrays that would
    touch their SHELX and PLATON installations before anything is executed.
    """
    preview: list[dict[str, object]] = []
    for branch in branches:
        if not registry.has(branch.kind):
            preview.append(
                {"branch": str(branch.branch_id), "solver": branch.kind.value, "plans": [], "note": "no backend registered"}
            )
            continue
        backend = registry.get(branch.kind)
        plans = backend.plan(branch, work_dir=work_dir)
        preview.append(
            {
                "branch": str(branch.branch_id),
                "solver": branch.kind.value,
                "plans": [plan.to_record() for plan in plans],
            }
        )
    return preview


CallableGuard = Callable[[], None]
