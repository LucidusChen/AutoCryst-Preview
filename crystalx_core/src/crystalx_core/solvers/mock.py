"""Deterministic mock backends and a scripted executor.

These exist so that the whole branch state machine, the workspace, the lineage and the gates can be
exercised **without a single proprietary binary**. They are shipped rather than kept in the test
directory because the Gradio development UI and the CLI ``--dry-run`` path use them too.

Nothing here approximates crystallography. A mock returns whatever it was told to return; it never
pretends to have solved a structure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Sequence

from crystalx_core.domain.chemistry import ChemicalPrior
from crystalx_core.domain.data import DataDiagnosis
from crystalx_core.domain.solving import (
    Applicability,
    ApplicabilityVerdict,
    BranchOutcome,
    BranchState,
    InitialSolutionBranch,
    SolverCapability,
    SolverKind,
    capability_for,
)
from crystalx_core.tools.runner import CommandExecutor, ExecutionOutcome, ExecutionPlan, RawResult

__all__ = ["MockBackend", "ScriptedExecutor", "ScriptedResponse", "always_applicable"]


def always_applicable(_diagnosis: DataDiagnosis, _prior: ChemicalPrior) -> Applicability:
    return Applicability(
        ApplicabilityVerdict.APPLICABLE, priority=50, reasons=("mock backend",)
    )


@dataclass
class MockBackend:
    """A backend that plans a fixed command and returns a fixed outcome."""

    kind: SolverKind
    argv: tuple[str, ...] = ("/bin/true",)
    expected_outputs: tuple[str, ...] = ()
    stdin_script: str | None = None
    timeout_s: float = 5.0
    applicability_fn: Callable[[DataDiagnosis, ChemicalPrior], Applicability] = always_applicable
    hypothesis_ids: tuple[str, ...] = ()
    figures_of_merit: Mapping[str, float] = field(default_factory=dict)
    n_plans: int = 1
    """Number of sequential invocations, so the SHELXC -> SHELXD -> SHELXE shape can be exercised."""

    def capability(self) -> SolverCapability:
        return capability_for(self.kind)

    def applicability(self, diagnosis: DataDiagnosis, prior: ChemicalPrior) -> Applicability:
        return self.applicability_fn(diagnosis, prior)

    def plan(
        self, branch: InitialSolutionBranch, *, work_dir: Path
    ) -> tuple[ExecutionPlan, ...]:
        return tuple(
            ExecutionPlan(
                argv=self.argv,
                cwd=work_dir,
                timeout_s=self.timeout_s,
                stdin_script=self.stdin_script,
                expected_outputs=self.expected_outputs if index == self.n_plans - 1 else (),
                label=f"{self.kind.value}_{index:02d}",
            )
            for index in range(self.n_plans)
        )

    def parse(
        self, branch: InitialSolutionBranch, outcomes: tuple[ExecutionOutcome, ...]
    ) -> BranchOutcome:
        if not outcomes:
            return BranchOutcome(
                branch_id=branch.branch_id,
                state=BranchState.FAILED,
                summary="no command was executed",
                failure_reason="no command was executed",
            )
        failures = [o for o in outcomes if not o.succeeded]
        if failures:
            worst = failures[0]
            return BranchOutcome(
                branch_id=branch.branch_id,
                state=BranchState.FAILED,
                summary=worst.diagnostic.message,
                failure_reason=worst.diagnostic.message,
                diagnostic_code=worst.diagnostic.kind.value,
                wall_clock_s=sum(o.duration_s for o in outcomes),
            )
        return BranchOutcome(
            branch_id=branch.branch_id,
            state=BranchState.SUCCEEDED,
            summary=f"{self.kind.value} completed ({len(outcomes)} step(s))",
            figures_of_merit=dict(self.figures_of_merit),
            produced_paths=tuple(
                str(Path(o.plan.cwd) / name) for o in outcomes for name in o.produced_outputs
            ),
            hypothesis_ids=self.hypothesis_ids,
            wall_clock_s=sum(o.duration_s for o in outcomes),
        )


@dataclass
class ScriptedResponse:
    """One pre-programmed reply from :class:`ScriptedExecutor`."""

    exit_code: int | None = 0
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    os_error: str = ""
    creates: Sequence[str] = ()
    """Files to create in the plan's ``cwd``, so expected-output checking can be exercised."""

    file_contents: Mapping[str, str] = field(default_factory=dict)


class ScriptedExecutor:
    """A :class:`CommandExecutor` that replays scripted responses instead of running anything.

    Records every plan it was given, so a test can assert on the exact argument arrays.
    """

    def __init__(self, responses: Sequence[ScriptedResponse] | None = None) -> None:
        self.responses = list(responses or [])
        self.calls: list[ExecutionPlan] = []
        self.envs: list[dict[str, str]] = []
        self.default = ScriptedResponse()

    #: Accepts the cancellation/log keywords; logs are left to the runner to write.
    supports_control = True

    def execute(
        self,
        plan: ExecutionPlan,
        env: Mapping[str, str],
        *,
        cancel=None,
        stdout_path=None,
        stderr_path=None,
    ) -> RawResult:
        if cancel is not None and cancel.cancelled:
            return RawResult(exit_code=None, stdout="", stderr="", cancelled=True)
        self.calls.append(plan)
        self.envs.append(dict(env))
        response = self.responses.pop(0) if self.responses else self.default
        for name in response.creates:
            target = Path(plan.cwd) / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(response.file_contents.get(name, ""), encoding="utf-8")
        return RawResult(
            exit_code=response.exit_code,
            stdout=response.stdout,
            stderr=response.stderr,
            timed_out=response.timed_out,
            os_error=response.os_error,
        )

    @property
    def argv_history(self) -> list[tuple[str, ...]]:
        return [plan.argv for plan in self.calls]


_ = CommandExecutor  # documents that ScriptedExecutor satisfies the protocol
