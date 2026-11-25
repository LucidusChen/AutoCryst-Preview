"""The initial-solution backend contract.

A backend answers three questions:

* **Can you be used on this data?** -- ``applicability()``. Returning ``NOT_APPLICABLE`` with a
  stated reason is a normal, recorded outcome; that is how "SHELXC-D-E is impossible on merged data"
  becomes visible at Stage 0 instead of failing obscurely later.
* **What exactly would you run?** -- ``plan()``, an argv array plus optional stdin script and the
  outputs that must appear.
* **What did it mean?** -- ``parse()``, turning the raw execution outcome into a
  :class:`BranchOutcome` with figures of merit.

No backend in P1 executes anything. Real adapters arrive in a later slice, behind this same contract.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from crystalx_core.domain.chemistry import ChemicalPrior
from crystalx_core.domain.data import DataDiagnosis
from crystalx_core.domain.solving import (
    Applicability,
    ApplicabilityVerdict,
    BackendReadiness,
    BranchOutcome,
    InitialSolutionBranch,
    SolverCapability,
    SolverKind,
    capability_for,
)
from crystalx_core.domain.tools import ToolAvailabilityReport, ToolKind
from crystalx_core.errors import StrategyError
from crystalx_core.tools.runner import ExecutionOutcome, ExecutionPlan

__all__ = [
    "InitialSolutionBackend",
    "BackendRegistry",
    "tool_requirements_met",
    "backend_readiness",
]


@runtime_checkable
class InitialSolutionBackend(Protocol):
    """One scientific route to an initial solution."""

    kind: SolverKind

    readiness: BackendReadiness
    """How far this backend has been taken. A backend that omits the attribute is treated as
    ``PRODUCTION``, so an existing backend does not silently become unrunnable."""

    def capability(self) -> SolverCapability:
        """Static description: role, prerequisites, figures of merit."""
        ...

    def applicability(self, diagnosis: DataDiagnosis, prior: ChemicalPrior) -> Applicability:
        """Whether this route is scientifically appropriate for these data."""
        ...

    def plan(self, branch: InitialSolutionBranch, *, work_dir: Path) -> tuple[ExecutionPlan, ...]:
        """The exact invocations this branch would perform, in order."""
        ...

    def parse(
        self, branch: InitialSolutionBranch, outcomes: tuple[ExecutionOutcome, ...]
    ) -> BranchOutcome:
        """Interpret what the programs produced, including when they produced nothing."""
        ...


def tool_requirements_met(
    kind: SolverKind, availability: ToolAvailabilityReport
) -> tuple[bool, tuple[str, ...]]:
    """Check that every executable a backend needs is actually usable on this machine.

    Returns ``(ok, missing_descriptions)``. The descriptions carry the resolver's diagnosis, so the
    user is told "shelxd is an x86_64 build and Rosetta 2 is missing" rather than "unavailable".
    """
    capability = capability_for(kind)
    missing: list[str] = []
    for tool_name in capability.required_tools:
        try:
            tool_kind = ToolKind(tool_name)
        except ValueError as exc:  # pragma: no cover - guards a typo in the capability table
            raise StrategyError(f"Unknown tool name {tool_name!r} for solver {kind.value}") from exc
        resolution = availability.get(tool_kind)
        if resolution is None:
            missing.append(f"{tool_name}: not checked")
        elif not resolution.is_usable:
            missing.append(f"{tool_name}: {resolution.diagnostic.message}")
    return (not missing, tuple(missing))


class BackendRegistry:
    """Lookup from :class:`SolverKind` to a backend implementation."""

    def __init__(self) -> None:
        self._backends: dict[SolverKind, InitialSolutionBackend] = {}

    def register(self, backend: InitialSolutionBackend) -> None:
        if backend.kind in self._backends:
            raise StrategyError(f"Backend already registered for {backend.kind.value}")
        self._backends[backend.kind] = backend

    def get(self, kind: SolverKind) -> InitialSolutionBackend:
        try:
            return self._backends[kind]
        except KeyError as exc:
            raise StrategyError(f"No backend registered for {kind.value}") from exc

    def has(self, kind: SolverKind) -> bool:
        return kind in self._backends

    @property
    def kinds(self) -> tuple[SolverKind, ...]:
        return tuple(self._backends)

    def applicability(
        self, diagnosis: DataDiagnosis, prior: ChemicalPrior
    ) -> dict[SolverKind, Applicability]:
        """Ask every registered backend whether it fits these data."""
        return {
            kind: backend.applicability(diagnosis, prior)
            for kind, backend in self._backends.items()
        }

    def applicable_kinds(
        self, diagnosis: DataDiagnosis, prior: ChemicalPrior
    ) -> tuple[SolverKind, ...]:
        verdicts = self.applicability(diagnosis, prior)
        return tuple(
            kind
            for kind, applicability in verdicts.items()
            if applicability.verdict is ApplicabilityVerdict.APPLICABLE
        )

    def __len__(self) -> int:
        return len(self._backends)


def backend_readiness(backend: object) -> BackendReadiness:
    """Readiness of a backend, defaulting to ``PRODUCTION`` when it declares none."""
    value = getattr(backend, "readiness", BackendReadiness.PRODUCTION)
    return value if isinstance(value, BackendReadiness) else BackendReadiness(value)
