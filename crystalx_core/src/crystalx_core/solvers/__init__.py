"""The initial-solution solver portfolio.

SHELXT is one route among several, not the entry point. Which routes run is decided by
:class:`~crystalx_core.solvers.selector.StrategySelector` from measurable properties of the user's
files, and the reasoning is recorded in a ``SelectionTrace``.
"""

from __future__ import annotations

from crystalx_core.solvers.base import (
    BackendRegistry,
    InitialSolutionBackend,
    backend_readiness,
    tool_requirements_met,
)
from crystalx_core.solvers.context import SolverContext, StagedCase
from crystalx_core.solvers.heavy_atom import (
    AnodeVerifyBackend,
    PlanOnlyBackend,
    ShelxcShelxdBackend,
    ShelxeDensityModBackend,
    ShelxsPattersonBackend,
    build_shelxc_instructions,
    heavy_atom_backends,
)
from crystalx_core.solvers.shelxt import ShelxtBackend, ShelxtSolution
from crystalx_core.solvers.machine import (
    BranchExecutionReport,
    InitialSolutionMachine,
    PortfolioResult,
    apply_rollback,
    budget_summary,
    decide_rollback,
    make_plan_preview,
)
from crystalx_core.solvers.mock import MockBackend, ScriptedExecutor, ScriptedResponse
from crystalx_core.solvers.selector import RULE_IDS, SelectionPolicy, StrategySelector

__all__ = [
    "RULE_IDS",
    "AnodeVerifyBackend",
    "BackendRegistry",
    "PlanOnlyBackend",
    "ShelxcShelxdBackend",
    "ShelxeDensityModBackend",
    "ShelxsPattersonBackend",
    "ShelxtBackend",
    "ShelxtSolution",
    "SolverContext",
    "StagedCase",
    "backend_readiness",
    "build_shelxc_instructions",
    "heavy_atom_backends",
    "BranchExecutionReport",
    "InitialSolutionBackend",
    "InitialSolutionMachine",
    "MockBackend",
    "PortfolioResult",
    "ScriptedExecutor",
    "ScriptedResponse",
    "SelectionPolicy",
    "StrategySelector",
    "apply_rollback",
    "budget_summary",
    "decide_rollback",
    "make_plan_preview",
    "tool_requirements_met",
]
