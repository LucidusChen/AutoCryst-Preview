"""The application layer: one service shared by every frontend.

``crystalx solve`` and the Gradio app (package ``crystalx_web``) both use
:class:`~crystalx_core.app.service.RunService`. Nothing in this package imports a third-party
library; a frontend brings its own.
"""

from __future__ import annotations

from crystalx_core.app.capabilities import capability_report
from crystalx_core.app.demo import DEMO_MARKER, is_demo_input, write_demo_case
from crystalx_core.app.mock_toolchain import MOCK_BANNER, MockToolchain
from crystalx_core.app.models import (
    STAGE_ORDER,
    TERMINAL_PHASES,
    ChemicalPriorInput,
    ModelAvailability,
    RefinementStatus,
    RunConfig,
    RunMode,
    RunPhase,
    SolveRequest,
    StageName,
    StageState,
    ValidationIssue,
)
from crystalx_core.app.service import RequestInvalid, RunHandle, RunService, UnknownRun
from crystalx_core.app.toolchain import RealToolchain, Toolchain

__all__ = [
    "DEMO_MARKER",
    "MOCK_BANNER",
    "STAGE_ORDER",
    "TERMINAL_PHASES",
    "ChemicalPriorInput",
    "MockToolchain",
    "ModelAvailability",
    "RealToolchain",
    "RefinementStatus",
    "RequestInvalid",
    "RunConfig",
    "RunHandle",
    "RunMode",
    "RunPhase",
    "RunService",
    "SolveRequest",
    "StageName",
    "StageState",
    "Toolchain",
    "UnknownRun",
    "ValidationIssue",
    "capability_report",
    "is_demo_input",
    "write_demo_case",
]
