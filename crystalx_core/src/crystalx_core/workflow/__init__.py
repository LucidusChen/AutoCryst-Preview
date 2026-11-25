"""Workflow policy and execution: eligibility gates, ordering, and the onion state machine."""

from __future__ import annotations

from crystalx_core.workflow.gates import evaluate_all, evaluate_gates
from crystalx_core.workflow.onion import (
    LockLedger,
    OnionPolicy,
    OnionWorkflow,
    StageExecutor,
    StageRequest,
    StageResult,
    WorkflowTrace,
)
from crystalx_core.workflow.ranking import (
    ORDERING_KEY_NAMES,
    RankingOutcome,
    describe_ordering,
    rank_candidates,
)

__all__ = [
    "ORDERING_KEY_NAMES",
    "LockLedger",
    "OnionPolicy",
    "OnionWorkflow",
    "RankingOutcome",
    "StageExecutor",
    "StageRequest",
    "StageResult",
    "WorkflowTrace",
    "describe_ordering",
    "evaluate_all",
    "evaluate_gates",
    "rank_candidates",
]
