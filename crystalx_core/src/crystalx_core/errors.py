"""Error taxonomy for crystalx_core.

Every error carries a machine-readable ``code``, an English message, the identifiers of the run,
stage, branch and candidate it belongs to, and the paths of the artefacts that evidence it. The GUI
layer is expected to map ``code`` to a localised explanation; the core stays English-only.

A ``PolicyStop`` is deliberately part of this tree but is **not** a failure: it is a planned,
scientifically motivated halt (for example "no candidate passed the eligibility gates"). Callers
must be able to distinguish it from a defect.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

__all__ = [
    "CrystalXError",
    "ConfigError",
    "InputValidationError",
    "WorkspaceError",
    "ToolResolutionError",
    "ToolExecutionError",
    "ParseError",
    "SolverError",
    "StrategyError",
    "InferenceError",
    "GateEvaluationError",
    "PolicyStop",
]


class CrystalXError(Exception):
    """Base class for every error raised by crystalx_core."""

    #: Stable, machine-readable identifier. Subclasses override it.
    code: str = "CRYSTALX_ERROR"

    def __init__(
        self,
        message: str,
        *,
        run_id: str | None = None,
        stage_id: str | None = None,
        branch_id: str | None = None,
        candidate_id: str | None = None,
        evidence_paths: Iterable[Path | str] = (),
        context: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.run_id = run_id
        self.stage_id = stage_id
        self.branch_id = branch_id
        self.candidate_id = candidate_id
        self.evidence_paths: tuple[Path, ...] = tuple(Path(p) for p in evidence_paths)
        self.context: dict[str, Any] = dict(context or {})

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable description, suitable for a run record."""
        return {
            "code": self.code,
            "message": self.message,
            "run_id": self.run_id,
            "stage_id": self.stage_id,
            "branch_id": self.branch_id,
            "candidate_id": self.candidate_id,
            "evidence_paths": [str(p) for p in self.evidence_paths],
            "context": self.context,
        }

    def __str__(self) -> str:  # pragma: no cover - trivial
        location = " ".join(
            f"{name}={value}"
            for name, value in (
                ("run", self.run_id),
                ("stage", self.stage_id),
                ("branch", self.branch_id),
                ("candidate", self.candidate_id),
            )
            if value
        )
        return f"[{self.code}] {self.message}" + (f" ({location})" if location else "")


class ConfigError(CrystalXError):
    """The user's configuration is missing or self-contradictory."""

    code = "CONFIG_ERROR"


class InputValidationError(CrystalXError):
    """A supplied input file or parameter is unusable."""

    code = "INPUT_VALIDATION_ERROR"


class WorkspaceError(CrystalXError):
    """The run workspace could not be created, or an immutability rule was violated."""

    code = "WORKSPACE_ERROR"


class ToolResolutionError(CrystalXError):
    """A required external program could not be resolved to a usable executable."""

    code = "TOOL_RESOLUTION_ERROR"

    def __init__(self, message: str, *, diagnostic: str | None = None, **kwargs: Any) -> None:
        super().__init__(message, **kwargs)
        self.diagnostic = diagnostic
        if diagnostic:
            self.context.setdefault("diagnostic", diagnostic)


class ToolExecutionError(CrystalXError):
    """An external program was found but the invocation did not succeed."""

    code = "TOOL_EXECUTION_ERROR"

    def __init__(
        self,
        message: str,
        *,
        diagnostic: str | None = None,
        exit_code: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.diagnostic = diagnostic
        self.exit_code = exit_code
        if diagnostic:
            self.context.setdefault("diagnostic", diagnostic)
        if exit_code is not None:
            self.context.setdefault("exit_code", exit_code)


class ParseError(CrystalXError):
    """A crystallographic file could not be parsed.

    Carries the offending file, the 1-based line number, and the raw text, so a user can be shown
    exactly what confused the reader instead of a generic failure.
    """

    code = "PARSE_ERROR"

    def __init__(
        self,
        message: str,
        *,
        source: Path | str | None = None,
        line_no: int | None = None,
        text: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.source = Path(source) if source is not None else None
        self.line_no = line_no
        self.text = text
        if self.source is not None:
            self.context.setdefault("source", str(self.source))
        if line_no is not None:
            self.context.setdefault("line_no", line_no)
        if text is not None:
            self.context.setdefault("text", text)


class SolverError(CrystalXError):
    """An initial-solution backend failed in a way that is not a tool-execution failure."""

    code = "SOLVER_ERROR"


class StrategyError(CrystalXError):
    """The strategy selector could not produce a usable branch plan."""

    code = "STRATEGY_ERROR"


class InferenceError(CrystalXError):
    """An inference backend failed or cannot honour the requested device."""

    code = "INFERENCE_ERROR"

    def __init__(
        self,
        message: str,
        *,
        backend: str | None = None,
        device: str | None = None,
        unsupported_op: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.backend = backend
        self.device = device
        self.unsupported_op = unsupported_op
        for key, value in (
            ("backend", backend),
            ("device", device),
            ("unsupported_op", unsupported_op),
        ):
            if value is not None:
                self.context.setdefault(key, value)


class GateEvaluationError(CrystalXError):
    """A scientific eligibility gate could not be evaluated (missing metric, not a gate failure)."""

    code = "GATE_EVALUATION_ERROR"


class PolicyStop(CrystalXError):
    """A planned, scientifically motivated halt. Not a defect.

    Raised (or reported) when the workflow correctly declines to continue: no trustworthy candidate,
    a budget exhausted, or a review checkpoint that requires a human crystallographer.
    """

    code = "POLICY_STOP"

    def __init__(self, message: str, *, reason_kind: str | None = None, **kwargs: Any) -> None:
        super().__init__(message, **kwargs)
        self.reason_kind = reason_kind
        if reason_kind:
            self.context.setdefault("reason_kind", reason_kind)
