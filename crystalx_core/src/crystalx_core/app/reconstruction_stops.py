"""Public reconstruction termination contract, independent of numerical providers.

Only keys of STOP_CODES are public terminal stop codes. Detailed guard identifiers
and controller messages remain in stop_detail rather than becoming undocumented
frontend codes. Numerical completion never certifies scientific convergence.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType


__all__ = ["STOP_CATEGORIES", "STOP_CODES", "StopCodeInfo", "stop_record", "empty_stop_record"]

STOP_CATEGORIES = (
    "numerical_stop", "fixed_rounds", "budget_exhausted", "cancelled", "blocked", "error",
)


@dataclass(frozen=True)
class StopCodeInfo:
    category: str
    phase: str
    description: str


STOP_CODES = MappingProxyType({
    "heavy_stage_completed": StopCodeInfo("fixed_rounds", "completed",
        "The explicitly selected initial heavy assignment and one anisotropic L.S. 16 stage completed; no iterative convergence, hydrogen completion or scientific validation is implied."),
    "no_acceptable_step_in_current_action_set": StopCodeInfo("blocked", "blocked",
        "Distinct proposals were rejected and the reduced action sets through pure refinement are exhausted; the last retained candidate remains unvalidated."),
    "anchor_displacement_unsupported": StopCodeInfo("blocked", "blocked",
        "The configured anchor-support policy blocked the trial after Uiso checks; the remaining supported anchor count was insufficient or block-any was requested."),
    "important_coordinates_and_r1_stable": StopCodeInfo("numerical_stop", "completed",
        "Legacy policy: selected important coordinates and R1 met the configured stability tolerances; the candidate remains scientifically unvalidated."),
    "movable_sites_r1_and_growth_stable": StopCodeInfo("numerical_stop", "completed",
        "Movable monitored sites and R1 met the configured tolerances, with two quiet rounds without accepted additions or deletions and the configured element-change rule; the candidate remains scientifically unvalidated."),
    "fixed_rounds_completed": StopCodeInfo("fixed_rounds", "completed",
        "The explicitly requested number of outer reconstruction rounds completed; this does not establish numerical or scientific convergence."),
    "failure_budget_exhausted": StopCodeInfo("budget_exhausted", "failed",
        "The configured unsuccessful-trial budget was exhausted; any exported candidate is the last retained unvalidated model."),
    "round_safety_budget_exhausted": StopCodeInfo("budget_exhausted", "failed",
        "The automatic round safety limit was reached without meeting the stopping policy."),
    "whole_chain_wall_budget_exhausted": StopCodeInfo("budget_exhausted", "failed",
        "The wall-clock budget for the complete initial-solution and refinement chain was exhausted."),
    "workflow_budget_exhausted": StopCodeInfo("budget_exhausted", "failed",
        "A controller resource budget was exhausted; stop_detail identifies the reported budget or controller message."),
    "cancelled": StopCodeInfo("cancelled", "cancelled",
        "The run was stopped on request; no automatic replay or scientific validation is implied."),
    "preview_blocked": StopCodeInfo("blocked", "blocked",
        "Static preview prerequisites are not satisfied; blockers list the missing inputs, tools or route requirements."),
    "initial_solution_blocked": StopCodeInfo("blocked", "blocked",
        "The initial route could not deliver a usable complete seed; blockers and stop_detail preserve the route evidence."),
    "reconstruction_blocked": StopCodeInfo("blocked", "blocked",
        "A reconstruction input, provenance, supported-model or consistency guard blocked further execution; stop_detail preserves its exact technical identifier."),
    "workflow_stopped": StopCodeInfo("blocked", "blocked",
        "The controller stopped without a reconstruction numerical or fixed-round completion; stop_detail preserves its reason."),
    "initial_solution_failed": StopCodeInfo("error", "failed",
        "The initial-solution execution failed; blockers and stop_detail preserve the route diagnostics."),
    "refinement_failed": StopCodeInfo("error", "failed",
        "The refinement process failed or its listing reported a fatal error; stop_detail preserves the diagnostic."),
    "backend_error": StopCodeInfo("error", "failed",
        "An unexpected backend error interrupted execution; stop_detail and error_type retain diagnostic context."),
    "owner_process_interrupted": StopCodeInfo("error", "interrupted",
        "The process that owned this queued or running job is no longer available; the job is not automatically replayed."),
})


def empty_stop_record() -> dict:
    return {"stop_code": None, "stop_category": None, "stop_description": None, "stop_detail": None}


def stop_record(code: str, *, detail: str | None = None, fallback_code: str = "reconstruction_blocked") -> dict:
    """Canonicalize a technical termination without losing its original identifier.

No code is classified by substring. The explicit native-error family is the only
technical prefix recognized here; all other unknown guards use the named fallback.
"""
    if code not in STOP_CODES:
        original = code
        if code.startswith(("shelxl_process_failed:", "shelxl_listing_reports_fatal_error:")) or code == "shelxl_listing_reports_fatal_error":
            code = "refinement_failed"
        else:
            code = fallback_code
        detail = original if detail is None else original + ": " + detail
    info = STOP_CODES[code]
    return {"phase": info.phase, "stop_code": code, "stop_category": info.category,
            "stop_description": info.description, "stop_detail": detail}
