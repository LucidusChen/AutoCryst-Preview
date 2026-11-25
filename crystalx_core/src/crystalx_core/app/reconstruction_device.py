"""Choose a run's inference device once, before any numerical work.

CPU selection at startup is distinct from per-operator fallback. The latter is
never enabled here, and the runtime provider only receives an explicit device.
"""
from __future__ import annotations

import platform


def _native_selection(requested: str):
    from crystalx_engine_apple_silicon.device import select_device
    return select_device(requested, allow_cpu_fallback=requested == "auto")


def resolve_inference_device(requested: str, *, synthetic: bool = False) -> dict:
    if requested not in ("auto", "cpu", "mps"):
        raise ValueError("unsupported inference device")
    record = {"device_requested": requested, "device_used": None,
              "device_selection_reason": "", "device_blocker": None,
              "operator_cpu_fallback_allowed": False}
    if requested == "cpu" or synthetic and requested == "auto":
        record.update(device_used="cpu", device_selection_reason=(
            "synthetic provider; no pretrained accelerator execution" if synthetic else "explicit CPU"))
        return record
    if synthetic:
        record["device_blocker"] = "synthetic_provider_does_not_load_weights_or_use_accelerators"
        return record
    apple_arm = platform.system() == "Darwin" and platform.machine().lower() in ("arm64", "aarch64")
    if not apple_arm:
        if requested == "auto":
            record.update(device_used="cpu", device_selection_reason="host is not Apple silicon")
        else:
            record["device_blocker"] = "mps_requires_available_apple_silicon_runtime"
        return record
    try:
        decision = _native_selection(requested)
        record.update(device_used=decision.used.value,
                      device_selection_reason=decision.reason or "strict MPS startup probe passed")
    except Exception as error:
        # An unavailable optional adapter/torch build is a startup CPU choice for
        # auto. An explicit MPS request is blocked, never silently changed.
        reason = type(error).__name__ + ": " + str(error)
        if requested == "auto":
            record.update(device_used="cpu", device_selection_reason=reason)
        else:
            record.update(device_blocker="requested_mps_unavailable_without_fallback",
                          device_selection_reason=reason)
    return record
