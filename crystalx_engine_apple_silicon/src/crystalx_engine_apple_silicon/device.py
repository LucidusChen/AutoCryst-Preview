"""Explicit device selection without implicit per-operator CPU fallback."""

from __future__ import annotations

from dataclasses import dataclass
import os

from crystalx_core.domain.inference import InferenceDevice


class BackendUnavailable(RuntimeError):
    """A requested execution backend cannot be used under the stated policy."""


@dataclass(frozen=True)
class DeviceDecision:
    requested: InferenceDevice
    used: InferenceDevice
    fell_back: bool = False
    reason: str = ""


def select_device(requested: InferenceDevice | str, *, allow_cpu_fallback: bool = True) -> DeviceDecision:
    import torch

    requested = InferenceDevice(requested)
    if requested == InferenceDevice.CUDA:
        raise BackendUnavailable("This adapter supports CPU/MPS, not CUDA; use the preserved CUDA baseline")
    if requested == InferenceDevice.CPU:
        return DeviceDecision(requested, InferenceDevice.CPU)
    reason = ""
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "1":
        reason = "Implicit per-operator MPS fallback is enabled; pure MPS execution cannot be attested"
    elif not torch.backends.mps.is_built():
        reason = "This torch build does not include MPS"
    elif not torch.backends.mps.is_available():
        reason = "MPS is unavailable in the current runtime (including possible sandbox restrictions)"
    else:
        try:
            probe = torch.ones(2, dtype=torch.float32, device="mps")
            _ = (probe + probe).cpu()
            torch.mps.synchronize()
        except (RuntimeError, NotImplementedError) as exc:
            reason = f"MPS startup probe failed: {exc}"
    if not reason:
        return DeviceDecision(requested, InferenceDevice.MPS)
    if not allow_cpu_fallback:
        raise BackendUnavailable(reason)
    return DeviceDecision(requested, InferenceDevice.CPU, True, reason)
