"""Facts about the machine, as far as running crystallographic executables is concerned.

``platform.machine()`` describes the *Python process*, not the hardware: an x86_64 Python started
under Rosetta 2 on an Apple Silicon Mac reports ``x86_64``. The hardware is asked directly
(``hw.optional.arm64``) and so is whether this process is translated (``sysctl.proc_translated``),
through ``sysctlbyname`` via :mod:`ctypes`. Rosetta marker files are reported as markers only --
a marker is not a launch test.
"""

from __future__ import annotations

import ctypes
import platform
import sys
from dataclasses import dataclass
from pathlib import Path

from crystalx_core.tools.resolver import ROSETTA_MARKER_PATHS

__all__ = ["HostFacts", "detect_host"]


@dataclass(frozen=True)
class HostFacts:
    system: str
    """``Darwin``, ``Linux``, ``Windows``."""

    hardware_arch: str
    """``arm64`` on Apple Silicon even when this Python itself runs translated."""

    process_arch: str
    process_translated: bool | None
    """``True`` when this Python process runs under Rosetta 2; ``None`` when not determinable."""

    macos_version: str | None
    rosetta_markers: tuple[str, ...]
    """Rosetta 2 marker files that exist. Presence is not proof that x86_64 programs launch."""

    python: str

    @property
    def is_apple_silicon(self) -> bool:
        return self.system == "Darwin" and self.hardware_arch == "arm64"

    def to_record(self) -> dict[str, object]:
        return {
            "system": self.system,
            "hardware_arch": self.hardware_arch,
            "process_arch": self.process_arch,
            "process_translated": self.process_translated,
            "macos_version": self.macos_version,
            "rosetta_markers": list(self.rosetta_markers),
            "rosetta_note": "marker files only; whether an x86_64 program launches needs a launch test",
            "python": self.python,
        }


def _sysctl_int(name: str) -> int | None:
    if sys.platform != "darwin":
        return None
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        sysctlbyname = libc.sysctlbyname
    except (OSError, AttributeError):
        return None
    value = ctypes.c_int(0)
    size = ctypes.c_size_t(ctypes.sizeof(value))
    sysctlbyname.argtypes = [ctypes.c_char_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.c_void_p, ctypes.c_size_t]
    sysctlbyname.restype = ctypes.c_int
    if sysctlbyname(name.encode(), ctypes.byref(value), ctypes.byref(size), None, 0) != 0:
        return None
    return value.value


def _normalise(machine: str) -> str:
    lowered = machine.lower()
    if lowered in ("x86_64", "amd64"):
        return "x86_64"
    if lowered in ("arm64", "aarch64"):
        return "arm64"
    return lowered or "unknown"


def detect_host(
    *,
    system: str | None = None,
    machine: str | None = None,
    exists=Path.exists,
) -> HostFacts:
    """Describe this machine. Arguments override detection, for tests."""
    resolved_system = system or platform.system()
    process_arch = _normalise(machine or platform.machine())
    translated = None
    hardware = process_arch
    macos = None
    if resolved_system == "Darwin" and system is None:
        flag = _sysctl_int("sysctl.proc_translated")
        translated = None if flag is None else bool(flag)
        arm64_hw = _sysctl_int("hw.optional.arm64")
        if arm64_hw == 1 or translated:
            hardware = "arm64"
        macos = platform.mac_ver()[0] or None
    markers = tuple(str(p) for p in ROSETTA_MARKER_PATHS if exists(p)) if resolved_system == "Darwin" else ()
    return HostFacts(
        system=resolved_system,
        hardware_arch=hardware,
        process_arch=process_arch,
        process_translated=translated,
        macos_version=macos,
        rosetta_markers=markers,
        python=platform.python_version(),
    )
