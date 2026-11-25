from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from crystalx_native_tools.host import HostFacts  # noqa: E402


def apple_silicon(markers: tuple[str, ...] = (), macos: str = "27.0.1") -> HostFacts:
    return HostFacts(
        system="Darwin",
        hardware_arch="arm64",
        process_arch="arm64",
        process_translated=False,
        macos_version=macos,
        rosetta_markers=markers,
        python="3.12",
    )


def linux_host() -> HostFacts:
    return HostFacts(system="Linux", hardware_arch="x86_64", process_arch="x86_64", process_translated=None,
                     macos_version=None, rosetta_markers=(), python="3.12")


def write_exe(path: Path, data: bytes, mode: int = 0o755) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    path.chmod(mode)
    return path


@pytest.fixture
def mac() -> HostFacts:
    return apple_silicon()
