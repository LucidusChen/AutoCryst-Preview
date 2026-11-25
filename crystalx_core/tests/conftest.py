"""Shared fixtures.

Every fixture here is synthetic. No proprietary binary is used, referenced, or executed, and the
"executables" the resolver tests inspect are byte patterns written by the tests themselves.
"""

from __future__ import annotations

import datetime as _dt
import os
import struct
from pathlib import Path

import pytest

# --------------------------------------------------------------------------------------
# deterministic time and identifiers
# --------------------------------------------------------------------------------------

FIXED_TIME = _dt.datetime(2026, 8, 18, 6, 45, 0, tzinfo=_dt.timezone.utc)


@pytest.fixture
def fixed_clock():
    """A clock that advances one second per call, so durations are non-zero but deterministic."""
    state = {"n": 0}

    def clock() -> _dt.datetime:
        value = FIXED_TIME + _dt.timedelta(seconds=state["n"])
        state["n"] += 1
        return value

    return clock


@pytest.fixture
def frozen_clock():
    def clock() -> _dt.datetime:
        return FIXED_TIME

    return clock


@pytest.fixture
def fixed_entropy():
    return lambda: "1a2b3c4d"


# --------------------------------------------------------------------------------------
# SHELX fixtures
# --------------------------------------------------------------------------------------

CUBIC_A = 10.0


def write_ins(
    path: Path,
    *,
    title: str = "test structure",
    wavelength: float = 0.71073,
    cell: tuple[float, float, float, float, float, float] = (
        CUBIC_A,
        CUBIC_A,
        CUBIC_A,
        90.0,
        90.0,
        90.0,
    ),
    latt: int | None = 1,
    symm: tuple[str, ...] = ("-X, -Y, Z",),
    sfac: tuple[str, ...] = ("C", "N", "O", "Ag"),
    unit: tuple[int, ...] = (24, 4, 8, 2),
    hklf: int | None = 4,
    extra_lines: tuple[str, ...] = (),
) -> Path:
    """Write a minimal but realistic SHELX instruction file."""
    lines = [f"TITL {title}", "CELL " + " ".join(f"{v:g}" for v in (wavelength, *cell))]
    lines.append("ZERR 2.00 0.0010 0.0010 0.0010 0.000 0.000 0.000")
    if latt is not None:
        lines.append(f"LATT {latt}")
    lines.extend(f"SYMM {op}" for op in symm)
    lines.append("SFAC " + " ".join(sfac))
    lines.append("UNIT " + " ".join(str(u) for u in unit))
    lines.extend(extra_lines)
    lines.append("L.S. 10")
    lines.append("FVAR 0.5")
    if hklf is not None:
        lines.append(f"HKLF {hklf}")
    lines.append("END")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_hkl(
    path: Path,
    *,
    reflections: list[tuple[int, int, int, float, float]] | None = None,
    include_friedel: bool = False,
    terminator: bool = True,
) -> Path:
    """Write a SHELX HKLF-4 reflection file in the fixed-format layout SHELX itself uses."""
    if reflections is None:
        reflections = []
        for h in range(1, 6):
            for k in range(0, 4):
                for l in range(0, 3):
                    reflections.append((h, k, l, 100.0 / (h + k + l + 1), 2.0))
        if include_friedel:
            reflections += [(-h, -k, -l, i * 0.98, s) for h, k, l, i, s in list(reflections)]
    lines = [
        f"{h:4d}{k:4d}{l:4d}{i:8.2f}{s:8.2f}" for h, k, l, i, s in reflections
    ]
    if terminator:
        lines.append(f"{0:4d}{0:4d}{0:4d}{0.0:8.2f}{0.0:8.2f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def merged_case(tmp_path: Path) -> dict[str, Path]:
    """A conventional, merged, heavy-atom-containing case with a known space group."""
    return {
        "ins": write_ins(tmp_path / "merged.ins"),
        "hkl": write_hkl(tmp_path / "merged.hkl", include_friedel=False),
    }


@pytest.fixture
def anomalous_case(tmp_path: Path) -> dict[str, Path]:
    """The same structure with unmerged Friedel pairs, which enables the SHELXC-D-E route."""
    return {
        "ins": write_ins(tmp_path / "anom.ins"),
        "hkl": write_hkl(tmp_path / "anom.hkl", include_friedel=True),
    }


@pytest.fixture
def light_unknown_sg_case(tmp_path: Path) -> dict[str, Path]:
    """Pure organic, no declared space group: only SHELXT can proceed."""
    return {
        "ins": write_ins(
            tmp_path / "organic.ins",
            sfac=("C", "H", "N", "O"),
            unit=(40, 60, 4, 8),
            latt=None,
            symm=(),
        ),
        "hkl": write_hkl(tmp_path / "organic.hkl"),
    }


# --------------------------------------------------------------------------------------
# synthetic executables (header bytes only; nothing is ever run unless a test says so)
# --------------------------------------------------------------------------------------

MACHO_MAGIC_64_LE = b"\xcf\xfa\xed\xfe"
CPU_X86_64 = 0x01000007
CPU_ARM64 = 0x0100000C


def macho_thin(cputype: int) -> bytes:
    """A 64-bit little-endian Mach-O header with the given cputype."""
    return MACHO_MAGIC_64_LE + struct.pack("<i", cputype) + b"\x00" * 24


def macho_fat(cputypes: list[int]) -> bytes:
    """A Mach-O universal (fat) header. The fat header is big-endian on disk."""
    head = struct.pack(">I", 0xCAFEBABE) + struct.pack(">I", len(cputypes))
    for cputype in cputypes:
        head += struct.pack(">i", cputype) + b"\x00" * 16
    return head + b"\x00" * 32


def elf(machine: int) -> bytes:
    """A little-endian 64-bit ELF header with the given e_machine."""
    head = bytearray(b"\x7fELF")
    head += bytes([2, 1, 1, 0])  # 64-bit, little-endian, version 1
    head += b"\x00" * 8
    head += struct.pack("<H", 2)  # e_type = ET_EXEC
    head += struct.pack("<H", machine)
    head += b"\x00" * 40
    return bytes(head)


def write_executable(path: Path, payload: bytes, *, mode: int = 0o755) -> Path:
    path.write_bytes(payload)
    os.chmod(path, mode)
    return path


def write_shell_stub(path: Path, body: str, *, mode: int = 0o755) -> Path:
    """A tiny POSIX shell script used to exercise the process runner.

    These are written by the test itself. They are not crystallographic software.
    """
    path.write_text("#!/bin/sh\n" + body, encoding="utf-8")
    os.chmod(path, mode)
    return path


@pytest.fixture
def toolbox(tmp_path: Path) -> Path:
    directory = tmp_path / "toolbox"
    directory.mkdir()
    return directory


# --------------------------------------------------------------------------------------
# real-tool boundary
# --------------------------------------------------------------------------------------
#
# CI and the default test run must never depend on SHELX, PLATON, cctbx or model weights.
# Anything that touches a real crystallographic binary is opt-in twice over: the environment
# variable must be set, *and* the tool must actually resolve on this machine. Otherwise the test
# skips with a reason that says which of the two was missing.

REAL_TOOLS_ENABLED = os.environ.get("CRYSTALX_REAL_TOOLS", "").strip().lower() in {"1", "true", "yes"}
REAL_TOOLBOX = os.environ.get("CRYSTALX_TOOLBOX", "").strip()


def real_tool_resolution(kind):
    """Resolve one real tool, or return ``None`` when real-tool testing is not enabled."""
    if not REAL_TOOLS_ENABLED:
        return None
    from crystalx_core.tools.resolver import ResolverConfig, ToolResolver

    config = ResolverConfig(toolbox_dir=Path(REAL_TOOLBOX) if REAL_TOOLBOX else None)
    return ToolResolver(config).resolve(kind)


@pytest.fixture
def require_real_tool():
    """Fixture factory: skip unless the named tool is genuinely available and opted in."""

    def _require(kind):
        if not REAL_TOOLS_ENABLED:
            pytest.skip(
                "real-tool tests are opt-in: set CRYSTALX_REAL_TOOLS=1 (and CRYSTALX_TOOLBOX "
                "if the binaries are not on PATH)"
            )
        resolution = real_tool_resolution(kind)
        if resolution is None or not resolution.is_usable:
            reason = resolution.diagnostic.message if resolution else "resolver returned nothing"
            pytest.skip(f"{kind.value} is not usable on this machine: {reason}")
        return resolution

    return _require
