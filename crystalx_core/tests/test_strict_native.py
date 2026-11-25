"""Strict native (no-Rosetta) resolution, explicit tool paths, and macOS quarantine detection.

The "executables" are header-only Mach-O files written by the tests; nothing is executed except
where a test says so (none here). The host is passed in explicitly so these run on any machine.
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

import pytest

from conftest import CPU_ARM64, CPU_X86_64, macho_fat, macho_thin, write_hkl, write_ins
from conftest import write_executable as _write_executable
from crystalx_core.app import RunConfig, RunPhase, RunService, SolveRequest, StageState
from crystalx_core.app.toolchain import RealToolchain
from crystalx_core.domain.tools import BinaryArch, HostPlatform, ToolDiagnosticKind, ToolKind
from crystalx_core.tools.resolver import STRICT_NATIVE_REMEDY, ResolverConfig, ToolResolver
from crystalx_core.tools.xattr import has_xattr

def write_executable(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return _write_executable(path, payload)


APPLE_SILICON = HostPlatform(system="Darwin", arch=BinaryArch.ARM64, rosetta_available=True)
NO_ROSETTA = HostPlatform(system="Darwin", arch=BinaryArch.ARM64, rosetta_available=False)


def _resolver(tmp_path: Path, *, host: HostPlatform, allow_translation: bool, extra: tuple[Path, ...] = ()) -> ToolResolver:
    return ToolResolver(
        ResolverConfig(
            toolbox_dir=tmp_path / "toolbox",
            search_path=[],
            include_known_locations=False,
            include_olex2=False,
            extra_dirs=extra,
            allow_translation=allow_translation,
        ),
        host=host,
        quarantine_checker=lambda _p: False,
    )


def test_strict_mode_refuses_an_x86_64_only_build_and_says_what_to_do(tmp_path: Path):
    write_executable(tmp_path / "toolbox" / "shelxt", macho_thin(CPU_X86_64))
    resolution = _resolver(tmp_path, host=APPLE_SILICON, allow_translation=False).resolve(ToolKind.SHELXT)
    assert not resolution.is_usable
    assert resolution.diagnostic.kind is ToolDiagnosticKind.TRANSLATION_REFUSED
    assert "Strict native mode" in resolution.diagnostic.message
    assert resolution.diagnostic.remedy == STRICT_NATIVE_REMEDY
    assert "cannot turn x86_64 code into arm64 code" in resolution.diagnostic.remedy


def test_strict_mode_keeps_searching_and_reuses_a_native_build_elsewhere(tmp_path: Path):
    write_executable(tmp_path / "toolbox" / "shelxl", macho_thin(CPU_X86_64))
    native = write_executable(tmp_path / "native" / "shelxl", macho_thin(CPU_ARM64))
    resolution = _resolver(
        tmp_path, host=APPLE_SILICON, allow_translation=False, extra=(tmp_path / "native",)
    ).resolve(ToolKind.SHELXL)
    assert resolution.is_usable
    assert resolution.identity.path == native
    assert resolution.diagnostic.kind is ToolDiagnosticKind.OK
    assert [d.kind for d in resolution.rejected] == [ToolDiagnosticKind.TRANSLATION_REFUSED]


def test_a_universal_build_with_arm64_is_native(tmp_path: Path):
    write_executable(tmp_path / "toolbox" / "shelxt", macho_fat([CPU_X86_64, CPU_ARM64]))
    resolution = _resolver(tmp_path, host=APPLE_SILICON, allow_translation=False).resolve(ToolKind.SHELXT)
    assert resolution.is_usable and resolution.diagnostic.kind is ToolDiagnosticKind.OK


def test_without_strict_mode_translation_is_planned_or_explained(tmp_path: Path):
    write_executable(tmp_path / "toolbox" / "shelxt", macho_thin(CPU_X86_64))
    planned = _resolver(tmp_path, host=APPLE_SILICON, allow_translation=True).resolve(ToolKind.SHELXT)
    assert planned.diagnostic.kind is ToolDiagnosticKind.OK_VIA_TRANSLATION
    missing = _resolver(tmp_path, host=NO_ROSETTA, allow_translation=True).resolve(ToolKind.SHELXT)
    assert missing.diagnostic.kind is ToolDiagnosticKind.ROSETTA_MISSING


def test_strict_mode_changes_nothing_on_an_intel_or_linux_host(tmp_path: Path):
    write_executable(tmp_path / "toolbox" / "shelxt", macho_thin(CPU_X86_64))
    intel = HostPlatform(system="Darwin", arch=BinaryArch.X86_64)
    assert _resolver(tmp_path, host=intel, allow_translation=False).resolve(ToolKind.SHELXT).is_usable


def test_run_config_carries_paths_and_policy_absolutely(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = RunConfig(tool_paths=(("shelxt", Path("bin/shelxt")),), extra_tool_dirs=(Path("more"),), strict_native=True)
    assert config.tool_paths == (("shelxt", tmp_path / "bin" / "shelxt"),)
    assert config.extra_tool_dirs == (tmp_path / "more",)
    record = config.to_dict()
    assert record["strict_native"] is True
    assert record["tool_paths"] == {"shelxt": str(tmp_path / "bin" / "shelxt")}
    with pytest.raises(ValueError, match="unknown tool"):
        RunConfig(tool_paths=(("not-a-tool", Path("x")),))


def test_the_toolchain_for_a_config_uses_its_locations_and_policy(tmp_path: Path):
    config = RunConfig(toolbox_dir=tmp_path, tool_paths=(("shelxl", tmp_path / "x"),), strict_native=True)
    toolchain = RealToolchain.for_config(config)
    resolver_config = toolchain.resolver.config
    assert resolver_config.allow_translation is False
    assert resolver_config.explicit_paths == {"shelxl": tmp_path / "x"}
    assert resolver_config.toolbox_dir == tmp_path
    assert resolver_config.include_olex2 is True and resolver_config.include_known_locations is True
    default = RealToolchain.for_config(RunConfig()).resolver.config
    assert default.allow_translation is True
    confined = RunConfig(include_olex2=False, include_known_locations=False)
    assert confined.to_dict()["include_olex2"] is False
    confined_resolver = RealToolchain.for_config(confined).resolver.config
    assert confined_resolver.include_olex2 is False and confined_resolver.include_known_locations is False


def test_capabilities_report_the_policy_a_run_would_use(tmp_path: Path):
    service = RunService(tmp_path / "runs")
    report = service.capabilities(config=RunConfig(toolbox_dir=tmp_path, strict_native=True))
    assert report["translation_allowed"] is False
    assert report["crystalx_inference"]["integrated"] is False
    assert "synthetic weights" in report["crystalx_inference"]["note"]


def test_a_strict_run_on_apple_silicon_blocks_before_starting_x86_64_shelxt(tmp_path: Path):
    write_executable(tmp_path / "toolbox" / "shelxt", macho_thin(CPU_X86_64))

    def factory(request: SolveRequest):
        return RealToolchain(
            resolver=_resolver(tmp_path, host=APPLE_SILICON, allow_translation=not request.config.strict_native)
        )

    service = RunService(tmp_path / "runs", toolchain_factory=factory)
    ins, hkl = write_ins(tmp_path / "c.ins"), write_hkl(tmp_path / "c.hkl")
    handle = service.submit(
        SolveRequest(ins_path=ins, hkl_path=hkl, config=RunConfig(strict_native=True), label="strict")
    )
    state = service.wait(handle.run_id, timeout=30)
    assert state["phase"] == RunPhase.BLOCKED.value
    shelxt = next(s for s in state["stages"] if s["name"] == "shelxt")
    assert shelxt["state"] == StageState.TOOL_UNAVAILABLE.value
    assert "Strict native mode refuses translation" in state["message"]
    assert not list((handle.run_dir).glob("stages/*/branches/*/provenance_*.json")), "nothing may start"


def test_crystalx_solve_accepts_strict_native(tmp_path: Path):
    from crystalx_core.cli.main import build_parser
    from crystalx_core.cli.solve import build_request

    args = build_parser().parse_args(["solve", "--demo", "--runs-dir", str(tmp_path), "--strict-native"])
    assert build_request(args).config.strict_native is True


# --------------------------------------------------------------------------------------
# extended attributes (the macOS quarantine flag was never visible to os.getxattr there)
# --------------------------------------------------------------------------------------


@pytest.mark.skipif(not hasattr(os, "setxattr"), reason="Linux user xattrs")
def test_has_xattr_on_linux(tmp_path: Path):
    target = tmp_path / "f"
    target.write_text("x")
    try:
        os.setxattr(target, "user.crystalx.test", b"1")
    except OSError:
        pytest.skip("filesystem without user xattrs")
    assert has_xattr(target, "user.crystalx.test") is True
    assert has_xattr(target, "user.crystalx.absent") is False
    assert has_xattr(target, "com.apple.quarantine") is False


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS only")
def test_has_xattr_sees_quarantine_on_macos(tmp_path: Path):
    target = tmp_path / "downloaded"
    target.write_text("x")
    assert has_xattr(target, "com.apple.quarantine") is False
    libc = ctypes.CDLL(None, use_errno=True)
    libc.setxattr.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_void_p, ctypes.c_size_t,
                              ctypes.c_uint32, ctypes.c_int]
    libc.setxattr.restype = ctypes.c_int
    value = b"0081;00000000;CrystalXTest;"
    rc = libc.setxattr(os.fsencode(str(target)), b"com.apple.quarantine", value, len(value), 0, 0)
    assert rc == 0, ctypes.get_errno()
    assert has_xattr(target, "com.apple.quarantine") is True
    from crystalx_core.tools.resolver import _default_quarantine_checker

    assert _default_quarantine_checker(target) is True
