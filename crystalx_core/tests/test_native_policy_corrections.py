"""Review findings D4 and D5 (2026-10-22), core side.

D4: strict native mode must not certify a wrapper script or a file of unknown format as native.
D5: one injected candidate check is used for the capability report and for actual runs, so a
deep-blocked program is never selected and a good fallback is.

Every executable here is a synthetic script or header; scripts only ever touch a marker file, and
the tests assert that none was started where nothing may start.
"""

from __future__ import annotations

import os
from pathlib import Path

from crystalx_core.app import RunConfig, RunMode, RunPhase, RunService, SolveRequest
from crystalx_core.app.service import default_toolchain
from crystalx_core.app.toolchain import RealToolchain
from crystalx_core.domain.tools import BinaryArch, HostPlatform, ToolDiagnostic, ToolDiagnosticKind, ToolKind
from crystalx_core.tools.resolver import NATIVE_UNVERIFIED_REMEDY, ResolverConfig, ToolResolver

from conftest import write_hkl, write_ins  # noqa: E402

MAC = HostPlatform(system="Darwin", arch=BinaryArch.ARM64, rosetta_available=True)
LINUX = HostPlatform(system="Linux", arch=BinaryArch.X86_64, rosetta_available=None)


def _tripwire(box: Path, marker: Path, names=("shelxt", "shelxl", "platon_special", "platon")) -> None:
    box.mkdir(parents=True, exist_ok=True)
    for name in names:
        program = box / name
        program.write_text(f"#!/bin/sh\necho \"$0\" >> '{marker}'\nexit 1\n", encoding="utf-8")
        program.chmod(0o755)


def _macho_arm64() -> bytes:
    import struct

    return struct.pack("<IiiIIIII", 0xFEEDFACF, 0x0100000C, 0, 2, 0, 0, 0, 0)


def _resolver(box: Path, *, strict: bool, host: HostPlatform, extra=(), check=None) -> ToolResolver:
    return ToolResolver(ResolverConfig(toolbox_dir=box, extra_dirs=tuple(extra), search_path=[],
                                       include_known_locations=False, include_olex2=False,
                                       allow_translation=not strict, candidate_check=check), host=host)


def test_strict_mode_refuses_wrapper_scripts_and_unknown_formats_on_any_host(tmp_path: Path):
    marker = tmp_path / "started"
    box = tmp_path / "toolbox"
    _tripwire(box, marker, names=("shelxt",))
    unknown = box / "shelxl"
    unknown.write_bytes(b"\x00\x01 neither script nor known binary")
    unknown.chmod(0o755)
    for host in (MAC, LINUX):
        resolver = _resolver(box, strict=True, host=host)
        for kind in (ToolKind.SHELXT, ToolKind.SHELXL):
            resolution = resolver.resolve(kind)
            assert resolution.identity is None
            assert resolution.diagnostic.kind is ToolDiagnosticKind.NATIVE_UNVERIFIED
            assert resolution.diagnostic.remedy == NATIVE_UNVERIFIED_REMEDY
    assert "cannot verify what it would run" in _resolver(box, strict=True, host=MAC).resolve(
        ToolKind.SHELXT).diagnostic.message
    # Outside strict mode a script is still usable, as before.
    assert _resolver(box, strict=False, host=MAC).resolve(ToolKind.SHELXT).is_usable
    assert not marker.exists()


def test_strict_mode_keeps_searching_past_a_wrapper_to_a_native_build(tmp_path: Path):
    box, native_dir = tmp_path / "toolbox", tmp_path / "native"
    _tripwire(box, tmp_path / "started", names=("shelxt",))
    native_dir.mkdir()
    native = native_dir / "shelxt"
    native.write_bytes(_macho_arm64())
    native.chmod(0o755)
    resolution = _resolver(box, strict=True, host=MAC, extra=(native_dir,)).resolve(ToolKind.SHELXT)
    assert resolution.is_usable and resolution.identity.path == native
    assert resolution.rejected[0].kind is ToolDiagnosticKind.NATIVE_UNVERIFIED


def test_a_strict_real_run_with_wrapper_scripts_starts_nothing(tmp_path: Path):
    marker = tmp_path / "started"
    box = tmp_path / "toolbox"
    _tripwire(box, marker)

    def factory(request: SolveRequest):
        return RealToolchain(resolver=_resolver(box, strict=request.config.strict_native, host=MAC))

    service = RunService(tmp_path / "runs", toolchain_factory=factory)
    ins, hkl = write_ins(tmp_path / "c.ins"), write_hkl(tmp_path / "c.hkl")
    handle = service.submit(SolveRequest(ins_path=ins, hkl_path=hkl, config=RunConfig(strict_native=True),
                                         label="strict-wrappers"))
    state = service.wait(handle.run_id, timeout=30)
    assert state["phase"] == RunPhase.BLOCKED.value
    assert "native" in state["message"].lower()
    assert not marker.exists(), "a wrapper script was started in strict native mode"


def test_one_injected_check_steers_capabilities_and_runs(tmp_path: Path):
    """D5: a deep-blocked platon_special yields to platon in the report and in the run's toolchain."""
    box = tmp_path / "toolbox"
    _tripwire(box, tmp_path / "started")
    seen: list[str] = []

    def check(kind: ToolKind, path: Path):
        seen.append(f"{kind.value}:{path.name}")
        if kind is ToolKind.PLATON_SPECIAL:  # any copy anywhere: keep the machine's own out of it
            return ToolDiagnostic(kind=ToolDiagnosticKind.LIBRARY_UNAVAILABLE, message=f"{path}: needs libX11",
                                  path=path, remedy="restore the library")
        return None

    config = RunConfig(toolbox_dir=box, strict_native=False, include_olex2=False, include_known_locations=False)
    service = RunService(tmp_path / "runs", candidate_check=check)
    report = service.capabilities(RunMode.REAL, config=config)
    assert report["tools"]["platon_special"]["usable_by_static_inspection"] is False
    assert report["tools"]["platon"]["usable_by_static_inspection"] is True
    assert "platon_special:platon_special" in seen

    request = SolveRequest(ins_path=tmp_path / "x.ins", hkl_path=tmp_path / "x.hkl", config=config)
    toolchain = service._toolchain_factory(request)
    assert toolchain.resolver.config.candidate_check is check
    toolchain.resolver = ToolResolver(
        ResolverConfig(**{**toolchain.resolver.config.__dict__, "search_path": []}), host=MAC)
    assert toolchain.resolver.resolve_platon().kind is ToolKind.PLATON
    assert default_toolchain(request).resolver.config.candidate_check is None  # nothing injected by default
    assert not (tmp_path / "started").exists()
