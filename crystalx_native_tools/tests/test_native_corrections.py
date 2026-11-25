"""Review findings D3 and D5 (2026-10-22): broken or wrong files are refused, and deep checks steer
selection everywhere -- the tool panel, the capability report and an actual run.

All files are synthetic; nothing is executed.
"""

from __future__ import annotations

import struct
from pathlib import Path

import machobuild as mb
from conftest import apple_silicon, write_exe
from crystalx_core.domain.tools import BinaryArch, HostPlatform, ToolDiagnosticKind, ToolKind
from crystalx_core.tools.resolver import ResolverConfig, ToolResolver
from crystalx_native_tools.config import ToolsConfig
from crystalx_native_tools.inspect import candidate_check_for, inspect_file
from crystalx_native_tools.inventory import build_inventory

MAC = HostPlatform("Darwin", BinaryArch.ARM64, False)


def _inventory(box: Path, tools=("shelxt",)):
    config = ToolsConfig(toolbox=box, include_known_locations=False, include_olex2=False)
    return build_inventory(config, tools=tools, strict_native=True, host=apple_silicon(), search_path=[])


def _run_choice(box: Path, kind: ToolKind):
    resolver = ToolResolver(ResolverConfig(toolbox_dir=box, search_path=[], include_known_locations=False,
                                           include_olex2=False, allow_translation=False,
                                           candidate_check=candidate_check_for(apple_silicon())), host=MAC)
    return resolver.resolve(kind)


def _truncated_load_commands() -> bytes:
    image = bytearray(mb.thin(mb.CPU_ARM64))
    ncmds = struct.unpack_from("<I", image, 16)[0]
    struct.pack_into("<I", image, 16, ncmds + 3)  # claims commands that are not there
    return bytes(image)


def _truncated_signature() -> bytes:
    image = mb.thin(mb.CPU_ARM64)
    return image[: len(image) - 40]  # the signature blob is cut short


CASES = {
    "not_an_executable": mb.thin(mb.CPU_ARM64, filetype=mb.MH_DYLIB),
    "wrong_platform": mb.thin(mb.CPU_ARM64, platform=2),  # iOS
    "malformed_binary": _truncated_load_commands(),
    "unsigned_arm64": _truncated_signature(),
}


def test_known_bad_files_are_blocked_in_the_panel_and_never_selected(tmp_path: Path):
    for code, image in CASES.items():
        box = tmp_path / code
        write_exe(box / "shelxt", image)
        status = _inventory(box).get("shelxt")
        assert status.static == "blocked", code
        assert status.selected is None and status.blockers[0].code == code, (code, status.blockers)
        assert _run_choice(box, ToolKind.SHELXT).identity is None, code
        assert _inventory(box).pipeline == "blocked"


def test_a_bad_file_yields_to_a_good_one_later_in_the_search(tmp_path: Path):
    box, more = tmp_path / "toolbox", tmp_path / "more"
    write_exe(box / "shelxt", CASES["not_an_executable"])
    good = write_exe(more / "shelxt", mb.thin(mb.CPU_ARM64))
    config = ToolsConfig(toolbox=box, search_dirs=(more,), include_known_locations=False, include_olex2=False)
    inventory = build_inventory(config, tools=("shelxt",), strict_native=True, host=apple_silicon(), search_path=[])
    assert inventory.get("shelxt").selected.path == str(good)
    resolver = ToolResolver(ResolverConfig(toolbox_dir=box, extra_dirs=(more,), search_path=[],
                                           include_known_locations=False, include_olex2=False,
                                           allow_translation=False,
                                           candidate_check=candidate_check_for(apple_silicon())), host=MAC)
    resolution = resolver.resolve(ToolKind.SHELXT)
    assert str(resolution.identity.path) == str(good)
    assert resolution.rejected[0].kind is ToolDiagnosticKind.NOT_AN_EXECUTABLE


def test_signature_presence_is_not_called_validity(tmp_path: Path):
    good = inspect_file(write_exe(tmp_path / "good", mb.thin(mb.CPU_ARM64)), apple_silicon())
    assert good.code_signature["present"] and good.code_signature["container_readable"]
    assert good.code_signature["cryptographically_verified"] is False  # never claimed
    cut = inspect_file(write_exe(tmp_path / "cut", _truncated_signature()), apple_silicon())
    assert cut.code_signature["present"] and not cut.code_signature["container_readable"]
    assert cut.first_blocking().code == "unsigned_arm64"


def test_an_x86_64_file_with_an_unreadable_signature_is_reported_but_not_blocked(tmp_path: Path):
    image = mb.thin(mb.CPU_X86_64, signed=True)
    report = inspect_file(write_exe(tmp_path / "intel", image[: len(image) - 40]), apple_silicon())
    assert report.first_blocking() is None
    assert any(f.code == "signature_unreadable" and not f.blocking for f in report.findings)


def test_the_good_file_has_no_findings(tmp_path: Path):
    report = inspect_file(write_exe(tmp_path / "ok", mb.thin(mb.CPU_ARM64)), apple_silicon())
    assert report.findings == () and candidate_check_for(apple_silicon())(ToolKind.SHELXT, tmp_path / "ok") is None
