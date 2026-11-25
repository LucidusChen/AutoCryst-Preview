"""The inventory: selection identical to a run, strict native policy, blockers, resources, provenance.

Every executable is a synthetic header-only file. Searches are confined to the test's directories.
"""

from __future__ import annotations

import json
from pathlib import Path

import machobuild as mb
from conftest import apple_silicon, write_exe
from crystalx_core.app import RunConfig
from crystalx_core.app.toolchain import RealToolchain
from crystalx_core.domain.tools import ToolKind
from crystalx_native_tools.config import ToolsConfig, load_config
from crystalx_native_tools.inventory import build_inventory
from crystalx_native_tools.probe import PROBE_SCHEMA, ProbeStore
from crystalx_native_tools.resources import checkdef_version, embedded_platon_dates

#: Exists on no machine, so the test does not depend on XQuartz being installed where it runs.
X11 = "/nonexistent/crystalx-test/X11/lib/libX11.6.dylib"


def _config(tmp_path: Path, **kwargs) -> ToolsConfig:
    return ToolsConfig(
        toolbox=tmp_path / "toolbox",
        include_olex2=False,
        include_known_locations=False,
        **kwargs,
    )


def _inventory(tmp_path: Path, *, strict: bool = True, host=None, **kwargs):
    return build_inventory(
        _config(tmp_path, **kwargs), strict_native=strict, host=host or apple_silicon(), search_path=[]
    )


def _all_native_toolbox(tmp_path: Path) -> None:
    for name in ("shelxt", "shelxl", "platon_special", "platon"):
        write_exe(tmp_path / "toolbox" / name, mb.thin(mb.CPU_ARM64, identifier=name))


def test_a_complete_native_toolbox_is_statically_ready_but_not_launch_tested(tmp_path: Path):
    _all_native_toolbox(tmp_path)
    inventory = _inventory(tmp_path)
    assert inventory.pipeline == "ready_static"
    assert inventory.validation_tool == "platon_special"
    for status in inventory.tools:
        assert status.static == "ok" and status.present
        assert status.launch == {"status": "not_tested"}
        assert status.validated == "not_verified"
        assert status.selected.native == "native_arm64"
        assert len(status.selected.inspection.sha256) == 64
    json.dumps(inventory.to_record())  # serialisable provenance


def test_strict_mode_blocks_x86_64_shelx_and_says_what_to_do(tmp_path: Path):
    _all_native_toolbox(tmp_path)
    for name in ("shelxt", "shelxl"):
        write_exe(tmp_path / "toolbox" / name, mb.thin(mb.CPU_X86_64))
    inventory = _inventory(tmp_path)
    assert inventory.pipeline == "blocked"
    assert inventory.pipeline_blockers == ("shelxt", "shelxl")
    shelxt = inventory.get("shelxt")
    assert shelxt.static == "blocked" and shelxt.selected is None
    assert shelxt.candidates[0].native == "x86_64_only"
    blocker = shelxt.blockers[0]
    assert blocker.code == "translation_refused"
    assert "cannot turn x86_64 code into arm64 code" in blocker.next_step
    assert shelxt.availability.native_arm64_build == "unknown"
    assert shelxt.availability.source_code == "distributed_as_executables"


def test_strict_mode_reuses_a_native_build_found_later(tmp_path: Path):
    _all_native_toolbox(tmp_path)
    write_exe(tmp_path / "toolbox" / "shelxt", mb.thin(mb.CPU_X86_64))
    native = write_exe(tmp_path / "native" / "shelxt", mb.thin(mb.CPU_ARM64))
    inventory = _inventory(tmp_path, search_dirs=(tmp_path / "native",))
    shelxt = inventory.get("shelxt")
    assert shelxt.static == "ok" and shelxt.selected.path == str(native)
    assert [c.resolver_verdict for c in shelxt.candidates] == ["translation_refused", "ok"]


def test_the_inventory_selects_exactly_what_a_run_would_use(tmp_path: Path):
    _all_native_toolbox(tmp_path)
    write_exe(tmp_path / "toolbox" / "shelxl", mb.thin(mb.CPU_X86_64))
    native = write_exe(tmp_path / "more" / "shelxl", mb.thin(mb.CPU_ARM64))
    inventory = _inventory(tmp_path, search_dirs=(tmp_path / "more",))
    import dataclasses

    from crystalx_native_tools.inventory import _core_host
    from crystalx_core.tools.resolver import ToolResolver

    run_config = RunConfig(toolbox_dir=tmp_path / "toolbox", extra_tool_dirs=(tmp_path / "more",), strict_native=True,
                           include_olex2=False, include_known_locations=False)
    # The configuration a run would use, with only the environment (PATH, host) pinned for the test.
    run_resolver_config = RealToolchain.for_config(run_config).resolver.config
    resolver = ToolResolver(dataclasses.replace(run_resolver_config, search_path=[]), host=_core_host(apple_silicon()))
    assert str(resolver.resolve(ToolKind.SHELXL).identity.path) == inventory.get("shelxl").selected.path == str(native)
    assert run_resolver_config.allow_translation is False and run_resolver_config.include_olex2 is False


def test_without_strict_mode_x86_64_is_marked_as_translation_dependent(tmp_path: Path):
    _all_native_toolbox(tmp_path)
    write_exe(tmp_path / "toolbox" / "shelxt", mb.thin(mb.CPU_X86_64))
    inventory = _inventory(tmp_path, strict=False, host=apple_silicon(markers=("/Library/Apple/usr/share/rosetta/rosetta",)))
    shelxt = inventory.get("shelxt")
    assert shelxt.static == "ok"
    assert any("Rosetta 2" in w and "not proof" in w for w in shelxt.selected.warnings)
    no_rosetta = _inventory(tmp_path, strict=False).get("shelxt")
    assert no_rosetta.static == "blocked" and no_rosetta.blockers[0].code == "rosetta_missing"


def test_a_platon_build_that_needs_missing_libraries_falls_back_to_platon(tmp_path: Path):
    """Review D5: the deep check steers selection, so a blocked special build yields to platon."""
    _all_native_toolbox(tmp_path)
    write_exe(tmp_path / "toolbox" / "platon_special",
              mb.thin(mb.CPU_ARM64, libraries=("/usr/lib/libSystem.B.dylib", X11), identifier="platon_special"))
    inventory = _inventory(tmp_path)
    special = inventory.get("platon_special")
    assert special.static == "blocked" and special.selected is None
    assert special.blockers[0].code == "library_unavailable" and "libX11" in special.blockers[0].message
    assert any("links X11 libraries" in n for n in special.resources.notes)
    assert inventory.validation_tool == "platon"
    assert inventory.pipeline == "ready_static"


def test_an_unsigned_arm64_build_is_blocked(tmp_path: Path):
    _all_native_toolbox(tmp_path)
    write_exe(tmp_path / "toolbox" / "shelxl", mb.thin(mb.CPU_ARM64, signed=False))
    shelxl = _inventory(tmp_path).get("shelxl")
    assert shelxl.static == "blocked" and shelxl.blockers[0].code == "unsigned_arm64"


def test_nothing_found_is_not_found(tmp_path: Path):
    inventory = _inventory(tmp_path)
    assert {t.static for t in inventory.tools} == {"not_found"}
    assert inventory.pipeline == "blocked"


def test_a_launch_probe_counts_only_for_the_exact_file(tmp_path: Path):
    _all_native_toolbox(tmp_path)
    store = ProbeStore(tmp_path / "probes.json")
    first = _inventory(tmp_path).get("platon_special").selected.inspection.sha256
    store.path.write_text(json.dumps([
        {"schema": PROBE_SCHEMA, "tool": "platon_special", "sha256": first, "outcome": "launched",
         "started": True, "probed_at": "2026-10-22T00:00:00+00:00", "message": "exit 0"},
    ]))
    inventory = build_inventory(_config(tmp_path), strict_native=True, host=apple_silicon(), search_path=[], probe_store=store)
    assert inventory.get("platon_special").launch["status"] == "launched"
    assert inventory.get("platon_special").validated == "not_verified"
    write_exe(tmp_path / "toolbox" / "platon_special", mb.thin(mb.CPU_ARM64, identifier="rebuilt"))
    again = build_inventory(_config(tmp_path), strict_native=True, host=apple_silicon(), search_path=[], probe_store=store)
    assert again.get("platon_special").launch == {"status": "not_tested"}


def test_platon_resources_and_version_heuristics(tmp_path: Path):
    _all_native_toolbox(tmp_path)
    platon = tmp_path / "toolbox" / "platon_special"
    platon.write_bytes(platon.read_bytes() + b"\0PLATON-Version 251106 (special)\0")
    (tmp_path / "toolbox" / "check.def").write_text("CHECK.DEF 251106\nVALIDATION TESTS\n")
    special = _inventory(tmp_path).get("platon_special")
    resources = special.resources
    assert resources.embedded_dates == ("251106",)
    found = [c for c in resources.checkdef if c.exists]
    assert [c.origin for c in found] == ["beside_executable"]
    assert found[0].version_date == "251106" and resources.version_match == "match"
    (tmp_path / "toolbox" / "check.def").write_text("CHECK.DEF 240101\n")
    assert _inventory(tmp_path).get("platon_special").resources.version_match == "mismatch"
    (tmp_path / "toolbox" / "check.def").unlink()
    none = _inventory(tmp_path).get("platon_special").resources
    assert none.version_match == "unknown"
    assert any("documented to create one" in n for n in none.notes)
    record = none.to_record()
    assert record["levels"]["can_start"].startswith("only the opt-in launch probe")


def test_date_heuristics_ignore_impossible_dates(tmp_path: Path):
    f = tmp_path / "check.def"
    f.write_text("header 991399 then 260230 and finally 250630\n")
    assert checkdef_version(f) == "250630"
    binary = tmp_path / "bin"
    binary.write_bytes(b"\0\0random 123456 text\0PLATON build 260229\0")
    assert embedded_platon_dates(binary) == ()  # 2026-02-29 does not exist


def test_source_line_numbers_are_not_mistaken_for_versions(tmp_path: Path):
    # Seen in the owner's platon_special: gfortran run-time messages carry source line numbers
    # that happen to look like yymmdd dates (130321 -> 2013-03-21).
    binary = tmp_path / "platon_special"
    binary.write_bytes(b"\0At line 130321 of file platon_special.f\0"
                       b"In file 'platon_special.f', around line 130321\0LINE 250101 platon\0")
    assert embedded_platon_dates(binary) == ()
    binary.write_bytes(binary.read_bytes() + b"\0PLATON-Version 251106\0")
    assert embedded_platon_dates(binary) == ("251106",)


def test_the_config_file_drives_the_inventory(tmp_path: Path):
    _all_native_toolbox(tmp_path)
    explicit = write_exe(tmp_path / "custom" / "shelxt-arm64", mb.thin(mb.CPU_ARM64))
    (tmp_path / "config.local.json").write_text(json.dumps({
        "tools": {"shelxt": "custom/shelxt-arm64", "nonsense": "x"},
        "toolbox": "toolbox",
        "strict_native": True,
        "include_olex2": False,
        "include_known_locations": False,
        "colour": "blue",
    }))
    config = load_config(tmp_path / "config.local.json")
    assert config.tools == {"shelxt": explicit}
    assert set(config.problems) == {"unknown key 'colour' ignored", "unknown tool 'nonsense' ignored"}
    inventory = build_inventory(config, host=apple_silicon(), search_path=[])
    assert inventory.strict_native is True
    assert inventory.get("shelxt").selected.origin == "explicit_config"
    assert any("unknown key" in n for n in inventory.notes)
