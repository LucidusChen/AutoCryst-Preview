"""Integration regressions use synthetic headers only; no crystallographic tools are run."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from crystalx_core.app.toolchain import RealToolchain
from crystalx_core.domain.tools import BinaryArch, HostPlatform
from crystalx_core.tools.resolver import ResolverConfig, ToolResolver
from crystalx_native_tools.config import ToolsConfig
from crystalx_native_tools.host import HostFacts
from crystalx_native_tools.inspect import candidate_check_for, resolve_dependency
from crystalx_native_tools.inventory import build_inventory

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "native_review_machobuild", REPO / "crystalx_native_tools/tests/machobuild.py"
)
mb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mb)


@pytest.mark.parametrize("name", ["@rpath/optional.dylib", "optional.dylib"])
def test_unresolved_weak_import_does_not_block_launch(tmp_path: Path, name: str):
    dependency = resolve_dependency(
        name,
        "weak",
        executable_dir=tmp_path,
        rpaths=(),
        arch="arm64",
        on_macos=True,
    )
    assert not dependency.blocks_launch, dependency.to_record()
    assert dependency.status == "missing_weak"


def test_unresolved_required_import_still_blocks_launch(tmp_path: Path):
    dependency = resolve_dependency(
        "@rpath/required.dylib",
        "load",
        executable_dir=tmp_path,
        rpaths=(),
        arch="arm64",
        on_macos=True,
    )
    assert dependency.blocks_launch and dependency.status == "unresolved"


def test_inventory_reports_actual_platon_choice_when_deep_check_blocks_it(tmp_path: Path):
    toolbox = tmp_path / "toolbox"
    toolbox.mkdir()
    for name in ("platon_special", "platon"):
        libraries = (str(tmp_path / "absent" / "libX11.dylib"),) if name == "platon_special" else ()
        path = toolbox / name
        path.write_bytes(mb.thin(mb.CPU_ARM64, libraries=libraries))
        path.chmod(0o755)
    host = HostFacts("Darwin", "arm64", "arm64", False, "27.0.1", (), "3.12")
    config = ToolsConfig(toolbox=toolbox, include_known_locations=False, include_olex2=False)
    inventory = build_inventory(
        config,
        tools=("platon_special", "platon"),
        host=host,
        search_path=[],
    )
    def runtime(candidate_check):
        resolver = ToolResolver(
            ResolverConfig(
                toolbox_dir=toolbox,
                search_path=[],
                include_known_locations=False,
                include_olex2=False,
                allow_translation=False,
                candidate_check=candidate_check,
            ),
            host=HostPlatform("Darwin", BinaryArch.ARM64, False),
        )
        return RealToolchain(resolver=resolver).platon()

    # Review D5, fixed: a run given the same injected check as the panel selects the same file.
    actual = runtime(candidate_check_for(host))
    assert actual is not None and actual.identity is not None
    assert inventory.get("platon_special").static == "blocked"
    assert inventory.get("platon").static == "ok"
    assert inventory.validation_tool == actual.kind.value == "platon"
    assert inventory.pipeline == "ready_static" and inventory.pipeline_blockers == ()
    # Without the injected check the core alone cannot see the missing library.
    assert runtime(None).kind.value == "platon_special"
