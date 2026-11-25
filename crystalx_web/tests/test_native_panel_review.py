"""Integration-review regressions for truthful static-tool readiness.

All candidates are local synthetic scripts. Nothing in this module starts a tool.
"""

from __future__ import annotations

import dataclasses
import subprocess
from pathlib import Path

import pytest

from crystalx_core.app.toolchain import RealToolchain
from crystalx_core.domain.tools import BinaryArch, HostPlatform
from crystalx_core.tools.resolver import ResolverConfig, ToolResolver
from crystalx_core.tools.runner import ProcessRunner
from crystalx_native_tools import inventory as inventory_module
from crystalx_native_tools.config import ToolsConfig
from crystalx_native_tools.host import HostFacts
from crystalx_native_tools.inspect import DependencyStatus
from crystalx_web.i18n import native_label, t
from crystalx_web.native_panel import native_status_md


def _record(*, blocked: bool) -> dict:
    return {
        "host": {"system": "Darwin", "hardware_arch": "arm64", "macos_version": "27.0"},
        "strict_native": True,
        "pipeline": "blocked" if blocked else "ready_static",
        "pipeline_blockers": ["platon_special/platon"] if blocked else [],
        "tools": [{
            "tool": "platon_special",
            "static": "blocked" if blocked else "ok",
            "launch": {"status": "not_tested"},
            "validated": "not_verified",
            "selected": {"path": "/synthetic/platon_special", "native": "native_arm64"},
            "blockers": ([{"code": "library_unavailable", "message": "synthetic missing library"}]
                         if blocked else []),
        }],
    }


@pytest.mark.parametrize(
    ("lang", "old_promise", "guidance"),
    [
        ("zh", "真实运行会在这些步骤前停止", "请先处理"),
        ("ja", "実行はこれらの段階の前で停止", "実行前に"),
        ("en", "A real run stops before these steps", "Resolve these blockers before a real run"),
    ],
)
def test_deep_blockers_do_not_promise_enforced_prelaunch_stop(lang, old_promise, guidance):
    text = native_status_md(_record(blocked=True), lang)
    assert "synthetic missing library" in text
    assert old_promise not in text
    assert guidance in text
    assert native_label("not_tested", lang) in text
    assert native_label("not_verified", lang) in text


@pytest.mark.parametrize("lang", ["zh", "ja", "en"])
def test_static_readiness_retains_unverified_launch_and_computation(lang):
    text = native_status_md(_record(blocked=False), lang)
    assert t("native_ready", lang) in text
    assert native_label("not_tested", lang) in text
    assert native_label("not_verified", lang) in text
    assert t("col_static_check", lang) in text
    assert t("col_launch", lang) in text
    assert t("col_validated", lang) in text


@pytest.mark.parametrize("lang", ["zh", "ja", "en"])
def test_configuration_problems_are_visible_and_escaped(lang):
    record = _record(blocked=True)
    record["config"] = {"problems": ["unknown option <unsafe>"]}
    text = native_status_md(record, lang)
    assert t("native_config_problems", lang) in text
    assert "unknown option &lt;unsafe&gt;" in text
    assert "unknown option <unsafe>" not in text


def _arm64_executable(libraries: tuple[str, ...] = ()) -> bytes:
    """A minimal signed-looking arm64 Mach-O header with LC_MAIN and the given LC_LOAD_DYLIBs."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "machobuild_for_web", Path(__file__).resolve().parents[2] / "crystalx_native_tools" / "tests" / "machobuild.py")
    mb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mb)
    return mb.thin(mb.CPU_ARM64, libraries=("/usr/lib/libSystem.B.dylib", *libraries))


def test_a_deep_blocked_platon_special_falls_back_identically_in_panel_and_run(tmp_path: Path, monkeypatch):
    """Review D5 (fixed): one injected check steers the panel and the run; the fallback is real."""
    from crystalx_native_tools.inspect import candidate_check_for

    box = tmp_path / "synthetic-tools"
    box.mkdir()
    for name in ("shelxt", "shelxl", "platon_special", "platon"):
        missing = (str(tmp_path / "absent" / "libX11.6.dylib"),) if name == "platon_special" else ()
        path = box / name
        path.write_bytes(_arm64_executable(missing))
        path.chmod(0o755)

    def refuse_launch(*args, **kwargs):
        raise AssertionError("Static inventory must not launch any process")

    monkeypatch.setattr(subprocess, "Popen", refuse_launch)
    monkeypatch.setattr(ProcessRunner, "run", refuse_launch)
    config = ToolsConfig(toolbox=box, include_olex2=False, include_known_locations=False)
    host = HostFacts(system="Darwin", hardware_arch="arm64", process_arch="arm64",
                     process_translated=False, macos_version="27.0", rosetta_markers=(), python="3.12")
    report = inventory_module.build_inventory(config, host=host, search_path=[], environ={})
    resolver = ToolResolver(
        ResolverConfig(toolbox_dir=box, search_path=[], include_olex2=False, include_known_locations=False,
                       allow_translation=False, candidate_check=candidate_check_for(host)),
        host=HostPlatform(system="Darwin", arch=BinaryArch.ARM64, rosetta_available=False),
        environ={},
    )
    actual = RealToolchain(resolver=resolver, allow_translation=False).platon()
    assert actual.kind.value == "platon" == report.validation_tool
    assert report.get("platon_special").static == "blocked"
    assert report.get("platon_special").blockers[0].code == "library_unavailable"
    assert report.get("platon").static == "ok"
    assert report.pipeline == "ready_static"
    text = native_status_md(report.to_record(), "en")
    assert t("native_ready", "en") in text and "library_unavailable" not in text.split("\n")[0]


def test_wrapper_scripts_cannot_make_the_page_ready_in_strict_mode(tmp_path: Path, monkeypatch):
    """Review D4: a script's target cannot be verified, so strict mode does not count it as native."""
    box = tmp_path / "synthetic-tools"
    box.mkdir()
    for name in ("shelxt", "shelxl", "platon_special", "platon"):
        path = box / name
        path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        path.chmod(0o755)
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no launch")))
    config = ToolsConfig(toolbox=box, include_olex2=False, include_known_locations=False)
    host = HostFacts(system="Darwin", hardware_arch="arm64", process_arch="arm64",
                     process_translated=False, macos_version="27.0", rosetta_markers=(), python="3.12")
    report = inventory_module.build_inventory(config, host=host, search_path=[], environ={})
    assert report.pipeline == "blocked"
    assert {s.blockers[0].code for s in report.tools} == {"native_unverified"}
    text = native_status_md(report.to_record(), "en")
    assert "**Blocked" in text and t("native_ready", "en") not in text
