"""The entry point and the page: local-only launch settings, the check mode, the page builds."""

from __future__ import annotations

import json
import os
from pathlib import Path

import gradio as gr
import pytest

from crystalx_core.app import RunService
from crystalx_web import launcher
from crystalx_web.app import WebSettings, build_app, launch_options


def test_launch_options_are_local_only_and_quiet():
    options = launch_options()
    assert options["server_name"] == "127.0.0.1"
    assert options["share"] is False
    assert options["footer_links"] == []
    assert options["run_history"] is False
    assert options["enable_monitoring"] is False
    assert options["mcp_server"] is False
    assert options["ssr_mode"] is False


def test_the_theme_uses_local_fonts_only():
    theme = launch_options()["theme"]
    fonts = [f for f in theme._font + theme._font_mono]
    assert not any(isinstance(f, gr.themes.GoogleFont) for f in fonts)


def test_the_page_builds_with_telemetry_off(tmp_path: Path):
    service = RunService(tmp_path / "runs")
    app = build_app(service, WebSettings(state_dir=tmp_path / "state"))
    assert isinstance(app, gr.Blocks)
    assert app.analytics_enabled is False


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.10", "example.com", "::"])
def test_a_non_loopback_address_is_refused(host, capsys):
    assert launcher.main(["--host", host, "--check"]) == 2
    assert "loopback" in capsys.readouterr().err


def test_check_mode_verifies_and_starts_nothing(tmp_path: Path, capsys, monkeypatch):
    monkeypatch.setenv("GRADIO_TEMP_DIR", os.environ.get("GRADIO_TEMP_DIR", ""))
    # Confined: neither the repository's toolbox nor programs installed on this machine are read.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    (tmp_path / "config.local.json").write_text(
        json.dumps({"include_olex2": False, "include_known_locations": False, "strict_native": True})
    )
    assert launcher.main(["--check", "--runs-dir", str(tmp_path / "runs")]) == 0
    out = capsys.readouterr().out
    assert "crystalx-web ready" in out
    assert "No server was started" in out
    assert "strict native mode at start: on" in out
    assert "pipeline blocked" in out and "shelxt" in out
    assert (tmp_path / "runs" / "_gradio_tmp").is_dir()


def test_a_run_uses_the_locations_and_policy_the_panel_inspects(tmp_path: Path):
    from crystalx_core.app import RunConfig
    from crystalx_core.app.toolchain import RealToolchain
    from crystalx_native_tools.config import ToolsConfig
    from crystalx_web.app import WebSettings, run_tool_kwargs

    shelxt = tmp_path / "bin" / "shelxt"
    tools = ToolsConfig(toolbox=tmp_path / "tb", tools={"shelxt": shelxt}, search_dirs=(tmp_path / "more",),
                        include_olex2=False, include_known_locations=False)
    config = RunConfig(**run_tool_kwargs(WebSettings(state_dir=tmp_path / "s", tools_config=tools), strict=True))
    resolver = RealToolchain.for_config(config).resolver.config
    assert resolver.toolbox_dir == tmp_path / "tb" and resolver.explicit_paths == {"shelxt": shelxt}
    assert resolver.extra_dirs == (tmp_path / "more",) and resolver.allow_translation is False
    assert resolver.include_olex2 is False and resolver.include_known_locations is False
    loose = RunConfig(**run_tool_kwargs(WebSettings(state_dir=tmp_path / "s"), strict=False))
    assert loose.include_olex2 and loose.include_known_locations and not loose.strict_native


def test_the_environment_is_prepared_inside_the_runs_directory(tmp_path: Path, monkeypatch):
    for name in ("GRADIO_ANALYTICS_ENABLED", "GRADIO_TEMP_DIR", "HF_HUB_DISABLE_TELEMETRY"):
        monkeypatch.setenv(name, os.environ.get(name, ""))
    state = launcher.prepare_environment(tmp_path / "runs")
    assert os.environ["GRADIO_ANALYTICS_ENABLED"] == "False"
    assert Path(os.environ["GRADIO_TEMP_DIR"]) == (tmp_path / "runs" / "_gradio_tmp").resolve()
    assert state == (tmp_path / "runs" / "_web").resolve()


def test_a_relative_toolbox_is_made_absolute_at_launch(tmp_path: Path, capsys, monkeypatch):
    """Codex review R1: programs run inside run directories, so the toolbox must be absolute."""
    monkeypatch.setenv("GRADIO_TEMP_DIR", os.environ.get("GRADIO_TEMP_DIR", ""))
    monkeypatch.chdir(tmp_path)
    (tmp_path / "tb").mkdir()
    assert launcher.main(["--check", "--runs-dir", "runs", "--toolbox", "tb"]) == 0
    assert f"toolbox: {tmp_path / 'tb'}" in capsys.readouterr().out
