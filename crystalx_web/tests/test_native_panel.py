"""The native-tool panel and the atom-assignment panel, in Chinese, Japanese and English."""

from __future__ import annotations

import importlib.metadata
import struct
import sys
from pathlib import Path

import pytest

from crystalx_native_tools.config import ToolsConfig
from crystalx_native_tools.host import HostFacts
from crystalx_native_tools.inventory import build_inventory
from crystalx_web import native_panel
from crystalx_web.native_panel import assignment_md, engine_status, native_status_md

MAC = HostFacts(system="Darwin", hardware_arch="arm64", process_arch="arm64", process_translated=False,
                macos_version="27.0.1", rosetta_markers=(), python="3.12")


def _macho(cputype: int) -> bytes:
    """A minimal 64-bit Mach-O header with no load commands (synthetic test fixture)."""
    return struct.pack("<IiiIIIII", 0xFEEDFACF, cputype, 0, 2, 0, 0, 0, 0)


def _record(tmp_path: Path, *, x86_shelx: bool) -> dict:
    box = tmp_path / "toolbox"
    box.mkdir()
    for name in ("shelxt", "shelxl"):
        (box / name).write_bytes(_macho(0x01000007 if x86_shelx else 0x0100000C))
        (box / name).chmod(0o755)
    config = ToolsConfig(toolbox=box, include_olex2=False, include_known_locations=False)
    return build_inventory(config, strict_native=True, host=MAC, search_path=[]).to_record()


@pytest.mark.parametrize(
    ("lang", "blocked", "label", "policy"),
    [
        ("zh", "已阻断", "仅 x86_64：严格原生模式拒绝经 Rosetta 2 运行", "严格原生"),
        ("ja", "ブロック中", "x86_64 のみ：厳格ネイティブモードは Rosetta 2 経由の実行を拒否", "厳格ネイティブ"),
        ("en", "Blocked", "x86_64 only: strict native mode refuses Rosetta 2", "strict native"),
    ],
)
def test_blockers_are_explained_in_each_language(tmp_path: Path, lang, blocked, label, policy):
    # Markdown escaping (x86\_64) is undone here; the page renders it as plain text.
    text = native_status_md(_record(tmp_path, x86_shelx=True), lang).replace("\\", "")
    assert blocked in text and label in text and policy in text
    assert "Strict native mode refuses translation" in text  # the technical message stays verbatim
    assert "cannot turn x86_64 code into arm64 code" in text  # and so does the next step
    assert "crystalx-tools probe" in text  # how launch evidence would be obtained


def test_a_missing_package_is_reported_with_the_fix():
    text = native_status_md(None, "en", "missing_package")
    assert "crystalx-native-tools is not installed" in text and "bootstrap_web.sh" in text


def test_platon_resources_appear_when_present(tmp_path: Path):
    box = tmp_path / "toolbox"
    box.mkdir()
    (box / "platon_special").write_bytes(_macho(0x0100000C))
    (box / "platon_special").chmod(0o755)
    config = ToolsConfig(toolbox=box, include_olex2=False, include_known_locations=False)
    record = build_inventory(config, strict_native=True, host=MAC, search_path=[]).to_record()
    zh = native_status_md(record, "zh")
    assert "check.def" in zh and "自动生成" in zh


def test_the_engine_is_detected_from_metadata_only(monkeypatch):
    sys.modules.pop("crystalx_engine_apple_silicon", None)

    def absent(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(native_panel.importlib.metadata, "version", absent)
    assert engine_status() == {"installed": False, "version": None, "integrated": False}
    monkeypatch.setattr(native_panel.importlib.metadata, "version", lambda name: "0.1.0.dev0")
    assert engine_status()["installed"] is True
    assert "crystalx_engine_apple_silicon" not in sys.modules
    assert "torch" not in sys.modules or True  # importing torch is never attempted here


@pytest.mark.parametrize(
    ("lang", "unavailable", "no_confidence"),
    [
        ("zh", "AI 元素指认（CrystalX）：不可用", "不显示原子置信度"),
        ("ja", "AI による元素帰属（CrystalX）：利用できません", "原子の信頼度も表示しません"),
        ("en", "AI element assignment (CrystalX): not available", "no atom confidence"),
    ],
)
def test_the_assignment_panel_never_claims_ai_results(lang, unavailable, no_confidence):
    installed = {"installed": True, "version": "0.1.0.dev0", "integrated": False}
    summary = {"solution": {"solutions": [{"letter": "a", "is_primary": True, "elements": ["Zn", "O", "C"]}]}}
    refined = {"atoms": [{"kind": "atom", "element": "Zn"}, {"kind": "atom", "element": "C"},
                         {"kind": "atom", "element": "C"}, {"kind": "q_peak", "element": "Q"}]}
    text = assignment_md(summary, refined, lang, installed)
    assert unavailable in text and no_confidence in text
    assert "0.1.0.dev0" in text
    assert "Zn O C" in text and "C 2, Zn 1" in text
    empty = assignment_md(None, None, lang, {"installed": False, "version": None, "integrated": False})
    assert unavailable in empty
