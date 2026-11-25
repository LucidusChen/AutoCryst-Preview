"""Rendering: escaping, unknown values, honest labels; the viewer's isolation and contents."""

from __future__ import annotations

import json
import re

import pytest

from crystalx_core.app.demo import write_demo_case
from crystalx_core.geometry.model import build_viewer_payload
from crystalx_core.io.shelx_res import read_res
from crystalx_web import render
from crystalx_web.viewer import EMPTY_VIEWER, viewer_document, viewer_html


def _payload(tmp_path, *, title="x", label="C1"):
    res = tmp_path / "m.res"
    res.write_text(
        f"TITL {title}\nCELL 0.71073 10 11 12 90 100 90\nZERR 2 0 0 0 0 0 0\nLATT -1\n"
        f"SFAC C O Xx\nUNIT 4 2 1\n{label} 1 0.1 0.2 0.3 11.0 0.05\nO1 2 0.15 0.25 0.3 11.0 0.04\n"
        "X1 3 0.5 0.5 0.5 11.0 0.05\nHKLF 4\nEND\nQ1 1 0.3 0.3 0.3 11.0 0.05 0.88\n",
        encoding="utf-8",
    )
    return build_viewer_payload(read_res(res), stage="refined", source_label="test")


def test_text_from_files_cannot_inject_markup():
    hostile = "<script>alert(1)</script> | **bold** `tick` [link](http://x)"
    escaped = render.esc(hostile)
    assert "<script>" not in escaped
    assert "\\|" in escaped and "\\*\\*" in escaped and "\\[" in escaped
    assert render.code("a`b<c>") == "`a'b&lt;c&gt;`"


def test_unparsed_numbers_are_shown_as_unknown_not_zero():
    summary = {
        "refinement": {
            "status": "ran_convergence_not_assessed",
            "settings": {"cycles": 4, "timeout_s": 900},
            "diagnostic": "ok",
            "duration_s": 1.0,
            "listing": {"r1_observed": None, "wr2": 0.0, "format_verified": False},
            "res_r_factors": None,
            "agreement": {},
            "changes": [],
            "reflections_note": "n",
            "preliminary_note": "p",
        }
    }
    text = render.refinement_md(summary)
    assert "| R1（Fo > 4σ） | 未知 | 未知 |" in text
    assert "| wR2 | 0 | 未知 |" in text  # a real 0.0 stays 0
    assert "尚未与本机真实 SHELXL 输出核对" in text


def test_the_facts_panel_keeps_the_four_facts_apart_and_marks_mock_runs():
    summary = {
        "mock": True,
        "process": {"phase": "completed", "message": "done"},
        "model_availability": "preliminary_refined_model",
        "refinement_status": "ran_convergence_not_assessed",
        "scientific_eligibility": {"reason": "gates not evaluated"},
        "warnings": [],
    }
    text = render.facts_md(summary)
    assert "模拟演示" in text and "不是真实结果" in text
    for label in ("过程完成情况", "可用模型", "精修状态", "科学资格"):
        assert label in text
    assert "未评估，不可排序或称为已验证" in text


def test_capabilities_say_static_inspection_is_not_a_launch_test(tmp_path):
    from crystalx_core.app import RunMode, RunService

    report = RunService(tmp_path / "runs").capabilities(RunMode.MOCK_DEMO)
    text = render.capabilities_md(report)
    assert "静态检查，不是启动测试" in text
    assert "仅规划（不执行）" in text and "未集成" in text and "未评估" in text


def test_platon_block_and_skip_are_explained():
    assert "被阻止" in render.platon_md({"platon": {"blocked": True, "reasons": ["no fcf"]}})
    assert "跳过" in render.platon_md({"platon": {"skipped": True, "reason": "no refinement"}})


def test_the_viewer_document_makes_no_network_request_and_cannot_be_broken_out_of(tmp_path):
    payload = _payload(tmp_path, title="</script><img src=x onerror=alert(1)>", label="C1")
    payload["source"] = "</script><script>alert(2)</script>"
    document = viewer_document(payload)
    assert "default-src 'none'" in document
    assert not re.search(r"https?://", document)
    assert "</script><script>alert(2)" not in document
    data = document.split('id="payload">', 1)[1].split("</script>", 1)[0]
    assert json.loads(data)["source"] == "</script><script>alert(2)</script>"


def test_the_viewer_iframe_is_sandboxed_without_same_origin(tmp_path):
    html = viewer_html(_payload(tmp_path))
    assert 'sandbox="allow-scripts"' in html
    assert "allow-same-origin" not in html
    assert viewer_html(None) == EMPTY_VIEWER


def test_the_viewer_payload_keeps_unknown_sites_and_q_peaks_distinct(tmp_path):
    payload = _payload(tmp_path)
    kinds = {a["label"]: a["kind"] for a in payload["atoms"]}
    assert kinds == {"C1": "atom", "O1": "atom", "X1": "unknown", "Q1": "q_peak"}
    rows = render.atom_rows(payload)
    assert [r[2] for r in rows] == ["原子", "原子", "未知位点", "Q 峰（差值电子密度）"]
    document = viewer_document(payload)
    assert "置信度：不可用" in document
    axes = json.loads(document.split('id="payload">', 1)[1].split("</script>", 1)[0])["axes"]
    assert axes["a"] == pytest.approx([10.0, 0.0, 0.0])


def test_the_demo_payload_lists_the_boundary_bond_it_does_not_draw(tmp_path):
    from crystalx_core.app.demo import demo_atoms

    assert any(a.label == "C5" for a in demo_atoms())
    ins, _ = write_demo_case(tmp_path)
    assert ins.exists()


@pytest.mark.parametrize("lang, words", [("zh", "过程完成情况"), ("ja", "処理の完了状況"), ("en", "Process")])
def test_every_panel_renders_in_each_language(lang, words):
    summary = {
        "process": {"phase": "completed", "message": "done"},
        "model_availability": "solution_model",
        "refinement_status": "not_run",
        "scientific_eligibility": {"reason": "r"},
    }
    assert words in render.facts_md(summary, lang)
    assert render.status_md(None, lang)
    assert render.platon_md({"platon": {"skipped": True, "reason": "x"}}, lang)


def test_no_interface_key_is_missing_in_any_language():
    from crystalx_web import i18n

    for key, row in i18n._T.items():
        assert len(row) == 3 and all(row), key
    for table in (i18n._PHASE, i18n._STAGE, i18n._MODEL, i18n._REFINEMENT, i18n._READINESS, i18n._ISSUE):
        assert all(len(v) == 3 and all(v) for v in table.values())
    from crystalx_web.viewer import VIEWER_TEXT

    assert VIEWER_TEXT["zh"].keys() == VIEWER_TEXT["ja"].keys() == VIEWER_TEXT["en"].keys()
