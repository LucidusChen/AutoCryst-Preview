"""The iterative-reconstruction tab: request mapping, honest presentation and pinned-seal access.

Synthetic providers only (generated inputs, stand-in programs and stand-in inference). The stand-ins
are supported backend providers restricted to the exact built-in generated inputs; no real
program, measurement or weight is touched.
"""

from __future__ import annotations

import dataclasses
import json
import re
import subprocess
import time
from pathlib import Path

import gradio as gr
import pytest

from crystalx_core.app.demo import is_demo_input, write_demo_case
from crystalx_core.app.reconstruction_synthetic import toolchain_factory, assigner_factory
from crystalx_core.app.reconstruction_backend import (
    FRONTEND_CONTRACT, ReconstructionBackendService, ReconstructionRequest,
)
from crystalx_core.app.reconstruction_stops import STOP_CODES
from crystalx_core.refinement.adaptive import AdaptivePolicy
from crystalx_core.workflow.refinement_cycle import CyclePolicy
from crystalx_web import reconstruction as R
from crystalx_web.reconstruction_tab import build_reconstruction_tab
from crystalx_web.reconstruction_text import (
    choice_label, decision_label, field_label, history_columns, rt, stop_codes_without_text, stop_explanation,
)


@pytest.fixture
def repo(tmp_path: Path):
    root = tmp_path.resolve() / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    (root / ".gitignore").write_text("/runs/\n")
    ins, hkl = write_demo_case(root / "runs" / "inputs")
    return root, ins, hkl


def _defaults(paths: list[str], **values) -> dict:
    """Form values as the page starts them, then the given overrides."""
    form = {}
    for spec in [s for group in R.cycle_groups().values() for s in group]:
        form[spec.path] = ("" if spec.default is None else spec.default) if spec.kind.startswith("optional_") else (
            ", ".join(spec.default) if spec.kind == "ids" else spec.default)
    form.update({"route.strategy": "shelxt", "route.trials": 100, "route.timeout_s": 600, "metal_only": False,
                 "anchor_elements": "", "candidate_elements": "", "candidate_element_source": "original_ins",
                 "device": "cpu", "strict_native": True, "label": "", "mode": "mock_demo"})
    form.update(values)
    return {p: form.get(p) for p in paths} if paths else form


# ---------------------------------------------------------------------------- contract
def test_every_backend_policy_field_is_in_the_form_once_and_described_in_three_languages():
    groups = R.cycle_groups()
    shown = [s.path for group in groups.values() for s in group]
    expected = ([f"cycle.{f.name}" for f in dataclasses.fields(CyclePolicy) if f.name != "adaptive"]
                + [f"cycle.adaptive.{f.name}" for f in dataclasses.fields(AdaptivePolicy)])
    assert sorted(shown) == sorted(expected) and len(shown) == len(set(shown))
    assert groups["other"] == []  # a new backend field lands here until it is placed and described
    assert not [s.path for group in groups.values() for s in group if s.kind == "unsupported"]
    for lang in ("zh", "ja", "en"):
        assert R.field_labels_missing(lang) == []


def test_choice_values_are_exactly_what_the_backend_accepts(repo):
    _, ins, hkl = repo
    for path, values in R.CHOICES.items():
        if not path.startswith("cycle."):  # request-level choice (e.g. heavy_preparation)
            for value in values:
                R.build_request(_defaults([], ins_path=str(ins), hkl_path=str(hkl), **{path: value}))
            with pytest.raises(R.FormError):
                R.build_request(_defaults([], ins_path=str(ins), hkl_path=str(hkl), **{path: "not-a-choice"}))
            continue
        name = path.rsplit(".", 1)[-1]
        build = (lambda v: AdaptivePolicy(**{name: v})) if ".adaptive." in path else (lambda v: CyclePolicy(**{name: v}))
        for value in values:
            build(value)
        with pytest.raises(ValueError):
            build("not-a-choice")


def test_form_defaults_reproduce_the_backend_defaults(repo):
    _, ins, hkl = repo
    request = R.build_request(_defaults([], ins_path=str(ins), hkl_path=str(hkl)))
    assert request.cycle == CyclePolicy() and request.cycle.iterations is None
    assert request.route.strategy.value == "shelxt" and request.metal_only is False
    assert request.anchor_elements == () and request.candidate_elements == ()
    assert request.candidate_element_source == "original_ins" and request.device == "cpu"


def test_rounds_cycles_and_tolerances_stay_distinct():
    assert R.parse_outer_rounds("") is None and R.parse_outer_rounds(None) is None
    assert R.parse_outer_rounds(" 3 ") == 3 and R.parse_outer_rounds(4.0) == 4
    for bad in ("2.5", "0", "-1", "x", 2.5, 0, True):
        with pytest.raises(ValueError):
            R.parse_outer_rounds(bad)
    record = R.request_record(_defaults([], ins_path="a.ins", hkl_path="a.hkl", **{
        "cycle.iterations": "5", "cycle.shelxl_cycles": 7, "cycle.r1_tolerance": 0.002,
        "cycle.coordinate_tolerance_angstrom": 0.03}))
    assert record["cycle"]["iterations"] == 5 and record["cycle"]["shelxl_cycles"] == 7
    assert record["cycle"]["r1_tolerance"] == 0.002  # a fraction, never converted to percent
    assert record["cycle"]["coordinate_tolerance_angstrom"] == 0.03


def test_an_optional_number_left_blank_stays_unset(tmp_path: Path):
    ins, hkl = write_demo_case(tmp_path)
    base = _defaults([], ins_path=str(ins), hkl_path=str(hkl))
    assert R.build_request(base).cycle.damp_factor is None
    assert R.build_request({**base, "cycle.damp_factor": " 0.7 "}).cycle.damp_factor == 0.7
    with pytest.raises(R.FormError, match="cycle.damp_factor"):
        R.build_request({**base, "cycle.damp_factor": "strong"})


def test_route_forms_serialise_to_the_backend_types(tmp_path: Path):
    ins, hkl = write_demo_case(tmp_path)
    base = _defaults([], ins_path=str(ins), hkl_path=str(hkl))
    shelxs = R.build_request({**base, "route.strategy": "small_molecule", "route.small_solver": "shelxs",
                              "route.small_method": "patterson", "route.trials": 40})
    assert shelxs.route.small_solver == "shelxs" and shelxs.route.small_method == "patterson" and shelxs.route.trials == 40
    shelxd = R.build_request({**base, "route.strategy": "small_molecule", "route.small_solver": "shelxd",
                              "route.find_sites": 8, "route.plop": "10, 12, 14"})
    assert shelxd.route.find_sites == 8 and shelxd.route.plop == (10, 12, 14)
    sad = tmp_path / "sad.hkl"
    sad.write_text("   1   0   0   10.0   1.0\n")
    protein = R.build_request({**base, "route.strategy": "protein_cde", "protein.protein_confirmed": True,
                               "protein.experiment": "SAD", "protein.space_group": "P212121",
                               "protein.find_sites": 4, "protein.heavy_element": "se", "protein.solvent_fraction": 0.5,
                               "protein.dataset.0.role": "SAD", "protein.dataset.0.path": str(sad),
                               "protein.dataset.0.format": "SHELX_INTENSITIES", "protein.dataset.0.friedel": True})
    config = protein.route.protein
    assert config.protein_confirmed and config.heavy_element == "Se" and config.solvent_fraction == 0.5
    assert len(config.datasets) == 1 and config.datasets[0].role == "SAD" and config.datasets[0].friedel_pairs_retained
    with pytest.raises(R.FormError, match="choose both a role and a file"):
        R.build_request({**base, "route.strategy": "protein_cde", "protein.dataset.1.role": "PEAK"})


def test_metal_anchoring_is_independent_of_the_route_and_off_by_default(tmp_path: Path):
    ins, hkl = write_demo_case(tmp_path)
    base = _defaults([], ins_path=str(ins), hkl_path=str(hkl))
    assert R.build_request(base).metal_only is False
    request = R.build_request({**base, "route.strategy": "small_molecule", "metal_only": True, "anchor_elements": "zn, CU"})
    assert request.metal_only is True and request.anchor_elements == ("Zn", "Cu")
    assert request.route.strategy.value == "small_molecule"


def test_no_expert_or_reference_field_can_reach_a_solve_request(tmp_path: Path):
    ins, hkl = write_demo_case(tmp_path)
    record = R.request_record(_defaults([], ins_path=str(ins), hkl_path=str(hkl), expert_path="/answer.res",
                                        reference="x"))
    allowed = {f.name for f in dataclasses.fields(ReconstructionRequest)}

    def keys(value):
        if isinstance(value, dict):
            for key, item in value.items():
                yield key
                yield from keys(item)
        elif isinstance(value, list):
            for item in value:
                yield from keys(item)

    assert set(record) <= allowed and "reference" not in record and "expert_path" not in record
    assert not [k for k in keys(record) if re.search(r"expert|answer|mentor|reference_(path|model|res)", k)]
    with pytest.raises(ValueError):
        ReconstructionRequest.from_record({**record, "expert_path": "/answer.res"})


def test_backend_validation_messages_are_passed_on_verbatim(tmp_path: Path):
    ins, hkl = write_demo_case(tmp_path)
    base = _defaults([], ins_path=str(ins), hkl_path=str(hkl))
    with pytest.raises(R.FormError) as caught:
        R.build_request({**base, "cycle.max_rounds": 0})
    assert "cycle integer budgets must be positive" in caught.value.messages[0]
    with pytest.raises(R.FormError) as caught:
        R.build_request({**base, "cycle.iterations": "2.5", "ins_path": None})
    assert any("ins_path" in m for m in caught.value.messages) and any("cycle.iterations" in m for m in caught.value.messages)


# ---------------------------------------------------------------------------- presentation
def test_a_missing_delta_is_unavailable_never_zero():
    assert R.format_iteration_delta(None, lang="en") == "— (no previous round)"
    assert R.format_iteration_delta(0.0) == "0.0000" and R.format_iteration_delta(0.0123, digits=3, unit=" Å") == "0.012 Å"
    rows = R.history_rows([{"round": 1, "r1": 0.3, "r1_change": None, "important_coordinate_max_change_angstrom": None,
                            "accepted_new_sites_if_retained": 0, "deleted_site_ids_if_retained": [], "element_changes": 0,
                            "stable_transition": False, "decision": "RETAIN_UNVALIDATED", "inference_protocol": "current_labels"}],
                          "en")
    assert rows[0][1] == "0.3000" and rows[0][2].startswith("—") and rows[0][3].startswith("—")


def test_stop_codes_are_verbatim_with_a_meaning_and_never_a_scientific_claim():
    for code in ("fixed_rounds_completed", "important_coordinates_and_r1_stable", "round_safety_budget_exhausted",
                 "failure_budget_exhausted", "whole_chain_wall_budget_exhausted", "cancelled", "tool_unavailable:shelxl",
                 "protein_space_group_missing", "pretrained_checkpoint_missing"):
        assert stop_explanation(code, "en"), code
    assert stop_explanation("some_new_code_nobody_described", "en") == ""
    text = R.summary_md({"phase": "completed", "stop_code": "important_coordinates_and_r1_stable",
                         "candidate_available": True, "seal_sha256": "a" * 64, "candidate_sha256": "b" * 64}, "en")
    assert "`important_coordinates_and_r1_stable`" in text and "numerically stable candidate" in text
    assert "not scientific convergence" in text and "not scientifically validated" in text
    unfinished = R.summary_md({"phase": "failed", "stop_code": "round_safety_budget_exhausted",
                               "candidate_available": True, "seal_sha256": "a" * 64, "candidate_sha256": "b" * 64}, "en")
    assert "not a converged result" in unfinished
    for summary in (text, unfinished):
        assert not re.search(r"(?<!not )(?<!un)(?<!scientifically )validated structure|\bPASS\b", summary)


def test_a_missing_checkcif_report_never_shows_zero_alerts():
    from crystalx_core.app.checkcif import CheckCifResult, CheckCifStatus

    result = CheckCifResult(CheckCifStatus.NOT_EVALUATED, ("explicit_checkdef_or_accepted_builtin_resource_policy_required",),
                            True, "a" * 64)
    text = R.present_local_alerts(result, "en")
    assert "`NOT_EVALUATED`" in text and "not \"zero alerts\"" in text and "No report was read" in text
    assert "Alert counts" not in text and "The candidate remains available." in text


# ---------------------------------------------------------------------------- end to end (synthetic)
def _tab(repo, tmp_path: Path):
    root, ins, hkl = repo
    service = ReconstructionBackendService(root, root / "runs" / "reconstruction",
                                           toolchain_factory=toolchain_factory, assigner_factory=assigner_factory)
    with gr.Blocks():
        tab = build_reconstruction_tab(lang0="en", service_factory=None, demo_service_factory=lambda: service,
                                       state_dir=root / "runs" / "_web",
                                       tool_kwargs=lambda strict: {"tool_paths": ()}, strict_default=True)
    return tab, service, ins, hkl


def _values(tab, **overrides):
    return [overrides.get(p, v) for p, v in _defaults(tab.paths).items()]


def _wait(tab, run_id, *, seconds=90):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        timer, status, rows, summary = tab.handlers["tick"](run_id, "en")
        if timer["active"] is False:
            return status, rows, summary
        time.sleep(0.2)
    raise AssertionError("the synthetic run did not finish")


def test_preview_submit_poll_summary_export_and_separate_checkcif(repo, tmp_path: Path):
    tab, service, ins, hkl = _tab(repo, tmp_path)
    values = _values(tab, ins_path=str(ins), hkl_path=str(hkl), **{"cycle.iterations": "2"})
    errors, preview, record = tab.handlers["preview"]("en", *values)
    assert errors == "" and record["expert_reference_used"] is False and record["scientifically_validated"] is False
    assert "Static check passed" in preview and "not a launch test" in preview
    errors, run_id, mode, status, rows, summary, timer = tab.handlers["start"]("en", *values)
    assert errors == "" and re.fullmatch(r"r-\d{8}T\d{6}Z-[a-f0-9]{8}", run_id) and mode == "mock_demo"
    status, rows, summary = _wait(tab, run_id)
    assert "`fixed_rounds_completed`" in status and "not a convergence judgement" in summary
    assert status.startswith("🧪 **Synthetic demo (MOCK)**") and "🧪 **Synthetic demo (MOCK)**" in summary
    assert "fixed rounds completed" in summary and "not scientifically validated" in summary
    history = rows["value"]
    assert len(history) == 2 and history[0][2].startswith("—")  # no previous round for the first delta
    request = json.loads(next((service.runs_root).glob(f"{run_id}*/request.json")).read_text())
    assert not [key for key in request if re.search(r"expert|answer|mentor|reference", key)]

    files, note, viewer = tab.handlers["export"](run_id, "en")
    names = sorted(Path(f).name for f in files)
    assert note == "" and names == ["ai_solved.cif", "ai_solved.fcf", "ai_solved.res"]
    assert all(Path(f).is_relative_to(service.runs_root.parent / "_web" / "downloads") for f in files)
    assert "<iframe" in viewer or "srcdoc" in viewer

    text, result = tab.handlers["checkcif"](run_id, "en", True, None, 30, True)
    assert result.status.value == "NOT_EVALUATED" and result.candidate_available
    assert "simulated_toolchain_cannot_establish_local_platon_validation" in text and "Alert counts" not in text
    assert tab.handlers["cancel"](run_id, "en").endswith("_This run is not an active job of this page, so it cannot be "
                                                         "cancelled (it may have ended or belong to another process)._")
    runs = tab.handlers["runs"]("en")["value"]
    assert runs and runs[0][0] == run_id


def test_blank_rounds_use_the_automatic_numerical_stop(repo, tmp_path: Path):
    tab, service, ins, hkl = _tab(repo, tmp_path)
    values = _values(tab, ins_path=str(ins), hkl_path=str(hkl), **{"cycle.max_rounds": 3})
    _, run_id, *_ = tab.handlers["start"]("en", *values)
    status, rows, summary = _wait(tab, run_id)
    request = json.loads(next(service.runs_root.glob(f"{run_id}*/request.json")).read_text())
    assert request["cycle"]["iterations"] is None and request["cycle"]["max_rounds"] == 3
    code = service.status(run_id)["stop_code"]
    assert f"`{code}`" in status and code != "fixed_rounds_completed"


def test_candidate_files_are_read_only_through_the_external_pin(repo, tmp_path: Path):
    tab, service, ins, hkl = _tab(repo, tmp_path)
    _, run_id, *_ = tab.handlers["start"]("en", *_values(tab, ins_path=str(ins), hkl_path=str(hkl),
                                                         **{"cycle.iterations": "1"}))
    _wait(tab, run_id)
    pin = service.runs_root / f"{run_id}.seal-pin.json"
    original = pin.read_text()
    record = json.loads(original)

    pin.write_text(json.dumps({**record, "seal_sha256": "0" * 64}))
    files, note, _ = tab.handlers["export"](run_id, "en")
    assert files is None and "could not be read" in note
    pin.write_text(json.dumps({**record, "run_root": str(tmp_path)}))
    assert tab.handlers["export"](run_id, "en")[0] is None
    pin.write_text(json.dumps({**record, "extra": 1}))
    assert tab.handlers["export"](run_id, "en")[0] is None
    pin.unlink()
    pin.symlink_to(tmp_path / "elsewhere.json")
    (tmp_path / "elsewhere.json").write_text(original)
    assert tab.handlers["export"](run_id, "en")[0] is None
    pin.unlink()
    pin.write_text(original)
    assert tab.handlers["export"](run_id, "en")[0] is not None

    sealed = Path(record["run_root"]) / "outputs" / "ai_solved.res"
    sealed.write_text(sealed.read_text() + "REM changed after sealing\n")
    files, note, _ = tab.handlers["export"](run_id, "en")
    assert files is None and "SealError" in note
    for bad in ("../x", "r-2026", "r-20261112T000000Z-zzzzzzzz"):
        assert tab.handlers["export"](bad, "en")[0] is None


def test_nonpositive_u_warnings_come_from_the_run_records(repo, tmp_path: Path):
    tab, service, ins, hkl = _tab(repo, tmp_path)
    _, run_id, *_ = tab.handlers["start"]("en", *_values(tab, ins_path=str(ins), hkl_path=str(hkl),
                                                         **{"cycle.iterations": "1"}))
    deadline = time.monotonic() + 90
    while service.status(run_id)["phase"] not in R.TERMINAL:  # not through the tab: it caches terminal runs
        assert time.monotonic() < deadline
        time.sleep(0.2)
    run_root = service.run_root(run_id)
    # Simulate a historical run without the additive history field.
    state_path = run_root / "state.json"
    state = json.loads(state_path.read_text())
    for row in state["history"]:
        row.pop("nonpositive_u_site_ids", None)
    state_path.write_text(json.dumps(state))
    record = run_root / "cycles" / "round-001" / "inference.json"
    data = json.loads(record.read_text())
    data["provenance"]["nonpositive_u_sites_preserved_for_geometry_only_inference"] = ["Zn1", "O2"]
    record.write_text(json.dumps(data))
    assert R.nonpositive_u_warnings(service, run_id) == {1: ["Zn1", "O2"]}
    assert "`Zn1`" in tab.handlers["load"](run_id, "en")[4]  # the summary shown for a loaded run
    text = R.summary_md(service.summary(run_id), "en", warnings=R.nonpositive_u_warnings(service, run_id))
    assert "Uiso ≤ 0" in text and "`Zn1`" in text and "not scientifically approved" in text
    record.unlink()
    record.symlink_to(tmp_path / "planted.json")
    (tmp_path / "planted.json").write_text(json.dumps(data))
    assert R.nonpositive_u_warnings(service, run_id) == {}


def test_route_specific_controls_follow_the_selected_route(tmp_path: Path):
    with gr.Blocks():
        tab = build_reconstruction_tab(lang0="en", service_factory=None, state_dir=tmp_path,
                                       tool_kwargs=lambda strict: {"tool_paths": ()}, strict_default=True)
    small, protein, trials = tab.handlers["strategy"]("shelxt")
    assert (small["visible"], protein["visible"], trials["visible"]) == (False, False, False)
    small, protein, trials = tab.handlers["strategy"]("small_molecule")
    assert (small["visible"], protein["visible"], trials["visible"]) == (True, False, True)
    small, protein, trials = tab.handlers["strategy"]("protein_cde")
    assert (small["visible"], protein["visible"], trials["visible"]) == (False, True, True)


def test_without_a_checkout_the_tab_explains_and_runs_nothing(tmp_path: Path):
    with gr.Blocks():
        tab = build_reconstruction_tab(lang0="en", service_factory=None, state_dir=tmp_path,
                                       tool_kwargs=lambda strict: {"tool_paths": ()}, strict_default=True)
    errors, preview, record = tab.handlers["preview"]("en", *[None] * len(tab.paths))
    assert record is None and errors.startswith("⛔")
    assert tab.handlers["runs"]("en")["value"] == []
    assert tab.handlers["export"]("", "en")[1] == "Run or load a reconstruction first."


def test_the_page_offers_both_modes_and_relabels_the_new_tab(tmp_path: Path):
    from crystalx_core.app import RunService
    from crystalx_web.app import WebSettings, build_app

    app = build_app(RunService(tmp_path / "runs"), WebSettings(state_dir=tmp_path / "state"))
    config = app.get_config_file()
    labels = {c.get("props", {}).get("label") for c in config["components"]}
    assert "One-shot solve" not in labels  # default language is Chinese
    assert {"单次求解", "迭代重构（预训练指认 + SHELXL 循环）"} <= labels
    assert field_label("cycle.r1_tolerance", "en").endswith("(fraction, not %)")


def test_every_backend_stop_code_is_explained_in_three_languages():
    """A code added to the backend registry fails here until the page can explain it (not just in English)."""
    assert stop_codes_without_text() == []
    for code, info in STOP_CODES.items():
        texts = {lang: stop_explanation(code, lang) for lang in ("zh", "ja", "en")}
        assert all(texts.values()), code
        assert texts["zh"] != info.description and texts["ja"] != info.description, code
        assert len(set(texts.values())) == 3, code


def test_the_page_was_written_against_the_backend_contract_in_use(monkeypatch):
    assert FRONTEND_CONTRACT == R.EXPECTED_CONTRACT and R.contract_note("en") == ""
    monkeypatch.setattr(R, "FRONTEND_CONTRACT", "2099-01-01")
    note = R.contract_note("en")
    assert "`2099-01-01`" in note and f"`{R.EXPECTED_CONTRACT}`" in note


def test_the_backend_stop_detail_is_shown_verbatim_bounded_and_explained():
    state = {"run_id": "r-x", "phase": "blocked", "stop_code": "reconstruction_blocked", "stop_category": "blocked",
             "stop_detail": "at_least_two_declared_nonhydrogen_candidate_elements_required"}
    text = R.status_md(state, "en")
    assert "`reconstruction_blocked`" in text and "Detail (backend text):" in text
    assert "`at_least_two_declared_nonhydrogen_candidate_elements_required`" in text
    assert "at least two declared non-hydrogen candidate elements are required" in text
    summary = R.summary_md({**state, "candidate_available": False}, "zh")
    assert "详情（后端原文）" in summary and "至少需要两个声明的非氢候选元素" in summary
    hostile = R.status_md({**state, "stop_detail": "<script>alert(1)</script> `x` " + "y" * 2000}, "en")
    assert "<script>" not in hostile and "&lt;script&gt;" in hostile and " …" in hostile
    assert "`x`" not in hostile and len(hostile) < 1500
    assert "Detail" not in R.status_md({**state, "stop_detail": None}, "en")


def test_the_candidate_is_named_after_the_backend_stop_category():
    base = {"candidate_available": True, "seal_sha256": "a" * 64, "candidate_sha256": "b" * 64}
    stable = R.summary_md({**base, "phase": "completed", "stop_code": "movable_sites_r1_and_growth_stable",
                           "stop_category": "numerical_stop"}, "en")
    assert "numerically stable candidate" in stable and "not scientific convergence" in stable
    blocked = R.summary_md({**base, "phase": "blocked", "stop_code": "no_acceptable_step_in_current_action_set",
                            "stop_category": "blocked"}, "en")
    assert "candidate kept without meeting a stop condition" in blocked and "not a converged result" in blocked
    fixed = R.summary_md({**base, "phase": "completed", "stop_code": "fixed_rounds_completed",
                          "stop_category": "fixed_rounds"}, "en")
    assert "fixed rounds completed" in fixed and "numerically stable" not in fixed


def test_every_choice_has_a_label_and_the_round_table_matches_its_headers():
    for path, values in {**R.CHOICES, "mode": R.MODES}.items():
        for value in values:
            for lang in ("zh", "ja", "en"):
                assert choice_label(path, value, lang) != value, (path, value, lang)
    row = R.history_rows([{"round": 3, "r1": 0.2, "backoff_level": 2, "refinement_method": "cgls",
                           "refinement_max_displacement_angstrom": 8.4914,
                           "refinement_displacement_above_threshold_count": 55}, {"round": 4, "r1": 0.2}], "en")
    assert row[0][-7:-3] == ["2", "cgls", "8.49", "55"] and row[1][-7:-3] == ["—", "—", "—", "—"]
    for lang in ("zh", "ja", "en"):
        assert len(history_columns(lang)) == len(row[0])


class _Recording:
    """Stands in front of a service and notes which methods the tab called on it."""

    def __init__(self, inner):
        self.inner, self.calls = inner, []

    def __getattr__(self, name):
        value = getattr(self.inner, name)
        if not callable(value):
            return value

        def call(*args, **kwargs):
            self.calls.append(name)
            return value(*args, **kwargs)
        return call


def test_real_and_synthetic_modes_keep_separate_services_and_mark_stand_in_results(repo, tmp_path: Path):
    root, ins, hkl = repo
    # Both stand on the backend's synthetic providers, so nothing on this machine is inspected or run;
    # those providers refuse a real-mode request, which is what the "real" side must receive.
    real = _Recording(ReconstructionBackendService(root, root / "runs" / "reconstruction",
                                                    toolchain_factory=toolchain_factory, assigner_factory=assigner_factory))
    demo = _Recording(ReconstructionBackendService(root, root / "runs" / "reconstruction_demo",
                                                    toolchain_factory=toolchain_factory, assigner_factory=assigner_factory))
    with gr.Blocks():
        tab = build_reconstruction_tab(lang0="en", service_factory=lambda: real, demo_service_factory=lambda: demo,
                                       state_dir=root / "runs" / "_web",
                                       tool_kwargs=lambda strict: {"tool_paths": (("shelxl", "/opt/x/shelxl"),)},
                                       strict_default=True)
    banner, button, _ = tab.handlers["mode"]("real", "en")
    assert banner["visible"] is False and banner["value"] == "" and button["visible"] is False
    banner, button, _ = tab.handlers["mode"]("mock_demo", "en")
    assert banner["visible"] and "MOCK" in banner["value"] and "not a real result" in banner["value"] and button["visible"]
    demo_ins, demo_hkl = (Path(p) for p in tab.handlers["demo"]())
    assert is_demo_input(demo_ins, demo_hkl) and demo_ins.is_relative_to(root / "runs" / "_web")

    values = _values(tab, ins_path=str(demo_ins), hkl_path=str(demo_hkl), mode="real", **{"cycle.iterations": "1"})
    errors, _, record = tab.handlers["preview"]("en", *values)
    assert real.calls == ["preview"] and demo.calls == [] and record is None
    assert "synthetic_provider_requires_mock_demo_mode" in errors  # the real side got a real-mode request

    values = _values(tab, ins_path=str(demo_ins), hkl_path=str(demo_hkl), mode="mock_demo", **{"cycle.iterations": "1"})
    errors, run_id, mode, *_ = tab.handlers["start"]("en", *values)
    assert errors == "" and mode == "mock_demo" and "submit" in demo.calls and "submit" not in real.calls
    request = json.loads(next(demo.runs_root.glob(f"{run_id}*/request.json")).read_text())
    assert request["mode"] == "mock_demo" and request["tool_paths"] == []  # tool locations belong to real runs only
    deadline = time.monotonic() + 90
    while True:
        timer, status, rows, summary = tab.handlers["tick"](run_id, "en", "mock_demo")
        if timer["active"] is False:
            break
        assert time.monotonic() < deadline
        time.sleep(0.2)
    assert status.startswith("🧪 **Synthetic demo (MOCK)**") and "🧪 **Synthetic demo (MOCK)**" in summary
    assert not list((root / "runs" / "reconstruction").glob("r-*"))  # nothing was written on the real side
    text, result = tab.handlers["checkcif"](run_id, "en", True, None, 30, True, "mock_demo")
    assert result.status.value == "NOT_EVALUATED"
    assert "simulated_toolchain_cannot_establish_local_platon_validation" in text
    assert tab.handlers["runs"]("en", "mock_demo")["value"][0][0] == run_id
    assert tab.handlers["runs"]("en", "real")["value"] == []
    # A run is looked up only in the service of the mode it is loaded under.
    loaded = tab.handlers["load"](run_id, "en", "real")
    assert loaded[2].startswith("⛔") and loaded[5]["active"] is False


def test_nonpositive_warning_uses_public_history_without_private_file_access():
    class HistoryService:
        def status(self, run_id):
            return {'history': [{'round': 1, 'nonpositive_u_site_ids': ['C1']},
                                {'round': 2, 'nonpositive_u_site_ids': []}]}
        def run_root(self, run_id):
            raise AssertionError('new history must not require inspecting inference files')
    assert R.nonpositive_u_warnings(HistoryService(), 'synthetic') == {1: ['C1']}


# ---------------------------------------------------------------------------- contract 2026-11-16
def test_input_files_are_named_through_their_real_folder(repo, tmp_path):
    """An upload folder reached through an alias (on macOS ``/var`` is one for ``/private/var``) must
    not block a run, and resolving the folder must not hide a file that is itself a link."""
    root, ins, hkl = repo
    alias = tmp_path / "alias"
    alias.symlink_to(ins.parent, target_is_directory=True)
    assert alias.resolve() == ins.parent and str(alias) != str(ins.parent)
    form = _defaults([], ins_path=str(alias / ins.name), hkl_path=[str(alias / hkl.name)])
    record = R.request_record(form, mode="mock_demo")
    assert record["ins_path"] == str(ins) and record["hkl_path"] == str(hkl)

    service = ReconstructionBackendService(root, root / "runs" / "reconstruction_demo",
                                           toolchain_factory=toolchain_factory, assigner_factory=assigner_factory)
    preview = service.preview(R.build_request(form, mode="mock_demo"))
    assert preview["blockers"] == [], preview["blockers"]

    link = ins.parent / "linked.ins"
    link.symlink_to(ins)
    named = R.request_record(_defaults([], ins_path=str(alias / link.name), hkl_path=str(hkl)), mode="mock_demo")["ins_path"]
    assert named == str(link) and Path(named).is_symlink()  # the link itself is left for the backend to refuse
    assert R._file("/no-such-folder-here/x.ins") == "/no-such-folder-here/x.ins"


def test_every_round_decision_of_the_backend_is_explained_in_three_languages():
    import inspect

    from crystalx_core.workflow import refinement_cycle

    codes = set(re.findall(r'"((?:RETAIN|REJECT|REJECTED|BLOCK|ACCEPT)_[A-Z_]+)"', inspect.getsource(refinement_cycle)))
    assert {"REJECT_REFINEMENT_DISPLACEMENT", "REJECT_R_REGRESSION", "RETAIN_UNVALIDATED"} <= codes
    for code in codes:
        texts = {lang: decision_label(code, lang) for lang in ("zh", "ja", "en")}
        assert all(text.startswith(code + " — ") for text in texts.values()), code
        assert len(set(texts.values())) == 3, code
        assert not re.search(r"converged|validated structure|correct", texts["en"].split(" — ", 1)[1]), code
    assert decision_label("SOMETHING_NEW", "zh") == "SOMETHING_NEW"  # an unknown code stays verbatim
    row = R.history_rows([{"round": 2, "decision": "REJECT_REFINEMENT_DISPLACEMENT", "max_refinement_net_shift_angstrom": 2.0,
                           "refinement_displacement_limit_exceeded": True}], "en")[0]
    assert row[8].startswith("REJECT_REFINEMENT_DISPLACEMENT — ") and "net-shift limit" in row[8]


def test_a_cif_blocked_after_cgls_is_explained_and_the_sealed_model_stays_on_offer():
    reason = "cgls_native_cif_requires_recorded_nonzero_ls_refinement"
    summary = {"phase": "blocked", "stop_code": "reconstruction_blocked", "stop_category": "blocked",
               "stop_detail": reason + "; retained_numerical_stop=fixed_rounds_completed", "blockers": [reason],
               "candidate_available": True, "seal_sha256": "a" * 64, "candidate_sha256": "b" * 64}
    for lang in ("zh", "ja", "en"):
        text = R.summary_md(summary, lang)
        assert f"`{reason}`" in text and stop_explanation(reason, lang) in text, lang
        assert "retained_numerical_stop=fixed_rounds_completed" in text  # the backend's detail, verbatim
        assert rt("candidate_from_unfinished_run", lang) in text and rt("candidate_missing", lang) not in text
    assert "cannot write a CIF under CGLS" in R.summary_md(summary, "en")


# ---------------------------------------------------------------- contract 2026-11-20
def test_device_is_auto_by_default_and_the_pinned_device_is_shown():
    assert R.request_record(_defaults([], ins_path="/x/a.ins", hkl_path="/x/a.hkl", device=None), mode="real")["device"] == "auto"
    state = {"run_id": "r-1", "phase": "running", "device_requested": "auto", "device_used": "mps",
             "device_selection_reason": "strict MPS startup probe passed", "device_blocker": None}
    text = R.status_md(state, "en")
    assert "requested `auto`" in text and "used in this run `mps`" in text and "startup probe passed" in text
    blocked = {"device_requested": "mps", "device_used": None, "device_selection_reason": "RuntimeError: no MPS",
               "device_blocker": "requested_mps_unavailable_without_fallback", "route": {}, "blockers": []}
    preview = R.preview_md(blocked, "zh")
    assert "`requested_mps_unavailable_without_fallback`" in preview and "不会改用 CPU" in preview
    assert R.device_lines({"phase": "completed"}, "en") == []  # records from before the contract: nothing invented


def test_the_initial_heavy_stage_is_shown_with_its_limits():
    stage = {"stage": "upstream_heavy_v1", "shelxl_cycles": 16, "anisotropic": True, "r1": 0.0512, "wr2": 0.1301,
             "goof": 1.04, "parameters": 289, "hydrogen_stage_performed": False}
    summary = {"phase": "completed", "stop_code": "heavy_stage_completed", "stop_category": "fixed_rounds",
               "stages": [stage], "candidate_available": True, "seal_sha256": "a" * 64, "candidate_sha256": "b" * 64}
    for lang in ("zh", "ja", "en"):
        text = R.summary_md(summary, lang)
        assert "`upstream_heavy_v1`" in text and "0.0512" in text and "289" in text
        assert rt("heavy_stage_note", lang).strip() in text and stop_explanation("heavy_stage_completed", lang) in text
    assert "not yet verified" in R.summary_md(summary, "en") and "converged" not in R.summary_md(summary, "en").lower()
    assert "initial heavy stage" in R.preview_md({"heavy_preparation": "upstream_heavy_v1", "route": {}}, "en")


def test_round_table_shows_listed_and_accepted_peaks_and_the_main_rejection():
    entry = {"round": 3, "r1": 0.45, "listed_output_peak_count": 400, "accepted_new_site_count": 13,
             "peak_rejection_reason_counts": {"no_bond_distance_contact_to_existing_scaffold": 385,
                                              "too_close_to_existing_scaffold": 1}}
    row = R.history_rows([entry], "en")[0]
    assert len(row) == len(history_columns("en")) and row[-3:] == ["400", "13", "no_bond_distance_contact_to_existing_scaffold ×385"]
    old = R.history_rows([{"round": 1, "r1": 0.5}], "en")[0]
    assert old[-3:] == ["—", "—", "—"]  # absent in older records: unknown, never 0


def test_new_technical_identifiers_are_explained():
    for code in ("non_difference_map_native_cif_requires_recorded_fmap2_refinement", "heavy_stage_completed",
                 "mps_requires_available_apple_silicon_runtime", "requested_mps_unavailable_without_fallback",
                 "synthetic_provider_does_not_load_weights_or_use_accelerators"):
        texts = [stop_explanation(code, lang) for lang in ("zh", "ja", "en")]
        assert all(texts) and len(set(texts)) == 3, code

