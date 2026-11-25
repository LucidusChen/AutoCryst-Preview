"""Generated synthetic expert-model copies only; no real program is invoked."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from crystalx_core.app.acceptance import verify_run
from crystalx_core.app.demo import demo_atoms, demo_case_texts
from crystalx_core.app.mock_toolchain import MockToolchain
from crystalx_core.app.models import ChemicalPriorInput, RunConfig, RunMode, SolveRequest
from crystalx_core.app.service import RunService
from crystalx_core.app.toolchain import Toolchain
from crystalx_core.domain.tools import ToolKind
from crystalx_core.private_acceptance.component import (
    ComponentRegressionPipeline, preview_component_request, verify_component_run,
)
from crystalx_core.tools.runner import CancelToken
from crystalx_core.workspace.hashing import sha256_file
from crystalx_core.workspace.store import RunWorkspace


class NoShelxtToolchain(MockToolchain):
    def adapter(self, kind):
        assert kind is not ToolKind.SHELXT, "component regression must not even request a SHELXT adapter"
        return super().adapter(kind)


def synthetic_expert_pair(tmp_path: Path):
    original = tmp_path / "synthetic expert originals"
    original.mkdir()
    header, reflections = demo_case_texts()
    sfac = ("C", "H", "N", "O", "Zn")
    atom_lines = []
    for atom in demo_atoms():
        x, y, z = atom.frac
        atom_lines.append(f"{atom.label} {sfac.index(atom.element) + 1} {x} {y} {z} 11.0 0.04")
    expert = header.replace("HKLF 4", "FVAR 1.0\n" + "\n".join(atom_lines) + "\nHKLF 4")
    model = original / "Original expert model.res"
    hkl = original / "Original paired intensities.hkl"
    model.write_text(expert, encoding="utf-8")
    hkl.write_text(reflections, encoding="utf-8")
    return model, hkl


def make_pipeline(tmp_path, *, prior=None, toolchain=None, mutate=None, cancel=None):
    model, hkl = synthetic_expert_pair(tmp_path)
    if mutate:
        model.write_text(mutate(model.read_text(encoding="utf-8")), encoding="utf-8")
    workspace = RunWorkspace.create(tmp_path / "runs", label="synthetic component")
    request = SolveRequest(model, hkl, prior=prior or ChemicalPriorInput(),
                           config=RunConfig(strict_native=True, shelxl_cycles=3), mode=RunMode.REAL)
    pipeline = ComponentRegressionPipeline(workspace, request, toolchain or NoShelxtToolchain(step_seconds=0.001),
                                           cancel=cancel or CancelToken())
    pipeline.write_initial_state()
    return pipeline, model, hkl


def test_component_reuses_service_and_shared_stages_without_initial_solution(tmp_path):
    model, hkl = synthetic_expert_pair(tmp_path)
    before = (model.read_bytes(), hkl.read_bytes())
    service = RunService(tmp_path / "runs", toolchain_factory=lambda request: NoShelxtToolchain(step_seconds=0.001),
                         pipeline_factory=ComponentRegressionPipeline)
    handle = service.submit(SolveRequest(model, hkl, config=RunConfig(strict_native=True, shelxl_cycles=3)))
    state = service.wait(handle.run_id, timeout=10)
    summary = service.summary(handle.run_id)
    assert state["phase"] == "completed"
    states = {stage["name"]: stage["state"] for stage in state["stages"]}
    assert states["shelxt"] == "skipped"
    assert states["shelxl"] == states["platon"] == states["viewer"] == "succeeded"
    assert summary["acceptance_track"] == "COMPONENT_REGRESSION"
    assert summary["solution"] is None
    assert summary["component_regression"]["initial_solution_performed"] is False
    assert summary["preconditions"]["space_group_from_solver"] is False
    assert summary["refinement"]["settings"]["cycles"] == 3
    assert summary["refinement"]["reflections_lineage"]["source"] == "expert_model_paired_hkl"
    assert "SHELXT" not in summary["refinement"]["reflections_note"]
    assert summary["scientific_eligibility"]["eligible_for_ranking"] is False
    assert set(summary["viewer"]) == {"expert_model", "refined"}
    assert summary["mock"] is True
    viewer = json.loads((handle.run_dir / summary["viewer"]["expert_model"]).read_text())
    assert viewer["acceptance_track"] == "COMPONENT_REGRESSION"
    assert "expert" in viewer["source"]
    assert (model.read_bytes(), hkl.read_bytes()) == before
    assert not (handle.run_dir / "stages/s1-initial_solution").exists()
    assert not list(handle.run_dir.rglob("provenance_00.json"))
    assert verify_component_run(handle.run_dir, require_real=False).accepted
    assert not verify_component_run(handle.run_dir).accepted
    assert not verify_run(handle.run_dir, require_real=False).accepted
    result = verify_component_run(handle.run_dir, require_real=False).to_record()
    assert result["initial_solution_verified"] is False
    assert result["scientifically_reviewed"] is False


def test_component_expected_input_hashes_required_both_roles(tmp_path):
    pipeline, model, hkl = make_pipeline(tmp_path)
    pipeline.run()
    root = pipeline.workspace.layout.root
    expected = {"ins": sha256_file(model), "hkl": sha256_file(hkl)}
    assert verify_component_run(root, require_real=False, expected_inputs=expected).accepted
    expected["hkl"] = "0" * 64
    assert not verify_component_run(root, require_real=False, expected_inputs=expected).accepted
    assert not verify_component_run(root, require_real=False, expected_inputs={"ins": sha256_file(model)}).accepted


def test_component_refuses_fixed_improper_symmetry_for_homochiral_prior(tmp_path):
    pipeline, _, _ = make_pipeline(tmp_path, prior=ChemicalPriorInput(peptide=True),
                                    mutate=lambda text: text.replace("LATT -1", "LATT 1"))
    summary = pipeline.run()
    assert summary["process"]["phase"] == "blocked"
    assert summary["preconditions"]["may_start"] is False
    assert "chirality_impossible" in {p["kind"] for p in summary["preconditions"]["items"]}
    assert not (pipeline.workspace.layout.root / "refinement").exists()


@pytest.mark.parametrize("remove", ["HKLF 4\n", "LATT -1\n"])
def test_component_never_infers_missing_fixed_model_instructions(tmp_path, remove):
    pipeline, _, _ = make_pipeline(tmp_path, mutate=lambda text: text.replace(remove, ""))
    summary = pipeline.run()
    assert summary["process"]["phase"] == "blocked"
    assert "no symmetry or reflection format is inferred" in summary["process"]["message"]


def test_component_refuses_composition_reassignment(tmp_path):
    pipeline, _, _ = make_pipeline(tmp_path, prior=ChemicalPriorInput(composition=(("C", 4.0),)))
    summary = pipeline.run()
    assert summary["process"]["phase"] == "blocked"
    assert "cannot replace SFAC/UNIT" in summary["process"]["message"]


@pytest.mark.parametrize("atom_change", [lambda text: text.replace("Zn1 5", "Zn1 99"),
                                         lambda text: text.replace("Zn1 5 0.25", "Zn1 5 nan")])
def test_component_invalid_atom_data_blocks_instead_of_guessing(tmp_path, atom_change):
    pipeline, _, _ = make_pipeline(tmp_path, mutate=atom_change)
    summary = pipeline.run()
    assert summary["process"]["phase"] == "blocked"
    assert not (pipeline.workspace.layout.root / "refinement").exists()


def test_component_include_directive_blocked_without_external_read(tmp_path):
    pipeline, _, _ = make_pipeline(tmp_path, mutate=lambda text: "+../held-out.res\n" + text)
    summary = pipeline.run()
    assert summary["process"]["phase"] == "blocked"
    assert "includes" in summary["process"]["message"]


def test_component_missing_fcf_blocks_required_platon_and_dedicated_acceptance(tmp_path):
    pipeline, _, _ = make_pipeline(tmp_path, toolchain=NoShelxtToolchain(step_seconds=0.001, omit_suffixes=(".fcf",)))
    summary = pipeline.run()
    assert summary["process"]["phase"] == "blocked"
    assert summary["platon"]["blocked"]
    assert not (pipeline.workspace.layout.root / "validation").exists()
    assert not verify_component_run(pipeline.workspace.layout.root, require_real=False).accepted


def test_component_prestart_cancellation_launches_nothing(tmp_path):
    cancel = CancelToken()
    cancel.cancel("synthetic stop")
    pipeline, _, _ = make_pipeline(tmp_path, cancel=cancel)
    summary = pipeline.run()
    assert summary["process"]["phase"] == "cancelled"
    assert not list(pipeline.workspace.layout.root.rglob("provenance*.json"))


@pytest.mark.parametrize("relative", ["component/input_model/{stem}.res", "component/input_model/{stem}.hkl",
                                     "refinement/shelxl-01/{stem}.hkl", "refinement/shelxl-01/{stem}.ins"])
def test_component_verifier_rehashes_retained_pair_and_derived_inputs(tmp_path, relative):
    pipeline, _, _ = make_pipeline(tmp_path)
    summary = pipeline.run()
    root = pipeline.workspace.layout.root
    target = root / relative.format(stem=summary["stem"])
    target.write_text(target.read_text() + "\nREM synthetic post-run change\n")
    verdict = verify_component_run(root, require_real=False)
    assert not verdict.accepted
    assert "component_lineage" in {check.name for check in verdict.failed}


def test_component_verifier_contains_recorded_pair_path(tmp_path):
    pipeline, _, _ = make_pipeline(tmp_path)
    pipeline.run()
    root = pipeline.workspace.layout.root
    summary = json.loads((root / "summary.json").read_text())
    summary["component_regression"]["input_model"]["file"] = "../answer.res"
    (root / "summary.json").write_text(json.dumps(summary))
    verdict = verify_component_run(root, require_real=False)
    assert not verdict.accepted
    assert "component_lineage" in {check.name for check in verdict.failed}


def test_component_verifier_rejects_inserted_shelxt_claim(tmp_path):
    pipeline, _, _ = make_pipeline(tmp_path)
    pipeline.run()
    root = pipeline.workspace.layout.root
    summary = json.loads((root / "summary.json").read_text())
    summary["component_regression"]["initial_solution_performed"] = True
    (root / "summary.json").write_text(json.dumps(summary))
    assert "component_track" in {check.name for check in verify_component_run(root, require_real=False).failed}


def test_component_verifier_empty_missing_record_does_not_pass(tmp_path):
    verdict = verify_component_run(tmp_path)
    assert not verdict.accepted
    assert verdict.failed[0].name == "record_readable"


@pytest.mark.parametrize("field", ["component_regression", "refinement", "inputs", "viewer"])
def test_malformed_component_archive_fails_closed(tmp_path, field):
    pipeline, _, _ = make_pipeline(tmp_path)
    pipeline.run()
    root = pipeline.workspace.layout.root
    summary = json.loads((root / "summary.json").read_text())
    summary[field] = ["synthetic malformed object"]
    (root / "summary.json").write_text(json.dumps(summary))
    assert not verify_component_run(root, require_real=False).accepted


@pytest.mark.parametrize("relative", ["summary.json", "refinement/shelxl-01/provenance.json"])
def test_component_verifier_does_not_follow_record_symlink_outside_run(tmp_path, relative, monkeypatch):
    pipeline, _, _ = make_pipeline(tmp_path)
    pipeline.run()
    root = pipeline.workspace.layout.root
    outside = tmp_path / "held-out-answer.res"
    outside.write_text("do not read this synthetic answer")
    path = root / relative
    path.unlink()
    path.symlink_to(outside)
    read_text = Path.read_text

    def guarded_read(self, *args, **kwargs):
        assert self.resolve() != outside.resolve()
        return read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read)
    assert not verify_component_run(root, require_real=False).accepted


def test_component_readonly_preview_opens_only_selected_input_pair(tmp_path, monkeypatch):
    model, hkl = synthetic_expert_pair(tmp_path)
    held_out = tmp_path / "held-out-answer.res"
    held_out.write_text("synthetic answer trap")
    before = {path: path.read_bytes() for path in (model, hkl, held_out)}
    original_open = __import__("builtins").open

    def guarded_open(path, *args, **kwargs):
        assert Path(path) in {model, hkl}
        assert not args or args[0] in {"r", "rb"}
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", guarded_open)
    monkeypatch.setattr(RunWorkspace, "create", lambda *a, **k: pytest.fail("preview must not create a workspace"))
    monkeypatch.setattr("crystalx_core.tools.runner.ProcessRunner.run", lambda *a, **k: pytest.fail("preview must not launch"))
    report = preview_component_request(SolveRequest(model, hkl))
    assert report["may_start"]
    assert report["fixed_model"]
    assert report["space_group_from_solver"] is False
    assert report["initial_solution_performed"] is False
    assert report["pair_compatibility_scientifically_verified"] is False
    assert {path: path.read_bytes() for path in (model, hkl, held_out)} == before


def test_component_preview_reports_empty_model_and_fixed_chirality(tmp_path):
    model, hkl = synthetic_expert_pair(tmp_path)
    source = model.read_text()
    model.write_text(source.replace("LATT -1", "LATT 1"))
    report = preview_component_request(SolveRequest(model, hkl, prior=ChemicalPriorInput(peptide=True)))
    assert not report["may_start"]
    assert "chirality_impossible" in report["blocker_codes"]
    model.write_text(demo_case_texts()[0])
    report = preview_component_request(SolveRequest(model, hkl))
    assert not report["may_start"]
    assert "no_model_atoms" in report["blocker_codes"]


def test_component_preview_bad_parse_returns_generic_diagnostic(tmp_path):
    model, hkl = synthetic_expert_pair(tmp_path)
    model.write_text("CELL SECRET-PARSE-EXCERPT")
    report = preview_component_request(SolveRequest(model, hkl))
    assert not report["may_start"]
    assert "component_input_unreadable" in report["blocker_codes"]
    assert "SECRET-PARSE-EXCERPT" not in json.dumps(report)


def test_real_component_service_path_is_blocked_before_any_adapter_access(tmp_path):
    class NoRealDispatch(Toolchain):
        mode = RunMode.REAL

        def availability(self):
            pytest.fail("offline component route must not inspect a real toolchain")

        def adapter(self, kind):
            pytest.fail("offline component route must not request a real adapter")

        def platon_adapter(self):
            pytest.fail("offline component route must not request PLATON")

    model, hkl = synthetic_expert_pair(tmp_path)
    service = RunService(tmp_path / "runs", toolchain_factory=lambda request: NoRealDispatch(),
                         pipeline_factory=ComponentRegressionPipeline)
    handle = service.submit(SolveRequest(model, hkl))
    state = service.wait(handle.run_id, timeout=10)
    summary = service.summary(handle.run_id)
    assert state["phase"] == "blocked"
    assert "authority is not implemented" in state["message"]
    assert summary["component_regression"]["real_launch_enabled"] is False
    assert not list(handle.run_dir.rglob("provenance*.json"))
    assert not (handle.run_dir / "refinement").exists()
    assert not verify_component_run(handle.run_dir).accepted
