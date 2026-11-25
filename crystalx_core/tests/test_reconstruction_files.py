"""The real file/process boundary is tested only with generated data and controlled children."""

from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess

import pytest

from crystalx_core.app.demo import write_demo_case
from crystalx_core.app.reconstruction import ReconstructionService
from crystalx_core.app.reconstruction_demo import synthetic_seed_text
from crystalx_core.app.reconstruction_file_demo import file_demo_providers, synthetic_file_assessment
from crystalx_core.app.reconstruction_files import FileEvidenceStore, QPeakDensity, ShelxlFileRefinement
from crystalx_core.domain.hypothesis import LockDirective, LockMode
from crystalx_core.refinement.locks import apply_locks
from crystalx_core.refinement.shelxl import RefinementSettings, prepare_refinement_ins
from crystalx_core.solvers.mock import ScriptedExecutor, ScriptedResponse
from crystalx_core.tools.runner import CancelToken, ProcessRunner
from crystalx_core.workflow.reconstruction import ReconstructionBlocked, RefinementTrial, read_snapshot
from crystalx_core.workspace.hashing import sha256_file


@pytest.fixture
def trial(tmp_path):
    root = tmp_path.resolve() / "trial"
    root.mkdir()
    _, original_hkl = write_demo_case(root / "source")
    hkl = root / "trial.hkl"
    hkl.write_bytes(original_hkl.read_bytes())
    seed = root / "seed.res"
    seed.write_text(synthetic_seed_text())
    model = read_snapshot(seed, reflection_sha256=sha256_file(hkl))
    baseline, _, _, _ = prepare_refinement_ins(model.text, RefinementSettings(cycles=2))
    ins = root / "trial.ins"
    ins.write_text(apply_locks(baseline, {}).text)
    return RefinementTrial(ins, hkl, root, model, {}, 2, 5., CancelToken(), True)


def test_child_artifacts_are_parsed_without_claiming_scientific_checks(trial):
    store = FileEvidenceStore()
    provider = ShelxlFileRefinement(store)
    result = provider.refine(trial)
    assert result.process_kind == "synthetic" and result.succeeded
    assert result.review_required and result.assessment.least_squares_converged is None
    assert {status for _, status in result.assessment.checks} == {"NOT_EVALUATED"}
    assert result.assessment.blockers() and not result.assessment.scientific_validation
    model = read_snapshot(trial.work_directory / "lock-free-model.res",
                          reflection_sha256=sha256_file(trial.hkl_path))
    density = QPeakDensity(store).observe(model)
    assert density.kind == "SYNTHETIC_PEAK_LIST" and density.coverage == "partial"
    assert [(peak.peak_id, peak.fractional, peak.height) for peak in density.peaks] == [
        ("Q1", (.45, .5, .35), 2.)]
    record = json.loads((trial.work_directory / "file-invocation.json").read_text())
    assert record["succeeded"] and record["executable_source"] == "synthetic_demo"
    assert record["argv"][1:4] == ["-I", "-S", "-c"]
    assert "DYLD_INSERT_LIBRARIES" in record["env_unset"] and "LD_PRELOAD" in record["env_unset"]
    assert record["limits"]["max_file_bytes"] == 8 * 1024 * 1024
    assert {item["relative_path"]: item["sha256"] for item in record["inputs"]} == {
        "trial.ins": sha256_file(trial.input_path), "trial.hkl": sha256_file(trial.hkl_path)}
    evidence = json.loads((trial.work_directory / "file-evidence.json").read_text())
    assert evidence["listing"]["format_verified"] is False


def test_real_mode_refuses_before_any_input_access(trial):
    absent = trial.work_directory / "does-not-exist"
    provider = ShelxlFileRefinement(FileEvidenceStore())
    with pytest.raises(ReconstructionBlocked, match="real_file_refinement_dispatch_disabled"):
        provider.refine(replace(trial, synthetic=False, input_path=absent, hkl_path=absent))
    assert not (trial.work_directory / "file-invocation.json").exists()


@pytest.mark.parametrize("change", ["hkl", "ins", "stale_res", "cancelled"])
def test_prelaunch_gates_do_not_start_any_child(trial, change):
    executor = ScriptedExecutor()
    provider = ShelxlFileRefinement(FileEvidenceStore(), runner=ProcessRunner(executor))
    if change == "hkl":
        trial.hkl_path.write_text("unapproved measured input")
    elif change == "ins":
        trial.input_path.write_text(trial.input_path.read_text().replace("L.S. 2", "L.S. 40"))
    elif change == "stale_res":
        (trial.work_directory / "trial.res").write_text("old output")
    else:
        trial.cancel.cancel()
    with pytest.raises(ReconstructionBlocked):
        provider.refine(trial)
    assert not executor.calls


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="POSIX FIFO boundary")
def test_nonregular_input_is_rejected_before_a_potentially_blocking_open(trial, monkeypatch):
    trial.input_path.unlink()
    os.mkfifo(trial.input_path)
    original = os.open

    def guarded(path, *args, **kwargs):
        assert Path(path) != trial.input_path, "FIFO must be rejected before open"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(os, "open", guarded)
    with pytest.raises(ReconstructionBlocked, match="regular_file"):
        ShelxlFileRefinement(FileEvidenceStore()).refine(trial)


@pytest.mark.parametrize("response,diagnostic", [
    (ScriptedResponse(timed_out=True, exit_code=None), "timeout"),
    (ScriptedResponse(exit_code=2), "nonzero_exit"),
    (ScriptedResponse(), "expected_output_missing"),
])
def test_subprocess_failure_is_retained_and_does_not_register_density(trial, response, diagnostic):
    store = FileEvidenceStore()
    executor = ScriptedExecutor([response])
    provider = ShelxlFileRefinement(store, runner=ProcessRunner(executor))
    with pytest.raises(ReconstructionBlocked, match="refinement_subprocess_failed"):
        provider.refine(trial)
    record = json.loads((trial.work_directory / "file-invocation.json").read_text())
    assert record["diagnostic"] == diagnostic
    assert executor.calls[0].timeout_s == trial.timeout_s
    with pytest.raises(ReconstructionBlocked, match="no_registered_output"):
        QPeakDensity(store).observe(trial.proposed_model)


def test_soft_constraint_output_is_unlocked_without_erasing_the_raw_file(trial):
    locks = {"lock-0": LockDirective(LockMode.SOFT_RESTRAINT, ("Zn1", "C1"), "synthetic only",
                                    instruction_lines=("DFIX 2.0 0.2 Zn1 C1",))}
    baseline, _, _, _ = prepare_refinement_ins(trial.proposed_model.text, RefinementSettings(cycles=2))
    trial.input_path.write_text(apply_locks(baseline, locks).text)
    result = ShelxlFileRefinement(FileEvidenceStore(), synthetic_assessor=synthetic_file_assessment).refine(
        replace(trial, active_locks=locks))
    raw = (trial.work_directory / "trial.res").read_text()
    assert "DFIX 2 0.2 Zn1 C1" in raw and "active constraints BEGIN" in raw
    assert "DFIX" not in result.lock_free_model_text and "active constraints BEGIN" not in result.lock_free_model_text
    assert not result.assessment.blockers()


@pytest.mark.parametrize("evaluated", [False, True])
def test_full_controller_uses_file_peaks_preserves_raw_outputs_and_rolls_back(tmp_path, evaluated):
    repo = tmp_path.resolve() / "checkout"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / ".gitignore").write_text("/runs/\n")
    ins, hkl = write_demo_case(repo / "runs/inputs")
    seed = ins.parent / "automatic.res"
    seed.write_text(synthetic_seed_text())
    service = ReconstructionService(repo, repo / "runs/output")
    result = service.run(ins_path=ins, hkl_path=hkl, initial_model=seed,
                         providers=file_demo_providers(evaluated=evaluated), synthetic=True)
    if not evaluated:
        assert result["status"] == "BLOCKED" and result["usage"].tool_calls == 1
        assert result["accepted_model_sha256"] is None and not result["accepted_model_available"]
        assert result["stop_code"] == "initial_assessment_incomplete_or_failed"
        root = Path(result["run_root"])
        assert not (root / "reconstruction/accepted-model.res").exists()
        assert (root / "reconstruction/retained-initial-model.res").is_file()
        return
    assert [item["decision"] for item in result["decisions"]] == [
        "ACCEPT_PROVISIONAL", "REJECT", "ACCEPT_PROVISIONAL"]
    assert result["usage"].tool_calls == 4 and result["real_invocations"] == 0
    assert result["stop_code"] == "no_progress_not_convergence"
    assert result["decisions"][1]["parent_sha256"] == result["decisions"][2]["parent_sha256"]
    root = Path(result["run_root"])
    for folder in ("baseline", "trial-0001/refine", "trial-0002/refine", "trial-0003/refine"):
        directory = root / "reconstruction" / folder
        record = json.loads((directory / "file-invocation.json").read_text())
        recorded = next(item["sha256"] for item in record["outputs"] if item["relative_path"] == "trial.res")
        assert sha256_file(directory / "trial.res") == recorded
        assert (directory / "lock-free-model.res").is_file()
    assert "X002" not in (root / "reconstruction/accepted-model.res").read_text()
    assert "X003" in (root / "reconstruction/accepted-model.res").read_text()
