"""Adversarial file-boundary checks using only the fixed generated Python fixture."""

import os
from dataclasses import replace

import pytest

from crystalx_core.app.demo import write_demo_case
from crystalx_core.app.reconstruction_demo import synthetic_seed_text
from crystalx_core.app.reconstruction_files import (
    FileEvidenceStore, QPeakDensity, ShelxlFileRefinement,
)
from crystalx_core.refinement.locks import apply_locks
from crystalx_core.refinement.shelxl import RefinementSettings, prepare_refinement_ins
from crystalx_core.tools.runner import CancelToken, ProcessRunner, SubprocessExecutor
from crystalx_core.workflow.reconstruction import (
    Assessment, DensityPeak, ReconstructionBlocked, RefinementTrial, digest, read_snapshot,
)
from crystalx_core.workspace.hashing import sha256_file


_CHECKS = ("geometry", "coordination", "occupancy_adp", "chirality_symmetry", "restraints",
           "heavy_site_drift", "residual_distribution")


@pytest.fixture
def file_trial(tmp_path):
    inputs = tmp_path / "generated-inputs"
    _, original_hkl = write_demo_case(inputs)
    seed = inputs / "synthetic-seed.res"
    seed.write_text(synthetic_seed_text())
    model = read_snapshot(seed, reflection_sha256=sha256_file(original_hkl))
    root = tmp_path / "trial"
    root.mkdir()
    baseline, _, _, _ = prepare_refinement_ins(model.text, RefinementSettings(cycles=2, timeout_s=5))
    application = apply_locks(baseline, {})
    ins, hkl = root / "trial.ins", root / "trial.hkl"
    ins.write_text(application.text)
    hkl.write_bytes(original_hkl.read_bytes())
    hkl.chmod(0o444)
    trial = RefinementTrial(ins, hkl, root, model, {}, 2, 5, CancelToken(), True)
    return trial, FileEvidenceStore()


def _software_assessment(model, listing, evidence):
    return Assessment(model.basis_key, listing.r1_observed, listing.wr2, listing.goof,
        (listing.highest_peak, abs(listing.deepest_hole)), listing.n_all, listing.parameters_refined,
        tuple((name, "PASS") for name in _CHECKS), evidence.source_sha256,
        least_squares_converged=True, model_sha256=model.sha256)


def _returned_model(trial):
    return read_snapshot(trial.work_directory / "lock-free-model.res", reflection_sha256=sha256_file(trial.hkl_path))


class NoRun:
    def run(self, *_args, **_kwargs):
        pytest.fail("invalid file request reached the child process")


def test_default_file_assessment_never_claims_scientific_checks_or_convergence(file_trial):
    trial, store = file_trial
    outcome = ShelxlFileRefinement(store).refine(trial)
    assert outcome.succeeded
    assert outcome.assessment.least_squares_converged is None
    assert all(status == "NOT_EVALUATED" for _, status in outcome.assessment.checks)
    assert not outcome.assessment.scientific_validation
    assert outcome.review_required
    model = _returned_model(trial)
    density = QPeakDensity(store).observe(model)
    assert density.coverage == "partial" and density.kind == "SYNTHETIC_PEAK_LIST"
    evidence = store.for_model(model)
    assert evidence.raw_res_path != evidence.normalized_path
    assert evidence.raw_res_path.stat().st_ino != evidence.normalized_path.stat().st_ino


def test_real_mode_refuses_before_opening_any_input(file_trial, monkeypatch):
    import crystalx_core.app.reconstruction_files as module
    trial, store = file_trial
    monkeypatch.setattr(module, "_regular", lambda *_a, **_k: pytest.fail("real input was opened"))
    with pytest.raises(ReconstructionBlocked, match="real_file_refinement_dispatch_disabled"):
        ShelxlFileRefinement(store, runner=NoRun()).refine(replace(trial, synthetic=False))


@pytest.mark.parametrize("name", ["trial.res", "trial.lst", "file-invocation.json", "lock-free-model.res", "file-evidence.json"])
def test_stale_output_prevents_subprocess(file_trial, name):
    trial, store = file_trial
    (trial.work_directory / name).write_text("stale synthetic artifact\n")
    with pytest.raises(ReconstructionBlocked, match="fresh"):
        ShelxlFileRefinement(store, runner=NoRun()).refine(trial)


@pytest.mark.parametrize("field", ["r1", "n_reflections", "source_sha256", "model_sha256", "basis_key", "scientific_validation"])
def test_assessor_cannot_change_file_facts_or_claim_science(file_trial, field):
    trial, store = file_trial
    def forged(model, listing, evidence):
        value = _software_assessment(model, listing, evidence)
        changes = {"r1": value.r1 / 2, "n_reflections": value.n_reflections + 1,
                   "source_sha256": digest("different source"), "model_sha256": digest("different model"),
                   "basis_key": "different basis", "scientific_validation": True}
        return replace(value, **{field: changes[field]})
    with pytest.raises(ReconstructionBlocked, match="assessor"):
        ShelxlFileRefinement(store, synthetic_assessor=forged).refine(trial)
    with pytest.raises(ReconstructionBlocked, match="no_registered"):
        store.for_model(_returned_model(trial))


@pytest.mark.parametrize("target", ["ins", "hkl", "raw_res", "lst", "invocation", "normalized"])
def test_assessor_mutation_cannot_register_evidence(file_trial, target):
    trial, store = file_trial
    def mutating(model, listing, evidence):
        path = {"ins": trial.input_path, "hkl": trial.hkl_path,
                "raw_res": evidence.raw_res_path, "lst": evidence.lst_path,
                "invocation": evidence.invocation_path, "normalized": evidence.normalized_path}[target]
        path.chmod(0o644)
        with path.open("ab") as handle:
            handle.write(b"\nsynthetic unrecorded mutation\n")
        return _software_assessment(model, listing, evidence)
    with pytest.raises(ReconstructionBlocked):
        ShelxlFileRefinement(store, synthetic_assessor=mutating).refine(trial)
    assert not store._by_model


def test_assessor_cancellation_cannot_register_evidence(file_trial):
    trial, store = file_trial
    def cancelled(model, listing, evidence):
        trial.cancel.cancel("synthetic assessor cancellation")
        return _software_assessment(model, listing, evidence)
    with pytest.raises(ReconstructionBlocked, match="cancelled"):
        ShelxlFileRefinement(store, synthetic_assessor=cancelled).refine(trial)
    assert not store._by_model


@pytest.mark.parametrize("target", ["raw_res", "lst", "invocation", "normalized", "hkl"])
def test_density_refuses_source_replacement_after_registration(file_trial, target):
    trial, store = file_trial
    ShelxlFileRefinement(store).refine(trial)
    model = _returned_model(trial)
    evidence = store.for_model(model)
    path = {"raw_res": evidence.raw_res_path, "lst": evidence.lst_path,
            "invocation": evidence.invocation_path, "normalized": evidence.normalized_path,
            "hkl": evidence.hkl_path}[target]
    path.chmod(0o644)
    with path.open("ab") as handle:
        handle.write(b"\nsynthetic replacement\n")
    with pytest.raises(ReconstructionBlocked):
        QPeakDensity(store).observe(model)


def test_density_never_substitutes_another_model(file_trial):
    trial, store = file_trial
    ShelxlFileRefinement(store).refine(trial)
    model = _returned_model(trial)
    another = replace(model, text=model.text + "REM another retained model\n")
    with pytest.raises(ReconstructionBlocked, match="no_registered"):
        QPeakDensity(store).observe(another)


def test_forged_registered_peak_coordinates_cannot_reuse_raw_source_identity(file_trial):
    trial, store = file_trial
    ShelxlFileRefinement(store).refine(trial)
    evidence = store.for_model(_returned_model(trial))
    forged = replace(evidence, peaks=(DensityPeak("Qforged", (0.1, 0.2, 0.3), 100.0),))
    with pytest.raises(ReconstructionBlocked):
        store.register(forged)


def test_controlled_fixture_evidence_cannot_be_relabelled_as_real_density(file_trial):
    trial, store = file_trial
    ShelxlFileRefinement(store).refine(trial)
    evidence = store.for_model(_returned_model(trial))
    with pytest.raises(ReconstructionBlocked):
        store.register(replace(evidence, synthetic=False))


class MutatingChild:
    supports_control = True
    def __init__(self, mutation):
        self.child = SubprocessExecutor()
        self.mutation = mutation

    def execute(self, plan, env, **controls):
        result = self.child.execute(plan, env, **controls)
        if result.exit_code == 0:
            self.mutation(plan.cwd)
        return result


@pytest.mark.parametrize("alias", ["symlink", "hardlink"])
def test_strict_adapter_refuses_output_alias_before_reading_outside_file(file_trial, tmp_path, monkeypatch, alias):
    import crystalx_core.tools.adapter as adapter
    trial, store = file_trial
    outside = tmp_path / "held-out-synthetic-marker.txt"
    outside.write_text("a synthetic file outside the trial boundary\n")
    original_hash = adapter.sha256_file
    reads = []
    def guarded_hash(path):
        if path.exists() and os.path.samefile(path, outside):
            reads.append(str(path))
            pytest.fail("an aliased outside file reached snapshot hashing")
        return original_hash(path)
    monkeypatch.setattr(adapter, "sha256_file", guarded_hash)
    def alias_output(root):
        output = root / "trial.res"
        output.unlink()
        if alias == "symlink":
            output.symlink_to(outside)
        else:
            os.link(outside, output)
    runner = ProcessRunner(executor=MutatingChild(alias_output))
    with pytest.raises((ReconstructionBlocked, ValueError)):
        ShelxlFileRefinement(store, runner=runner).refine(trial)
    assert not reads and not store._by_model


@pytest.mark.parametrize("mutation", ["missing_listing", "missing_res", "mismatched_metrics", "fatal_listing"])
def test_required_process_outputs_and_consistent_metrics_are_not_optional(file_trial, mutation):
    trial, store = file_trial
    def change_outputs(root):
        if mutation == "missing_listing":
            (root / "trial.lst").unlink()
        elif mutation == "missing_res":
            (root / "trial.res").unlink()
        elif mutation == "mismatched_metrics":
            path = root / "trial.lst"
            path.write_text(path.read_text().replace("R1 = 0.450000", "R1 = 0.250000"))
        else:
            path = root / "trial.lst"
            path.write_text(path.read_text() + "\n** FATAL synthetic failure **\n")
    runner = ProcessRunner(executor=MutatingChild(change_outputs))
    with pytest.raises(ReconstructionBlocked):
        ShelxlFileRefinement(store, runner=runner).refine(trial)
    assert not store._by_model
