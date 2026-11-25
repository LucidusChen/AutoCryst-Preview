"""Runtime contracts; optional actual pretrained probes use generated models only."""

from dataclasses import replace
import importlib
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from conftest import write_ins
from crystalx_core.app.reconstruction_runtime_inference import (
    OFFICIAL_HEAVY_SHA256, RuntimeElementAssigner, _complete_symmetry,
)
from crystalx_core.domain.inference import InferenceDevice
from crystalx_core.domain.inference import ElementInferenceRequest
from crystalx_core.io.shelx_res import read_res
from crystalx_core.workflow.reconstruction import ReconstructionBlocked

REPO = Path(__file__).resolve().parents[2]
SOURCE = REPO / "crystalx_engine/inference_repo"
WEIGHTS = SOURCE / "weights/crystalx-heavy.pth"


def _model(tmp_path, *, latt=-1, symm=(), peaks=False):
    path = write_ins(tmp_path / "synthetic-runtime.res", latt=latt, symm=symm,
                     sfac=("C", "N", "O"), unit=(12, 2, 4),
                     extra_lines=("C1 1 0.20 0.20 0.20 11.0 0.03", "O1 3 0.35 0.20 0.20 11.0 0.03"))
    if peaks:
        with path.open("a") as handle:
            handle.write("Q1 1 0.50 0.30 0.20 11.0 0.05 2.0\n")
    return path


@pytest.fixture
def upstream_ratio_source():
    if not (SOURCE / "crystalx_infer/common/heavy.py").is_file():
        pytest.skip("optional upstream ratio-helper source absent; tests never download it")
    pytest.importorskip("numpy")
    return SOURCE


@pytest.fixture(scope="module")
def assigner():
    if not WEIGHTS.is_file():
        pytest.skip("optional owner-local published checkpoint absent; tests never download weights")
    pytest.importorskip("torch")
    pytest.importorskip("gemmi")
    pytest.importorskip("crystalx_engine_apple_silicon")
    return RuntimeElementAssigner(SOURCE, WEIGHTS, device=InferenceDevice.CPU)


def test_runtime_module_import_keeps_torch_optional():
    result = subprocess.run([sys.executable, "-I", "-c",
        "import sys; import crystalx_core.app.reconstruction_runtime_inference; "
        "assert 'torch' not in sys.modules; assert 'gemmi' not in sys.modules"],
        capture_output=True, check=False)
    assert result.returncode == 0, result.stderr.decode()


def test_missing_weights_block_before_optional_engine_import(tmp_path, monkeypatch):
    original = importlib.import_module
    def guard(name, *args, **kwargs):
        if name.startswith("crystalx_engine_apple_silicon"):
            pytest.fail("absent checkpoint reached optional tensor implementation")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(importlib, "import_module", guard)
    with pytest.raises(ReconstructionBlocked, match="required_local_inference_artifact_missing"):
        RuntimeElementAssigner(tmp_path, tmp_path / "absent.pth")


@pytest.mark.parametrize("latt,count", [(-1, 1), (1, 2), (-2, 2), (2, 4), (-3, 3), (-4, 4), (-5, 2), (-6, 2), (-7, 2)])
def test_complete_symmetry_contains_implicit_lattice_operations(tmp_path, latt, count):
    pytest.importorskip("gemmi")
    operators, provenance = _complete_symmetry(read_res(_model(tmp_path, latt=latt)))
    assert len(operators) == count
    assert all(abs(op.determinant) == 1 for op in operators)
    assert "LATT/SYMM" in provenance


def test_missing_latt_is_not_silently_assumed_p1(tmp_path):
    with pytest.raises(ReconstructionBlocked, match="declared_supported_latt"):
        _complete_symmetry(read_res(_model(tmp_path, latt=None)))


def test_actual_pretrained_cpu_preserves_main_ids_and_uncertainty(assigner, tmp_path):
    output = assigner.assign(_model(tmp_path, peaks=True), candidate_elements=("C", "N", "O"))
    assert output.result.model_identity.weights_sha256 == OFFICIAL_HEAVY_SHA256
    assert output.result.device_used is InferenceDevice.CPU
    assert output.site_ids == ("C1", "O1", "Q1")
    assert output.request.site_ids[:3] == output.site_ids
    assert len(output.peak_mapping) == 1 and output.peak_mapping[0]["prediction_index"] == 2
    assert all(len(row.topk) >= 2 for row in output.result.predictions)
    assert output.provenance["actual_pretrained_weights_executed"]
    assert not output.provenance["model_training_performed"]
    assert not output.provenance["upstream_prediction_parity_verified"]
    assert not output.provenance["scientific_validation_performed"]
    assert not output.result.fell_back and not output.result.probabilities_calibrated


def test_negative_displacement_is_reported_but_not_used_as_a_geometry_inference_gate(assigner,tmp_path):
    path=_model(tmp_path)
    path.write_text(path.read_text().replace("11.0 0.03","11.0 -0.01"))
    result=assigner.assign(path,candidate_elements=("C","N","O"))
    assert result.provenance["nonpositive_u_sites_preserved_for_geometry_only_inference"]
    assert result.provenance["nonpositive_u_scientifically_approved"] is False


def test_shelxl_creation_banner_does_not_hide_atoms_or_block_geometry(assigner,tmp_path):
    path=_model(tmp_path)
    rows=path.read_text().splitlines()
    rows[1:1]=["    trial.res", "    created by SHELXL-2025/1 at 12:34:56 on 10-Nov-2026"]
    path.write_text("\n".join(rows)+"\n")
    before=path.read_bytes()
    result=assigner.assign(path,candidate_elements=("C","N","O"))
    assert result.site_ids == ("C1","O1") and path.read_bytes()==before


@pytest.mark.parametrize("banner", ["created 1 0.2 broken atom", "created by some unknown program"])
def test_unrecognised_banner_still_blocks_rather_than_suppressing_parse_warning(assigner,tmp_path,banner):
    path=_model(tmp_path)
    rows=path.read_text().splitlines();rows.insert(1,banner)
    path.write_text("\n".join(rows)+"\n")
    with pytest.raises(ReconstructionBlocked,match="explicit_crystallographic_preprocessing_blocked"):
        assigner.assign(path,candidate_elements=("C","N","O"))


def test_peak_exclusion_is_explicit_and_does_not_rewrite_model(assigner, tmp_path):
    path = _model(tmp_path, peaks=True)
    before = path.read_bytes()
    output = assigner.assign(path, candidate_elements=("C", "N", "O"), include_q_peaks=False)
    assert output.site_ids == ("C1", "O1") and output.peak_mapping == ()
    assert path.read_bytes() == before


def test_hydrogen_is_explicitly_omitted_only_from_heavy_output_candidates(assigner, tmp_path):
    path = _model(tmp_path)
    before = path.read_bytes()
    output = assigner.assign(path, candidate_elements=("H", "C", "N", "O"))
    assert output.result.candidate_atomic_numbers == (6, 7, 8)
    assert output.provenance["hydrogen_output_candidate_explicitly_omitted"]
    assert path.read_bytes() == before


@pytest.mark.parametrize("elements", [("H", "C"), ("Cf", "C"), ("not_an_element", "C")])
def test_candidate_scope_cannot_exceed_actual_checkpoint(assigner, tmp_path, elements):
    with pytest.raises(ReconstructionBlocked):
        assigner.assign(_model(tmp_path), candidate_elements=elements)


@pytest.mark.parametrize("change", ["stale_request", "wrong_checkpoint", "fallback", "science_claim", "nonfinite_probability"])
def test_runtime_rejects_changed_prediction_identity_or_scope(assigner, tmp_path, monkeypatch, change):
    original = assigner.backend.predict_elements
    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        if change == "stale_request":
            return replace(result, input_sha256="0" * 64)
        if change == "wrong_checkpoint":
            return replace(result, model_identity=replace(result.model_identity, weights_sha256="0" * 64))
        if change == "fallback":
            return replace(result, fell_back=True)
        if change == "science_claim":
            return replace(result, scientific_validation_performed=True)
        first = result.predictions[0]
        bad = replace(first.topk[0], probability=float("nan"))
        prediction = replace(first, topk=(bad, *first.topk[1:]))
        return replace(result, predictions=(prediction, *result.predictions[1:]))
    monkeypatch.setattr(assigner.backend, "predict_elements", changed)
    with pytest.raises(ReconstructionBlocked):
        assigner.assign(_model(tmp_path), candidate_elements=("C", "N", "O"))


def test_model_replacement_during_prediction_is_not_returned(assigner, tmp_path, monkeypatch):
    path = _model(tmp_path)
    original = assigner.backend.predict_elements
    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        with path.open("a") as handle:
            handle.write("REM changed after inference\n")
        return result
    monkeypatch.setattr(assigner.backend, "predict_elements", changed)
    with pytest.raises(ReconstructionBlocked, match="digest_or_snapshot_changed"):
        assigner.assign(path, candidate_elements=("C", "N", "O"))


@pytest.mark.parametrize("coordinate", [-2.2, 2.2])
def test_ordinary_outside_cell_main_coordinates_are_preserved(assigner, tmp_path, coordinate):
    path = _model(tmp_path)
    path.write_text(path.read_text().replace("C1 1 0.20", f"C1 1 {coordinate}"))
    before = path.read_bytes()
    output = assigner.assign(path, candidate_elements=("C", "N", "O"))
    assert output.request.positions_angstrom[0][0] == pytest.approx(10.0 * coordinate)
    assert output.site_ids == ("C1", "O1")
    assert output.provenance["main_coordinates_preserved_without_wrapping"]
    assert path.read_bytes() == before


@pytest.mark.parametrize("coordinate", [-5.0, 5.0, 10.2])
def test_extreme_or_encoded_main_coordinates_remain_blocked(assigner, tmp_path, coordinate):
    path = _model(tmp_path)
    path.write_text(path.read_text().replace("C1 1 0.20", f"C1 1 {coordinate}"))
    with pytest.raises(ReconstructionBlocked, match="unsupported_disorder_free_variable_or_encoded_coordinate"):
        assigner.assign(path, candidate_elements=("C", "N", "O"))


def test_outside_domain_q_peak_blocks_instead_of_silent_wrapping_or_drop(assigner, tmp_path):
    path = _model(tmp_path, peaks=True)
    path.write_text(path.read_text().replace("Q1 1 0.50", "Q1 1 5.50"))
    before = path.read_bytes()
    with pytest.raises(ReconstructionBlocked, match="q_peaks_outside_supported_fractional_domain"):
        assigner.assign(path, candidate_elements=("C", "N", "O"))


def test_ordinary_unwrapped_q_peak_is_kept_with_its_physical_coordinate(assigner,tmp_path):
    path=_model(tmp_path,peaks=True)
    path.write_text(path.read_text().replace("Q1 1 0.50","Q1 1 1.50"))
    before=path.read_bytes()
    result=assigner.assign(path,candidate_elements=("C","N","O"))
    assert "Q1" in result.site_ids and path.read_bytes()==before
    q_index=result.request.site_ids.index("Q1")
    assert result.request.positions_angstrom[q_index][0]>10
    assert path.read_bytes() == before


def _protocol_assigner():
    value=RuntimeElementAssigner.__new__(RuntimeElementAssigner)
    value.q_descriptor=6;value.source_root=SOURCE
    value.backend=SimpleNamespace(shape=SimpleNamespace(max_z=100))
    return value


def test_initial_labels_protocol_avoids_refined_label_feedback_and_records_new_site_fallback():
    request=ElementInferenceRequest(('C1','O1','X001','C1/context/copy'),(8,6,8,8),
        ((0.,0.,0.),(1.,0.,0.),(2.,0.,0.),(0.,1.,0.)),(6,7,8),main_site_count=3)
    kwargs=dict(input_protocol='initial_labels',original_ins_path=None,initial_labels={'C1':'C','O1':'O'},protocol_seed=42)
    mapped,record=_protocol_assigner()._input_protocol(request,**kwargs)
    assert mapped.atomic_numbers==(6,8,6,6)
    changed=replace(request,atomic_numbers=(7,7,7,7))
    assert _protocol_assigner()._input_protocol(changed,**kwargs)[0].atomic_numbers==mapped.atomic_numbers
    assert record['new_site_fallback_ids']==['X001'] and record['new_site_fallback_atomic_number']==6
    assert not record['randomized_initialization']


def test_upstream_protocol_uses_original_ratio_only_for_main_and_initial_labels_for_context(tmp_path, upstream_ratio_source):
    original=write_ins(tmp_path/'original.ins',sfac=('C','N','O'),unit=(12,2,4))
    request=ElementInferenceRequest(('C1','N1','N1/context/copy'),(8,8,8),
        ((0.,0.,0.),(1.,0.,0.),(0.,1.,0.)),(6,7,8),main_site_count=2)
    mapped,record=_protocol_assigner()._input_protocol(request,input_protocol='upstream_ins',
        original_ins_path=original,initial_labels={'C1':'C','N1':'N'},protocol_seed=42)
    assert mapped.atomic_numbers==(8,6,7)
    assert mapped.candidate_atomic_numbers==(6,7,8)
    assert record['original_ins_sha256'] and record['forced_element_coverage']
    assert record['ratio_initialization']['original_unit']==(12.,2.,4.)
    assert record['second_pass_input']=='all_first_pass_graph_predictions'
    assert record['full_upstream_preprocessing_parity_verified'] is False


def test_zero_unit_class_is_not_forced_unlike_unique_upstream_graph_descriptors(tmp_path, upstream_ratio_source):
    original=write_ins(tmp_path/'original.ins',sfac=('C','N','O'),unit=(1,0,1))
    request=ElementInferenceRequest(('C1','O1'),(6,8),((0.,0.,0.),(1.,0.,0.)),(6,7,8))
    mapped,record=_protocol_assigner()._input_protocol(request,input_protocol='upstream_ins',
        original_ins_path=original,initial_labels={'C1':'C','O1':'O'},protocol_seed=42)
    assert mapped.candidate_atomic_numbers==(6,8)
    assert record['requested_candidate_atomic_numbers']==(6,7,8)


def test_original_atom_model_is_rejected_as_ratio_protocol_input(tmp_path):
    original=_model(tmp_path)
    request=ElementInferenceRequest(('C1','O1'),(6,8),((0.,0.,0.),(1.,0.,0.)),(6,7,8))
    with pytest.raises(ReconstructionBlocked,match='atom_free_original_ins'):
        _protocol_assigner()._input_protocol(request,input_protocol='upstream_ins',original_ins_path=original,
            initial_labels={'C1':'C','O1':'O'},protocol_seed=42)


def test_missing_original_ins_reports_a_protocol_block(tmp_path):
    request=ElementInferenceRequest(('C1','O1'),(6,8),((0.,0.,0.),(1.,0.,0.)),(6,7,8))
    with pytest.raises(ReconstructionBlocked,match='input_protocol_original_ins_unreadable'):
        _protocol_assigner()._input_protocol(request,input_protocol='upstream_ins',original_ins_path=tmp_path/'missing.ins',
            initial_labels={'C1':'C','O1':'O'},protocol_seed=42)


def test_actual_upstream_protocol_keeps_ranked_probabilities_separate_from_coverage(assigner,tmp_path):
    original=write_ins(tmp_path/'original.ins',sfac=('C','N','O'),unit=(1,1,1))
    model=_model(tmp_path,peaks=True);before=model.read_bytes()
    result=assigner.assign(model,candidate_elements=('C','N','O'),input_protocol='upstream_ins',
        original_ins_path=original,initial_labels={'C1':'C','O1':'O'})
    assert set(result.provenance['protocol_element_choices'].values())=={'C','N','O'}
    assert result.provenance['input_protocol']=='upstream_ins'
    assert result.provenance['coverage_choices_are_not_ranked_probabilities']
    assert result.request.atomic_numbers[:3]==(8,7,6)
    assert all(all(a.probability>=b.probability for a,b in zip(p.topk,p.topk[1:])) for p in result.result.predictions)
    assert not result.result.probabilities_calibrated and model.read_bytes()==before


def test_pinned_ratio_helper_is_exact_and_changed_source_is_rejected(tmp_path, upstream_ratio_source):
    from crystalx_core.app.inference_protocols import pinned_upstream_helper
    ratio=pinned_upstream_helper(SOURCE,'build_sorted_init_z_from_ratio')
    assert ratio(('C','H','N','O','Zn'),(6,12,2,1,1),7)==[30,8,7,6,6,6,6]
    target=tmp_path/'crystalx_infer/common/heavy.py';target.parent.mkdir(parents=True)
    target.write_bytes((SOURCE/'crystalx_infer/common/heavy.py').read_bytes()+b'\n# changed\n')
    with pytest.raises(ValueError,match='SHA-256 changed'):
        pinned_upstream_helper(tmp_path,'build_sorted_init_z_from_ratio')
