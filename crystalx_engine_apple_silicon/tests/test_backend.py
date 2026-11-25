from dataclasses import replace
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from crystalx_core.domain.inference import ElementInferenceRequest, InferenceDevice
from crystalx_engine_apple_silicon.backend import AppleSiliconBackend
from crystalx_engine_apple_silicon.device import BackendUnavailable, DeviceDecision, select_device
from crystalx_engine_apple_silicon.source import load_model_classes


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "crystalx_engine" / "inference_repo"


@pytest.fixture
def weights(tmp_path):
    torch.manual_seed(42)
    classes = load_model_classes(SOURCE)
    model = classes.TorchMD_Net(
        classes.TorchMD_ET(hidden_channels=16, num_layers=2, num_rbf=8),
        classes.EquivariantScalar(16, num_classes=32),
    )
    path = tmp_path / "synthetic.pth"
    torch.save({"state_dict": model.state_dict()}, path)
    return path


@pytest.fixture
def request_sites():
    return ElementInferenceRequest(("a", "b", "c"), (6, 7, 8),
                                   ((0., 0., 0.), (1.2, 0., 0.), (0., 1.3, 0.)), (6, 7, 8))


def test_cpu_two_pass_and_identity(weights, request_sites):
    engine = AppleSiliconBackend(SOURCE, weights)
    result = engine.predict_elements(request_sites, device="cpu")
    assert len(result.predictions) == 3
    assert result.model_identity.hidden_channels == 16
    assert result.model_identity.num_classes == 32
    assert result.device_used == InferenceDevice.CPU and not result.fell_back
    assert len(result.model_identity.weights_sha256) == 64
    assert len(result.model_identity.source_files) == 4
    assert len(result.input_sha256) == 64
    assert result.candidate_atomic_numbers == (6, 7, 8)
    assert result.torch_version == torch.__version__
    assert not result.probabilities_calibrated
    assert not result.scientific_validation_performed
    assert not result.upstream_prediction_parity_verified
    for prediction in result.predictions:
        assert prediction.runner_up == prediction.topk[1]
        assert sum(p.probability for p in prediction.topk) == pytest.approx(1.)


def test_main_sites_are_subset_not_context(weights, request_sites):
    result = AppleSiliconBackend(SOURCE, weights).predict_elements(replace(request_sites, main_site_count=2), device="cpu")
    assert [p.site_id for p in result.predictions] == ["a", "b"]


def test_weight_digest_and_strict_state(weights):
    with pytest.raises(ValueError, match="SHA-256"):
        AppleSiliconBackend(SOURCE, weights, expected_sha256="0" * 64)
    state = torch.load(weights, weights_only=True)["state_dict"]
    del state["representation_model.embedding.weight"]
    torch.save(state, weights)
    with pytest.raises(ValueError, match="absent"):
        AppleSiliconBackend(SOURCE, weights)


def test_mps_unavailable_is_explicit(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    decision = select_device("mps")
    assert decision.used == InferenceDevice.CPU and decision.fell_back and decision.reason
    with pytest.raises(BackendUnavailable):
        select_device("mps", allow_cpu_fallback=False)
    with pytest.raises(BackendUnavailable):
        select_device("cuda")


def test_implicit_fallback_environment_cannot_claim_pure_mps(monkeypatch):
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    assert "Implicit" in select_device("mps").reason
    with pytest.raises(BackendUnavailable):
        select_device("mps", allow_cpu_fallback=False)


def test_full_model_failure_reruns_on_cpu(weights, request_sites, monkeypatch):
    import crystalx_engine_apple_silicon.backend as module
    engine = AppleSiliconBackend(SOURCE, weights)
    real_run = engine._run
    monkeypatch.setattr(module, "select_device", lambda *a, **k: DeviceDecision(InferenceDevice.MPS, InferenceDevice.MPS))
    def fail_mps(request, used):
        if used == InferenceDevice.MPS:
            raise NotImplementedError("synthetic unsupported MPS operator")
        return real_run(request, used)
    monkeypatch.setattr(engine, "_run", fail_mps)
    result = engine.predict_elements(request_sites, device="mps")
    assert result.fell_back and result.device_used == InferenceDevice.CPU
    assert "whole model" in result.fallback_reason
    with pytest.raises(BackendUnavailable):
        engine.predict_elements(request_sites, device="mps", allow_cpu_fallback=False)


@pytest.mark.parametrize("positions", [((0., 0., 0.),) * 3, ((1e50, 0., 0.), (1., 0., 0.), (0., 1., 0.))])
def test_invalid_float32_geometry_rejected(weights, request_sites, positions):
    with pytest.raises(ValueError):
        AppleSiliconBackend(SOURCE, weights).predict_elements(replace(request_sites, positions_angstrom=positions), device="cpu")


@pytest.mark.mps
@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable in this runtime")
def test_real_mps_full_model_matches_native_cpu(weights, request_sites):
    engine = AppleSiliconBackend(SOURCE, weights)
    cpu = engine.predict_elements(request_sites, device="cpu")
    mps = engine.predict_elements(request_sites, device="mps", allow_cpu_fallback=False)
    assert mps.device_used == InferenceDevice.MPS and not mps.fell_back
    for a, b in zip(cpu.predictions, mps.predictions):
        assert a.topk[0].atomic_number == b.topk[0].atomic_number
        expected = {p.atomic_number: p.probability for p in a.topk}
        assert {p.atomic_number: p.probability for p in b.topk} == pytest.approx(expected, abs=1e-4)


def test_package_and_core_contract_do_not_import_torch():
    script = "import sys; import crystalx_engine_apple_silicon; import crystalx_core.domain.inference; assert 'torch' not in sys.modules"
    completed = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr


def test_optional_forced_coverage_preserves_raw_ranked_probability_results(weights,request_sites):
    engine=AppleSiliconBackend(SOURCE,weights)
    raw=engine.predict_elements(request_sites,device='cpu')
    covered,choices=engine.predict_elements_with_coverage(request_sites,device='cpu',allow_cpu_fallback=False)
    assert set(choices)=={6,7,8}
    assert covered.predictions==raw.predictions
    assert covered.input_sha256==raw.input_sha256
    assert any('adjusted choices are separate' in warning for warning in covered.warnings)
    assert not covered.probabilities_calibrated


def test_pinned_coverage_can_select_a_class_outside_raw_top_five():
    from crystalx_core.app.inference_protocols import pinned_upstream_helper
    coverage=pinned_upstream_helper(SOURCE,'enforce_coverage_by_prob')
    probabilities=torch.tensor([[.9,.02,.02,.02,.02,.02]]*6)
    labels=torch.tensor([6,7,8,9,16,17]);raw=labels[probabilities.argmax(dim=1)]
    before=probabilities.clone();chosen=coverage(probabilities,labels,raw.clone())
    assert set(chosen.tolist())==set(labels.tolist())
    assert chosen.tolist()==[7,8,9,16,17,6]
    assert torch.equal(probabilities,before) and torch.equal(raw,torch.full((6,),6))
