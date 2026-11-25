"""Reject lossy checkpoint casts and unsupported audited model layouts."""

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from crystalx_core.domain.inference import ElementInferenceRequest
from crystalx_engine_apple_silicon.backend import AppleSiliconBackend, infer_model_shape
from crystalx_engine_apple_silicon.source import load_model_classes


SOURCE = Path(__file__).resolve().parents[2] / "crystalx_engine" / "inference_repo"
EMBEDDING = "representation_model.embedding.weight"
RBF = "representation_model.distance_expansion.means"
CLASS_BIAS = "output_model.output_network.1.update_net.2.bias"


@pytest.fixture(scope="module")
def audited_state():
    if not (SOURCE / "crystalx_infer" / "models" / "torchmd_et.py").exists():
        pytest.skip("Local upstream inference source is separately supplied")
    classes = load_model_classes(SOURCE)
    model = classes.TorchMD_Net(
        classes.TorchMD_ET(hidden_channels=16, num_layers=1, num_rbf=8),
        classes.EquivariantScalar(16, num_classes=32),
    )
    return model.state_dict()


@pytest.mark.parametrize("dtype", [
    torch.float16, torch.bfloat16, torch.float64,
    torch.complex64, torch.complex128,
    torch.int8, torch.int32, torch.int64, torch.uint8, torch.bool,
])
def test_no_lossy_checkpoint_tensor_casts(audited_state, dtype):
    state = dict(audited_state)
    value = state[EMBEDDING].to(dtype)
    if value.is_complex():
        value = value + 1j  # Dropping this component must not be silently allowed.
    state[EMBEDDING] = value
    with pytest.raises(ValueError, match="float32"):
        infer_model_shape(state)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_checkpoint_rejected(audited_state, value):
    state = dict(audited_state)
    state[EMBEDDING] = state[EMBEDDING].clone()
    state[EMBEDDING][0, 0] = value
    with pytest.raises(ValueError, match="nonfinite"):
        infer_model_shape(state)


@pytest.mark.parametrize("state", [None, [], {}, {"bad": "not a tensor"}, {1: torch.ones(1)}])
def test_checkpoint_must_be_nonempty_string_to_tensor_mapping(state):
    with pytest.raises(ValueError, match="tensor state_dict"):
        infer_model_shape(state)


@pytest.mark.parametrize("key", [EMBEDDING, RBF, CLASS_BIAS])
def test_missing_required_shape_tensor_rejected(audited_state, key):
    state = dict(audited_state)
    del state[key]
    with pytest.raises(ValueError, match="absent"):
        infer_model_shape(state)


@pytest.mark.parametrize("key,shape", [
    (EMBEDDING, (100,)),
    (EMBEDDING, (100, 15)),
    (EMBEDDING, (1, 16)),
    (RBF, (2, 4)),
    (RBF, (1,)),
    (CLASS_BIAS, (63,)),
    (CLASS_BIAS, (2,)),
    (CLASS_BIAS, ()),
])
def test_unsupported_shape_layout_rejected(audited_state, key, shape):
    state = dict(audited_state)
    state[key] = torch.zeros(shape, dtype=torch.float32)
    with pytest.raises(ValueError, match="Unsupported model"):
        infer_model_shape(state)


def test_absent_or_noncontiguous_attention_layers_rejected(audited_state):
    without_layers = {key: value for key, value in audited_state.items()
                      if not key.startswith("representation_model.attention_layers.")}
    with pytest.raises(ValueError, match="Unsupported model"):
        infer_model_shape(without_layers)
    gap = {key.replace("attention_layers.0.", "attention_layers.1."): value
           for key, value in audited_state.items()}
    with pytest.raises(ValueError, match="Unsupported model"):
        infer_model_shape(gap)


def test_shape_inference_does_not_substitute_for_strict_load(audited_state, tmp_path):
    state = dict(audited_state)
    state["representation_model.attention_layers.0.q_proj.weight"] = torch.zeros(1, 1)
    # The dedicated shape tensors still describe a plausible model, but an
    # inconsistent individual parameter must fail strict state_dict loading.
    assert infer_model_shape(state).hidden_channels == 16
    checkpoint = tmp_path / "inconsistent.pth"
    torch.save(state, checkpoint)
    backend = AppleSiliconBackend(SOURCE, checkpoint)
    request = ElementInferenceRequest(("a", "b"), (6, 8), ((0., 0., 0.), (1., 0., 0.)), (6, 8))
    with pytest.raises(RuntimeError, match="size mismatch"):
        backend.predict_elements(request, device="cpu")


def test_audited_checkpoint_has_only_float32_state(audited_state):
    assert all(value.dtype == torch.float32 for value in audited_state.values())
    shape = infer_model_shape(audited_state)
    assert (shape.hidden_channels, shape.num_classes, shape.num_layers, shape.num_rbf, shape.max_z) == (16, 32, 1, 8, 100)
