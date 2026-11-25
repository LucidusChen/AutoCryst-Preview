"""Snapshot compatibility tests; no weights or proprietary programs are used."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from crystalx_engine_apple_silicon.source import (
    AUDITED_SHA256,
    SourceRevisionError,
    load_model_classes,
)


@pytest.fixture
def source_root():
    root = Path(__file__).resolve().parents[2] / "crystalx_engine" / "inference_repo"
    if not (root / "crystalx_infer" / "models" / "torchmd_et.py").exists():
        pytest.skip("Local upstream inference snapshot is not distributed with this adapter")
    return root


def _copy_sources(source_root, tmp_path):
    model_dir = tmp_path / "crystalx_infer" / "models"
    model_dir.mkdir(parents=True)
    for filename in AUDITED_SHA256:
        shutil.copyfile(source_root / "crystalx_infer" / "models" / filename, model_dir / filename)
    return tmp_path


def _tiny_model(loaded):
    representation = loaded.TorchMD_ET(
        hidden_channels=8, num_layers=1, num_rbf=4, num_heads=2,
        max_num_neighbors=32, max_z=100, layernorm_on_vec=None,
    )
    head = loaded.EquivariantScalar(hidden_channels=8, num_classes=5)
    return loaded.TorchMD_Net(representation, head).eval()


def test_all_snapshot_sources_verified_without_mutating_originals(source_root):
    pytest.importorskip("torch")
    before = {filename: (source_root / "crystalx_infer" / "models" / filename).read_bytes()
              for filename in AUDITED_SHA256}
    loaded = load_model_classes(source_root)
    assert dict(loaded.identity) == dict(AUDITED_SHA256)
    assert loaded.source_root == source_root.resolve()
    for filename, data in before.items():
        assert hashlib.sha256(data).hexdigest() == AUDITED_SHA256[filename]
        assert (source_root / "crystalx_infer" / "models" / filename).read_bytes() == data
    with pytest.raises(TypeError):
        loaded.identity["torchmd_et.py"] = "changed"


@pytest.mark.parametrize("filename", tuple(AUDITED_SHA256))
def test_each_modified_source_rejected_before_snapshot_execution(source_root, tmp_path, filename):
    root = _copy_sources(source_root, tmp_path)
    path = root / "crystalx_infer" / "models" / filename
    path.write_bytes(path.read_bytes() + b"\nraise RuntimeError('must never execute')\n")
    with pytest.raises(SourceRevisionError, match="Unsupported source revision"):
        load_model_classes(root)


@pytest.mark.parametrize("filename", tuple(AUDITED_SHA256))
def test_each_missing_source_rejected(source_root, tmp_path, filename):
    root = _copy_sources(source_root, tmp_path)
    (root / "crystalx_infer" / "models" / filename).unlink()
    with pytest.raises(SourceRevisionError, match="Cannot read required inference source"):
        load_model_classes(root)


def test_inference_forward_is_finite_and_state_dict_compatible(source_root):
    torch = pytest.importorskip("torch")
    torch.manual_seed(17)
    loaded = load_model_classes(source_root)
    model = _tiny_model(loaded)
    z = torch.tensor([6, 7, 8, 47], dtype=torch.long)
    pos = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [9., 9., 9.]])
    with torch.inference_mode():
        result = model(z, pos)
    assert result.shape == (4, 5)
    assert torch.isfinite(result).all()
    # The unused whitened branch needs float64/SVD and must not be enabled for MPS.
    assert model.representation_model.layernorm_on_vec is None
    state = model.state_dict()
    assert state["representation_model.embedding.weight"].shape == (100, 8)
    assert state["representation_model.neighbor_embedding.embedding.weight"].shape == (100, 8)
    assert state["representation_model.attention_layers.0.q_proj.weight"].shape == (8, 8)
    assert state["output_model.output_network.1.update_net.2.weight"].shape == (10, 4)
    assert not any("_cx_" in key for key in state)
    another = _tiny_model(load_model_classes(source_root))
    another.load_state_dict(state, strict=True)
    with torch.inference_mode():
        torch.testing.assert_close(another(z, pos), result)


def test_no_global_namespace_or_cross_instance_graph_injection(source_root):
    torch = pytest.importorskip("torch")
    from crystalx_engine_apple_silicon.ops import radius_graph

    calls = []

    def graph_one(*args, **kwargs):
        calls.append("one")
        return radius_graph(*args, **kwargs)

    def graph_two(*args, **kwargs):
        calls.append("two")
        return radius_graph(*args, **kwargs)

    before = {key: value for key, value in sys.modules.items() if key.startswith("crystalx_infer")}
    one = load_model_classes(source_root, radius_graph_fn=graph_one)
    two = load_model_classes(source_root, radius_graph_fn=graph_two)
    assert one.TorchMD_ET.__module__ != two.TorchMD_ET.__module__
    assert one.TorchMD_ET.__module__ not in sys.modules
    assert two.TorchMD_ET.__module__ not in sys.modules
    models = (_tiny_model(one), _tiny_model(two), _tiny_model(one))
    for model in models:
        with torch.inference_mode():
            model(torch.tensor([6, 8]), torch.tensor([[0., 0., 0.], [1., 0., 0.]]))
    assert calls == ["one", "two", "one"]
    assert before == {key: value for key, value in sys.modules.items() if key.startswith("crystalx_infer")}


def test_loader_does_not_import_pyg_extensions_ase_or_upstream_package(source_root):
    pytest.importorskip("torch")
    code = r'''
import importlib.abc
import json
from pathlib import Path
import sys

blocked = {"torch_geometric", "torch_cluster", "torch_scatter", "ase", "crystalx_infer"}
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in blocked:
            raise AssertionError("Forbidden dependency import: " + fullname)
sys.meta_path.insert(0, Guard())
from crystalx_engine_apple_silicon.source import load_model_classes
loaded = load_model_classes(Path(sys.argv[1]))
model = loaded.TorchMD_ET(hidden_channels=8, num_layers=1, num_rbf=4, num_heads=2)
assert not any(key.split(".")[0] in blocked for key in sys.modules)
print(json.dumps({"loaded": sorted(loaded.identity), "class": type(model).__name__}))
'''
    env = os.environ.copy()
    package_src = Path(__file__).resolve().parents[1] / "src"
    env["PYTHONPATH"] = str(package_src) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run([sys.executable, "-c", code, str(source_root)], env=env,
                            capture_output=True, text=True, timeout=60, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)["loaded"] == sorted(AUDITED_SHA256)


def test_message_passing_preserves_trailing_isolated_node():
    torch = pytest.importorskip("torch")
    from crystalx_engine_apple_silicon.message_passing import MessagePassing

    class Neighbor(MessagePassing):
        def message(self, x_j, W):
            return x_j * W

    layer = Neighbor()
    edges = torch.tensor([[0, 2], [1, 1]], dtype=torch.long)
    x = torch.tensor([[1., 2.], [3., 4.], [5., 6.], [7., 8.]])
    result = layer.propagate(edges, x=x, W=torch.ones(2, 2), size=None)
    torch.testing.assert_close(result, torch.tensor([[0., 0.], [6., 8.], [0., 0.], [0., 0.]]))
    assert layer.jittable() is layer
    assert not layer.state_dict()


def test_message_passing_preserves_nodes_with_no_edges():
    torch = pytest.importorskip("torch")
    from crystalx_engine_apple_silicon.message_passing import MessagePassing

    class Neighbor(MessagePassing):
        def message(self, x_j, W):
            return x_j * W

    layer = Neighbor()
    out = layer.propagate(torch.empty((2, 0), dtype=torch.long), x=torch.ones(3, 2), W=torch.ones(0, 2))
    assert out.shape == (3, 2)
    assert not bool(out.any())


def test_attention_pattern_gathers_target_i_source_j_and_uses_tuple_hooks():
    torch = pytest.importorskip("torch")
    from crystalx_engine_apple_silicon.message_passing import MessagePassing
    from crystalx_engine_apple_silicon.ops import scatter

    class Attention(MessagePassing):
        def message(self, q_i, k_j, v_j, vec_j, dk, dv, r_ij, d_ij):
            return q_i + k_j + v_j, vec_j

        def aggregate(self, features, index, ptr, dim_size):
            assert ptr is None
            assert dim_size == 4
            return tuple(scatter(value, index, dim=0, dim_size=dim_size) for value in features)

        def update(self, inputs):
            return inputs[0] + 10, inputs[1] + 20

    nodes = torch.tensor([[1.], [2.], [3.], [4.]])
    edges = torch.tensor([[0, 2], [1, 1]], dtype=torch.long)
    scalars, vectors = Attention().propagate(
        edges, q=nodes, k=nodes * 2, v=nodes * 3, vec=nodes[:, None, :],
        dk=None, dv=None, r_ij=torch.ones(2), d_ij=torch.ones(2, 3),
    )
    torch.testing.assert_close(scalars, torch.tensor([[10.], [34.], [10.], [10.]]))
    torch.testing.assert_close(vectors, torch.tensor([[[20.]], [[24.]], [[20.]], [[20.]]]))


def test_message_passing_rejects_unsupported_api():
    torch = pytest.importorskip("torch")
    from crystalx_engine_apple_silicon.message_passing import MessagePassing

    with pytest.raises(NotImplementedError):
        MessagePassing(aggr="mean")
    with pytest.raises(NotImplementedError):
        MessagePassing(node_dim=-2)

    class Other(MessagePassing):
        def message(self, x_i, x_j):
            return x_i + x_j

    with pytest.raises(NotImplementedError, match="signature"):
        Other().propagate(torch.empty((2, 0), dtype=torch.long), x=torch.ones(3, 2))
