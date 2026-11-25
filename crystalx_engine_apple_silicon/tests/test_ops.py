"""Mathematical operation tests; these do not establish checkpoint parity."""

from dataclasses import FrozenInstanceError
import math

import pytest
import torch

from crystalx_engine_apple_silicon.ops import (
    NEIGHBOR_SELECTION_POLICY,
    radius_graph,
    radius_graph_with_diagnostics,
    scatter,
)


def reference_edges(points, radius, *, loop=False, limit=32, batch=None):
    edges = []
    for target, center in enumerate(points):
        candidates = []
        for source, point in enumerate(points):
            if not loop and source == target:
                continue
            if batch is not None and batch[source] != batch[target]:
                continue
            distance = sum((a - b) ** 2 for a, b in zip(point, center))
            if distance < radius * radius:
                candidates.append((distance, source))
        for _, source in sorted(candidates)[:limit]:
            edges.append((source, target))
    return torch.tensor(edges, dtype=torch.long).T.contiguous() if edges else torch.empty((2, 0), dtype=torch.long)


def test_scatter_scalar_sum_and_add():
    src = torch.tensor([1.0, 2.0, 4.0, 8.0])
    index = torch.tensor([2, 0, 2, 0])
    expected = torch.tensor([10.0, 0.0, 5.0, 0.0])
    torch.testing.assert_close(scatter(src, index, dim_size=4), expected)
    torch.testing.assert_close(scatter(src, index, dim_size=4, reduce="add"), expected)


def test_scatter_vector_and_autograd():
    src = torch.arange(12, dtype=torch.float32).reshape(4, 3).requires_grad_()
    result = scatter(src, torch.tensor([1, 0, 1, 0]))
    torch.testing.assert_close(result, torch.stack((src[1] + src[3], src[0] + src[2])))
    result.sum().backward()
    torch.testing.assert_close(src.grad, torch.ones_like(src))


def test_scatter_arbitrary_dim_and_existing_out():
    src = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    out = torch.ones((2, 2))
    result = scatter(src, torch.tensor([1, 0, 1]), dim=-1, out=out)
    assert result is out
    torch.testing.assert_close(result, torch.tensor([[3.0, 5.0], [6.0, 11.0]]))


@pytest.mark.parametrize("shape,dim", [((0,), 0), ((0, 3), 0), ((2, 0, 3), 1)])
def test_scatter_empty(shape, dim):
    src = torch.empty(shape)
    result = scatter(src, torch.empty(0, dtype=torch.long), dim=dim)
    assert result.shape == src.shape
    result = scatter(src, torch.empty(0, dtype=torch.long), dim=dim, dim_size=4)
    expected_shape = list(shape)
    expected_shape[dim] = 4
    assert result.shape == tuple(expected_shape)
    assert not bool(result.any())


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"reduce": "mean"}, "supports only"),
        ({"dim": 2}, "outside"),
        ({"dim_size": 1}, "outside the output"),
        ({"dim_size": -1}, "dim_size"),
        ({"dim_size": True}, "integer"),
        ({"out": torch.zeros(2, 2)}, "out shape"),
        ({"out": torch.zeros(2, dtype=torch.float64)}, "dtype"),
        ({"out": torch.zeros(2), "dim_size": 3}, "conflicts"),
    ],
)
def test_scatter_rejects_unsupported_or_inconsistent_arguments(kwargs, match):
    with pytest.raises((ValueError, TypeError), match=match):
        scatter(torch.ones(2), torch.tensor([0, 1]), **kwargs)


@pytest.mark.parametrize("index", [torch.tensor([-1, 0]), torch.tensor([0]), torch.tensor([0.0, 1.0]), torch.tensor([[0, 1]])])
def test_scatter_rejects_invalid_indices(index):
    with pytest.raises(ValueError):
        scatter(torch.ones(2), index)


def test_scatter_rejects_scalar_src():
    with pytest.raises(ValueError, match="at least one"):
        scatter(torch.tensor(2.0), torch.tensor([0]))


@pytest.mark.parametrize("loop", [False, True])
@pytest.mark.parametrize("limit", [0, 1, 3, 32])
def test_radius_graph_matches_independent_ground_truth(loop, limit):
    points = [[0.0, 0.0, 0.0], [0.3, 0.4, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-0.5, 0.0, 0.0]]
    edge, diag = radius_graph_with_diagnostics(torch.tensor(points), 1.0, loop=loop, max_num_neighbors=limit, chunk_size=2)
    torch.testing.assert_close(edge, reference_edges(points, 1.0, loop=loop, limit=limit))
    assert diag.policy == NEIGHBOR_SELECTION_POLICY
    assert diag.node_count == 5
    assert diag.edge_count == edge.shape[1]
    assert diag.cutoff_angstrom == 1.0
    assert diag.neighbor_limit == limit


def test_radius_distinct_coincident_nodes_are_not_silently_removed():
    x = torch.zeros((2, 3))
    torch.testing.assert_close(radius_graph(x, 0.1), torch.tensor([[1, 0], [0, 1]]))


def test_radius_cutoff_is_strict():
    x = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    assert radius_graph(x, 1.0).shape == (2, 0)
    assert radius_graph(x, 1.001).shape == (2, 2)


def test_radius_unsorted_gapped_batches_and_reverse_flow():
    points = [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0], [0.3, 0.0, 0.0]]
    batch = torch.tensor([3, 0, 3, 0])
    forward = radius_graph(torch.tensor(points), 1.0, batch=batch, batch_size=4)
    torch.testing.assert_close(forward, reference_edges(points, 1.0, batch=batch.tolist()))
    backward = radius_graph(torch.tensor(points), 1.0, batch=batch, flow="target_to_source")
    torch.testing.assert_close(backward, forward.flip(0))


def test_radius_truncation_ties_policy_and_diagnostics_are_explicit():
    x = torch.zeros((40, 3))
    edge, diag = radius_graph_with_diagnostics(x, 5.0)
    assert edge.shape == (2, 40 * 32)
    assert diag.truncated_nodes == 40
    assert diag.max_neighbors_observed == 39
    torch.testing.assert_close(edge[0, edge[1] == 39], torch.arange(32))
    torch.testing.assert_close(edge[0, edge[1] == 0], torch.arange(1, 33))
    with pytest.raises(FrozenInstanceError):
        diag.edge_count = 0


def test_radius_distance_precedes_original_index():
    x = torch.tensor([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    edge = radius_graph(x, 5.0, max_num_neighbors=2)
    torch.testing.assert_close(edge[0, edge[1] == 0], torch.tensor([2, 3]))


def test_radius_self_loop_counts_toward_cap():
    x = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
    edge, diag = radius_graph_with_diagnostics(x, 1.0, loop=True, max_num_neighbors=1)
    torch.testing.assert_close(edge, torch.tensor([[0, 1], [0, 1]]))
    assert diag.max_neighbors_observed == 2
    assert diag.truncated_nodes == 2


def test_radius_empty_graph():
    edge, diag = radius_graph_with_diagnostics(torch.empty((0, 3)), 5.0, batch=torch.empty(0, dtype=torch.long), batch_size=0)
    assert edge.shape == (2, 0)
    assert edge.dtype == torch.long
    assert diag.edge_count == diag.node_count == diag.max_neighbors_observed == diag.truncated_nodes == 0


def test_radius_chunk_limits_do_not_change_result():
    x = torch.arange(33, dtype=torch.float32).reshape(11, 3) / 9
    normal = radius_graph_with_diagnostics(x, 1.5, max_num_neighbors=3)
    bounded = radius_graph_with_diagnostics(x, 1.5, max_num_neighbors=3, chunk_size=8, max_pairwise_elements=11)
    torch.testing.assert_close(normal[0], bounded[0])
    assert normal[1] == bounded[1]


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"r": 0.0}, "positive"),
        ({"r": -1.0}, "positive"),
        ({"r": float("nan")}, "finite"),
        ({"r": float("inf")}, "finite"),
        ({"r": True}, "real number"),
        ({"r": 1e30}, "r squared"),
        ({"r": 1e-40}, "r squared"),
        ({"loop": 1}, "bool"),
        ({"flow": "invalid"}, "flow"),
        ({"max_num_neighbors": -1}, "max_num_neighbors"),
        ({"max_num_neighbors": 1.5}, "integer"),
        ({"num_workers": 0}, "num_workers"),
        ({"batch_size": 0}, "batch_size"),
        ({"chunk_size": 0}, "chunk_size"),
        ({"max_nodes": 1}, "exceeds max_nodes"),
        ({"max_pairwise_elements": 1}, "neighbor row"),
    ],
)
def test_radius_rejects_invalid_options(kwargs, match):
    options = {"r": 1.0, **kwargs}
    with pytest.raises((ValueError, TypeError), match=match):
        radius_graph(torch.zeros((2, 3)), **options)


@pytest.mark.parametrize("x", [torch.zeros(3), torch.zeros(2, 2), torch.ones((2, 3), dtype=torch.long), torch.tensor([[math.nan, 0, 0]]), torch.tensor([[math.inf, 0, 0]])])
def test_radius_rejects_bad_coordinates(x):
    with pytest.raises(ValueError):
        radius_graph(x, 1.0)


@pytest.mark.parametrize("batch", [torch.tensor([-1, 0]), torch.tensor([0]), torch.tensor([0.0, 1.0]), torch.tensor([[0, 1]])])
def test_radius_rejects_bad_batches(batch):
    with pytest.raises(ValueError):
        radius_graph(torch.zeros((2, 3)), 1.0, batch=batch)


def test_radius_rejects_batch_size_smaller_than_gapped_ids():
    with pytest.raises(ValueError, match="batch_size"):
        radius_graph(torch.zeros((2, 3)), 1.0, batch=torch.tensor([0, 4]), batch_size=2)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is not available in this test environment")
def test_mps_scatter_matches_cpu_and_backpropagates():
    src = torch.arange(15, dtype=torch.float32).reshape(5, 3)
    index = torch.tensor([0, 2, 0, 1, 2])
    mps_src = src.to("mps").requires_grad_()
    actual = scatter(mps_src, index.to("mps"), dim_size=4)
    torch.testing.assert_close(actual.cpu(), scatter(src, index, dim_size=4))
    actual.sum().backward()
    torch.testing.assert_close(mps_src.grad.cpu(), torch.ones_like(src))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is not available in this test environment")
def test_mps_graph_matches_cpu_away_from_cutoff():
    x = torch.tensor([[0.0, 0.0, 0.0], [0.2, 0.3, 0.1], [0.0, 0.0, 0.0], [0.5, 0.4, 0.0], [8.0, 8.0, 8.0]])
    batch = torch.tensor([0, 0, 0, 0, 1])
    cpu_edges, cpu_diag = radius_graph_with_diagnostics(x, 2.0, batch=batch, loop=True, max_num_neighbors=3, chunk_size=2)
    mps_edges, mps_diag = radius_graph_with_diagnostics(x.to("mps"), 2.0, batch=batch.to("mps"), loop=True, max_num_neighbors=3, chunk_size=2)
    torch.testing.assert_close(mps_edges.cpu(), cpu_edges)
    assert mps_diag == cpu_diag
