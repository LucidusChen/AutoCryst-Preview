"""Native PyTorch graph operations for the Apple Silicon backend.

The neighbor cap has an explicit, deterministic policy: nearest distance first,
then original node index.  This is *not* equivalent to the implementation-defined
truncation order of torch_cluster on CPU or CUDA.  Checkpoint prediction parity
must therefore be measured on representative datasets, not inferred from these
operation-level tests.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

import torch


NEIGHBOR_SELECTION_POLICY = "nearest-distance-then-original-index-v1"


@dataclass(frozen=True)
class GraphDiagnostics:
    """Observed topology, including candidates removed by the neighbor cap."""

    policy: str
    node_count: int
    edge_count: int
    truncated_nodes: int
    max_neighbors_observed: int
    cutoff_angstrom: float
    neighbor_limit: int


def _integer(name: str, value: object, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def scatter(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = 0,
    out: torch.Tensor | None = None,
    dim_size: int | None = None,
    reduce: str = "sum",
) -> torch.Tensor:
    """Sum slices using native ``index_add_``; only 1-D indices are supported.

    This deliberately implements the scalar/vector aggregation used by CrystalX,
    not the complete torch_scatter API.  Existing ``out`` contents are preserved
    and added to.  No implicit device transfer or reduction substitution occurs.
    """

    if not isinstance(src, torch.Tensor) or not isinstance(index, torch.Tensor):
        raise TypeError("src and index must be torch tensors")
    if reduce not in ("sum", "add"):
        raise ValueError("native scatter supports only reduce='sum' or 'add'")
    if src.ndim == 0:
        raise ValueError("src must have at least one dimension")
    if isinstance(dim, bool) or not isinstance(dim, int):
        raise TypeError("dim must be an integer")
    if not -src.ndim <= dim < src.ndim:
        raise ValueError("dim is outside src dimensions")
    dim %= src.ndim
    if index.ndim != 1 or index.dtype != torch.long:
        raise ValueError("index must be a one-dimensional torch.long tensor")
    if index.device != src.device:
        raise ValueError("index and src must be on the same device")
    if index.numel() != src.shape[dim]:
        raise ValueError("index length must equal src.shape[dim]")
    if index.numel() and bool((index < 0).any().item()):
        raise ValueError("index must not contain negative values")
    inferred_size = int(index.max().item()) + 1 if index.numel() else 0
    if dim_size is not None:
        dim_size = _integer("dim_size", dim_size)
    if out is not None:
        if not isinstance(out, torch.Tensor):
            raise TypeError("out must be a torch tensor")
        if out.device != src.device or out.dtype != src.dtype:
            raise ValueError("out must have the same device and dtype as src")
        if out.ndim != src.ndim or any(
            out.shape[axis] != src.shape[axis]
            for axis in range(src.ndim)
            if axis != dim
        ):
            raise ValueError("out shape must match src except along dim")
        if dim_size is not None and dim_size != out.shape[dim]:
            raise ValueError("dim_size conflicts with out.shape[dim]")
        target_size = out.shape[dim]
    else:
        target_size = inferred_size if dim_size is None else dim_size
    if inferred_size > target_size:
        raise ValueError("index contains a value outside the output dimension")
    if out is None:
        shape = list(src.shape)
        shape[dim] = target_size
        out = src.new_zeros(shape)
    return out.index_add_(dim, index, src)


def radius_graph(
    x: torch.Tensor,
    r: float,
    batch: torch.Tensor | None = None,
    loop: bool = False,
    max_num_neighbors: int = 32,
    flow: str = "source_to_target",
    num_workers: int = 1,
    batch_size: int | None = None,
    *,
    chunk_size: int = 128,
    max_nodes: int = 8192,
    max_pairwise_elements: int = 1_000_000,
) -> torch.Tensor:
    """Return a capped radius graph; see ``radius_graph_with_diagnostics``."""

    edge_index, _ = radius_graph_with_diagnostics(
        x,
        r,
        batch,
        loop,
        max_num_neighbors,
        flow,
        num_workers,
        batch_size,
        chunk_size=chunk_size,
        max_nodes=max_nodes,
        max_pairwise_elements=max_pairwise_elements,
    )
    return edge_index


@torch.no_grad()
def radius_graph_with_diagnostics(
    x: torch.Tensor,
    r: float,
    batch: torch.Tensor | None = None,
    loop: bool = False,
    max_num_neighbors: int = 32,
    flow: str = "source_to_target",
    num_workers: int = 1,
    batch_size: int | None = None,
    *,
    chunk_size: int = 128,
    max_nodes: int = 8192,
    max_pairwise_elements: int = 1_000_000,
) -> tuple[torch.Tensor, GraphDiagnostics]:
    """Construct edges with a strict distance cutoff and explicit truncation.

    In ``source_to_target`` flow each column is ``[neighbor, center]``.  Centers
    follow original node order; their neighbors are sorted by distance and then
    original index.  ``loop`` counts self edges toward the cap.  Distinct nodes
    at identical coordinates remain neighbors: callers must handle zero-length
    direction vectors separately.  Batch IDs may be unsorted or contain gaps.

    The algorithm performs quadratic work but bounds the pairwise workspace by
    the chunk and element limits.  ``max_nodes`` independently prevents accidental
    unbounded quadratic work; larger inputs require an explicit caller override.
    ``num_workers`` is accepted for API compatibility, while PyTorch controls its
    own execution scheduling.  Topology is discrete and carries no gradients.
    """

    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a torch tensor")
    if x.ndim != 2 or x.shape[1] != 3:
        raise ValueError("x must have shape (N, 3)")
    if not x.is_floating_point():
        raise ValueError("x must use a floating-point dtype")
    if not bool(torch.isfinite(x).all().item()):
        raise ValueError("x must contain only finite coordinates")
    if isinstance(r, bool) or not isinstance(r, Real):
        raise TypeError("r must be a real number")
    radius = float(r)
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("r must be finite and positive")
    if not isinstance(loop, bool):
        raise TypeError("loop must be a bool")
    if flow not in ("source_to_target", "target_to_source"):
        raise ValueError("flow must be 'source_to_target' or 'target_to_source'")
    neighbor_limit = _integer("max_num_neighbors", max_num_neighbors)
    _integer("num_workers", num_workers, minimum=1)
    chunk_size = _integer("chunk_size", chunk_size, minimum=1)
    max_nodes = _integer("max_nodes", max_nodes, minimum=1)
    max_pairwise_elements = _integer(
        "max_pairwise_elements", max_pairwise_elements, minimum=1
    )
    if batch_size is not None:
        batch_size = _integer("batch_size", batch_size)
    node_count = x.shape[0]
    if node_count > max_nodes:
        raise ValueError(
            f"node count {node_count} exceeds max_nodes={max_nodes}; "
            "explicitly raise the limit only after budgeting quadratic work"
        )
    if node_count > max_pairwise_elements:
        raise ValueError(
            "max_pairwise_elements cannot hold even one complete neighbor row"
        )
    if batch is not None:
        if not isinstance(batch, torch.Tensor):
            raise TypeError("batch must be a torch tensor")
        if batch.ndim != 1 or batch.shape[0] != node_count:
            raise ValueError("batch must have shape (N,)")
        if batch.dtype != torch.long:
            raise ValueError("batch must use torch.long")
        if batch.device != x.device:
            raise ValueError("batch and x must be on the same device")
        if batch.numel() and bool((batch < 0).any().item()):
            raise ValueError("batch IDs must be nonnegative")
        required_batches = int(batch.max().item()) + 1 if node_count else 0
    else:
        required_batches = 1 if node_count else 0
    if batch_size is not None and batch_size < required_batches:
        raise ValueError("batch_size does not include every batch ID")
    if node_count == 0:
        return torch.empty((2, 0), dtype=torch.long, device=x.device), GraphDiagnostics(
            NEIGHBOR_SELECTION_POLICY, 0, 0, 0, 0, radius, neighbor_limit
        )

    # Broadcast subtraction avoids cdist's matrix-multiplication cancellation at
    # nearby coordinates.  Promote low-precision input to float32, not float64:
    # MPS does not support float64 tensors.
    work = x if x.dtype in (torch.float32, torch.float64) else x.float()
    squared_cutoff = torch.tensor(radius * radius, dtype=work.dtype, device=x.device)
    if not bool(torch.isfinite(squared_cutoff).item()) or squared_cutoff.item() == 0:
        raise ValueError("r squared must be finite and nonzero in the coordinate dtype")
    effective_chunk = min(chunk_size, max_pairwise_elements // node_count)
    node_indices = torch.arange(node_count, device=x.device)
    pieces: list[torch.Tensor] = []
    truncated_nodes = 0
    max_neighbors_observed = 0
    for start in range(0, node_count, effective_chunk):
        stop = min(start + effective_chunk, node_count)
        centers = node_indices[start:stop]
        offsets = work[start:stop, None, :] - work[None, :, :]
        squared_distance = (offsets * offsets).sum(dim=-1)
        eligible = squared_distance < squared_cutoff
        if not loop:
            eligible &= centers[:, None] != node_indices[None, :]
        if batch is not None:
            eligible &= batch[start:stop, None] == batch[None, :]
        counts = eligible.sum(dim=1)
        max_neighbors_observed = max(
            max_neighbors_observed, int(counts.max().item())
        )
        truncated_nodes += int((counts > neighbor_limit).sum().item())
        if neighbor_limit == 0:
            continue
        # Stable sorting retains ascending original index for exact distance ties.
        ranked = torch.argsort(
            squared_distance.masked_fill(~eligible, float("inf")),
            dim=1,
            stable=True,
        )[:, : min(neighbor_limit, node_count)]
        selected = eligible.gather(1, ranked)
        sources = ranked[selected]
        targets = centers[:, None].expand_as(ranked)[selected]
        pieces.append(torch.stack((sources, targets), dim=0))
    edge_index = (
        torch.cat(pieces, dim=1)
        if pieces
        else torch.empty((2, 0), dtype=torch.long, device=x.device)
    )
    if flow == "target_to_source":
        edge_index = edge_index.flip(0)
    return edge_index, GraphDiagnostics(
        policy=NEIGHBOR_SELECTION_POLICY,
        node_count=node_count,
        edge_count=edge_index.shape[1],
        truncated_nodes=truncated_nodes,
        max_neighbors_observed=max_neighbors_observed,
        cutoff_angstrom=radius,
        neighbor_limit=neighbor_limit,
    )
