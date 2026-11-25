"""The two message-passing operations used by the pinned inference snapshot.

This is deliberately NOT a replacement for the public PyG MessagePassing API.
Only NeighborEmbedding and EquivariantMultiHeadAttention from the audited
CrystalX inference model are supported. Unsupported signatures fail explicitly.
There are no parameters/buffers here, so upstream state_dict keys are preserved.
"""

from __future__ import annotations

import inspect

import torch
from torch import nn

from .ops import scatter


class MessagePassing(nn.Module):
    """Scoped source-to-target sum propagation with nodes on dimension zero."""

    def __init__(self, aggr="add", node_dim=0, flow="source_to_target", **kwargs):
        super().__init__()
        if kwargs or aggr not in ("add", "sum") or node_dim != 0 or flow != "source_to_target":
            raise NotImplementedError("Only node_dim=0, source_to_target, add/sum propagation is supported")
        self.aggr = aggr
        self.node_dim = node_dim
        self.flow = flow

    def jittable(self):
        """Retain the upstream eager-mode call site; no TorchScript is promised."""
        return self

    def propagate(self, edge_index, size=None, **kwargs):
        if size is not None:
            raise NotImplementedError("Explicit/bipartite sizes are outside the pinned-model scope")
        if not isinstance(edge_index, torch.Tensor) or edge_index.dtype != torch.long:
            raise ValueError("edge_index must be a torch.long tensor")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape (2, E)")

        pattern = tuple(inspect.signature(self.message).parameters)
        if pattern == ("x_j", "W"):
            expected = {"x", "W"}
            nodes = kwargs.get("x")
            node_names = ("x",)
        elif pattern == ("q_i", "k_j", "v_j", "vec_j", "dk", "dv", "r_ij", "d_ij"):
            expected = {"q", "k", "v", "vec", "dk", "dv", "r_ij", "d_ij"}
            nodes = kwargs.get("q")
            node_names = ("q", "k", "v", "vec")
        else:
            raise NotImplementedError(f"Unsupported pinned-model message signature: {pattern!r}")
        if set(kwargs) != expected:
            raise ValueError(f"Unexpected message arguments: expected {sorted(expected)!r}")
        if not isinstance(nodes, torch.Tensor) or nodes.ndim < 2:
            raise ValueError("Node features must have a leading node dimension")
        num_nodes = nodes.shape[0]
        for name in node_names:
            feature = kwargs[name]
            if not isinstance(feature, torch.Tensor) or feature.ndim < 2 or feature.shape[0] != num_nodes:
                raise ValueError(f"{name} does not have the expected node dimension")
            if feature.device != edge_index.device:
                raise ValueError("Node features and edge_index must share a device")
        if edge_index.numel() and (bool((edge_index < 0).any()) or bool((edge_index >= num_nodes).any())):
            raise ValueError("edge_index contains an out-of-range node")

        source, target = edge_index[0], edge_index[1]
        if pattern == ("x_j", "W"):
            messages = self.message(kwargs["x"].index_select(0, source), kwargs["W"])
        else:
            messages = self.message(
                kwargs["q"].index_select(0, target),
                kwargs["k"].index_select(0, source),
                kwargs["v"].index_select(0, source),
                kwargs["vec"].index_select(0, source),
                kwargs["dk"], kwargs["dv"], kwargs["r_ij"], kwargs["d_ij"],
            )
        # Crucially, dim_size is the feature tensor's node count, not max(edge)+1.
        # Isolated nodes (including the last node) must retain an output row.
        aggregated = self.aggregate(messages, target, ptr=None, dim_size=num_nodes)
        return self.update(aggregated)

    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        if ptr is not None:
            raise NotImplementedError("CSR message aggregation is unsupported")
        return scatter(inputs, index, dim=self.node_dim, dim_size=dim_size, reduce="sum")

    def update(self, inputs):
        return inputs
