"""MindSpore implementations of simplified GIN and GAT graph classifiers."""

from __future__ import annotations

import numpy as np
import mindspore as ms
from mindspore import Tensor, nn, ops


class MLP(nn.Cell):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.0):
        super().__init__()
        cells = [
            nn.Dense(input_dim, hidden_dim),
            nn.ReLU(),
        ]
        if dropout > 0:
            cells.append(nn.Dropout(p=dropout))
        cells += [
            nn.Dense(hidden_dim, output_dim),
            nn.ReLU(),
        ]
        self.net = nn.SequentialCell(cells)

    def construct(self, x):
        return self.net(x)


class GINLayer(nn.Cell):
    """GIN layer: MLP((1 + eps) * h_v + sum_{u in N(v)} h_u)."""

    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.0):
        super().__init__()
        self.eps = ms.Parameter(Tensor(np.asarray([0.0], dtype=np.float32)), name="eps")
        self.mlp = MLP(input_dim, output_dim, output_dim, dropout=dropout)

    def construct(self, node_feature, adjacency, node_mask):
        neighbor_feature = ops.matmul(adjacency, node_feature)
        updated = (1.0 + self.eps) * node_feature + neighbor_feature
        updated = self.mlp(updated)
        return updated * ops.expand_dims(node_mask, -1)


class GINClassifier(nn.Cell):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layer: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        layers = []
        dim = input_dim
        for _ in range(num_layer):
            layers.append(GINLayer(dim, hidden_dim, dropout=dropout))
            dim = hidden_dim
        self.layers = nn.CellList(layers)
        self.classifier = nn.SequentialCell([
            nn.Dense(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Dense(hidden_dim, 1),
        ])

    def construct(self, node_feature, adjacency, node_mask):
        h = node_feature
        for layer in self.layers:
            h = layer(h, adjacency, node_mask)
        graph_feature = masked_mean_pool(h, node_mask)
        return self.classifier(graph_feature)


class GATLayer(nn.Cell):
    """Dense masked multi-head graph attention layer."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_head: int = 4,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
    ):
        super().__init__()
        if output_dim % num_head != 0:
            raise ValueError("output_dim must be divisible by num_head")
        self.output_dim = output_dim
        self.num_head = num_head
        self.head_dim = output_dim // num_head
        self.negative_slope = negative_slope

        self.proj = nn.Dense(input_dim, output_dim, has_bias=False)
        self.attn_src = ms.Parameter(
            Tensor(np.random.normal(0, 0.1, size=(num_head, self.head_dim)).astype(np.float32)),
            name="attn_src",
        )
        self.attn_dst = ms.Parameter(
            Tensor(np.random.normal(0, 0.1, size=(num_head, self.head_dim)).astype(np.float32)),
            name="attn_dst",
        )
        self.activation = nn.ELU()
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None
        self.softmax = ops.Softmax(axis=-1)
        self.reduce_sum = ops.ReduceSum(keep_dims=False)

    def construct(self, node_feature, adjacency, node_mask):
        batch_size, num_node, _ = node_feature.shape
        h = self.proj(node_feature)
        h = ops.reshape(h, (batch_size, num_node, self.num_head, self.head_dim))
        h = ops.transpose(h, (0, 2, 1, 3))

        src_score = self.reduce_sum(h * ops.reshape(self.attn_src, (1, self.num_head, 1, self.head_dim)), -1)
        dst_score = self.reduce_sum(h * ops.reshape(self.attn_dst, (1, self.num_head, 1, self.head_dim)), -1)
        score = ops.expand_dims(dst_score, -1) + ops.expand_dims(src_score, -2)
        score = ops.maximum(score, score * self.negative_slope)

        eye = ops.eye(num_node, num_node, ms.float32)
        adjacency_with_self = adjacency + ops.expand_dims(eye, 0)
        adjacency_with_self = ops.minimum(adjacency_with_self, ops.ones_like(adjacency_with_self))
        valid_pair = ops.expand_dims(node_mask, 1) * ops.expand_dims(node_mask, 2)
        edge_mask = ops.expand_dims(adjacency_with_self * valid_pair, 1)
        score = score + (1.0 - edge_mask) * -1e9

        attention = self.softmax(score)
        if self.dropout is not None:
            attention = self.dropout(attention)
        output = ops.matmul(attention, h)
        output = ops.transpose(output, (0, 2, 1, 3))
        output = ops.reshape(output, (batch_size, num_node, self.output_dim))
        output = self.activation(output)
        return output * ops.expand_dims(node_mask, -1)


class GATClassifier(nn.Cell):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layer: int = 4,
        num_head: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        layers = []
        dim = input_dim
        for _ in range(num_layer):
            layers.append(GATLayer(dim, hidden_dim, num_head=num_head, dropout=dropout))
            dim = hidden_dim
        self.layers = nn.CellList(layers)
        self.classifier = nn.SequentialCell([
            nn.Dense(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Dense(hidden_dim, 1),
        ])

    def construct(self, node_feature, adjacency, node_mask):
        h = node_feature
        for layer in self.layers:
            h = layer(h, adjacency, node_mask)
        graph_feature = masked_mean_pool(h, node_mask)
        return self.classifier(graph_feature)


def masked_mean_pool(node_feature, node_mask):
    mask = ops.expand_dims(node_mask, -1)
    summed = ops.ReduceSum(keep_dims=False)(node_feature * mask, 1)
    count = ops.ReduceSum(keep_dims=False)(mask, 1)
    count = ops.maximum(count, ops.ones_like(count))
    return summed / count


def build_model(model_name: str, input_dim: int, hidden_dim: int, num_layer: int, num_head: int, dropout: float):
    if model_name == "gin":
        return GINClassifier(input_dim, hidden_dim=hidden_dim, num_layer=num_layer, dropout=dropout)
    if model_name == "gat":
        return GATClassifier(input_dim, hidden_dim=hidden_dim, num_layer=num_layer, num_head=num_head, dropout=dropout)
    raise ValueError(f"Unknown model `{model_name}`")
