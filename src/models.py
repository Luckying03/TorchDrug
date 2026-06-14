"""MindSpore GIN / GAT models, with a TorchDrug-like reproduction variant."""

from __future__ import annotations

from typing import List

import numpy as np
import mindspore as ms
from mindspore import Tensor, nn, ops


class MLP(nn.Cell):
    """TorchDrug-style MLP: no activation / BN / dropout after the last layer."""

    def __init__(self, input_dim: int, hidden_dims: List[int], dropout: float = 0.0):
        super().__init__()
        dims = [input_dim] + list(hidden_dims)
        self.layers = nn.CellList([nn.Dense(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None

    def construct(self, x):
        hidden = x
        for i, layer in enumerate(self.layers):
            hidden = layer(hidden)
            if i < len(self.layers) - 1:
                hidden = self.activation(hidden)
                if self.dropout is not None:
                    hidden = self.dropout(hidden)
        return hidden


class NodeBatchNorm(nn.Cell):
    """BatchNorm over padded node tensors of shape [batch, num_node, dim]."""

    def __init__(self, dim: int):
        super().__init__()
        self.bn = nn.BatchNorm1d(dim)

    def construct(self, x):
        batch_size, num_node, dim = x.shape
        flat = ops.reshape(x, (batch_size * num_node, dim))
        flat = self.bn(flat)
        return ops.reshape(flat, (batch_size, num_node, dim))


class TorchDrugGINLayer(nn.Cell):
    """GIN convolution aligned with TorchDrug's GraphIsomorphismConv."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        edge_input_dim: int,
        num_mlp_layer: int = 2,
        batch_norm: bool = True,
    ):
        super().__init__()
        self.eps = Tensor(np.asarray([0.0], dtype=np.float32), ms.float32)
        hidden_dims = [output_dim] * (num_mlp_layer - 1) + [output_dim]
        self.mlp = MLP(input_dim, hidden_dims)
        self.edge_linear = nn.Dense(edge_input_dim, input_dim) if edge_input_dim > 0 else None
        self.batch_norm = NodeBatchNorm(output_dim) if batch_norm else None
        self.activation = nn.ReLU()
        self.reduce_sum = ops.ReduceSum(keep_dims=False)

    def construct(self, node_feature, adjacency, edge_feature, node_mask):
        neighbor_feature = ops.matmul(adjacency, node_feature)
        if self.edge_linear is not None:
            edge_message = self.edge_linear(edge_feature)
            edge_message = edge_message * ops.expand_dims(adjacency, -1)
            neighbor_feature = neighbor_feature + self.reduce_sum(edge_message, 2)

        output = self.mlp((1.0 + self.eps) * node_feature + neighbor_feature)
        if self.batch_norm is not None:
            output = self.batch_norm(output)
        output = self.activation(output)
        return output * ops.expand_dims(node_mask, -1)


class TorchDrugGATLayer(nn.Cell):
    """GAT convolution aligned with TorchDrug's GraphAttentionConv."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        edge_input_dim: int,
        num_head: int = 4,
        negative_slope: float = 0.2,
        batch_norm: bool = True,
    ):
        super().__init__()
        if output_dim % num_head != 0:
            raise ValueError("output_dim must be divisible by num_head")
        self.output_dim = output_dim
        self.num_head = num_head
        self.head_dim = output_dim // num_head
        self.negative_slope = negative_slope

        self.linear = nn.Dense(input_dim, output_dim)
        self.edge_linear = nn.Dense(edge_input_dim, output_dim) if edge_input_dim > 0 else None
        query = np.random.normal(0, 0.1, size=(num_head, self.head_dim * 2)).astype(np.float32)
        self.query = ms.Parameter(Tensor(query, ms.float32), name="query")
        self.batch_norm = NodeBatchNorm(output_dim) if batch_norm else None
        self.activation = nn.ReLU()
        self.softmax = ops.Softmax(axis=-1)
        self.reduce_sum = ops.ReduceSum(keep_dims=False)

    def construct(self, node_feature, adjacency, edge_feature, node_mask):
        batch_size, num_node, _ = node_feature.shape
        hidden = self.linear(node_feature)
        hidden = ops.reshape(hidden, (batch_size, num_node, self.num_head, self.head_dim))
        hidden = ops.transpose(hidden, (0, 2, 1, 3))

        source_hidden = ops.expand_dims(hidden, 2)
        target_hidden = ops.expand_dims(hidden, 3)
        if self.edge_linear is not None:
            projected_edge = self.edge_linear(edge_feature)
            projected_edge = projected_edge * ops.expand_dims(adjacency, -1)
            projected_edge = ops.reshape(projected_edge, (batch_size, num_node, num_node, self.num_head, self.head_dim))
            projected_edge = ops.transpose(projected_edge, (0, 3, 1, 2, 4))
            source_hidden = source_hidden + projected_edge
            target_hidden = target_hidden + projected_edge

        q_source = ops.reshape(self.query[:, :self.head_dim], (1, self.num_head, 1, 1, self.head_dim))
        q_target = ops.reshape(self.query[:, self.head_dim:], (1, self.num_head, 1, 1, self.head_dim))
        score = self.reduce_sum(source_hidden * q_source + target_hidden * q_target, -1)
        score = ops.maximum(score, score * self.negative_slope)

        eye = ops.eye(num_node, num_node, ms.float32)
        adjacency_with_self = adjacency + ops.expand_dims(eye, 0)
        adjacency_with_self = ops.minimum(adjacency_with_self, ops.ones_like(adjacency_with_self))
        valid_pair = ops.expand_dims(node_mask, 1) * ops.expand_dims(node_mask, 2)
        edge_mask = ops.expand_dims(adjacency_with_self * valid_pair, 1)
        score = score + (1.0 - edge_mask) * -1e9

        attention = self.softmax(score)
        output = ops.matmul(attention, hidden)
        output = ops.transpose(output, (0, 2, 1, 3))
        output = ops.reshape(output, (batch_size, num_node, self.output_dim))
        if self.batch_norm is not None:
            output = self.batch_norm(output)
        output = self.activation(output)
        return output * ops.expand_dims(node_mask, -1)


class GraphClassifier(nn.Cell):
    def __init__(
        self,
        layers: List[nn.Cell],
        hidden_dim: int,
        num_layer: int,
        concat_hidden: bool,
        short_cut: bool,
        readout: str,
        num_mlp_layer: int,
        dropout: float,
    ):
        super().__init__()
        self.layers = nn.CellList(layers)
        self.concat_hidden = concat_hidden
        self.short_cut = short_cut
        self.readout = readout
        graph_dim = hidden_dim * num_layer if concat_hidden else hidden_dim
        if num_mlp_layer <= 1:
            self.classifier = nn.Dense(graph_dim, 1)
        else:
            self.classifier = MLP(graph_dim, [graph_dim] * (num_mlp_layer - 1) + [1], dropout=dropout)
        self.concat = ops.Concat(axis=-1)
        self.reduce_sum = ops.ReduceSum(keep_dims=False)

    def construct(self, node_feature, adjacency, edge_feature, node_mask):
        h = node_feature
        hiddens = []
        for layer in self.layers:
            hidden = layer(h, adjacency, edge_feature, node_mask)
            if self.short_cut and hidden.shape == h.shape:
                hidden = hidden + h
            hiddens.append(hidden)
            h = hidden

        node_output = self.concat(hiddens) if self.concat_hidden else hiddens[-1]
        graph_feature = self.pool(node_output, node_mask)
        return self.classifier(graph_feature)

    def pool(self, node_feature, node_mask):
        mask = ops.expand_dims(node_mask, -1)
        summed = self.reduce_sum(node_feature * mask, 1)
        if self.readout == "sum":
            return summed
        count = self.reduce_sum(mask, 1)
        count = ops.maximum(count, ops.ones_like(count))
        return summed / count


def build_model(
    model_name: str,
    input_dim: int,
    edge_input_dim: int,
    hidden_dim: int,
    num_layer: int,
    num_head: int,
    dropout: float,
    variant: str = "torchdrug_like",
    readout: str = "sum",
    num_mlp_layer: int = 1,
):
    if model_name not in {"gin", "gat"}:
        raise ValueError(f"Unknown model `{model_name}`")

    if variant == "simple":
        readout = "mean"
        short_cut = False
        batch_norm = False
        concat_hidden = False
        use_edge_dim = 0
        head_layers = 2
    elif variant == "torchdrug_like":
        short_cut = True
        batch_norm = True
        concat_hidden = True
        use_edge_dim = edge_input_dim
        head_layers = num_mlp_layer
    else:
        raise ValueError(f"Unknown variant `{variant}`")

    layers = []
    dim = input_dim
    for _ in range(num_layer):
        if model_name == "gin":
            layers.append(TorchDrugGINLayer(dim, hidden_dim, use_edge_dim, batch_norm=batch_norm))
        else:
            layers.append(TorchDrugGATLayer(dim, hidden_dim, use_edge_dim, num_head=num_head, batch_norm=batch_norm))
        dim = hidden_dim

    return GraphClassifier(
        layers=layers,
        hidden_dim=hidden_dim,
        num_layer=num_layer,
        concat_hidden=concat_hidden,
        short_cut=short_cut,
        readout=readout,
        num_mlp_layer=head_layers,
        dropout=dropout,
    )
