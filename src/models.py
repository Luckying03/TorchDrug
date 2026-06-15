"""MindSpore edge-list GIN / GAT models aligned with TorchDrug."""

from __future__ import annotations

from typing import List

import numpy as np
import mindspore as ms
from mindspore import Tensor, nn, ops


class MLP(nn.Cell):
    """TorchDrug-style MLP: no activation / dropout after the last layer."""

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


class SegmentOps(nn.Cell):
    def __init__(self):
        super().__init__()
        self.segment_sum = ops.UnsortedSegmentSum()
        self.gather = ops.Gather()

    def sum(self, data, segment_ids, num_segments):
        return self.segment_sum(data, segment_ids, num_segments)

    def gather_rows(self, data, indices):
        return self.gather(data, indices, 0)


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
        self.batch_norm = nn.BatchNorm1d(output_dim) if batch_norm else None
        self.activation = nn.ReLU()
        self.segment = SegmentOps()

    def construct(self, node_feature, edge_list, edge_feature, node_index=None):
        # node_index is accepted for a uniform layer interface but unused here:
        # GIN aggregates only over real edges and has no explicit self-loops.
        num_node = node_feature.shape[0]
        node_in = edge_list[:, 0]
        node_out = edge_list[:, 1]
        message = self.segment.gather_rows(node_feature, node_in)
        if self.edge_linear is not None:
            message = message + self.edge_linear(edge_feature)
        update = self.segment.sum(message, node_out, num_node)

        output = self.mlp((1.0 + self.eps) * node_feature + update)
        if self.batch_norm is not None:
            output = self.batch_norm(output)
        return self.activation(output)


class TorchDrugGATLayer(nn.Cell):
    """GAT convolution aligned with TorchDrug's GraphAttentionConv."""

    eps = 1e-10

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
        self.batch_norm = nn.BatchNorm1d(output_dim) if batch_norm else None
        self.activation = nn.ReLU()
        self.segment = SegmentOps()
        self.reduce_sum = ops.ReduceSum(keep_dims=False)
        self.concat0 = ops.Concat(axis=0)
        self.concat_feature = ops.Concat(axis=2)
        self.score_min = Tensor(np.asarray(-20.0, dtype=np.float32), ms.float32)
        self.score_max = Tensor(np.asarray(20.0, dtype=np.float32), ms.float32)

    def construct(self, node_feature, edge_list, edge_feature, node_index):
        num_node = node_feature.shape[0]
        hidden = self.linear(node_feature)
        hidden_heads = ops.reshape(hidden, (num_node, self.num_head, self.head_dim))

        node_in = edge_list[:, 0]
        node_out = edge_list[:, 1]
        self_index = node_index
        node_in = self.concat0((node_in, self_index))
        node_out = self.concat0((node_out, self_index))

        source = self.segment.gather_rows(hidden_heads, node_in)
        target = self.segment.gather_rows(hidden_heads, node_out)
        if self.edge_linear is not None:
            edge_hidden = self.edge_linear(edge_feature)
            edge_hidden = ops.reshape(edge_hidden, (-1, self.num_head, self.head_dim))
            zero_edge = ops.zeros((num_node, self.num_head, self.head_dim), ms.float32)
            edge_hidden = self.concat0((edge_hidden, zero_edge))
            source = source + edge_hidden
            target = target + edge_hidden

        key = self.concat_feature((source, target))
        score = self.reduce_sum(key * ops.expand_dims(self.query, 0), -1)
        score = ops.maximum(score, score * self.negative_slope)
        score = ops.minimum(ops.maximum(score, self.score_min), self.score_max)

        exp_score = ops.exp(score)
        normalizer = self.segment.sum(exp_score, node_out, num_node)
        attention = exp_score / (self.segment.gather_rows(normalizer, node_out) + self.eps)

        value = self.segment.gather_rows(hidden_heads, node_in)
        message = attention[:, :, None] * value
        message = ops.reshape(message, (-1, self.output_dim))
        update = self.segment.sum(message, node_out, num_node)

        if self.batch_norm is not None:
            update = self.batch_norm(update)
        return self.activation(update)


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
        self.concat = ops.Concat(axis=1)
        self.segment = SegmentOps()

    def construct(self, node_feature, edge_list, edge_feature, node2graph, node_index, num_graph):
        h = node_feature
        hiddens = []
        for layer in self.layers:
            hidden = layer(h, edge_list, edge_feature, node_index)
            if self.short_cut and hidden.shape == h.shape:
                hidden = hidden + h
            hiddens.append(hidden)
            h = hidden

        node_output = self.concat(hiddens) if self.concat_hidden else hiddens[-1]
        graph_feature = self.pool(node_output, node2graph, num_graph)
        return self.classifier(graph_feature)

    def pool(self, node_feature, node2graph, num_graph):
        summed = self.segment.sum(node_feature, node2graph, num_graph)
        if self.readout == "sum":
            return summed
        ones = ops.ones((node_feature.shape[0], 1), ms.float32)
        count = self.segment.sum(ones, node2graph, num_graph)
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
