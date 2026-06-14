"""MindSpore training and evaluation loops."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import mindspore as ms
import numpy as np
from mindspore import Tensor, nn

from .dataset import MoleculeDataset, batch_iterator
from .metrics import binary_classification_metrics


@dataclass
class TrainConfig:
    epoch: int = 50
    batch_size: int = 64
    lr: float = 1e-3
    seed: int = 0


def to_tensor_batch(batch):
    node_feature, adjacency, node_mask, label = batch
    return (
        Tensor(node_feature, ms.float32),
        Tensor(adjacency, ms.float32),
        Tensor(node_mask, ms.float32),
        Tensor(label, ms.float32),
    )


def train_one_epoch(model: nn.Cell, optimizer: nn.Optimizer, dataset: MoleculeDataset, config: TrainConfig) -> float:
    loss_fn = nn.BCEWithLogitsLoss(reduction="mean")
    model.set_train(True)

    def forward_fn(node_feature, adjacency, node_mask, label):
        logits = model(node_feature, adjacency, node_mask)
        loss = loss_fn(logits, label)
        return loss

    grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters)
    losses = []
    for batch_id, batch in enumerate(batch_iterator(dataset, config.batch_size, shuffle=True, seed=config.seed)):
        node_feature, adjacency, node_mask, label = to_tensor_batch(batch)
        loss, grads = grad_fn(node_feature, adjacency, node_mask, label)
        optimizer(grads)
        losses.append(float(loss.asnumpy()))
    return float(np.mean(losses)) if losses else float("nan")


def evaluate(model: nn.Cell, dataset: MoleculeDataset, batch_size: int) -> Dict[str, float]:
    model.set_train(False)
    logits_list, labels_list = [], []
    for batch in batch_iterator(dataset, batch_size, shuffle=False, seed=0):
        node_feature, adjacency, node_mask, label = to_tensor_batch(batch)
        logits = model(node_feature, adjacency, node_mask)
        logits_list.append(logits.asnumpy())
        labels_list.append(label.asnumpy())
    logits = np.concatenate(logits_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    return binary_classification_metrics(logits, labels)


def fit(
    model: nn.Cell,
    train_set: MoleculeDataset,
    valid_set: MoleculeDataset,
    test_set: MoleculeDataset,
    config: TrainConfig,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    optimizer = nn.Adam(model.trainable_params(), learning_rate=config.lr)
    best_valid = {"auroc": -1.0, "auprc": -1.0}
    best_test = {"auroc": float("nan"), "auprc": float("nan")}

    for epoch in range(1, config.epoch + 1):
        train_loss = train_one_epoch(model, optimizer, train_set, config)
        valid_metrics = evaluate(model, valid_set, config.batch_size)
        test_metrics = evaluate(model, test_set, config.batch_size)
        if np.isnan(best_valid["auroc"]) or valid_metrics["auroc"] > best_valid["auroc"]:
            best_valid = valid_metrics
            best_test = test_metrics
        print(
            f"epoch {epoch:03d} | loss {train_loss:.4f} | "
            f"valid AUROC {valid_metrics['auroc']:.4f} AUPRC {valid_metrics['auprc']:.4f} | "
            f"test AUROC {test_metrics['auroc']:.4f} AUPRC {test_metrics['auprc']:.4f}"
        )

    return best_valid, best_test
