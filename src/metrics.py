"""Binary classification metrics."""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def binary_classification_metrics(logits: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    y_true = labels.reshape(-1).astype(np.int32)
    y_score = sigmoid(logits.reshape(-1))
    if len(np.unique(y_true)) < 2:
        return {"auroc": float("nan"), "auprc": float("nan")}
    return {
        "auroc": float(roc_auc_score(y_true, y_score)),
        "auprc": float(average_precision_score(y_true, y_score)),
    }
