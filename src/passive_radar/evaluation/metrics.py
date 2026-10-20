"""Prediction collection and classification metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader

POSITIVE_CLASS_INDEX = 1  # "target"


@dataclass(frozen=True)
class Predictions:
    """Ground truth, predicted labels and class probabilities for a dataset."""

    y_true: np.ndarray  # (N,)
    probabilities: np.ndarray  # (N, num_classes)

    @property
    def y_pred(self) -> np.ndarray:
        """Arg-max predicted class indices."""
        return self.probabilities.argmax(axis=1)

    @property
    def positive_probability(self) -> np.ndarray:
        """Probability of the ``target`` class."""
        return self.probabilities[:, POSITIVE_CLASS_INDEX]


@torch.no_grad()
def collect_predictions(model: nn.Module, loader: DataLoader, device: torch.device) -> Predictions:
    """Run ``model`` in eval mode over ``loader`` and gather softmax probabilities."""
    model.eval()
    labels, probabilities = [], []
    for images, targets in loader:
        logits = model(images.to(device))
        probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
        labels.append(targets.numpy())
    return Predictions(y_true=np.concatenate(labels), probabilities=np.concatenate(probabilities))


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray | None = None) -> dict:
    """Accuracy, precision, recall, F1 (positive class = target), ROC-AUC and confusion matrix.

    ``zero_division=0`` keeps metrics defined when a class is never predicted.
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=POSITIVE_CLASS_INDEX, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=POSITIVE_CLASS_INDEX, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=POSITIVE_CLASS_INDEX, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "support": int(len(y_true)),
    }
    if y_score is not None and len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
    return metrics


def metrics_from_predictions(predictions: Predictions) -> dict:
    """Convenience wrapper around :func:`compute_classification_metrics`."""
    return compute_classification_metrics(
        predictions.y_true, predictions.y_pred, predictions.positive_probability
    )
