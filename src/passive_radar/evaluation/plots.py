"""Matplotlib figures shared by the notebooks."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

if TYPE_CHECKING:  # avoid importing torch just for a type hint
    from passive_radar.training.trainer import TrainingHistory


def _save(figure: Figure, path: Path | None) -> Figure:
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=120, bbox_inches="tight")
    return figure


def plot_training_history(history: TrainingHistory, save_path: Path | None = None) -> Figure:
    """Train/validation loss and validation accuracy/F1 per epoch; best epoch marked."""
    epochs = [result.epoch for result in history.epochs]
    figure, (loss_axis, metric_axis) = plt.subplots(1, 2, figsize=(12, 4))
    loss_axis.plot(epochs, [r.train_loss for r in history.epochs], marker="o", label="train loss")
    loss_axis.plot(epochs, [r.val_loss for r in history.epochs], marker="o", label="val loss")
    metric_axis.plot(epochs, [r.val_accuracy for r in history.epochs], marker="o", label="val accuracy")
    metric_axis.plot(epochs, [r.val_f1 for r in history.epochs], marker="o", label="val F1")
    for axis, title in ((loss_axis, "Cross-entropy loss"), (metric_axis, "Validation metrics")):
        if history.best_epoch is not None:
            axis.axvline(history.best_epoch, color="grey", linestyle="--", label="best epoch")
        axis.set_xlabel("epoch")
        axis.set_title(title)
        axis.grid(alpha=0.3)
        axis.legend()
    metric_axis.set_ylim(0, 1)
    figure.tight_layout()
    return _save(figure, save_path)


def plot_confusion_matrix(
    matrix: Sequence[Sequence[int]], class_names: Sequence[str], save_path: Path | None = None
) -> Figure:
    """Annotated confusion matrix (rows = true class, columns = predicted class)."""
    values = np.asarray(matrix)
    figure, axis = plt.subplots(figsize=(4.5, 4))
    axis.imshow(values, cmap="Blues")
    for (row, column), count in np.ndenumerate(values):
        color = "white" if count > values.max() / 2 else "black"
        axis.text(column, row, str(count), ha="center", va="center", color=color, fontsize=12)
    axis.set_xticks(range(len(class_names)), class_names)
    axis.set_yticks(range(len(class_names)), class_names)
    axis.set_xlabel("predicted")
    axis.set_ylabel("true")
    axis.set_title("Confusion matrix")
    figure.tight_layout()
    return _save(figure, save_path)


def plot_image_grid(
    images: Sequence[np.ndarray], titles: Sequence[str], columns: int = 2, save_path: Path | None = None
) -> Figure:
    """Show images (H, W, 3 arrays) in a grid with titles."""
    if not images:
        raise ValueError("No images to plot")
    rows = int(np.ceil(len(images) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(7 * columns, 2.2 * rows), squeeze=False)
    for axis in axes.flat:
        axis.axis("off")
    for axis, image, title in zip(axes.flat, images, titles):
        axis.imshow(image, aspect="auto")
        axis.set_title(title, fontsize=9)
    figure.tight_layout()
    return _save(figure, save_path)


def plot_rd_maps(
    maps: Sequence[np.ndarray],
    titles: Sequence[str],
    max_range_km: float,
    max_doppler_hz: float,
    markers: Sequence[tuple[float, float] | None] | None = None,
    save_path: Path | None = None,
) -> Figure:
    """Show (Doppler x range) maps with physical axes; optional (range_km, doppler_hz) markers."""
    if not maps:
        raise ValueError("No maps to plot")
    figure, axes = plt.subplots(1, len(maps), figsize=(5.5 * len(maps), 4), squeeze=False)
    extent = (0.0, max_range_km, -max_doppler_hz, max_doppler_hz)
    for index, (axis, rd_map, title) in enumerate(zip(axes.flat, maps, titles)):
        image = axis.imshow(rd_map, origin="lower", aspect="auto", extent=extent, cmap="jet")
        marker = markers[index] if markers else None
        if marker is not None:
            axis.plot(*marker, marker="o", markersize=14, markerfacecolor="none", markeredgecolor="white")
        axis.set_xlabel("bistatic range [km]")
        axis.set_ylabel("Doppler [Hz]")
        axis.set_title(title, fontsize=10)
        figure.colorbar(image, ax=axis)
    figure.tight_layout()
    return _save(figure, save_path)
