"""Training loop with validation-based early stopping and best-model checkpointing."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from passive_radar.config import TrainingConfig
from passive_radar.data.transforms import InputConfig
from passive_radar.evaluation.metrics import compute_classification_metrics
from passive_radar.models.checkpoint import save_checkpoint
from passive_radar.models.cnn import RangeDopplerCNN


@dataclass(frozen=True)
class EpochResult:
    """Losses and validation metrics of one epoch."""

    epoch: int
    train_loss: float
    val_loss: float
    val_accuracy: float
    val_f1: float


@dataclass
class TrainingHistory:
    """All epoch results plus the epoch chosen as best (lowest validation loss)."""

    epochs: list[EpochResult] = field(default_factory=list)
    best_epoch: int | None = None
    stopped_early: bool = False

    def as_dict(self) -> dict:
        """JSON-serialisable representation."""
        return {
            "epochs": [asdict(result) for result in self.epochs],
            "best_epoch": self.best_epoch,
            "stopped_early": self.stopped_early,
        }

    def save_json(self, path: Path) -> None:
        """Write the history to ``path`` as JSON."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.as_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load_json(cls, path: Path) -> "TrainingHistory":
        """Read a history written by :meth:`save_json`."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            epochs=[EpochResult(**item) for item in data["epochs"]],
            best_epoch=data["best_epoch"],
            stopped_early=data["stopped_early"],
        )


class EarlyStopping:
    """Signals a stop when the monitored loss has not improved for ``patience`` epochs."""

    def __init__(self, patience: int, min_delta: float = 0.0) -> None:
        if patience < 1:
            raise ValueError("patience must be >= 1")
        self.patience = patience
        self.min_delta = min_delta
        self.best_value = math.inf
        self.epochs_without_improvement = 0

    def update(self, value: float) -> bool:
        """Record a new loss value; return ``True`` if it is a new best."""
        if value < self.best_value - self.min_delta:
            self.best_value = value
            self.epochs_without_improvement = 0
            return True
        self.epochs_without_improvement += 1
        return False

    @property
    def should_stop(self) -> bool:
        """Whether training should stop now."""
        return self.epochs_without_improvement >= self.patience


class Trainer:
    """Trains a :class:`RangeDopplerCNN` with Adam + cross-entropy.

    The model with the lowest validation loss is written to ``checkpoint_path``.
    The test split is never seen here.
    """

    def __init__(
        self,
        model: RangeDopplerCNN,
        config: TrainingConfig,
        device: torch.device,
        checkpoint_path: Path,
        class_names: tuple[str, ...],
        preprocessing_config: InputConfig,
    ) -> None:
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.checkpoint_path = Path(checkpoint_path)
        self.class_names = class_names
        self.preprocessing_config = preprocessing_config
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, verbose: bool = True) -> TrainingHistory:
        """Train until ``max_epochs`` or early stopping; return the history."""
        history = TrainingHistory()
        stopper = EarlyStopping(self.config.early_stopping_patience, self.config.min_delta)
        for epoch in range(1, self.config.max_epochs + 1):
            train_loss = self._train_one_epoch(train_loader)
            val_loss, val_metrics = self._validate(val_loader)
            result = EpochResult(epoch, train_loss, val_loss, val_metrics["accuracy"], val_metrics["f1"])
            history.epochs.append(result)

            improved = stopper.update(val_loss)
            if improved:
                history.best_epoch = epoch
                self._save_best(result)
            if verbose:
                marker = " *best*" if improved else ""
                print(
                    f"epoch {epoch:02d} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f} "
                    f"| val_acc {result.val_accuracy:.3f} | val_f1 {result.val_f1:.3f}{marker}"
                )
            if stopper.should_stop:
                history.stopped_early = True
                if verbose:
                    print(f"Early stopping: no val_loss improvement for {stopper.patience} epochs.")
                break
        return history

    def _train_one_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss, total_samples = 0.0, 0
        for images, labels in loader:
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            loss = self.criterion(self.model(images), labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)
        return total_loss / max(total_samples, 1)

    @torch.no_grad()
    def _validate(self, loader: DataLoader) -> tuple[float, dict]:
        self.model.eval()
        total_loss, total_samples = 0.0, 0
        all_labels, all_predictions = [], []
        for images, labels in loader:
            images, labels = images.to(self.device), labels.to(self.device)
            logits = self.model(images)
            total_loss += self.criterion(logits, labels).item() * labels.size(0)
            total_samples += labels.size(0)
            all_labels.append(labels.cpu())
            all_predictions.append(logits.argmax(dim=1).cpu())
        metrics = compute_classification_metrics(
            torch.cat(all_labels).numpy(), torch.cat(all_predictions).numpy()
        )
        return total_loss / max(total_samples, 1), metrics

    def _save_best(self, result: EpochResult) -> None:
        save_checkpoint(
            self.checkpoint_path,
            self.model,
            self.class_names,
            self.preprocessing_config,
            metadata={"best_epoch": result.epoch, "val_loss": result.val_loss,
                      "val_accuracy": result.val_accuracy, "val_f1": result.val_f1,
                      "training_config": asdict(self.config)},
        )
