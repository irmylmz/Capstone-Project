"""Saving and loading model checkpoints together with everything needed for inference."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from passive_radar.config import ModelConfig, PreprocessingConfig, RangeDopplerMapConfig
from passive_radar.data.transforms import InputConfig
from passive_radar.models.cnn import RangeDopplerCNN

CHECKPOINT_FORMAT_VERSION = 2
_SUPPORTED_VERSIONS = (1, CHECKPOINT_FORMAT_VERSION)
_INPUT_KINDS = {"image": PreprocessingConfig, "rd_map": RangeDopplerMapConfig}


class ModelNotFoundError(FileNotFoundError):
    """Raised when a checkpoint file does not exist (the model has not been trained yet)."""


@dataclass
class LoadedCheckpoint:
    """A model restored from disk along with its metadata."""

    model: RangeDopplerCNN
    class_names: tuple[str, ...]
    model_config: ModelConfig
    preprocessing_config: InputConfig  # PreprocessingConfig (images) or RangeDopplerMapConfig (maps)
    metadata: dict[str, Any] = field(default_factory=dict)


def _tuplify(values: dict[str, Any]) -> dict[str, Any]:
    """Convert list values back to tuples so frozen config dataclasses round-trip exactly."""
    return {key: tuple(value) if isinstance(value, list) else value for key, value in values.items()}


def _input_kind(config: InputConfig) -> str:
    for kind, config_class in _INPUT_KINDS.items():
        if isinstance(config, config_class):
            return kind
    raise TypeError(f"Unsupported input configuration type: {type(config).__name__}")


def save_checkpoint(
    path: Path,
    model: RangeDopplerCNN,
    class_names: tuple[str, ...],
    preprocessing_config: InputConfig,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Write weights, class names, model and input-preprocessing configuration to ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "class_names": list(class_names),
        "model_config": asdict(model.config),
        "input_kind": _input_kind(preprocessing_config),
        "preprocessing_config": asdict(preprocessing_config),
        "metadata": metadata or {},
    }
    torch.save(payload, path)
    return path


def load_checkpoint(path: Path, device: torch.device | str = "cpu") -> LoadedCheckpoint:
    """Rebuild the model from a checkpoint written by :func:`save_checkpoint`.

    Raises:
        ModelNotFoundError: If ``path`` does not exist.
        ValueError: If the file is not a compatible checkpoint.
    """
    path = Path(path)
    if not path.is_file():
        raise ModelNotFoundError(
            f"Model checkpoint not found: {path}. Train a model first (notebooks/03_model_training.ipynb)."
        )
    payload = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(payload, dict) or payload.get("format_version") not in _SUPPORTED_VERSIONS:
        raise ValueError(f"Unsupported or corrupt checkpoint format: {path}")

    model_config = ModelConfig(**_tuplify(payload["model_config"]))
    config_class = _INPUT_KINDS[payload.get("input_kind", "image")]  # v1 checkpoints: images only
    preprocessing_config = config_class(**_tuplify(payload["preprocessing_config"]))
    model = RangeDopplerCNN(model_config)
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    return LoadedCheckpoint(
        model=model,
        class_names=tuple(payload["class_names"]),
        model_config=model_config,
        preprocessing_config=preprocessing_config,
        metadata=dict(payload.get("metadata", {})),
    )
