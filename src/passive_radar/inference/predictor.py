"""Single-image inference with a trained checkpoint.

Example::

    from passive_radar.inference.predictor import predict_image

    result = predict_image("sample_range_doppler.png")
    print(result)

Only the target / no-target decision and class probabilities are returned; no
range, velocity or position is estimated.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from passive_radar.config import ProjectConfig
from passive_radar.data.transforms import input_file_to_tensor
from passive_radar.models.checkpoint import load_checkpoint
from passive_radar.training.runtime import resolve_device


@dataclass(frozen=True)
class PredictionResult:
    """Outcome of classifying one image."""

    image_path: str
    predicted_class: str
    target_present: bool
    probabilities: dict[str, float]

    def __str__(self) -> str:
        probabilities = ", ".join(f"{name}={value:.3f}" for name, value in self.probabilities.items())
        return f"{Path(self.image_path).name}: {self.predicted_class} ({probabilities})"


class RangeDopplerPredictor:
    """Loads a checkpoint once and classifies images with the training preprocessing."""

    TARGET_CLASS_NAME = "target"

    def __init__(self, checkpoint_path: str | Path, device: str = "auto") -> None:
        self.device = resolve_device(device)
        loaded = load_checkpoint(Path(checkpoint_path), device=self.device)
        self.model = loaded.model
        self.class_names = loaded.class_names
        self.preprocessing_config = loaded.preprocessing_config
        self.metadata = loaded.metadata

    @torch.no_grad()
    def predict(self, image_path: str | Path) -> PredictionResult:
        """Classify one image file (or ``.npy`` Range-Doppler map for simulation models).

        Raises:
            ImageLoadError: If the image is missing or corrupt.
        """
        tensor = input_file_to_tensor(image_path, self.preprocessing_config).unsqueeze(0).to(self.device)
        probabilities = torch.softmax(self.model(tensor), dim=1).squeeze(0).cpu().tolist()
        best_index = max(range(len(probabilities)), key=probabilities.__getitem__)
        predicted_class = self.class_names[best_index]
        return PredictionResult(
            image_path=str(image_path),
            predicted_class=predicted_class,
            target_present=predicted_class == self.TARGET_CLASS_NAME,
            probabilities={name: float(p) for name, p in zip(self.class_names, probabilities)},
        )


def predict_image(
    image_path: str | Path, checkpoint_path: str | Path | None = None, device: str = "auto"
) -> PredictionResult:
    """Classify a single image with the default (or given) checkpoint.

    Raises:
        ModelNotFoundError: If no trained checkpoint exists.
    """
    path = Path(checkpoint_path) if checkpoint_path is not None else ProjectConfig.default().paths.checkpoint_path
    return RangeDopplerPredictor(path, device=device).predict(image_path)
