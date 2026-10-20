"""Convolutional network for binary Range-Doppler target detection."""

from __future__ import annotations

import torch
from torch import nn

from passive_radar.config import ModelConfig

_POOL_FACTOR = 2


def conv_block(in_channels: int, out_channels: int, kernel_size: int) -> nn.Sequential:
    """Conv -> BatchNorm -> ReLU -> 2x2 MaxPool."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(_POOL_FACTOR),
    )


def _adaptive_pool(kind: str, output_size: tuple[int, int]) -> nn.Module:
    """Adaptive average or max pooling to a fixed spatial size."""
    if kind == "avg":
        return nn.AdaptiveAvgPool2d(output_size)
    if kind == "max":
        return nn.AdaptiveMaxPool2d(output_size)
    raise ValueError(f"Unknown global_pool '{kind}'. Use 'avg' or 'max'.")


class RangeDopplerCNN(nn.Module):
    """Stack of convolution blocks followed by a small fully connected classifier.

    Adaptive pooling makes the classifier independent of the exact input size
    while keeping a coarse range/Doppler layout (``config.pooled_size``).
    Outputs raw logits of shape ``(batch, num_classes)`` for ``CrossEntropyLoss``.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        channels = (config.in_channels, *config.conv_channels)
        self.features = nn.Sequential(
            *(conv_block(c_in, c_out, config.kernel_size) for c_in, c_out in zip(channels[:-1], channels[1:]))
        )
        self.pool = _adaptive_pool(config.global_pool, config.pooled_size)
        pooled_height, pooled_width = config.pooled_size
        flattened = config.conv_channels[-1] * pooled_height * pooled_width
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(config.dropout),
            nn.Linear(flattened, config.hidden_units),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_units, config.num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return class logits for a batch of images shaped (N, C, H, W)."""
        if images.ndim != 4 or images.shape[1] != self.config.in_channels:
            raise ValueError(
                f"Expected input of shape (N, {self.config.in_channels}, H, W), got {tuple(images.shape)}"
            )
        return self.classifier(self.pool(self.features(images)))


def count_trainable_parameters(model: nn.Module) -> int:
    """Number of trainable parameters in ``model``."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
