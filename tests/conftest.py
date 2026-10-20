"""Shared pytest fixtures. All test images are synthetic and written to tmp_path;
the real dataset under data/ is never read or modified."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from passive_radar.config import CLASS_NAMES, PreprocessingConfig


def make_figure_image(config: PreprocessingConfig, seed: int, bright_spot: bool = False) -> Image.Image:
    """Create an RGBA image mimicking the dataset layout: white figure, coloured plot box."""
    rng = np.random.default_rng(seed)
    width, height = config.reference_size
    canvas = np.full((height, width, 4), 255, dtype=np.uint8)
    left, top, right, bottom = config.plot_box
    # Row-wise colour bands (not per-pixel noise): keeps PNG encoding fast while staying non-white.
    row_colours = rng.integers(0, 200, size=(bottom - top, 1, 3), dtype=np.uint8)
    plot = np.repeat(row_colours, right - left, axis=1)
    if bright_spot:
        plot[150:200, 600:700] = (200, 0, 0)
    canvas[top:bottom, left:right, :3] = plot
    return Image.fromarray(canvas, mode="RGBA")


@pytest.fixture
def preprocessing_config() -> PreprocessingConfig:
    return PreprocessingConfig()


@pytest.fixture
def figure_path(tmp_path: Path, preprocessing_config: PreprocessingConfig) -> Path:
    path = tmp_path / "Example Song (128kbit_AAC)Part001.png"
    make_figure_image(preprocessing_config, seed=0).save(path)
    return path


@pytest.fixture
def corrupt_path(tmp_path: Path) -> Path:
    path = tmp_path / "corrupt.png"
    path.write_bytes(b"this is not a png")
    return path


@pytest.fixture
def labeled_dir(tmp_path: Path, preprocessing_config: PreprocessingConfig) -> Path:
    """Labelled folder with 8 recordings x 2 segments per class (32 images)."""
    root = tmp_path / "train"
    seed = 0
    for label, class_name in enumerate(CLASS_NAMES):
        class_dir = root / class_name
        class_dir.mkdir(parents=True)
        for recording in range(8):
            for part in range(1, 3):
                seed += 1
                image = make_figure_image(preprocessing_config, seed=seed, bright_spot=bool(label))
                image.save(class_dir / f"Recording {recording} (128kbit_AAC)Part{part:03d}.png")
    return root
