"""Tests for image loading and preprocessing."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from passive_radar.config import PreprocessingConfig
from passive_radar.data.preprocessing import (
    ImageLoadError,
    compute_channel_stats,
    crop_plot_area,
    inspect_image,
    letterbox_resize,
    load_and_preprocess,
    load_channel_stats,
    load_rgb_image,
    save_channel_stats,
    scaled_plot_box,
)
from passive_radar.data.transforms import array_to_tensor, image_file_to_tensor


def test_load_rgb_image_drops_alpha(figure_path):
    image = load_rgb_image(figure_path)
    assert image.mode == "RGB"
    assert image.size == (2240, 740)


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(ImageLoadError, match="not found"):
        load_rgb_image(tmp_path / "missing.png")


def test_load_corrupt_file_raises(corrupt_path):
    with pytest.raises(ImageLoadError, match="Could not decode"):
        load_rgb_image(corrupt_path)


def test_crop_keeps_only_plot_area(figure_path, preprocessing_config):
    cropped = np.asarray(crop_plot_area(load_rgb_image(figure_path), preprocessing_config))
    left, top, right, bottom = preprocessing_config.plot_box
    assert cropped.shape == (bottom - top, right - left, 3)
    assert not np.all(cropped == 255, axis=2).any(), "white figure margin leaked into the crop"


def test_plot_box_scales_with_image_size(preprocessing_config):
    half_size = (1120, 370)
    assert scaled_plot_box(half_size, preprocessing_config) == (80, 100, 1024, 290)


def test_letterbox_preserves_aspect_ratio():
    image = Image.new("RGB", (1000, 100), (255, 0, 0))
    result = np.asarray(letterbox_resize(image, (640, 128), pad_value=0))
    assert result.shape == (128, 640, 3)
    content_rows = np.where(result[:, :, 0].max(axis=1) > 0)[0]
    assert len(content_rows) == 64  # 1000x100 scaled by 0.64 -> 640x64, rest is padding


def test_letterbox_rejects_unknown_resample():
    with pytest.raises(ValueError, match="resample"):
        letterbox_resize(Image.new("RGB", (10, 10)), (5, 5), resample="magic")


def test_load_and_preprocess_output_shape(figure_path, preprocessing_config):
    array = load_and_preprocess(figure_path, preprocessing_config)
    width, height = preprocessing_config.target_size
    assert array.shape == (height, width, 3)
    assert array.dtype == np.uint8


def test_alternative_input_size_is_configurable(figure_path):
    config = PreprocessingConfig(target_size=(512, 128))
    assert load_and_preprocess(figure_path, config).shape == (128, 512, 3)


def test_channel_stats_in_unit_range(figure_path, preprocessing_config):
    mean, std = compute_channel_stats([figure_path], preprocessing_config)
    assert len(mean) == len(std) == 3
    assert all(0.0 <= value <= 1.0 for value in mean)
    assert all(value > 0.0 for value in std)


def test_channel_stats_empty_list_raises(preprocessing_config):
    with pytest.raises(ValueError):
        compute_channel_stats([], preprocessing_config)


def test_tensor_conversion_normalises(preprocessing_config):
    config = PreprocessingConfig(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    white = np.full((128, 640, 3), 255, dtype=np.uint8)
    tensor = array_to_tensor(white, config)
    assert tensor.shape == (3, 128, 640)
    assert torch.allclose(tensor, torch.ones_like(tensor))


def test_tensor_conversion_rejects_wrong_shape(preprocessing_config):
    with pytest.raises(ValueError):
        array_to_tensor(np.zeros((128, 640), dtype=np.uint8), preprocessing_config)


def test_image_file_to_tensor_matches_config(figure_path, preprocessing_config):
    tensor = image_file_to_tensor(figure_path, preprocessing_config)
    assert tensor.shape == (3, 128, 640)
    assert tensor.dtype == torch.float32


def test_inspect_image_reports_valid_and_corrupt(figure_path, corrupt_path):
    valid = inspect_image(figure_path)
    assert valid.is_valid and (valid.width, valid.height, valid.mode) == (2240, 740, "RGBA")
    broken = inspect_image(corrupt_path)
    assert not broken.is_valid and broken.error


def test_channel_stats_round_trip(tmp_path):
    save_channel_stats(tmp_path / "n.json", (0.1, 0.2, 0.3), (0.4, 0.5, 0.6))
    assert load_channel_stats(tmp_path / "n.json") == ((0.1, 0.2, 0.3), (0.4, 0.5, 0.6))


def test_missing_channel_stats_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="02_preprocessing"):
        load_channel_stats(tmp_path / "missing.json")
