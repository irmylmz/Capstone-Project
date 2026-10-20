"""Tests for the CNN, checkpoint I/O, training loop and predictor."""

from __future__ import annotations

import pytest
import torch

from passive_radar.config import CLASS_NAMES, ModelConfig, PreprocessingConfig, TrainingConfig
from passive_radar.data.dataset import RangeDopplerDataset, make_data_loader
from passive_radar.data.indexing import scan_labeled_directory
from passive_radar.data.preprocessing import ImageLoadError
from passive_radar.inference.predictor import RangeDopplerPredictor, predict_image
from passive_radar.models.checkpoint import ModelNotFoundError, load_checkpoint, save_checkpoint
from passive_radar.models.cnn import RangeDopplerCNN, count_trainable_parameters
from passive_radar.training.trainer import EarlyStopping, Trainer, TrainingHistory


@pytest.fixture
def model() -> RangeDopplerCNN:
    torch.manual_seed(0)
    return RangeDopplerCNN(ModelConfig()).eval()


@pytest.mark.parametrize("height,width", [(128, 640), (128, 512)])
def test_forward_output_shape(model, height, width):
    logits = model(torch.randn(2, 3, height, width))
    assert logits.shape == (2, len(CLASS_NAMES))


def test_forward_rejects_wrong_channel_count(model):
    with pytest.raises(ValueError, match="Expected input"):
        model(torch.randn(1, 1, 128, 640))


def test_model_has_trainable_parameters(model):
    assert count_trainable_parameters(model) > 0


def test_checkpoint_round_trip(model, tmp_path):
    preprocessing = PreprocessingConfig(mean=(0.1, 0.2, 0.3), std=(0.4, 0.5, 0.6))
    path = save_checkpoint(tmp_path / "m.pt", model, CLASS_NAMES, preprocessing, metadata={"best_epoch": 3})
    loaded = load_checkpoint(path)
    batch = torch.randn(2, 3, 128, 640)
    with torch.no_grad():
        assert torch.allclose(model(batch), loaded.model(batch))
    assert loaded.class_names == CLASS_NAMES
    assert loaded.model_config == model.config
    assert loaded.preprocessing_config == preprocessing
    assert loaded.metadata["best_epoch"] == 3


def test_missing_checkpoint_raises_clear_error(tmp_path):
    with pytest.raises(ModelNotFoundError, match="Train a model first"):
        load_checkpoint(tmp_path / "missing.pt")


def test_corrupt_checkpoint_raises(tmp_path):
    path = tmp_path / "bad.pt"
    torch.save({"something": 1}, path)
    with pytest.raises(ValueError, match="Unsupported"):
        load_checkpoint(path)


def test_early_stopping_counts_epochs_without_improvement():
    stopper = EarlyStopping(patience=2, min_delta=0.0)
    assert stopper.update(1.0) is True
    assert stopper.update(1.1) is False and not stopper.should_stop
    assert stopper.update(1.2) is False and stopper.should_stop
    assert stopper.update(0.5) is True and not stopper.should_stop


def test_trainer_saves_best_checkpoint(labeled_dir, tmp_path):
    preprocessing = PreprocessingConfig()
    dataset = RangeDopplerDataset(scan_labeled_directory(labeled_dir), preprocessing, cache_images=True)
    loader = make_data_loader(dataset, batch_size=8, shuffle=True, seed=0)
    config = TrainingConfig(max_epochs=2, early_stopping_patience=1, device="cpu")
    checkpoint = tmp_path / "best.pt"
    trainer = Trainer(RangeDopplerCNN(ModelConfig(conv_channels=(4, 8))), config, torch.device("cpu"),
                      checkpoint, CLASS_NAMES, preprocessing)
    history = trainer.fit(loader, loader, verbose=False)
    assert checkpoint.is_file()
    assert 1 <= len(history.epochs) <= 2 and history.best_epoch is not None
    history.save_json(tmp_path / "history.json")
    assert TrainingHistory.load_json(tmp_path / "history.json") == history


def test_predictor_returns_probabilities(model, figure_path, tmp_path):
    checkpoint = save_checkpoint(tmp_path / "m.pt", model, CLASS_NAMES, PreprocessingConfig())
    result = predict_image(figure_path, checkpoint_path=checkpoint, device="cpu")
    assert result.predicted_class in CLASS_NAMES
    assert result.target_present == (result.predicted_class == "target")
    assert set(result.probabilities) == set(CLASS_NAMES)
    assert sum(result.probabilities.values()) == pytest.approx(1.0, abs=1e-5)


def test_predictor_without_model_raises(tmp_path, figure_path):
    with pytest.raises(ModelNotFoundError):
        predict_image(figure_path, checkpoint_path=tmp_path / "none.pt", device="cpu")


def test_predictor_with_corrupt_image_raises(model, corrupt_path, tmp_path):
    checkpoint = save_checkpoint(tmp_path / "m.pt", model, CLASS_NAMES, PreprocessingConfig())
    with pytest.raises(ImageLoadError):
        RangeDopplerPredictor(checkpoint, device="cpu").predict(corrupt_path)
