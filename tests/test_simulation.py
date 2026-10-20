"""Tests for the FM passive radar simulation (synthetic signals only)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from passive_radar.config import CLASS_NAMES, ModelConfig, RangeDopplerMapConfig, SimulationConfig
from passive_radar.data.transforms import normalize_rd_map, rd_map_to_tensor
from passive_radar.models.checkpoint import load_checkpoint, save_checkpoint
from passive_radar.models.cnn import RangeDopplerCNN
from passive_radar.simulation.ambiguity import cross_ambiguity, doppler_bin_count, doppler_step_hz
from passive_radar.simulation.fm import fm_modulate, fm_signal_from_audio
from passive_radar.simulation.scenario import (
    TargetParameters,
    delay_in_samples,
    range_cells,
    synthesize_channels,
)

SAMPLE_RATE = 220_500.0


def _noise_illuminator(num_samples: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.exp(1j * 2 * np.pi * rng.random(num_samples))  # unit-modulus, white


def _peak_location(rd_map: np.ndarray, sample_rate: float, num_samples: int, max_doppler: float):
    row, col = np.unravel_index(np.abs(rd_map).argmax(), rd_map.shape)
    bins = doppler_bin_count(sample_rate, num_samples, max_doppler)
    return (row - bins) * doppler_step_hz(sample_rate, num_samples), col


def test_cross_ambiguity_peak_at_target_delay_and_doppler():
    num_samples, history = 2**14, 80
    illuminator = _noise_illuminator(num_samples + history)
    target = TargetParameters(bistatic_range_km=20.0, doppler_hz=-60.0, snr_db=0.0, delay_samples=15)
    reference, surveillance = synthesize_channels(illuminator, history, SAMPLE_RATE, target, np.random.default_rng(1))
    rd_map = cross_ambiguity(reference, surveillance, SAMPLE_RATE, max_doppler_hz=185.0, max_range_cells=73)
    doppler, cell = _peak_location(rd_map, SAMPLE_RATE, num_samples, 185.0)
    assert cell == 15
    assert doppler == pytest.approx(-60.0, abs=doppler_step_hz(SAMPLE_RATE, num_samples))


def test_cross_ambiguity_shape_and_validation():
    signal = _noise_illuminator(4096)
    rd_map = cross_ambiguity(signal, signal, SAMPLE_RATE, 185.0, 50)
    assert rd_map.shape == (2 * doppler_bin_count(SAMPLE_RATE, 4096, 185.0) + 1, 50)
    with pytest.raises(ValueError):
        cross_ambiguity(signal, signal[:-1], SAMPLE_RATE, 185.0, 50)


def test_cross_ambiguity_matches_pyapril():
    detector = pytest.importorskip("pyapril.detector")
    reference, surveillance = _noise_illuminator(4096, 1), _noise_illuminator(4096, 2)
    ours = cross_ambiguity(reference, surveillance, SAMPLE_RATE, 185.0, 40)
    theirs = detector.cc_detector_fd(reference, surveillance, SAMPLE_RATE, 185.0, 40)
    np.testing.assert_allclose(ours, theirs, rtol=1e-6, atol=1e-6)


def test_fm_modulation_is_unit_modulus_with_bounded_deviation():
    t = np.arange(10_000) / SAMPLE_RATE
    signal = fm_modulate(np.sin(2 * np.pi * 1_000 * t), SAMPLE_RATE, deviation_hz=75e3)
    np.testing.assert_allclose(np.abs(signal), 1.0)
    instantaneous_hz = np.diff(np.unwrap(np.angle(signal))) * SAMPLE_RATE / (2 * np.pi)
    assert np.max(np.abs(instantaneous_hz)) <= 75e3 * 1.001


def test_fm_modulation_rejects_silence():
    with pytest.raises(ValueError, match="silent"):
        fm_modulate(np.zeros(100), SAMPLE_RATE, 75e3)


def test_fm_signal_from_audio_length_and_bounds():
    audio = np.sin(np.arange(10_000) / 7.0)
    signal, rate = fm_signal_from_audio(audio, 44_100, start=100, num_samples=5_000, upsample_factor=5, deviation_hz=75e3)
    assert signal.size == 5_000 and rate == 220_500.0
    with pytest.raises(ValueError):
        fm_signal_from_audio(audio, 44_100, start=9_999, num_samples=5_000, upsample_factor=5, deviation_hz=75e3)


def test_range_geometry():
    assert range_cells(100.0, SAMPLE_RATE) == 73  # 100 km / (c / fs)
    assert delay_in_samples(0.0, SAMPLE_RATE) == 0
    assert delay_in_samples(50.0, SAMPLE_RATE) == 37


def test_no_target_channel_is_noise_only():
    illuminator = _noise_illuminator(5_000)
    reference, surveillance = synthesize_channels(illuminator, 100, SAMPLE_RATE, None, np.random.default_rng(0))
    assert reference.size == surveillance.size == 4_900
    assert np.mean(np.abs(surveillance) ** 2) == pytest.approx(1.0, rel=0.1)
    assert abs(np.vdot(reference, surveillance)) / reference.size < 0.05  # uncorrelated with the illuminator


def test_target_delay_beyond_history_raises():
    target = TargetParameters(10.0, 0.0, 0.0, delay_samples=200)
    with pytest.raises(ValueError, match="history"):
        synthesize_channels(_noise_illuminator(1_000), 100, SAMPLE_RATE, target, np.random.default_rng(0))


def test_simulation_config_doppler_axis():
    assert SimulationConfig().max_doppler_hz == pytest.approx(185.3, abs=0.1)  # 1000 km/h at 100 MHz


def test_rd_map_normalisation():
    db_map = np.random.default_rng(0).normal(0, 3, size=(37, 73)).astype(np.float32)
    db_map[10, 20] += 60
    config = RangeDopplerMapConfig(dynamic_range_db=40.0)
    normalised = normalize_rd_map(db_map, config)
    assert normalised.min() >= 0.0 and normalised.max() == pytest.approx(1.0)
    tensor = rd_map_to_tensor(db_map, config)
    assert tensor.shape == (1, 37, 73) and tensor.dtype == torch.float32
    with pytest.raises(ValueError):
        rd_map_to_tensor(db_map[0], config)


def test_single_channel_model_and_checkpoint(tmp_path):
    model_config = ModelConfig(in_channels=1, pooled_size=(2, 2), global_pool="max")
    model = RangeDopplerCNN(model_config).eval()
    batch = torch.randn(2, 1, 109, 73)
    assert model(batch).shape == (2, len(CLASS_NAMES))
    path = save_checkpoint(tmp_path / "sim.pt", model, CLASS_NAMES, RangeDopplerMapConfig(30.0))
    loaded = load_checkpoint(path)
    assert loaded.preprocessing_config == RangeDopplerMapConfig(30.0)
    with torch.no_grad():
        assert torch.allclose(model(batch), loaded.model(batch))


def test_unknown_global_pool_raises():
    with pytest.raises(ValueError, match="global_pool"):
        RangeDopplerCNN(ModelConfig(global_pool="median"))


def test_generate_dataset_end_to_end(tmp_path):
    soundfile = pytest.importorskip("soundfile")
    from passive_radar.simulation.dataset_builder import generate_dataset, list_audio_files, load_simulated_records

    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    t = np.arange(44_100) / 44_100
    for name, freq in (("Song A (128kbit_AAC)Part001.wav", 440), ("Song A (128kbit_AAC)Part002.wav", 660)):
        soundfile.write(audio_dir / name, 0.5 * np.sin(2 * np.pi * freq * t), 44_100)
    config = SimulationConfig(integration_time_s=0.05, segments_per_recording=2, target_snr_db=(-10.0, -5.0))
    samples = generate_dataset(list_audio_files(audio_dir), tmp_path / "sim", config, progress=None)

    assert len(samples) == 2 * 2 * 2  # files x segments x {target, no_target}
    assert {s.source_id for s in samples} == {"song a"}  # parts of one recording share a group
    records = load_simulated_records(tmp_path / "sim")
    assert sorted(r.label for r in records) == [0, 0, 0, 0, 1, 1, 1, 1]
    first = np.load(records[0].path)
    assert first.ndim == 2 and first.dtype == np.float32
