"""Single-target passive radar scenario: reference / surveillance channel synthesis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from passive_radar.config import SPEED_OF_LIGHT_M_S, SimulationConfig


@dataclass(frozen=True)
class TargetParameters:
    """Ground truth of one simulated target echo."""

    bistatic_range_km: float
    doppler_hz: float
    snr_db: float
    delay_samples: int


def range_cells(max_range_km: float, sample_rate: float) -> int:
    """Number of bistatic range cells (one cell = c / fs metres) up to ``max_range_km``."""
    return int(max_range_km * 1e3 / SPEED_OF_LIGHT_M_S * sample_rate)


def delay_in_samples(bistatic_range_km: float, sample_rate: float) -> int:
    """Bistatic range -> integer sample delay of the echo relative to the reference."""
    return int(round(bistatic_range_km * 1e3 / SPEED_OF_LIGHT_M_S * sample_rate))


def sample_target(rng: np.random.Generator, config: SimulationConfig, sample_rate: float) -> TargetParameters:
    """Draw a target uniformly in range, Doppler and SNR within the configured limits."""
    range_km = float(rng.uniform(*config.target_range_km))
    doppler_limit = config.max_doppler_hz - config.target_doppler_margin_hz
    return TargetParameters(
        bistatic_range_km=range_km,
        doppler_hz=float(rng.uniform(-doppler_limit, doppler_limit)),
        snr_db=float(rng.uniform(*config.target_snr_db)),
        delay_samples=delay_in_samples(range_km, sample_rate),
    )


def complex_noise(rng: np.random.Generator, num_samples: int) -> np.ndarray:
    """Circular complex white Gaussian noise with unit power per sample."""
    return (rng.standard_normal(num_samples) + 1j * rng.standard_normal(num_samples)) / np.sqrt(2)


def synthesize_channels(
    illuminator: np.ndarray,
    history: int,
    sample_rate: float,
    target: TargetParameters | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Build (reference, surveillance) channels of length ``illuminator.size - history``.

    * reference = the noise-free illuminator (ideal reference channel)
    * surveillance = noise [+ echo], where the echo is the illuminator delayed by
      ``target.delay_samples``, Doppler-shifted by ``target.doppler_hz`` and scaled to
      ``target.snr_db`` relative to the unit-power noise.

    ``history`` leading samples of ``illuminator`` are only used to form delayed echoes.

    Raises:
        ValueError: If the target delay exceeds the available history.
    """
    num_samples = illuminator.size - history
    reference = illuminator[history:]
    surveillance = complex_noise(rng, num_samples)
    if target is not None:
        if not 0 <= target.delay_samples <= history:
            raise ValueError(f"Target delay {target.delay_samples} exceeds signal history {history}")
        start = history - target.delay_samples
        delayed = illuminator[start:start + num_samples]
        doppler = np.exp(1j * 2 * np.pi * target.doppler_hz / sample_rate * np.arange(num_samples))
        amplitude = np.sqrt(10 ** (target.snr_db / 10) / np.mean(np.abs(delayed) ** 2))
        surveillance = surveillance + amplitude * delayed * doppler
    return reference, surveillance
