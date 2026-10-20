"""FM illuminator-of-opportunity (IoO) signal generation from audio recordings.

Adapted from the FM IoO simulator notebook (pyAPRiL ``sim.FMSimulator``), with one
deliberate change: the baseband signal is generated as a *complex* exponential
(I/Q). A real-valued ``sin`` baseband has a mirror-symmetric spectrum, which makes
positive and negative Doppler shifts indistinguishable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

AUDIO_EXTENSIONS: tuple[str, ...] = (".mp3", ".wav", ".flac", ".ogg")


class AudioLoadError(RuntimeError):
    """Raised when an audio file is missing or cannot be decoded."""


def load_audio_mono(path: str | Path) -> tuple[np.ndarray, int]:
    """Read an audio file as a mono ``float64`` signal and its sample rate.

    Uses ``soundfile`` (libsndfile >= 1.1 decodes MP3).

    Raises:
        AudioLoadError: If the file is missing, unsupported or corrupt.
    """
    audio_path = Path(path)
    if not audio_path.is_file():
        raise AudioLoadError(f"Audio file not found: {audio_path}")
    try:
        import soundfile  # imported lazily: only needed for dataset generation
    except ImportError as error:  # pragma: no cover - environment specific
        raise AudioLoadError("The 'soundfile' package is required to read audio files") from error
    try:
        samples, sample_rate = soundfile.read(audio_path, dtype="float64", always_2d=True)
    except (RuntimeError, soundfile.LibsndfileError) as error:
        raise AudioLoadError(f"Could not decode audio file {audio_path}: {error}") from error
    return samples.mean(axis=1), int(sample_rate)


def upsample(signal: np.ndarray, factor: int) -> np.ndarray:
    """Polyphase up-sampling by an integer ``factor`` (anti-imaging filter included)."""
    if factor < 1:
        raise ValueError("Upsample factor must be >= 1")
    return signal if factor == 1 else resample_poly(signal, factor, 1)


def fm_modulate(modulating: np.ndarray, sample_rate: float, deviation_hz: float) -> np.ndarray:
    """Complex baseband FM: ``exp(j 2 pi k_fm / fs * cumsum(m))`` with peak deviation ``deviation_hz``.

    Raises:
        ValueError: If the modulating signal is silent (all zeros).
    """
    peak = np.max(np.abs(modulating))
    if peak == 0:
        raise ValueError("Modulating signal is silent; cannot normalise it")
    normalised = modulating / peak
    phase = 2 * np.pi * deviation_hz / sample_rate * np.cumsum(normalised)
    return np.exp(1j * phase)


def fm_signal_from_audio(
    audio: np.ndarray, audio_rate: int, start: int, num_samples: int, upsample_factor: int, deviation_hz: float
) -> tuple[np.ndarray, float]:
    """Cut ``audio[start:]``, up-sample it and FM-modulate it to ``num_samples`` baseband samples.

    Returns:
        The complex FM signal and its sample rate.

    Raises:
        ValueError: If the requested window does not fit in ``audio``.
    """
    needed = int(np.ceil(num_samples / upsample_factor)) + 1
    if start < 0 or start + needed > audio.size:
        raise ValueError(f"Audio window [{start}, {start + needed}) exceeds recording length {audio.size}")
    modulating = upsample(audio[start:start + needed], upsample_factor)[:num_samples]
    sample_rate = float(audio_rate * upsample_factor)
    return fm_modulate(modulating, sample_rate, deviation_hz), sample_rate
