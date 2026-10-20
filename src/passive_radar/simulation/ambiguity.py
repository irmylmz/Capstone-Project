"""Cross-ambiguity (Range-Doppler) computation for passive radar.

NumPy re-implementation of pyAPRiL ``detector.cc_detector_fd`` (frequency-domain
cross-correlation detector) so the project does not depend on pyFFTW. The output
layout is identical: rows = Doppler bins (negative -> positive), columns = range
cells (bistatic delay in samples).
"""

from __future__ import annotations

import numpy as np


def doppler_bin_count(sample_rate: float, num_samples: int, max_doppler_hz: float) -> int:
    """Number of Doppler bins on each side of zero (bin width = fs / (2N))."""
    return int(max_doppler_hz / doppler_step_hz(sample_rate, num_samples))


def doppler_step_hz(sample_rate: float, num_samples: int) -> float:
    """Doppler resolution of the zero-padded cross-correlation."""
    return sample_rate / (2 * num_samples)


def cross_ambiguity(
    reference: np.ndarray,
    surveillance: np.ndarray,
    sample_rate: float,
    max_doppler_hz: float,
    max_range_cells: int,
    chunk_size: int = 32,
) -> np.ndarray:
    """Range-Doppler map of ``surveillance`` against ``reference``.

    Args:
        reference: Reference-channel samples (complex, length N).
        surveillance: Surveillance-channel samples (complex, length N).
        sample_rate: Sampling frequency [Hz].
        max_doppler_hz: Largest |Doppler| to compute [Hz].
        max_range_cells: Number of range (delay) cells to keep.
        chunk_size: Doppler bins processed per vectorised inverse FFT.

    Returns:
        Complex matrix of shape ``(2 * D + 1, max_range_cells)`` with D Doppler bins per side.
    """
    if reference.shape != surveillance.shape or reference.ndim != 1:
        raise ValueError("reference and surveillance must be 1-D arrays of equal length")
    num_samples = reference.size
    if max_range_cells > num_samples:
        raise ValueError("max_range_cells cannot exceed the number of samples")

    bins_per_side = doppler_bin_count(sample_rate, num_samples, max_doppler_hz)
    zeros = np.zeros(num_samples, dtype=complex)
    surveillance_spectrum = np.fft.fft(np.concatenate((zeros, surveillance)))
    reference_spectrum = np.fft.fft(np.concatenate((reference, zeros)))

    shifts = np.arange(-bins_per_side, bins_per_side + 1)
    rd_map = np.empty((shifts.size, max_range_cells), dtype=complex)
    for start in range(0, shifts.size, chunk_size):
        chunk = shifts[start:start + chunk_size]
        shifted = np.stack([np.roll(reference_spectrum, shift) for shift in chunk])
        correlation = np.fft.ifft(surveillance_spectrum[None, :] * np.conj(shifted), axis=1)
        rd_map[start:start + chunk.size] = correlation[:, num_samples:num_samples + max_range_cells]
    return rd_map
