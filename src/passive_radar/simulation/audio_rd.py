"""Audio file -> Range-Doppler maps, with exactly the processing used to build ``data/dataset_*``.

A sound file alone is only the *illuminator* (the FM broadcast). Whether a target exists depends on
what the surveillance antenna receives, so the surveillance channel is simulated here:

    reference    x_r = s + noise
    surveillance x_s = direct path + static clutter + [target echo] + noise
    ECA clutter cancellation -> cross-ambiguity -> dB map relative to the noise floor

Parameters are read from the dataset's ``config.json`` so maps match what the model was trained on.
Torch-free; needs numpy and ``ffmpeg`` (falls back to soundfile + scipy if ffmpeg is missing).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np




@dataclass(frozen=True)
class TargetSpec:
    """A simulated target echo (``None`` fields are drawn at random within the dataset's ranges)."""

    range_km: float | None = None
    doppler_hz: float | None = None
    snr_in_db: float | None = None


@dataclass
class SegmentMap:
    """One coherent processing interval and its Range-Doppler map."""

    index: int
    t_start_s: float
    t_end_s: float
    db_map: np.ndarray  # (Doppler, range), row 0 = +max Doppler, dB above noise floor
    has_target: bool
    range_km: float | None = None
    doppler_hz: float | None = None
    snr_in_db: float | None = None
    snr_out_db: float | None = None


def load_sim_config(dataset_dir: Path) -> dict:
    return json.loads((Path(dataset_dir) / "config.json").read_text(encoding="utf-8"))


def load_audio(path: str | Path, fs: int) -> np.ndarray:
    """Mono audio resampled to ``fs`` (float64). ``~`` in the path is expanded.

    Raises:
        FileNotFoundError: If the audio file does not exist.
    """
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Ses dosyasi bulunamadi: {path}")
    path = str(path)
    ffmpeg = shutil.which("ffmpeg") or next((p for p in ("/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg")
                                             if Path(p).exists()), None)
    if ffmpeg:
        out = subprocess.run([ffmpeg, "-v", "error", "-i", path, "-ac", "1", "-ar", str(fs), "-f", "f32le", "-"],
                             stdout=subprocess.PIPE, check=True).stdout
        return np.frombuffer(out, dtype=np.float32).astype(np.float64)
    import soundfile as sf  # fallback (mp3 needs libsndfile >= 1.1)
    from math import gcd
    from scipy.signal import resample_poly
    data, fs_in = sf.read(path, always_2d=True)
    mono = data.mean(axis=1)
    g = gcd(fs, fs_in)
    return resample_poly(mono, fs // g, fs_in // g)


def _cnoise(rng, n, power=1.0):
    return np.sqrt(power / 2) * (rng.standard_normal(n) + 1j * rng.standard_normal(n))


def _eca(xr, xs, K):
    N = len(xr)
    X = np.zeros((N, K), dtype=complex)
    for k in range(K):
        X[k:, k] = xr[:N - k]
    w, *_ = np.linalg.lstsq(X, xs, rcond=None)
    return xs - X @ w


def _caf(xr, xs, fs, r_max, fD_max, L):
    N = len(xr)
    nb = N // L
    nfft = int(round((fs / L) / (fs / (2 * N))))
    df = (fs / L) / nfft
    D = int(fD_max / df)
    rd = np.zeros((2 * D + 1, r_max), dtype=np.complex64)
    for tau in range(r_max):
        ref = np.zeros(N, dtype=complex)
        ref[tau:] = xr[:N - tau]
        p = (xs * np.conj(ref))[: nb * L].reshape(nb, L).sum(axis=1)
        spec = np.fft.fftshift(np.fft.fft(p, nfft))
        c = nfft // 2
        rd[:, tau] = spec[c - D: c + D + 1]
    return rd, df, D


def audio_to_rd_maps(
    audio_path: str | Path,
    cfg: dict,
    target: str = "random",
    target_spec: TargetSpec = TargetSpec(),
    max_segments: int | None = None,
    start_s: float = 0.0,
    seed: int = 0,
) -> list[SegmentMap]:
    """Cut ``audio_path`` into CPIs and return one simulated RD map per CPI.

    ``target``: ``"none"`` (no echo), ``"always"`` (echo in every CPI) or ``"random"`` (50 %).
    """
    if target not in ("none", "always", "random"):
        raise ValueError("target must be 'none', 'always' or 'random'")
    rng = np.random.default_rng(seed)
    fs, N, r_max, fD_max = cfg["fs"], cfg["N"], cfg["r_max"], cfg["fD_max"]
    audio = load_audio(audio_path, fs)
    s_all = np.exp(1j * 2 * np.pi * cfg["fdev"] / fs * np.cumsum(audio / (np.max(np.abs(audio)) + 1e-12)))
    pad = r_max + 1
    first = int(round(start_s * fs / N))
    n_seg = (len(s_all) - pad) // N
    segs = range(first, n_seg if max_segments is None else min(n_seg, first + max_segments))
    out = []
    for k in segs:
        st = pad + k * N
        s = s_all[st:st + N]
        if np.sqrt(np.mean(audio[st:st + N] ** 2)) < cfg["silence_rms"]:
            continue  # silent audio -> no FM bandwidth -> meaningless map
        xr = s + _cnoise(rng, N, 10 ** (-cfg["ref_snr_db"] / 10))
        xs = _cnoise(rng, N, 1.0)
        xs += 10 ** (rng.uniform(*cfg["dnr_db"]) / 20) * s * np.exp(1j * rng.uniform(0, 2 * np.pi))
        for _ in range(rng.integers(cfg["n_clutter"][0], cfg["n_clutter"][1] + 1)):
            d = int(rng.integers(1, cfg["clutter_max_delay"] + 1))
            xs += 10 ** (rng.uniform(*cfg["clutter_cnr_db"]) / 20) * s_all[st - d: st - d + N] \
                  * np.exp(1j * rng.uniform(0, 2 * np.pi))

        has_t = target == "always" or (target == "random" and rng.random() < 0.5)
        seg = SegmentMap(k, st / fs, (st + N) / fs, None, has_t)
        if has_t:
            cell = cfg["range_cell_m"] / 1e3
            tau = (int(round(target_spec.range_km / cell)) if target_spec.range_km is not None
                   else int(rng.integers(cfg["tgt_min_delay"], r_max - 1)))
            tau = int(np.clip(tau, 1, r_max - 1))
            fd = (target_spec.doppler_hz if target_spec.doppler_hz is not None
                  else float(rng.uniform(cfg["tgt_min_abs_fd"], fD_max * 0.95)) * rng.choice([-1, 1]))
            snr_in = (target_spec.snr_in_db if target_spec.snr_in_db is not None
                      else float(rng.uniform(*cfg["tgt_snr_in_db"])))
            n = np.arange(N)
            xs += 10 ** (snr_in / 20) * s_all[st - tau: st - tau + N] \
                  * np.exp(1j * (2 * np.pi * fd / fs * (n + st) + rng.uniform(0, 2 * np.pi)))
            seg.range_km, seg.doppler_hz, seg.snr_in_db = tau * cell, fd, snr_in

        rd, df, D = _caf(xr, _eca(xr, xs, cfg["eca_K"]), fs, r_max, fD_max, cfg["batch_L"])
        P = np.abs(rd) ** 2
        noise = np.ones_like(P, dtype=bool)
        noise[D - 3: D + 4, :] = False
        rel_db = 10 * np.log10(P + 1e-30) - 10 * np.log10(np.median(P[noise]))
        if has_t:
            db = int(round(seg.doppler_hz / df)) + D
            r = int(round(seg.range_km / (cfg["range_cell_m"] / 1e3)))
            seg.snr_out_db = float(rel_db[max(0, db - 2): db + 3, max(0, r - 1): r + 2].max())
        seg.db_map = np.flipud(rel_db).astype(np.float16)  # same layout/dtype as rd_npy in the dataset
        out.append(seg)
    return out
