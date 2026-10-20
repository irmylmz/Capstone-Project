"""Runtime helpers: device selection and reproducibility."""

from __future__ import annotations

import random

import numpy as np
import torch

_SUPPORTED_DEVICES = ("auto", "cpu", "mps", "cuda")


def resolve_device(preference: str = "auto") -> torch.device:
    """Pick a torch device. ``"auto"`` prefers CUDA, then Apple Silicon MPS, then CPU.

    Raises:
        ValueError: For an unknown preference.
        RuntimeError: If an explicitly requested accelerator is unavailable.
    """
    if preference not in _SUPPORTED_DEVICES:
        raise ValueError(f"Unknown device '{preference}'. Choose from {_SUPPORTED_DEVICES}")
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if preference == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if preference == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available on this machine")
    return torch.device(preference)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch RNGs (results on MPS may still vary slightly)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
