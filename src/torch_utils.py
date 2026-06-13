"""Shared Torch/numpy helpers used by multiple model wrappers.

- ``default_device`` / ``describe_device``: pick and label a torch device.
- ``Standardizer``: float64 z-score for X and y, fit on training rows only.
- ``to_tensor``: small wrapper around ``torch.as_tensor``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

def default_device(pref="auto"):
    """Resolve a torch device.

    - `auto`: prefer CUDA > MPS > CPU. CUDA is preferred when both are
      present (e.g., a CUDA workstation running PyTorch nightly with MPS
      build flags) because dense MHSA throughput is consistently higher
      on CUDA than on MPS.
    - `cuda` / `mps`: require the backend to be available, raise otherwise.
    - `cpu`: always succeeds.
    """
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "--ft_device cuda requested but torch.cuda.is_available() is False. "
                "Pass --ft_device auto to fall back, or install a CUDA-enabled PyTorch."
            )
        return torch.device("cuda")
    if pref == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                "--ft_device mps requested but torch.backends.mps.is_available() is False."
            )
        return torch.device("mps")
    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def describe_device(device):
    """Human-readable label for the resolved device, e.g.
    'cuda (NVIDIA A100-SXM4-80GB)' or 'mps (Apple Silicon GPU)' or 'cpu'."""
    device = torch.device(device) if not isinstance(device, torch.device) else device
    if device.type == "cuda":
        idx = device.index if device.index is not None else torch.cuda.current_device()
        return f"cuda:{idx} ({torch.cuda.get_device_name(idx)})"
    if device.type == "mps":
        return "mps (Apple Silicon GPU)"
    return "cpu"


# ---------------------------------------------------------------------------
# Standardizer
# ---------------------------------------------------------------------------

@dataclass
class Standardizer:
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: float
    y_std: float

    @classmethod
    def fit(cls, X: np.ndarray, y: np.ndarray) -> "Standardizer":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        x_mean = X.mean(axis=0)
        x_std = X.std(axis=0, ddof=0)
        x_std[x_std == 0] = 1.0
        y_mean = float(y.mean())
        y_std = float(y.std(ddof=0))
        if y_std == 0:
            y_std = 1.0
        return cls(x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std)

    def transform_X(self, X):
        return (np.asarray(X, dtype=np.float64) - self.x_mean) / self.x_std

    def transform_y(self, y):
        return (np.asarray(y, dtype=np.float64) - self.y_mean) / self.y_std

    def inverse_y(self, yz):
        return np.asarray(yz, dtype=np.float64) * self.y_std + self.y_mean


# ---------------------------------------------------------------------------
# Tensor helper
# ---------------------------------------------------------------------------

def to_tensor(arr, dtype, device):
    return torch.as_tensor(np.asarray(arr), dtype=dtype, device=device)
