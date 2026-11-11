"""Reproducibility helpers for reliability-calibration experiments."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_reproducibility_seed(seed: int) -> int:
    """Seed Python, NumPy, and PyTorch RNGs and return the normalized seed."""

    normalized_seed = int(seed)

    if normalized_seed < 0:
        raise ValueError("seed cannot be negative.")

    random.seed(normalized_seed)
    np.random.seed(normalized_seed)
    torch.manual_seed(normalized_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(normalized_seed)

    return normalized_seed
