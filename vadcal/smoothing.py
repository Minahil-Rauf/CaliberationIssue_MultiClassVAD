"""Temporal median smoothing of frame posteriors, as deployed VADs do (E4)."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter


def median_smooth(probs: np.ndarray, window: int) -> np.ndarray:
    """probs: (clips, frames, K). Median over `window` frames per class, then renormalised."""
    if window <= 1:
        return probs
    out = median_filter(probs, size=(1, window, 1), mode="nearest")
    return out / np.maximum(out.sum(-1, keepdims=True), 1e-12)
