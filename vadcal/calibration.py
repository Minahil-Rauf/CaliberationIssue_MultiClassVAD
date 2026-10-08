"""Post-hoc calibration fitted on validation logits (E3).

Temperature scaling: p = softmax(z / T). One number; never changes the predicted class.
Vector scaling: p = softmax(z * w + b) with per-class w, b. Can change predictions.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize_scalar

from .metrics import log_softmax, softmax


def _nll(z: np.ndarray, y: np.ndarray, T: float) -> float:
    return float(-log_softmax(z / T)[np.arange(len(y)), y].mean())


def fit_temperature(z: np.ndarray, y: np.ndarray, grid: np.ndarray | None = None) -> float:
    """z: (n, K) logits, y: (n,). Grid search on NLL, then a bounded refinement."""
    z = z.astype(np.float64)
    y = y.astype(np.int64)
    grid = np.geomspace(0.05, 20.0, 120) if grid is None else grid
    losses = [_nll(z, y, T) for T in grid]
    i = int(np.argmin(losses))
    lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    res = minimize_scalar(lambda t: _nll(z, y, t), bounds=(lo, hi), method="bounded")
    return float(res.x) if res.fun <= losses[i] else float(grid[i])


def apply_temperature(z: np.ndarray, T: float) -> np.ndarray:
    return softmax(z.astype(np.float64) / T)


def fit_vector_scaling(z: np.ndarray, y: np.ndarray, max_iter: int = 200) -> tuple[np.ndarray, np.ndarray]:
    zt = torch.from_numpy(z.astype(np.float32))
    yt = torch.from_numpy(y.astype(np.int64))
    k = z.shape[1]
    w = torch.ones(k, requires_grad=True)
    b = torch.zeros(k, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=1.0, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(zt * w + b, yt)
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach().numpy().astype(np.float64), b.detach().numpy().astype(np.float64)


def apply_vector_scaling(z: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    return softmax(z.astype(np.float64) * w + b)
