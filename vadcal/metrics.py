"""Accuracy and calibration metrics on flat frame arrays: probs (n, K), y (n,).

ECE: top-label expected calibration error with equal-width bins.
Classwise-ECE: per-class ECE on p_k vs 1[y == k], averaged over classes (Kull et al., 2019).
Macro-F1 averages over classes that occur in y.
"""
from __future__ import annotations

import numpy as np


def softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def log_softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    z = z - z.max(axis=axis, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=axis, keepdims=True))


def bin_index(p: np.ndarray, n_bins: int) -> np.ndarray:
    return np.minimum((p * n_bins).astype(np.int64), n_bins - 1)


def ece(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    conf = probs.max(1)
    correct = (probs.argmax(1) == y).astype(np.float64)
    b = bin_index(conf, n_bins)
    return float(np.abs(np.bincount(b, correct, n_bins) - np.bincount(b, conf, n_bins)).sum() / len(y))


def classwise_ece(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> tuple[float, np.ndarray]:
    per_class = np.zeros(probs.shape[1])
    for k in range(probs.shape[1]):
        b = bin_index(probs[:, k], n_bins)
        gap = np.bincount(b, (y == k).astype(np.float64), n_bins) - np.bincount(b, probs[:, k], n_bins)
        per_class[k] = np.abs(gap).sum() / len(y)
    return float(per_class.mean()), per_class


def nll(probs: np.ndarray, y: np.ndarray) -> float:
    return float(-np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, None)).mean())


def brier(probs: np.ndarray, y: np.ndarray) -> float:
    onehot = np.eye(probs.shape[1])[y]
    return float(((probs - onehot) ** 2).sum(1).mean())


def confusion(y: np.ndarray, pred: np.ndarray, n_classes: int) -> np.ndarray:
    """C[true, pred]."""
    return np.bincount(y * n_classes + pred, minlength=n_classes ** 2).reshape(n_classes, n_classes)


def f1_from_confusion(C: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-class F1 (NaN for a class absent from y) and macro-F1 over present classes, C: (..., K, K)."""
    C = np.asarray(C, dtype=np.float64)
    tp = np.diagonal(C, axis1=-2, axis2=-1).copy()
    fp = C.sum(-2) - tp
    fn = C.sum(-1) - tp
    denom = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    present = C.sum(-1) > 0
    macro = (f1 * present).sum(-1) / np.maximum(present.sum(-1), 1)
    return np.where(present, f1, np.nan), macro


def quick_metrics(logits: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> dict:
    """Macro-F1, NLL and ECE from (clips, frames, K) logits; used for early stopping and logs."""
    k = logits.shape[-1]
    probs = softmax(logits.reshape(-1, k).astype(np.float64))
    y = labels.reshape(-1).astype(np.int64)
    _, macro = f1_from_confusion(confusion(y, probs.argmax(1), k))
    return {"macro_f1": float(macro), "nll": nll(probs, y), "ece": ece(probs, y, n_bins)}
