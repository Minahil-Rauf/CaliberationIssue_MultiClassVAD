"""Clip-level bootstrap via per-clip sufficient statistics.

Neighbouring frames are nearly identical, so resampling frames would give falsely narrow
confidence intervals. We resample clips. Every metric is a function of per-clip sums
(bin counts, confusion matrix, NLL/Brier sums), so a resample is a weighted sum of rows:
`weights @ stats`, which makes 1,000 resamples take seconds.
"""
from __future__ import annotations

import numpy as np

from .config import CLASSES
from .metrics import bin_index, f1_from_confusion


def layout(n_bins: int, k: int) -> tuple[dict[str, slice], int]:
    sizes = [("n", n_bins), ("sconf", n_bins), ("scorr", n_bins), ("cw_sp", k * n_bins),
             ("cw_st", k * n_bins), ("conf", k * k), ("nll", 1), ("brier", 1), ("frames", 1)]
    out, start = {}, 0
    for name, size in sizes:
        out[name] = slice(start, start + size)
        start += size
    return out, start


def clip_stats(probs: np.ndarray, y: np.ndarray, n_bins: int) -> np.ndarray:
    """probs (clips, frames, K), y (clips, frames) -> stats (clips, D)."""
    n_clips, n_frames, k = probs.shape
    P = probs.reshape(-1, k).astype(np.float64)
    Y = y.reshape(-1).astype(np.int64)
    c = np.repeat(np.arange(n_clips), n_frames)
    size = n_clips * n_bins

    conf = P.max(1)
    pred = P.argmax(1)
    idx = c * n_bins + bin_index(conf, n_bins)
    n = np.bincount(idx, minlength=size).reshape(n_clips, n_bins)
    sconf = np.bincount(idx, conf, size).reshape(n_clips, n_bins)
    scorr = np.bincount(idx, (pred == Y).astype(np.float64), size).reshape(n_clips, n_bins)

    cw_sp = np.zeros((n_clips, k, n_bins))
    cw_st = np.zeros((n_clips, k, n_bins))
    for j in range(k):
        idj = c * n_bins + bin_index(P[:, j], n_bins)
        cw_sp[:, j] = np.bincount(idj, P[:, j], size).reshape(n_clips, n_bins)
        cw_st[:, j] = np.bincount(idj, (Y == j).astype(np.float64), size).reshape(n_clips, n_bins)

    cm = np.bincount(c * k * k + Y * k + pred, minlength=n_clips * k * k).reshape(n_clips, k * k)
    p_true = P[np.arange(len(Y)), Y]
    nll = np.bincount(c, -np.log(np.clip(p_true, 1e-12, None)), n_clips)
    brier = np.bincount(c, (P ** 2).sum(1) - 2 * p_true + 1, n_clips)
    frames = np.full(n_clips, n_frames, dtype=np.float64)
    return np.concatenate([n, sconf, scorr, cw_sp.reshape(n_clips, -1), cw_st.reshape(n_clips, -1),
                           cm, nll[:, None], brier[:, None], frames[:, None]], axis=1).astype(np.float64)


def metrics_from_stats(S: np.ndarray, n_bins: int, k: int) -> dict[str, np.ndarray]:
    """S: (D,) or (B, D) summed stats -> dict of (B,) metric arrays."""
    S = np.atleast_2d(np.asarray(S, dtype=np.float64))
    L, _ = layout(n_bins, k)
    B = S.shape[0]
    N = S[:, L["frames"]][:, 0]
    ece = np.abs(S[:, L["scorr"]] - S[:, L["sconf"]]).sum(1) / N
    cw = np.abs(S[:, L["cw_st"]] - S[:, L["cw_sp"]]).reshape(B, k, n_bins).sum(2) / N[:, None]
    C = S[:, L["conf"]].reshape(B, k, k)
    f1, macro = f1_from_confusion(C)
    out = {"macro_f1": macro, "accuracy": np.trace(C, axis1=1, axis2=2) / N, "ece": ece,
           "cw_ece": cw.mean(1), "nll": S[:, L["nll"]][:, 0] / N, "brier": S[:, L["brier"]][:, 0] / N}
    for j, name in enumerate(CLASSES[:k]):
        out[f"f1_{name}"] = f1[:, j]
        out[f"cwece_{name}"] = cw[:, j]
    return out


def reliability(S: np.ndarray, n_bins: int, k: int) -> dict[str, np.ndarray]:
    """Per-bin count, accuracy and mean confidence from summed stats (D,)."""
    L, _ = layout(n_bins, k)
    n = S[L["n"]]
    with np.errstate(invalid="ignore", divide="ignore"):
        return {"bin": np.arange(n_bins), "lo": np.arange(n_bins) / n_bins, "hi": (np.arange(n_bins) + 1) / n_bins,
                "count": n, "accuracy": S[L["scorr"]] / n, "confidence": S[L["sconf"]] / n}


def resample_weights(n_clips: int, n_resamples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.multinomial(n_clips, np.full(n_clips, 1.0 / n_clips), size=n_resamples).astype(np.float64)


def _point(S: np.ndarray, n_bins: int, k: int) -> dict[str, float]:
    return {m: float(v[0]) for m, v in metrics_from_stats(S.sum(0), n_bins, k).items()}


def bootstrap_metrics(S: np.ndarray, n_bins: int, k: int, n_resamples: int, seed: int, alpha: float = 0.05):
    """Point estimates plus percentile CIs (lo, hi) over clip resamples."""
    point = _point(S, n_bins, k)
    if n_resamples <= 0:
        return point, {}, {}
    boot = metrics_from_stats(resample_weights(S.shape[0], n_resamples, seed) @ S, n_bins, k)
    lo = {m: float(np.nanpercentile(v, 100 * alpha / 2)) for m, v in boot.items()}
    hi = {m: float(np.nanpercentile(v, 100 * (1 - alpha / 2))) for m, v in boot.items()}
    return point, lo, hi


def paired_difference(S_a: np.ndarray, S_b: np.ndarray, n_bins: int, k: int, n_resamples: int, seed: int,
                      metrics: tuple[str, ...], alpha: float = 0.05) -> list[dict]:
    """a - b on the same test clips, with a paired clip-bootstrap CI."""
    assert S_a.shape == S_b.shape, "paired comparison needs the same clips"
    pa, pb = _point(S_a, n_bins, k), _point(S_b, n_bins, k)
    W = resample_weights(S_a.shape[0], max(n_resamples, 1), seed)
    ba, bb = metrics_from_stats(W @ S_a, n_bins, k), metrics_from_stats(W @ S_b, n_bins, k)
    rows = []
    for m in metrics:
        d = ba[m] - bb[m]
        lo, hi = np.nanpercentile(d, 100 * alpha / 2), np.nanpercentile(d, 100 * (1 - alpha / 2))
        rows.append({"metric": m, "a": pa[m], "b": pb[m], "diff": pa[m] - pb[m],
                     "diff_lo": float(lo), "diff_hi": float(hi), "ci_excludes_zero": bool(lo > 0 or hi < 0)})
    return rows
