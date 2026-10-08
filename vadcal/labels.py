"""Frame-level labels from the clean source signals (before mixing)."""
from __future__ import annotations

import numpy as np


def frame_db(x: np.ndarray, hop: int) -> np.ndarray:
    """Power in dB (= 20*log10 RMS) of non-overlapping hop-sized frames."""
    n = len(x) // hop
    if n == 0:
        return np.zeros(0)
    frames = x[: n * hop].astype(np.float64).reshape(n, hop)
    return 10.0 * np.log10(np.mean(frames ** 2, axis=1) + 1e-12)


def _runs(mask: np.ndarray, value: bool) -> list[tuple[int, int]]:
    """(start, end) of each run of `value` in a boolean mask."""
    m = np.concatenate([[False], mask == value, [False]]).astype(np.int8)
    d = np.diff(m)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def close_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    """Fill inactive runs of length <= max_gap that have activity on both sides."""
    out = mask.copy()
    for s, e in _runs(mask, False):
        if s > 0 and e < len(mask) and e - s <= max_gap:
            out[s:e] = True
    return out


def remove_islands(mask: np.ndarray, min_len: int) -> np.ndarray:
    """Drop active runs shorter than min_len."""
    out = mask.copy()
    for s, e in _runs(mask, True):
        if e - s < min_len:
            out[s:e] = False
    return out


def speech_activity(x: np.ndarray, hop: int, sr: int, lcfg: dict) -> np.ndarray:
    db = frame_db(x, hop)
    if db.size == 0:
        return np.zeros(0, dtype=bool)
    active = db > max(db.max() + lcfg["speech_rel_db"], lcfg["speech_abs_dbfs"])
    frames_per_ms = sr / hop / 1000.0
    active = close_gaps(active, int(round(lcfg["close_gap_ms"] * frames_per_ms)))
    return remove_islands(active, max(1, int(round(lcfg["min_island_ms"] * frames_per_ms))))


def music_activity(x: np.ndarray, hop: int, sr: int, lcfg: dict) -> np.ndarray:
    active = frame_db(x, hop) > lcfg["music_abs_dbfs"]
    return close_gaps(active, int(round(lcfg["close_gap_ms"] * sr / hop / 1000.0)))


def combine(speech: np.ndarray, music: np.ndarray) -> np.ndarray:
    """0 background, 1 speech, 2 music, 3 speech+music."""
    return (speech.astype(np.int8) + 2 * music.astype(np.int8)).astype(np.int8)


def class_shares(labels: np.ndarray, n_classes: int = 4) -> np.ndarray:
    counts = np.bincount(labels.ravel().astype(np.int64), minlength=n_classes)
    return counts / max(counts.sum(), 1)
