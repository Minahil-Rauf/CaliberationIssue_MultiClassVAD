"""Synthetic stand-ins for MUSAN, LibriSpeech, ESC-50 and GTZAN, in the same folder layout.

Used by the smoke test and the unit tests so the whole pipeline can be checked
in minutes without downloading anything. The signals are crude (harmonic "speech"
with syllable gating, chord "music", filtered noise) and say nothing about real results.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import lfilter


def _normalise(x: np.ndarray, peak: float = 0.5) -> np.ndarray:
    return (peak * x / (np.abs(x).max() + 1e-9)).astype(np.float32)


def speech_like(rng: np.random.Generator, seconds: float, sr: int) -> np.ndarray:
    n = int(seconds * sr)
    t = np.arange(n) / sr
    f0 = rng.uniform(100, 230) * (1 + 0.03 * np.sin(2 * np.pi * rng.uniform(3, 6) * t))
    phase = 2 * np.pi * np.cumsum(f0) / sr
    x = sum((0.7 ** h) * np.sin(h * phase) for h in range(1, 12))
    syllables = np.abs(np.sin(2 * np.pi * rng.uniform(3, 5) * t)) ** 0.5
    gate, i = np.zeros(n), 0
    while i < n:
        on, off = int(rng.uniform(0.3, 1.2) * sr), int(rng.uniform(0.1, 0.6) * sr)
        gate[i:i + on] = 1
        i += on + off
    return _normalise(x * syllables * gate)


def music_like(rng: np.random.Generator, seconds: float, sr: int) -> np.ndarray:
    n = int(seconds * sr)
    scale = 220 * 2 ** (np.array([0, 2, 4, 5, 7, 9, 11, 12]) / 12)
    out, i = np.zeros(n), 0
    while i < n:
        d = int(rng.uniform(0.25, 1.0) * sr)
        tt = np.arange(d) / sr
        notes = rng.choice(scale, size=3) * rng.choice([1, 2])
        seg = sum(np.sin(2 * np.pi * f * tt) for f in notes) * np.exp(-tt * rng.uniform(0.5, 3))
        out[i:i + d] += seg[: n - i]
        i += d
    return _normalise(out)


def noise_like(rng: np.random.Generator, seconds: float, sr: int) -> np.ndarray:
    n = int(seconds * sr)
    a = rng.uniform(0.5, 0.98)
    y = lfilter([1 - a], [1, -a], rng.standard_normal(n))
    return _normalise(y * (1 + 0.5 * np.sin(2 * np.pi * rng.uniform(0.1, 1) * np.arange(n) / sr)))


def generate_all(raw: Path, seed: int = 0, n_musan: int = 15, musan_seconds: float = 20.0,
                 n_libri: int = 8, n_esc: int = 10, n_gtzan: int = 4) -> None:
    marker = raw / ".done_fake"
    if marker.exists():
        print(f"synthetic sources already in {raw}, skipping")
        return
    rng = np.random.default_rng(seed)
    make = {"speech": speech_like, "music": music_like, "noise": noise_like}
    sub = {"speech": "librivox", "music": "fma", "noise": "free-sound"}
    for kind, fn in make.items():
        d = raw / "musan" / kind / sub[kind]
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n_musan):
            sf.write(d / f"{kind}-{sub[kind]}-{i:04d}.wav", fn(rng, musan_seconds, 16000), 16000)

    for i in range(n_libri):
        spk, chap = 1000 + i % 3, 2000 + i % 2
        d = raw / "LibriSpeech" / "test-clean" / str(spk) / str(chap)
        d.mkdir(parents=True, exist_ok=True)
        sf.write(d / f"{spk}-{chap}-{i:04d}.flac", speech_like(rng, 12.0, 16000), 16000)

    esc = raw / "ESC-50-master"
    (esc / "audio").mkdir(parents=True, exist_ok=True)
    (esc / "meta").mkdir(parents=True, exist_ok=True)
    cats = ["rain", "dog", "engine", "wind", "coughing"]  # "coughing" must be filtered out
    rows = []
    for i in range(n_esc):
        cat = cats[i % len(cats)]
        fn = f"{1 + i % 5}-{100000 + i}-A-{i % len(cats)}.wav"
        sf.write(esc / "audio" / fn, noise_like(rng, 5.0, 44100), 44100)
        rows.append({"filename": fn, "fold": 1 + i % 5, "target": i % len(cats), "category": cat,
                     "esc10": False, "src_file": 100000 + i, "take": "A"})
    pd.DataFrame(rows).to_csv(esc / "meta" / "esc50.csv", index=False)

    for genre in ("blues", "rock", "hiphop"):  # "hiphop" must be filtered out
        d = raw / "genres" / genre
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n_gtzan):
            sf.write(d / f"{genre}.{i:05d}.wav", music_like(rng, 10.0, 22050), 22050)

    marker.write_text("ok")
    print(f"wrote synthetic sources to {raw}")
