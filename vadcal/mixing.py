"""Build labelled 10 s mixtures event by event.

Each clip has a quiet noise bed plus a sequence of 1-4 s events. Each event's class
(background / speech / music / speech+music) is drawn from `mixing.event_probs`.
Frame labels come from the clean speech and music signals, so they are exact.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from math import gcd
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly
from tqdm import tqdm

from .config import CLASSES, SPLITS, audio_dims
from .labels import class_shares, combine, frame_db, music_activity, speech_activity


@dataclass(frozen=True)
class Source:
    path: str
    sr: int
    frames: int


def load_pools(index: pd.DataFrame, raw: Path, split: str) -> dict[str, list[Source]]:
    sub = index[index["split"] == split]
    return {kind: [Source(str(raw / p), int(s), int(f))
                   for p, s, f in zip(g["path"], g["sr"], g["frames"])]
            for kind, g in sub.groupby("kind")}


def read_segment(src: Source, n: int, sr: int, rng: np.random.Generator, loop: bool = False) -> np.ndarray:
    """A random n-sample excerpt at `sr`. Short sources are looped (loop=True) or zero-padded."""
    n_src = int(np.ceil(n * src.sr / sr)) + 16
    start = int(rng.integers(0, src.frames - n_src + 1)) if src.frames > n_src else 0
    x, file_sr = sf.read(src.path, start=start, frames=min(n_src, src.frames),
                         dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if file_sr != sr:
        g = gcd(sr, file_sr)
        x = resample_poly(x, sr // g, file_sr // g).astype(np.float32)
    if len(x) < n:
        if loop and len(x) > 0:
            x = np.tile(x, int(np.ceil(n / len(x))))
        else:
            x = np.pad(x, (0, n - len(x)))
    return x[:n]


def set_level(x: np.ndarray, target_dbfs: float, hop: int, rel_db: float = -35.0) -> np.ndarray:
    """Scale so the power of frames within `rel_db` of the loudest frame equals target_dbfs."""
    db = frame_db(x, hop)
    if db.size == 0 or db.max() < -100:
        return np.zeros_like(x)
    current = 10 * np.log10(np.mean(10 ** (db[db > db.max() + rel_db] / 10)))
    return (x * 10 ** ((target_dbfs - current) / 20)).astype(np.float32)


def fade(x: np.ndarray, n: int) -> np.ndarray:
    n = min(n, len(x) // 2)
    if n > 0:
        ramp = (0.5 - 0.5 * np.cos(np.pi * np.arange(n) / n)).astype(np.float32)
        x[:n] *= ramp
        x[-n:] *= ramp[::-1]
    return x


def build_clip(rng: np.random.Generator, pools: dict[str, list[Source]], cfg: dict):
    sr, hop, n_samples, n_frames = audio_dims(cfg)
    m, lcfg = cfg["mixing"], cfg["labels"]
    probs = np.array([m["event_probs"][c] for c in CLASSES], dtype=float)
    probs /= probs.sum()
    fade_n = int(sr * m["fade_ms"] / 1000)

    def pick(kind: str) -> Source:
        pool = pools[kind]
        return pool[int(rng.integers(len(pool)))]

    mix = np.zeros(n_samples, dtype=np.float32)
    speech_mask = np.zeros(n_frames, dtype=bool)
    music_mask = np.zeros(n_frames, dtype=bool)

    bed_src = pick("noise")
    bed = set_level(read_segment(bed_src, n_samples, sr, rng, loop=True),
                    rng.uniform(*m["noise_bed_dbfs"]), hop, rel_db=-200)
    mix += bed

    events, f0 = [], 0
    while f0 < n_frames:
        n_ev = int(round(rng.uniform(*m["event_seconds"]) * sr / hop))
        n_ev = max(1, min(n_ev, n_frames - f0))
        s0, n_s = f0 * hop, n_ev * hop
        cls = CLASSES[int(rng.choice(len(CLASSES), p=probs))]
        ev = {"cls": cls, "start": round(s0 / sr, 2), "dur": round(n_s / sr, 2)}
        if cls == "background" and rng.random() < m["noise_burst_prob"]:
            src = pick("noise")
            x = set_level(read_segment(src, n_s, sr, rng, loop=True),
                          rng.uniform(*m["noise_burst_dbfs"]), hop, rel_db=-200)
            mix[s0:s0 + n_s] += fade(x, fade_n)
            ev["noise"] = Path(src.path).name
        speech_db = None
        if cls in ("speech", "speech_music"):
            src = pick("speech")
            speech_db = rng.uniform(*m["speech_level_dbfs"])
            x = fade(set_level(read_segment(src, n_s, sr, rng), speech_db, hop), fade_n)
            mix[s0:s0 + n_s] += x
            speech_mask[f0:f0 + n_ev] |= speech_activity(x, hop, sr, lcfg)
            ev["speech"] = Path(src.path).name
        if cls in ("music", "speech_music"):
            src = pick("music")
            if cls == "speech_music":
                smr = float(rng.choice(m["smr_db"]))
                target = speech_db - smr
                ev["smr_db"] = smr
            else:
                target = rng.uniform(*m["music_level_dbfs"])
            x = fade(set_level(read_segment(src, n_s, sr, rng, loop=True), target, hop), fade_n)
            mix[s0:s0 + n_s] += x
            music_mask[f0:f0 + n_ev] |= music_activity(x, hop, sr, lcfg)
            ev["music"] = Path(src.path).name
        events.append(ev)
        f0 += n_ev

    peak = float(np.abs(mix).max())
    if peak > 0.99:
        mix *= 0.99 / peak
    labels = combine(speech_mask, music_mask)
    shares = class_shares(labels)
    meta = {"noise_bed": Path(bed_src.path).name, "n_events": len(events),
            "events": json.dumps(events),
            **{f"share_{c}": round(float(s), 4) for c, s in zip(CLASSES, shares)}}
    return mix, labels, meta


def build_split(index: pd.DataFrame, raw: Path, out_dir: Path, split: str, cfg: dict,
                force: bool = False) -> np.ndarray:
    labels_path = out_dir / "labels.npy"
    if labels_path.exists() and not force:
        print(f"[{split}] mixtures exist, skipping (use --force to rebuild)")
        return class_shares(np.load(labels_path))
    sr, _, _, n_frames = audio_dims(cfg)
    n_clips = int(cfg["mixing"]["n_clips"][split])
    pools = load_pools(index, raw, split)
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    all_labels = np.zeros((n_clips, n_frames), dtype=np.int8)
    metas = []
    split_id = SPLITS.index(split)
    for i in tqdm(range(n_clips), desc=f"mixing {split}"):
        rng = np.random.default_rng([cfg["seed"], split_id, i])  # one stream per clip: reproducible
        audio, labels, meta = build_clip(rng, pools, cfg)
        clip_id = f"{split}_{i:05d}"
        sf.write(audio_dir / f"{clip_id}.wav", audio, sr, subtype="PCM_16")
        all_labels[i] = labels
        metas.append({"clip_id": clip_id, **meta})
    pd.DataFrame(metas).to_csv(out_dir / "meta.csv", index=False)
    np.save(labels_path, all_labels)  # written last: its presence marks the split as complete
    return class_shares(all_labels)
