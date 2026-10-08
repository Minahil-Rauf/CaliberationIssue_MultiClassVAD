"""Log-mel features, one float16 memmap per split, plus train-set normalisation stats."""
from __future__ import annotations

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

from .config import audio_dims
from .paths import Paths


def logmel(y: np.ndarray, cfg: dict) -> np.ndarray:
    """(n_frames, n_mels) log-mel. Frame i is centred on sample i*hop + hop/2, the centre of label frame i."""
    sr, hop, _, n_frames = audio_dims(cfg)
    f = cfg["features"]
    assert f["hop_length"] == hop, "features.hop_length must equal the label hop"
    y = y[hop // 2:]  # with center=True this aligns feature frame i with label samples [i*hop, (i+1)*hop)
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=f["n_fft"], hop_length=f["hop_length"], win_length=f["win_length"],
        n_mels=f["n_mels"], fmin=f["fmin"], fmax=f["fmax"], power=2.0, center=True)
    out = np.log(mel + 1e-10).T
    if out.shape[0] < n_frames:
        out = np.pad(out, ((0, n_frames - out.shape[0]), (0, 0)), mode="edge")
    return out[:n_frames].astype(np.float32)


def extract_split(paths: Paths, split: str, cfg: dict, force: bool = False) -> None:
    done = paths.feats_done(split)
    if done.exists() and not force:
        print(f"[{split}] features exist, skipping")
        return
    _, _, _, n_frames = audio_dims(cfg)
    clip_ids = pd.read_csv(paths.meta(split))["clip_id"].tolist()
    paths.features.mkdir(parents=True, exist_ok=True)
    arr = np.lib.format.open_memmap(paths.feats(split), mode="w+", dtype=np.float16,
                                    shape=(len(clip_ids), n_frames, cfg["features"]["n_mels"]))
    for i, cid in enumerate(tqdm(clip_ids, desc=f"features {split}")):
        y, _ = sf.read(paths.audio_dir(split) / f"{cid}.wav", dtype="float32")
        arr[i] = logmel(y, cfg)
    arr.flush()
    del arr
    done.write_text("ok")


def compute_norm_stats(paths: Paths, chunk: int = 256) -> None:
    X = np.load(paths.feats("train"), mmap_mode="r")
    total = np.zeros(X.shape[-1])
    total_sq = np.zeros(X.shape[-1])
    count = 0
    for i in range(0, X.shape[0], chunk):
        x = np.asarray(X[i:i + chunk], dtype=np.float64).reshape(-1, X.shape[-1])
        total += x.sum(0)
        total_sq += (x ** 2).sum(0)
        count += x.shape[0]
    mean = total / count
    std = np.sqrt(np.maximum(total_sq / count - mean ** 2, 1e-8))
    np.savez(paths.norm_stats, mean=mean.astype(np.float32), std=std.astype(np.float32))
