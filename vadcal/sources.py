"""Scan the raw datasets into one source index and assign the file-level splits.

MUSAN -> train / val / test_in (split by file, separately for speech, music and noise).
LibriSpeech test-clean, GTZAN and ESC-50 -> test_shift (sources never seen in training).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

COLUMNS = ["path", "dataset", "kind", "split", "sr", "frames", "duration", "group"]
GTZAN_DIRS = ("genres", "genres_original", "Data/genres_original")


def _musan(raw: Path) -> list[dict]:
    base = raw / "musan"
    rows = []
    for kind in ("speech", "music", "noise"):
        for p in sorted((base / kind).rglob("*.wav")):
            rows.append({"path": p, "dataset": "musan", "kind": kind, "group": p.parent.name})
    return rows


def _librispeech(raw: Path) -> list[dict]:
    base = raw / "LibriSpeech" / "test-clean"
    return [{"path": p, "dataset": "librispeech", "kind": "speech", "group": p.parts[-3]}
            for p in sorted(base.rglob("*.flac"))]


def _esc50(raw: Path, exclude: list[str]) -> list[dict]:
    base = raw / "ESC-50-master"
    meta_csv = base / "meta" / "esc50.csv"
    if not meta_csv.exists():
        return []
    meta = pd.read_csv(meta_csv)
    meta = meta[~meta["category"].isin(exclude)]
    return [{"path": base / "audio" / fn, "dataset": "esc50", "kind": "noise", "group": cat}
            for fn, cat in zip(meta["filename"], meta["category"])]


def _gtzan(raw: Path, exclude: list[str]) -> list[dict]:
    base = next((raw / d for d in GTZAN_DIRS if (raw / d).exists()), None)
    if base is None:
        return []
    return [{"path": p, "dataset": "gtzan", "kind": "music", "group": p.parent.name}
            for p in sorted(base.rglob("*"))
            if p.suffix.lower() in (".wav", ".au") and p.parent.name not in exclude]


def _musan_splits(n: int, shares: dict, rng: np.random.Generator) -> np.ndarray:
    n_val = max(1, int(round(shares["val"] * n)))
    n_test = max(1, int(round(shares["test_in"] * n)))
    n_train = n - n_val - n_test
    labels = np.array(["train"] * n_train + ["val"] * n_val + ["test_in"] * n_test, dtype=object)
    return labels[rng.permutation(n)]


def build_index(raw: Path, cfg: dict) -> pd.DataFrame:
    scfg = cfg["sources"]
    rows = (_musan(raw) + _librispeech(raw) + _esc50(raw, scfg["esc50_exclude"])
            + _gtzan(raw, scfg["gtzan_exclude"]))
    kept, bad = [], []
    for r in rows:
        try:
            info = sf.info(str(r["path"]))
        except Exception as e:  # e.g. the known corrupt GTZAN file jazz.00054
            bad.append(f"{r['path']}: {e}")
            continue
        duration = info.frames / info.samplerate
        if duration < scfg["min_source_seconds"]:
            continue
        r.update(sr=info.samplerate, frames=info.frames, duration=duration,
                 path=Path(r["path"]).relative_to(raw).as_posix())
        kept.append(r)
    if bad:
        print(f"skipped {len(bad)} unreadable files:\n  " + "\n  ".join(bad))
    df = pd.DataFrame(kept)
    if df.empty:
        raise RuntimeError(f"no audio found under {raw}. Run scripts/01_download_data.py first.")
    df = df.sort_values("path").reset_index(drop=True)
    df["split"] = "test_shift"
    rng = np.random.default_rng(cfg["seed"])
    for kind in ("speech", "music", "noise"):
        idx = df.index[(df["dataset"] == "musan") & (df["kind"] == kind)]
        if len(idx):
            df.loc[idx, "split"] = _musan_splits(len(idx), scfg["musan_split"], rng)
    _check(df)
    return df[COLUMNS]


def _check(df: pd.DataFrame) -> None:
    missing = [f"{split}/{kind}" for split in ("train", "val", "test_in", "test_shift")
               for kind in ("speech", "music", "noise")
               if not ((df["split"] == split) & (df["kind"] == kind)).any()]
    if missing:
        raise RuntimeError("no source files for: " + ", ".join(missing)
                           + ". Check that every dataset was downloaded and extracted.")


def summary(df: pd.DataFrame) -> pd.DataFrame:
    return (df.groupby(["split", "kind", "dataset"])
              .agg(files=("path", "size"), hours=("duration", lambda d: round(d.sum() / 3600, 2)))
              .reset_index())
