"""Load the YAML protocol (with `extends:` inheritance) and derived audio sizes."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ("background", "speech", "music", "speech_music")
N_CLASSES = len(CLASSES)
SPLITS = ("train", "val", "test_in", "test_shift")
EVAL_SPLITS = ("test_in", "test_shift")


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    if (Path.cwd() / path).exists():
        return (Path.cwd() / path).resolve()
    return (ROOT / path).resolve()


def _load(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    parent = cfg.pop("extends", None)
    if parent:
        cfg = _merge(_load(path.parent / parent), cfg)
    return cfg


def load_config(path: str | Path | None = None) -> dict:
    path = _resolve(path) if path else ROOT / "configs" / "default.yaml"
    cfg = _load(path)
    cfg.setdefault("smoke", False)
    cfg["_config_path"] = str(path)
    return cfg


def audio_dims(cfg: dict) -> tuple[int, int, int, int]:
    """Return (sample_rate, hop_samples, clip_samples, n_frames)."""
    a = cfg["audio"]
    sr = int(a["sample_rate"])
    hop = sr * int(a["hop_ms"]) // 1000
    clip = int(round(float(a["clip_seconds"]) * sr))
    return sr, hop, clip, clip // hop
