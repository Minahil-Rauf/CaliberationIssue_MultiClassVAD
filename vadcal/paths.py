"""Every file location used by the pipeline, derived from the config."""
from __future__ import annotations

from pathlib import Path

from .config import ROOT


class Paths:
    def __init__(self, cfg: dict):
        self.data = ROOT / cfg["paths"]["data_dir"]
        self.out = ROOT / cfg["paths"]["output_dir"]
        self.raw = self.data / "raw"
        self.downloads = self.data / "downloads"
        self.index_csv = self.data / "index" / "sources.csv"
        self.mixtures = self.data / "mixtures"
        self.features = self.data / "features"
        self.norm_stats = self.features / "norm_stats.npz"
        self.runs = self.out / "runs"
        self.tables = self.out / "tables"
        self.figures = self.out / "figures"

    def mix_dir(self, split: str) -> Path:
        return self.mixtures / split

    def audio_dir(self, split: str) -> Path:
        return self.mix_dir(split) / "audio"

    def labels(self, split: str) -> Path:
        return self.mix_dir(split) / "labels.npy"

    def meta(self, split: str) -> Path:
        return self.mix_dir(split) / "meta.csv"

    def feats(self, split: str) -> Path:
        return self.features / f"{split}.npy"

    def feats_done(self, split: str) -> Path:
        return self.features / f"{split}.done"

    def run_dir(self, loss: str, seed: int) -> Path:
        return self.runs / f"{loss}_seed{seed}"
