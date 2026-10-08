"""PyTorch dataset over the feature memmaps."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class ClipDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray, mean: np.ndarray, std: np.ndarray,
                 crop_frames: int | None = None, seed: int = 0):
        assert features.shape[:2] == labels.shape, "features and labels do not match"
        self.X, self.Y = features, labels
        self.mean, self.std = mean.astype(np.float32), std.astype(np.float32)
        self.crop = crop_frames if crop_frames and crop_frames < labels.shape[1] else None
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, i: int):
        x = np.asarray(self.X[i], dtype=np.float32)
        y = np.asarray(self.Y[i], dtype=np.int64)
        if self.crop:
            s = int(self.rng.integers(0, len(y) - self.crop + 1))
            x, y = x[s:s + self.crop], y[s:s + self.crop]
        x = (x - self.mean) / self.std
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(y))
