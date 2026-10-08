"""The four training objectives compared in E2."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

LOSSES = ("ce", "weighted_ce", "focal", "label_smoothing")


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logp_t = F.log_softmax(logits, dim=-1).gather(1, target[:, None]).squeeze(1)
        return (-((1 - logp_t.exp()) ** self.gamma) * logp_t).mean()


def class_weights(labels: np.ndarray, n_classes: int) -> np.ndarray:
    """Inverse class frequency, normalised to mean 1."""
    counts = np.maximum(np.bincount(labels.ravel().astype(np.int64), minlength=n_classes), 1)
    w = counts.sum() / counts
    return (w / w.mean()).astype(np.float32)


def make_loss(name: str, tcfg: dict, weights: torch.Tensor | None = None) -> nn.Module:
    if name == "ce":
        return nn.CrossEntropyLoss()
    if name == "weighted_ce":
        return nn.CrossEntropyLoss(weight=weights)
    if name == "focal":
        return FocalLoss(tcfg["focal_gamma"])
    if name == "label_smoothing":
        return nn.CrossEntropyLoss(label_smoothing=tcfg["label_smoothing"])
    raise ValueError(f"unknown loss {name!r}; choose from {LOSSES}")
