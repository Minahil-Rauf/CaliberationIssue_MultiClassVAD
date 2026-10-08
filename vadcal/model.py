"""Small CRNN frame classifier (~86K parameters). Pools only along frequency, so every
10 ms input frame gets its own 4-class output."""
from __future__ import annotations

import torch
from torch import nn


class CRNN(nn.Module):
    def __init__(self, n_mels: int = 64, n_classes: int = 4, channels=(16, 32, 64),
                 freq_pool=(4, 2, 2), gru_hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        layers, c_in, f = [], 1, n_mels
        for c, p in zip(channels, freq_pool):
            layers += [nn.Conv2d(c_in, c, kernel_size=3, padding=1), nn.BatchNorm2d(c),
                       nn.ReLU(inplace=True), nn.MaxPool2d((1, p))]
            c_in, f = c, f // p
        self.cnn = nn.Sequential(*layers)
        self.drop = nn.Dropout(dropout)
        self.gru = nn.GRU(c_in * f, gru_hidden, batch_first=True)
        self.fc = nn.Linear(gru_hidden, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, time, n_mels) -> logits (batch, time, n_classes)."""
        x = self.cnn(x.unsqueeze(1))                      # (B, C, T, F')
        b, c, t, f = x.shape
        x = x.permute(0, 2, 1, 3).reshape(b, t, c * f)
        x, _ = self.gru(self.drop(x))
        return self.fc(self.drop(x))


def build_model(cfg: dict, n_classes: int = 4) -> CRNN:
    m = cfg["model"]
    return CRNN(n_mels=cfg["features"]["n_mels"], n_classes=n_classes, channels=tuple(m["channels"]),
                freq_pool=tuple(m["freq_pool"]), gru_hidden=m["gru_hidden"], dropout=m["dropout"])


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
