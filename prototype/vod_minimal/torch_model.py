"""Trainable PyTorch updater for the minimal VOD prototype."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def _smooth_torch(x: torch.Tensor) -> torch.Tensor:
    """Shape-agnostic local smoothing over all non-batch dimensions."""
    out = x
    for dim in range(1, x.ndim):
        out = (out + torch.roll(out, shifts=1, dims=dim) + torch.roll(out, shifts=-1, dims=dim)) / 3.0
    return out


class SharedPointUpdater(nn.Module):
    """Shared learned local update for image/video/audio/text toy projections.

    The same tiny MLP is applied pointwise to every medium. This preserves the
    minimal VOD claim: different media projections share one field update rule.
    """

    def __init__(self, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        self.step_logit = nn.Parameter(torch.tensor(0.0))

    def forward_step(self, current: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        smooth = _smooth_torch(current)
        diffusivity = 0.15 + 0.85 * target.abs() / (target.abs().amax().clamp_min(1e-6))
        features = torch.stack(
            [
                current,
                target,
                smooth - current,
                diffusivity,
            ],
            dim=-1,
        )
        delta = self.net(features).squeeze(-1)
        step = torch.sigmoid(self.step_logit)
        return current + step * delta

    def forward_path(self, noisy: torch.Tensor, target: torch.Tensor, steps: int) -> list[torch.Tensor]:
        current = noisy
        path = [current]
        for _ in range(steps):
            current = self.forward_step(current, target)
            path.append(current)
        return path


@dataclass
class TorchVODResult:
    model: SharedPointUpdater
    train_loss: float
    test_loss: float
    train_before: float
    train_after: float
    test_before: float
    test_after: float


def to_tensor(array: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(array.astype(np.float32)).to(device)


def normalized_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    denom = target.pow(2).mean().detach().clamp_min(1e-4)
    return F.mse_loss(pred, target) / denom
