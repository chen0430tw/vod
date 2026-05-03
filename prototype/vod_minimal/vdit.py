"""Tiny VDiT skeleton for the minimal VOD prototype.

This is intentionally small. It borrows the MMDiT/DiT idea of projecting field
samples into a transformer, mixing them with condition/boundary features, and
predicting a field update. It does not import Open-Sora; it keeps the VOD
prototype dependency-light while preserving the architecture direction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


MEDIA_TO_ID = {
    "image": 0,
    "video": 1,
    "audio": 2,
    "text": 3,
}


def _smooth_torch(x: torch.Tensor) -> torch.Tensor:
    out = x
    for dim in range(1, x.ndim):
        out = (out + torch.roll(out, shifts=1, dims=dim) + torch.roll(out, shifts=-1, dims=dim)) / 3.0
    return out


def _flatten_features(
    current: torch.Tensor,
    target: torch.Tensor,
    medium: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    smooth = _smooth_torch(current)
    diffusivity = 0.15 + 0.85 * target.abs() / target.abs().amax().clamp_min(1e-6)
    n = current.numel()
    pos = torch.linspace(0.0, 1.0, n, device=current.device, dtype=current.dtype)
    medium_id = torch.full((n,), MEDIA_TO_ID[medium], device=current.device, dtype=torch.long)
    feats = torch.stack(
        [
            current.reshape(-1),
            target.reshape(-1),
            (smooth - current).reshape(-1),
            diffusivity.reshape(-1),
            pos,
        ],
        dim=-1,
    )
    return feats, medium_id


class VDiTBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 2.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, int(hidden_size * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(hidden_size * mlp_ratio), hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn
        x = x + self.mlp(self.norm2(x))
        return x


@dataclass(frozen=True)
class VDiTConfig:
    hidden_size: int = 64
    depth: int = 3
    num_heads: int = 4
    mlp_ratio: float = 2.0
    max_tokens: int = 512
    chunk_tokens: int = 512


class TinyVDiT(nn.Module):
    """Small trainable VOD Diffusion Transformer skeleton."""

    def __init__(self, config: VDiTConfig = VDiTConfig()):
        super().__init__()
        self.config = config
        self.input_proj = nn.Linear(5, config.hidden_size)
        self.media_embed = nn.Embedding(len(MEDIA_TO_ID), config.hidden_size)
        self.pos_embed = nn.Sequential(
            nn.Linear(2, config.hidden_size),
            nn.SiLU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )
        self.blocks = nn.ModuleList(
            [
                VDiTBlock(
                    hidden_size=config.hidden_size,
                    num_heads=config.num_heads,
                    mlp_ratio=config.mlp_ratio,
                )
                for _ in range(config.depth)
            ]
        )
        self.final = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, 1),
        )
        self.step_logit = nn.Parameter(torch.tensor(-0.4))

    def _forward_tokens(
        self,
        feats: torch.Tensor,
        medium_ids: torch.Tensor,
        token_positions: torch.Tensor,
    ) -> torch.Tensor:
        # feats: [N, 5]
        phase_pos = torch.stack(
            [
                torch.sin(2.0 * torch.pi * token_positions),
                torch.cos(2.0 * torch.pi * token_positions),
            ],
            dim=-1,
        )
        x = self.input_proj(feats)
        x = x + self.media_embed(medium_ids)
        x = x + self.pos_embed(phase_pos)
        x = x.unsqueeze(0)
        for block in self.blocks:
            x = block(x)
        return self.final(x.squeeze(0)).squeeze(-1)

    def forward_sampled(
        self,
        current: torch.Tensor,
        target: torch.Tensor,
        medium: str,
        token_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        feats, medium_ids = _flatten_features(current, target, medium)
        n = feats.shape[0]
        if token_indices is None:
            if n > self.config.max_tokens:
                token_indices = torch.randperm(n, device=current.device)[: self.config.max_tokens]
                token_indices, _ = token_indices.sort()
            else:
                token_indices = torch.arange(n, device=current.device)
        token_positions = token_indices.to(current.dtype) / max(1, n - 1)
        delta = self._forward_tokens(feats[token_indices], medium_ids[token_indices], token_positions)
        pred = current.reshape(-1)[token_indices] + torch.sigmoid(self.step_logit) * delta
        return pred, target.reshape(-1)[token_indices]

    @torch.no_grad()
    def forward_full(self, current: torch.Tensor, target: torch.Tensor, medium: str) -> torch.Tensor:
        feats, medium_ids = _flatten_features(current, target, medium)
        n = feats.shape[0]
        out = current.reshape(-1).clone()
        step = torch.sigmoid(self.step_logit)
        for start in range(0, n, self.config.chunk_tokens):
            end = min(start + self.config.chunk_tokens, n)
            idx = torch.arange(start, end, device=current.device)
            token_positions = idx.to(current.dtype) / max(1, n - 1)
            delta = self._forward_tokens(feats[idx], medium_ids[idx], token_positions)
            out[idx] = out[idx] + step * delta
        return out.reshape_as(current)

    @torch.no_grad()
    def denoise_views(
        self,
        noisy_views: dict[str, np.ndarray],
        target_views: dict[str, np.ndarray],
        device: torch.device,
        steps: int,
    ) -> dict[str, np.ndarray]:
        result = {}
        self.eval()
        for medium, noisy in noisy_views.items():
            current = torch.from_numpy(noisy.astype(np.float32)).to(device)
            target = torch.from_numpy(target_views[medium].astype(np.float32)).to(device)
            for _ in range(steps):
                current = self.forward_full(current, target, medium)
            result[medium] = current.detach().cpu().numpy()
        return result
