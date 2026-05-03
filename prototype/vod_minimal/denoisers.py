"""Denoiser backbones for the native VOD substrate.

This module factors the field denoiser out of `native.py` so that
`NativeVOD` can swap backbones while keeping every substrate semantic
intact:

  * Input:  `(B, T, H, W, C)` field tensor U plus a `(B, T, H, W, 3)`
            normalized (t, y, x) position grid that the substrate
            already attaches to every voxel.
  * Output: `(B, T, H, W, C)` delta tensor matching `dtype`/`device`
            of the input.

`PointwiseMLPDenoiser` preserves the legacy pointwise MLP behaviour
exactly (3-tap + 5-tap smoothing of u_noisy concatenated with the
position grid before a per-voxel MLP). It is kept for ablation under
`NativeVODConfig(backbone="mlp")`.

`UNetDenoiser` is the new default backbone. It is a small 3-level
spatial UNet over (H, W) with a single 1-D convolution along T at the
bottleneck for multi-frame context. No self-attention. No new
pip dependency. Replicate-padding is used to align the latent grid to
multiples of 4 when needed.

Both backbones operate on the SAME features (u_noisy, smooth3,
smooth5, position grid) so the only thing that changes when you flip
`config.backbone` is the spatial / temporal structure of the
denoiser itself.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


# --------------------------------------------------------------------------- #
#  Pointwise MLP denoiser (legacy)
# --------------------------------------------------------------------------- #

class PointwiseMLPDenoiser(nn.Module):
    """Per-voxel MLP over [u_noisy, smooth3, smooth5, pos_t, pos_y, pos_x].

    Feature dim = 3 * channels + position_dims (default 3). This is the
    exact pointwise architecture that was inlined inside
    `NativeVOD.__init__` before the UNet upgrade.
    """

    def __init__(self, channels: int, hidden: int = 32, position_dims: int = 3,
                  extra_dims: int = 0):
        super().__init__()
        self.channels = channels
        self.position_dims = position_dims
        self.extra_dims = extra_dims
        feat_in = 3 * channels + position_dims + extra_dims
        self.net = nn.Sequential(
            nn.Linear(feat_in, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, channels),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """features: (..., 3C + position_dims + extra_dims). Output: (..., C)."""
        return self.net(features)


# --------------------------------------------------------------------------- #
#  Spatial UNet denoiser
# --------------------------------------------------------------------------- #

def _pad_to_multiple(x: torch.Tensor, multiple: int) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
    """Replicate-pad a (B, C, H, W) tensor so H, W are multiples of `multiple`.

    Returns the padded tensor and the (left, right, top, bottom) pad
    extents so the caller can crop after the up-path.
    """
    _, _, h, w = x.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return x, (0, 0, 0, 0)
    # F.pad order for last two dims: (W_left, W_right, H_top, H_bottom)
    left = pad_w // 2
    right = pad_w - left
    top = pad_h // 2
    bottom = pad_h - top
    return F.pad(x, (left, right, top, bottom), mode="replicate"), (left, right, top, bottom)


def _crop(x: torch.Tensor, pads: tuple[int, int, int, int]) -> torch.Tensor:
    """Inverse of `_pad_to_multiple` for a (B, C, H, W) tensor."""
    left, right, top, bottom = pads
    if left == right == top == bottom == 0:
        return x
    h, w = x.shape[-2], x.shape[-1]
    return x[..., top : h - bottom, left : w - right]


class _ConvBlock(nn.Module):
    """Two SiLU-activated 3x3 convs (replicate padding)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, padding_mode="replicate")
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, padding_mode="replicate")
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        return x


class UNetDenoiser(nn.Module):
    """Small 3-level spatial UNet with 1-D temporal mixing at bottleneck.

    Input feature layout (matches `PointwiseMLPDenoiser`):

        features.shape == (B, T, H, W, 3*C + position_dims)

    The first `C` features are u_noisy, the next 2*C are smooth3 and
    smooth5, and the trailing `position_dims` are the (t, y, x)
    position grid. The UNet mixes all of them; it does NOT enforce any
    asymmetric path between them — the substrate semantics (encoders /
    decoders / smoothing taps) are upstream of this module.

    Architecture:
        rearrange (B, T, H, W, F) -> (B*T, F, H, W)
        down1: ConvBlock(F, hidden)
        down2: AvgPool2d(2) + ConvBlock(hidden, 2*hidden)
        bot:   AvgPool2d(2) + ConvBlock(2*hidden, 4*hidden)
        temporal-1d: rearrange to (B, 4*hidden, T, H/4 * W/4)
                     -> Conv1d along T (kernel=3, replicate pad)
                     -> rearrange back
        up2:   Upsample(2x) + concat(skip down2)
                              + ConvBlock(4*hidden + 2*hidden, 2*hidden)
        up1:   Upsample(2x) + concat(skip down1)
                              + ConvBlock(2*hidden + hidden, hidden)
        out:   Conv2d(hidden, C, 1)
        rearrange back to (B, T, H, W, C)
    """

    def __init__(
        self,
        channels: int,
        hidden: int = 32,
        position_dims: int = 3,
        extra_dims: int = 0,
    ):
        super().__init__()
        if hidden < 1:
            raise ValueError(f"hidden must be >= 1, got {hidden}")
        self.channels = channels
        self.hidden = hidden
        self.position_dims = position_dims
        self.extra_dims = extra_dims
        feat_in = 3 * channels + position_dims + extra_dims

        self.down1 = _ConvBlock(feat_in, hidden)
        self.pool1 = nn.AvgPool2d(2)
        self.down2 = _ConvBlock(hidden, 2 * hidden)
        self.pool2 = nn.AvgPool2d(2)
        self.bot = _ConvBlock(2 * hidden, 4 * hidden)

        # Temporal 1-D mix at bottleneck. Kernel=3 with replicate
        # padding so a single frame run still works (T can be 1).
        self.temporal_conv = nn.Conv1d(
            4 * hidden, 4 * hidden, kernel_size=3, padding=1, padding_mode="replicate"
        )
        self.temporal_act = nn.SiLU()

        self.up2 = _ConvBlock(4 * hidden + 2 * hidden, 2 * hidden)
        self.up1 = _ConvBlock(2 * hidden + hidden, hidden)
        self.out_conv = nn.Conv2d(hidden, channels, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """features: (B, T, H, W, 3*C + position_dims). Output: (B, T, H, W, C)."""
        if features.ndim != 5:
            raise ValueError(
                f"UNetDenoiser expects 5-D (B, T, H, W, F), got shape {tuple(features.shape)}"
            )
        b, t, h, w, _ = features.shape
        if h < 4 or w < 4:
            raise ValueError(
                f"UNetDenoiser requires H >= 4 and W >= 4, got H={h} W={w}"
            )

        # (B, T, H, W, F) -> (B*T, F, H, W)
        x = features.permute(0, 1, 4, 2, 3).reshape(b * t, -1, h, w)

        # Pad H, W to multiples of 4 for two AvgPool2d(2) layers.
        x, pads = _pad_to_multiple(x, 4)

        # Down path
        d1 = self.down1(x)                 # (B*T, hidden, H', W')
        d2 = self.down2(self.pool1(d1))    # (B*T, 2h, H'/2, W'/2)
        bot = self.bot(self.pool2(d2))     # (B*T, 4h, H'/4, W'/4)

        # Temporal mix: (B*T, 4h, H'/4, W'/4) -> (B, 4h, T, H'/4 * W'/4)
        bh, bc, hh, ww = bot.shape
        bot_bt = bot.view(b, t, bc, hh, ww).permute(0, 2, 1, 3, 4)        # (B, 4h, T, H'/4, W'/4)
        bot_flat = bot_bt.reshape(b, bc, t, hh * ww)                       # (B, 4h, T, H'/4*W'/4)
        bot_t = bot_flat.permute(0, 3, 1, 2).reshape(b * hh * ww, bc, t)   # (B*HW, 4h, T)
        bot_t = self.temporal_act(self.temporal_conv(bot_t))
        bot_back = bot_t.reshape(b, hh * ww, bc, t).permute(0, 2, 3, 1)    # (B, 4h, T, H'/4*W'/4)
        bot_back = bot_back.reshape(b, bc, t, hh, ww).permute(0, 2, 1, 3, 4).reshape(bh, bc, hh, ww)

        # Up path with skip-concat
        u2 = F.interpolate(bot_back, scale_factor=2, mode="nearest")
        u2 = torch.cat([u2, d2], dim=1)
        u2 = self.up2(u2)

        u1 = F.interpolate(u2, scale_factor=2, mode="nearest")
        u1 = torch.cat([u1, d1], dim=1)
        u1 = self.up1(u1)

        out = self.out_conv(u1)            # (B*T, C, H', W')
        out = _crop(out, pads)             # back to (B*T, C, H, W)

        # (B*T, C, H, W) -> (B, T, H, W, C)
        out = out.view(b, t, self.channels, h, w).permute(0, 1, 3, 4, 2).contiguous()
        return out


__all__ = [
    "PointwiseMLPDenoiser",
    "UNetDenoiser",
]
