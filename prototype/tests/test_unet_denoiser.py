"""Tests for the spatial UNet denoiser backbone.

Covers shape contracts and a 2-epoch smoke training. The substrate
semantics (encoders / decoders / smoothing taps / position grid /
loss plumbing) are tested elsewhere; this file only checks that
swapping in the UNet backbone preserves the public contract of
`NativeVOD` and survives a tiny end-to-end optimization step.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from vod_minimal.core import build_projection_batch
from vod_minimal.denoisers import PointwiseMLPDenoiser, UNetDenoiser
from vod_minimal.native import (
    LATENT_HW,
    LATENT_T,
    NativeLossWeights,
    NativeVOD,
    NativeVODConfig,
    native_total_loss,
    views_to_torch,
)


def _toy_sample(seed: int = 0):
    return build_projection_batch(
        np.random.default_rng(seed),
        batch_size=1,
        size=LATENT_HW,
        frames=LATENT_T,
        spacetime=True,
    ).samples[0]


class TestUNetDenoiserStandalone:
    def test_constructs_with_default_args(self):
        m = UNetDenoiser(channels=4)
        assert isinstance(m, torch.nn.Module)
        # Sanity: at C=4, hidden=32 the architecture's bottleneck
        # (4*hidden = 128 channels with two 3x3 convs) dominates
        # the parameter count. The actual figure is ~524k; we keep
        # an upper bound here to catch accidental 5M+ blowups while
        # being flexible enough not to be a stylistic trap.
        n_params = sum(p.numel() for p in m.parameters())
        assert 100_000 <= n_params <= 1_000_000, (
            f"UNetDenoiser default params={n_params}, expected 100k-1M"
        )

    def test_forward_shape(self):
        torch.manual_seed(0)
        B, T, H, W, C = 2, 4, 16, 16, 4
        m = UNetDenoiser(channels=C, hidden=32)
        # Features: 3*C + position_dims = 15 channels.
        feats = torch.randn(B, T, H, W, 3 * C + 3)
        out = m(feats)
        assert out.shape == (B, T, H, W, C)
        assert out.dtype == feats.dtype

    def test_forward_finite_gradient(self):
        torch.manual_seed(0)
        B, T, H, W, C = 2, 4, 16, 16, 4
        m = UNetDenoiser(channels=C, hidden=16)
        feats = torch.randn(B, T, H, W, 3 * C + 3, requires_grad=True)
        out = m(feats)
        loss = out.pow(2).mean()
        loss.backward()
        # Every UNet parameter must receive a finite gradient.
        for name, p in m.named_parameters():
            assert p.grad is not None, name
            assert torch.isfinite(p.grad).all(), name
        # Input gradient also finite (shape contract for caller side).
        assert feats.grad is not None
        assert torch.isfinite(feats.grad).all()

    def test_rejects_too_small_grid(self):
        m = UNetDenoiser(channels=4, hidden=16)
        feats = torch.randn(1, 2, 2, 2, 3 * 4 + 3)  # H=W=2 < 4
        with pytest.raises(ValueError, match="H >= 4 and W >= 4"):
            m(feats)

    def test_handles_non_multiple_of_four_grid(self):
        # H, W not divisible by 4 should still work via replicate-pad.
        torch.manual_seed(0)
        m = UNetDenoiser(channels=4, hidden=8)
        feats = torch.randn(1, 2, 6, 10, 3 * 4 + 3)
        out = m(feats)
        assert out.shape == (1, 2, 6, 10, 4)


class TestNativeVODBackboneSwitch:
    def test_unet_default_active(self):
        m = NativeVOD()
        assert m.config.backbone == "unet"
        assert isinstance(m.denoiser, UNetDenoiser)

    def test_mlp_legacy_still_constructible(self):
        m = NativeVOD(NativeVODConfig(backbone="mlp"))
        assert isinstance(m.denoiser, PointwiseMLPDenoiser)

    def test_invalid_backbone_raises(self):
        with pytest.raises(ValueError, match="backbone"):
            NativeVOD(NativeVODConfig(backbone="transformer"))

    def test_unet_forward_matches_mlp_contract(self):
        """Same public contract: forward(noisy) → predicted_views, U_pred,
        with the exact shapes demanded by `decode`."""
        torch.manual_seed(0)
        sample = _toy_sample()
        device = torch.device("cpu")
        noisy = views_to_torch(sample.noisy_views, device)

        for backbone in ("unet", "mlp"):
            m = NativeVOD(NativeVODConfig(channels=4, hidden=16, backbone=backbone))
            pred, u_pred = m(noisy)
            assert pred["image"].shape == (LATENT_HW, LATENT_HW), backbone
            assert pred["video"].shape == (LATENT_T, LATENT_HW, LATENT_HW), backbone
            assert u_pred.shape == (LATENT_T, LATENT_HW, LATENT_HW, 4), backbone
            assert u_pred.dtype == noisy["video"].dtype, backbone

    def test_unet_two_epoch_smoke_no_nan(self):
        """A 2-epoch smoke training with the UNet backbone must not
        produce NaN/Inf in the loss or in any parameter."""
        torch.manual_seed(0)
        samples = build_projection_batch(
            np.random.default_rng(0),
            batch_size=2,
            size=LATENT_HW,
            frames=LATENT_T,
            spacetime=True,
        ).samples
        m = NativeVOD(NativeVODConfig(channels=4, hidden=16, backbone="unet"))
        opt = torch.optim.AdamW(m.parameters(), lr=2e-3)
        device = torch.device("cpu")
        weights = NativeLossWeights()

        for _ in range(2):
            opt.zero_grad(set_to_none=True)
            losses = []
            for s in samples:
                noisy = views_to_torch(s.noisy_views, device)
                target = views_to_torch(s.target_views, device)
                loss, components = native_total_loss(m, noisy, target, weights=weights)
                assert math.isfinite(float(loss.detach()))
                for k, v in components.items():
                    if isinstance(v, float):
                        # binary_twin_symbol_accuracy can be NaN when
                        # text is disabled; everything else must be finite.
                        if k == "binary_twin_symbol_accuracy":
                            continue
                        assert math.isfinite(v), f"{k}={v}"
                losses.append(loss)
            torch.stack(losses).mean().backward()
            opt.step()

        for name, p in m.named_parameters():
            assert torch.isfinite(p).all(), f"non-finite param after smoke: {name}"
