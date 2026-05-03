"""Tests for the blocky scattering stress / diagnostic dataset.

The stress data exists to verify that the artifact-suppression stack
responds to coherent tile residue. These tests pin down:

    - blocky_scattering_mask shape contract (2-D, video, 1-D no-op)
    - injected views show elevated tile_residue
    - oc_four_over_e actually changes a blocky view (no-op or
      inflate would both be regressions)
    - the differentiable artifact_regularization_loss fires AND back-
      propagates against a blocky pred / smooth target pair
    - a tiny optimizer step on a blocky batch with weight>0 moves
      parameters away from the unpenalised baseline
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from vod_minimal.artifacts import oc_four_over_e, tile_residue
from vod_minimal.blocky_scattering import (
    SPATIAL_MEDIA,
    blocky_scattering_mask,
    build_blocky_scattering_batch,
    inject_blocky_scattering,
)
from vod_minimal.core import (
    MEDIA,
    ProjectionBatch,
    ProjectionSample,
    build_projection_batch,
    projection_loss,
)
from vod_minimal.metrics import artifact_metrics
from vod_minimal.torch_artifacts import (
    artifact_regularization_loss,
    artifact_train_loss,
    torch_tile_residue,
)
from vod_minimal.torch_model import SharedPointUpdater


# --------------------------------------------------------------------------- #
#  blocky_scattering_mask
# --------------------------------------------------------------------------- #

class TestBlockyScatteringMask:
    def test_2d_shape(self):
        m = blocky_scattering_mask((32, 32), tile=8, strength=1.0, rng=np.random.default_rng(0))
        assert m.shape == (32, 32)
        assert m.dtype == np.float64

    def test_video_shape(self):
        m = blocky_scattering_mask((10, 32, 32), tile=8, rng=np.random.default_rng(0))
        assert m.shape == (10, 32, 32)
        # All frames share the same 2-D mask by design.
        np.testing.assert_array_equal(m[0], m[1])

    def test_one_dim_returns_zero(self):
        m = blocky_scattering_mask((2048,), tile=8, rng=np.random.default_rng(0))
        assert m.shape == (2048,)
        assert np.all(m == 0.0)

    def test_too_small_returns_zero(self):
        # Spatial side smaller than tile period → no useful pattern → zero mask.
        m = blocky_scattering_mask((4, 4), tile=8, rng=np.random.default_rng(0))
        assert m.shape == (4, 4)
        assert np.all(m == 0.0)

    def test_strength_scales_linearly(self):
        rng_a = np.random.default_rng(7)
        rng_b = np.random.default_rng(7)
        a = blocky_scattering_mask((32, 32), tile=8, strength=1.0, rng=rng_a)
        b = blocky_scattering_mask((32, 32), tile=8, strength=2.0, rng=rng_b)
        np.testing.assert_allclose(b, 2.0 * a, rtol=1e-12)

    def test_invalid_tile_raises(self):
        with pytest.raises(ValueError, match="tile"):
            blocky_scattering_mask((32, 32), tile=1, rng=np.random.default_rng(0))

    def test_negative_strength_raises(self):
        with pytest.raises(ValueError, match="strength"):
            blocky_scattering_mask((32, 32), tile=8, strength=-0.1, rng=np.random.default_rng(0))


# --------------------------------------------------------------------------- #
#  inject_blocky_scattering — residue elevation
# --------------------------------------------------------------------------- #

class TestInjectBlockyScattering:
    def test_residue_higher_than_smooth_baseline(self):
        rng = np.random.default_rng(0)
        smooth = rng.standard_normal((32, 32)) * 0.3
        blocky = inject_blocky_scattering(smooth, np.random.default_rng(1), tile=8, strength=0.5)
        assert tile_residue(blocky, tile=8) > tile_residue(smooth, tile=8)

    def test_video_input_residue_elevated(self):
        rng = np.random.default_rng(0)
        v = rng.standard_normal((10, 32, 32)) * 0.3
        v_blocky = inject_blocky_scattering(v, np.random.default_rng(1), tile=8, strength=0.5)
        # On a (frames, H, W) tensor tile_residue averages over frames internally.
        assert tile_residue(v_blocky, tile=8) > tile_residue(v, tile=8)

    def test_one_dim_no_op(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal((2048,))
        out = inject_blocky_scattering(x, np.random.default_rng(1), tile=8, strength=0.5)
        np.testing.assert_array_equal(out, x.astype(np.float64))


# --------------------------------------------------------------------------- #
#  oc_four_over_e actually changes a blocky view
# --------------------------------------------------------------------------- #

class TestSuppressionRespondsToBlocky:
    def test_suppression_changes_blocky_output(self):
        rng = np.random.default_rng(0)
        smooth = rng.standard_normal((32, 32)) * 0.2
        blocky = inject_blocky_scattering(smooth, np.random.default_rng(1), tile=8, strength=0.5)
        suppressed = oc_four_over_e(
            blocky, np.random.default_rng(2), beta=0.05, tile=8
        )
        assert not np.allclose(suppressed, blocky)
        assert np.isfinite(suppressed).all()


# --------------------------------------------------------------------------- #
#  Differentiable artifact_regularization_loss on stress data
# --------------------------------------------------------------------------- #

class TestArtifactLossOnStress:
    def test_loss_positive_on_blocky_pred_smooth_target(self):
        rng = np.random.default_rng(0)
        smooth = torch.zeros((32, 32))
        blocky = torch.from_numpy(
            inject_blocky_scattering(np.zeros((32, 32)), rng, tile=8, strength=0.6)
        ).float()
        loss = artifact_regularization_loss(blocky, smooth, tile=8)
        assert loss.item() > 0.0

    def test_loss_backward_produces_finite_gradient(self):
        rng = np.random.default_rng(0)
        blocky = torch.from_numpy(
            inject_blocky_scattering(np.zeros((32, 32)), rng, tile=8, strength=0.6)
        ).float()
        blocky.requires_grad_(True)
        smooth = torch.zeros((32, 32))
        loss = artifact_regularization_loss(blocky, smooth, tile=8)
        loss.backward()
        assert blocky.grad is not None
        assert torch.isfinite(blocky.grad).all()
        assert blocky.grad.abs().sum().item() > 0


# --------------------------------------------------------------------------- #
#  build_blocky_scattering_batch — end-to-end batch contract
# --------------------------------------------------------------------------- #

class TestBlockyBatch:
    def test_blocky_batch_residue_higher_than_clean(self):
        clean = build_projection_batch(
            np.random.default_rng(0), batch_size=2, size=32, noise_scale=0.24
        )
        blocky = build_blocky_scattering_batch(
            np.random.default_rng(0), batch_size=2, size=32, noise_scale=0.24,
            artifact_strength=0.3, tile=8,
        )
        clean_a = artifact_metrics(clean.samples[0].noisy_views, tile=8)
        blocky_a = artifact_metrics(blocky.samples[0].noisy_views, tile=8)
        assert blocky_a["mean_tile_residue"] > clean_a["mean_tile_residue"]
        assert blocky_a["artifact_score"] <= clean_a["artifact_score"]

    def test_blocky_batch_preserves_audio_text(self):
        # Stress data must NOT touch non-spatial media.
        clean = build_projection_batch(
            np.random.default_rng(0), batch_size=1, size=32, noise_scale=0.24
        )
        blocky = build_blocky_scattering_batch(
            np.random.default_rng(0), batch_size=1, size=32, noise_scale=0.24,
            artifact_strength=0.3, tile=8,
        )
        for medium in ("audio", "text"):
            np.testing.assert_array_equal(
                blocky.samples[0].noisy_views[medium],
                clean.samples[0].noisy_views[medium],
            )

    def test_spatial_only_score_is_sensitive_to_stress(self):
        # Core invariant of the spatial-only redesign: the main artifact
        # score (image+video) must move on stress data, while the
        # non-spatial diagnostic must stay essentially flat between
        # clean and blocky (audio/text are untouched by injection).
        clean = build_projection_batch(
            np.random.default_rng(0), batch_size=2, size=32, noise_scale=0.24,
        )
        blocky = build_blocky_scattering_batch(
            np.random.default_rng(0), batch_size=2, size=32, noise_scale=0.24,
            artifact_strength=0.6, tile=8,
        )
        clean_a = artifact_metrics(clean.samples[0].noisy_views, tile=8)
        blocky_a = artifact_metrics(blocky.samples[0].noisy_views, tile=8)

        # Spatial side: meaningful drop in score, meaningful rise in residue.
        assert clean_a["artifact_score"] - blocky_a["artifact_score"] > 0.01
        assert blocky_a["mean_tile_residue"] > clean_a["mean_tile_residue"] + 0.05

        # Non-spatial side: audio + text untouched, so the non-spatial
        # diagnostic must NOT have shifted significantly.
        assert abs(
            blocky_a["non_spatial_mean_tile_residue"]
            - clean_a["non_spatial_mean_tile_residue"]
        ) < 1e-6


# --------------------------------------------------------------------------- #
#  Train stress: weight=0 == baseline; weight>0 changes parameters.
# --------------------------------------------------------------------------- #

class TestBlockyTrainStress:
    def _step(self, *, weight: float):
        torch.manual_seed(430)
        batch = build_blocky_scattering_batch(
            np.random.default_rng(430), batch_size=2, size=32, noise_scale=0.24,
            artifact_strength=0.4, tile=8,
        )
        model = SharedPointUpdater(hidden=8)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
        opt.zero_grad(set_to_none=True)

        def update_fn(current, target, medium):
            del medium
            return model.forward_step(current, target)

        loss = projection_loss(update_fn, batch, steps=2, device=torch.device("cpu"))
        if weight > 0:
            art = artifact_train_loss(update_fn, batch, steps=2, device=torch.device("cpu"), tile=8)
            loss = loss + weight * art
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        return [p.detach().clone() for p in model.parameters()]

    def test_weight_zero_baseline(self):
        # weight=0 short-circuit must not touch parameters relative to no-branch
        torch.manual_seed(430)
        batch = build_blocky_scattering_batch(
            np.random.default_rng(430), batch_size=2, size=32, noise_scale=0.24,
            artifact_strength=0.4, tile=8,
        )
        model = SharedPointUpdater(hidden=8)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
        opt.zero_grad(set_to_none=True)

        def update_fn(current, target, medium):
            del medium
            return model.forward_step(current, target)

        loss = projection_loss(update_fn, batch, steps=2, device=torch.device("cpu"))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        baseline = [p.detach().clone() for p in model.parameters()]

        skip = self._step(weight=0.0)
        for a, b in zip(baseline, skip):
            assert torch.equal(a, b)

    def test_artifact_loss_nonzero_on_blocky_batch(self):
        # On stress data the inner penalty must be > 0 (the whole point of
        # this dataset). Otherwise the regularizer cannot drive learning.
        torch.manual_seed(430)
        batch = build_blocky_scattering_batch(
            np.random.default_rng(430), batch_size=2, size=32, noise_scale=0.24,
            artifact_strength=0.4, tile=8,
        )
        model = SharedPointUpdater(hidden=8)

        def update_fn(current, target, medium):
            del medium
            return model.forward_step(current, target)

        art = artifact_train_loss(update_fn, batch, steps=1, device=torch.device("cpu"), tile=8)
        assert art.item() > 0.0

    def test_weight_positive_changes_parameters(self):
        baseline = self._step(weight=0.0)
        with_pen = self._step(weight=0.5)
        assert any(not torch.equal(a, b) for a, b in zip(baseline, with_pen))
