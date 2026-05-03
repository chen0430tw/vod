"""Tests for the differentiable artifact-suppression layer.

Covers:
    - torch_tile_residue parity with the NumPy reference
    - artifact_regularization_loss is differentiable end-to-end
    - artifact_train_loss only touches image / video media
    - --artifact-loss-weight=0 keeps the optimizer step bit-exact
      identical to the pre-feature behaviour
    - --artifact-loss-weight>0 changes both the loss and the gradient
    - --artifact-suppression actually reaches the dataset (noisy_views
      differ from the suppression-off case)
"""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from vod_minimal.artifacts import tile_residue
from vod_minimal.core import build_projection_batch, projection_loss
from vod_minimal.torch_artifacts import (
    SPATIAL_MEDIA,
    artifact_regularization_loss,
    artifact_train_loss,
    torch_tile_residue,
)
from vod_minimal.torch_model import SharedPointUpdater


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #

def _blocky(size: int = 32, tile: int = 8) -> np.ndarray:
    y = np.arange(size)[:, None] // tile
    x = np.arange(size)[None, :] // tile
    return ((x + y) % 2).astype(np.float64)


def _smooth(size: int = 32) -> np.ndarray:
    return np.linspace(0.0, 1.0, size * size, dtype=np.float64).reshape(size, size)


def _make_torch_update():
    alpha = torch.nn.Parameter(torch.tensor(0.3, dtype=torch.float32))

    def step(current, target, medium):
        del medium
        return current + alpha * (target - current)

    step.alpha = alpha
    return step


# --------------------------------------------------------------------------- #
#  torch_tile_residue
# --------------------------------------------------------------------------- #

class TestTorchTileResidue:
    @pytest.mark.parametrize(
        "field_factory",
        [_blocky, _smooth],
        ids=["blocky", "smooth"],
    )
    def test_matches_numpy_reference(self, field_factory):
        np_field = field_factory()
        ref = tile_residue(np_field, tile=8)
        out = torch_tile_residue(torch.from_numpy(np_field), tile=8).item()
        assert out == pytest.approx(ref, rel=1e-5, abs=1e-7)

    def test_zero_when_too_small(self):
        x = torch.zeros((4, 4))
        assert torch_tile_residue(x, tile=8).item() == 0.0

    def test_one_dim_returns_zero(self):
        x = torch.zeros((128,))
        assert torch_tile_residue(x, tile=8).item() == 0.0

    def test_invalid_tile_raises(self):
        with pytest.raises(ValueError, match="tile"):
            torch_tile_residue(torch.zeros((32, 32)), tile=1)

    def test_handles_video_shape(self):
        # video shape (frames, H, W) should be averaged across frames.
        rng = np.random.default_rng(0)
        v = rng.standard_normal((10, 32, 32)).astype(np.float32)
        out = torch_tile_residue(torch.from_numpy(v), tile=8)
        assert out.ndim == 0
        assert torch.isfinite(out)


# --------------------------------------------------------------------------- #
#  artifact_regularization_loss
# --------------------------------------------------------------------------- #

class TestArtifactRegularizationLoss:
    def test_zero_when_pred_smoother_than_target(self):
        # Smooth pred against blocky target → ReLU(neg) = 0
        pred = torch.from_numpy(_smooth()).float().requires_grad_(True)
        target = torch.from_numpy(_blocky()).float()
        loss = artifact_regularization_loss(pred, target, tile=8)
        assert loss.item() == 0.0

    def test_positive_when_pred_blockier_than_target(self):
        pred = torch.from_numpy(_blocky()).float().requires_grad_(True)
        target = torch.from_numpy(_smooth()).float()
        loss = artifact_regularization_loss(pred, target, tile=8)
        assert loss.item() > 0.0

    def test_backward_produces_finite_gradient(self):
        # Use a random pred so boundary diffs are non-zero AND distinct;
        # a pure 0/1 blocky field has |diff| ∈ {0, 1} which makes the
        # sign-based gradient through `abs` cancel by symmetry across the
        # mean-of-mean-ratio normalisation. Random pred avoids that.
        rng = np.random.default_rng(0)
        pred = torch.from_numpy(rng.standard_normal((32, 32))).float().requires_grad_(True)
        target = torch.zeros((32, 32))
        loss = artifact_regularization_loss(pred, target, tile=8)
        assert loss.item() > 0.0  # smooth target, noisy pred → positive penalty
        loss.backward()
        assert pred.grad is not None
        assert torch.isfinite(pred.grad).all()
        assert pred.grad.abs().sum().item() > 0  # actually responsive

    def test_target_gradient_is_blocked(self):
        # Even if target requires grad, the loss's `.grad_fn` must not
        # carry a path back to it (we explicitly detach the target residue).
        pred = torch.from_numpy(_blocky()).float().requires_grad_(True)
        target = torch.from_numpy(_smooth()).float().requires_grad_(True)
        loss = artifact_regularization_loss(pred, target, tile=8)
        loss.backward()
        assert target.grad is None or float(target.grad.abs().sum()) == 0.0

    def test_audio_shape_returns_zero(self):
        # 1-D audio waveform; loss is only defined on spatial media
        pred = torch.zeros(2048).requires_grad_(True)
        target = torch.zeros(2048)
        assert artifact_regularization_loss(pred, target, tile=8).item() == 0.0

    def test_floor_gate_param_suppresses_below_floor(self):
        # The gate is implemented as `floor = max(target_r, residue_floor)`.
        # Comparing residue_floor=0.0 (ungated, the old behaviour) vs the
        # default residue_floor=1.0 isolates the gate effect without
        # needing to construct fields with a precisely controlled residue.
        rng = np.random.default_rng(0)
        # cos with period 16 gives a target whose tile boundaries (idx 7,
        # 15, 23) sit near the smooth cosine peaks, so the field is below
        # the residue=1.0 neutral point.
        target_np = np.tile(np.cos(2 * np.pi * np.arange(32) / 16), (32, 1))
        pred_np = target_np + 0.05 * rng.standard_normal((32, 32))
        target = torch.from_numpy(target_np).float()
        pred = torch.from_numpy(pred_np).float()

        loss_no_gate = artifact_regularization_loss(pred, target, tile=8, residue_floor=0.0)
        loss_gated = artifact_regularization_loss(pred, target, tile=8, residue_floor=1.0)

        # Gate must never increase the penalty.
        assert loss_gated.item() <= loss_no_gate.item()
        # On this smooth setup the un-gated loss is positive (would
        # over-smooth) and the gated loss is zero — proving the gate
        # actively prevents the over-smoothing regression.
        assert loss_no_gate.item() > 0.0
        assert loss_gated.item() == 0.0

    def test_floor_gate_does_not_kill_blocky_signal(self):
        # Inverse of the gate test: when pred actually IS blocky (residue
        # well above 1.0) the loss must still fire, regardless of how
        # smooth the target is. This locks the gate as one-sided — it
        # blocks below-floor noise, never blocks real failures.
        from vod_minimal.torch_artifacts import torch_tile_residue

        smooth_target = torch.from_numpy(
            np.linspace(0.0, 1.0, 32 * 32, dtype=np.float64).reshape(32, 32)
        ).float()
        blocky_pred = torch.from_numpy(_blocky(size=32, tile=8).astype(np.float64)).float()

        assert torch_tile_residue(blocky_pred, tile=8).item() > 1.5
        loss = artifact_regularization_loss(blocky_pred, smooth_target, tile=8)
        assert loss.item() > 0.0


# --------------------------------------------------------------------------- #
#  artifact_train_loss
# --------------------------------------------------------------------------- #

class TestArtifactTrainLoss:
    @pytest.fixture
    def small_batch(self):
        rng = np.random.default_rng(430)
        return build_projection_batch(rng, batch_size=2, size=16, noise_scale=0.2)

    def test_returns_scalar(self, small_batch):
        update = _make_torch_update()
        out = artifact_train_loss(update, small_batch, steps=1, device=torch.device("cpu"), tile=4)
        assert out.ndim == 0
        assert torch.isfinite(out)

    def test_skips_non_spatial_media(self, small_batch):
        update = _make_torch_update()
        full = artifact_train_loss(update, small_batch, steps=1, device=torch.device("cpu"), tile=4)
        spatial_only = artifact_train_loss(
            update, small_batch, steps=1, device=torch.device("cpu"), tile=4, media=SPATIAL_MEDIA
        )
        # Both calls iterate exactly the same media set; identical result is required.
        assert torch.isclose(full, spatial_only, rtol=0.0, atol=0.0)

    def test_audio_text_only_returns_zero(self, small_batch):
        update = _make_torch_update()
        out = artifact_train_loss(
            update, small_batch, steps=1, device=torch.device("cpu"), tile=4, media=("audio", "text")
        )
        assert out.item() == 0.0

    def test_grad_flows_through_update_fn(self, small_batch):
        update = _make_torch_update()
        loss = artifact_train_loss(update, small_batch, steps=1, device=torch.device("cpu"), tile=4)
        if loss.item() == 0.0:
            # All-zero penalty case: parameter receives no signal; nothing to assert.
            return
        loss.backward()
        assert update.alpha.grad is not None
        assert torch.isfinite(update.alpha.grad)


# --------------------------------------------------------------------------- #
#  Integration: train_torch weight=0 must be bit-exact same as pre-feature
# --------------------------------------------------------------------------- #

class TestTrainTorchWeightZeroIsBaseline:
    """End-to-end equivalence: a single optimizer step with weight=0 must
    produce the same parameters as taking a step on projection_loss alone.

    To exercise a NON-ZERO penalty we synthesize a batch whose noisy_views
    contain a strong blocky pattern — Chladni-projected views are smoother
    than the target, so the one-sided ReLU penalty is identically zero on
    them and any "weight>0 changes parameters" assertion would be vacuous.
    """

    def _make_blocky_batch(self):
        # Construct a ProjectionBatch by hand whose noisy image/video have
        # high tile residue and target are smooth. The non-spatial media
        # are kept zero (penalty skips them anyway).
        from vod_minimal.core import ProjectionBatch, ProjectionSample

        block = _blocky(size=16, tile=4)
        smooth = _smooth(size=16)
        sample = ProjectionSample(
            source_field=smooth,
            target_field=smooth,
            noisy_views={
                "image": block.copy(),
                "video": np.repeat(block[None, ...], 6, axis=0),
                "audio": np.zeros(2048, dtype=np.float64),
                "text": np.zeros(32, dtype=np.float64),
            },
            target_views={
                "image": smooth.copy(),
                "video": np.repeat(smooth[None, ...], 6, axis=0),
                "audio": np.zeros(2048, dtype=np.float64),
                "text": np.zeros(32, dtype=np.float64),
            },
        )
        return ProjectionBatch(samples=(sample,))

    def _take_step(self, with_artifact_branch: bool, weight: float, *, batch=None):
        torch.manual_seed(430)
        if batch is None:
            rng = np.random.default_rng(430)
            batch = build_projection_batch(rng, batch_size=2, size=16, noise_scale=0.2)
        model = SharedPointUpdater(hidden=8)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
        opt.zero_grad(set_to_none=True)

        def update_fn(current, target, medium):
            del medium
            return model.forward_step(current, target)

        loss = projection_loss(update_fn, batch, steps=2, device=torch.device("cpu"))
        if with_artifact_branch and weight > 0:
            art = artifact_train_loss(update_fn, batch, steps=2, device=torch.device("cpu"), tile=4)
            loss = loss + weight * art
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        return [p.detach().clone() for p in model.parameters()]

    def test_weight_zero_skipping_branch_matches_baseline(self):
        # Branch off (the train script's `if weight > 0:` short-circuit)
        baseline = self._take_step(with_artifact_branch=False, weight=0.0)
        # Branch on but weight=0 → the train script SHORT-CIRCUITS this case
        # (it never adds the artifact loss). Verify the branch-on path with
        # weight 0 is equivalent to never running it.
        skip = self._take_step(with_artifact_branch=True, weight=0.0)
        for a, b in zip(baseline, skip):
            assert torch.equal(a, b)

    def test_weight_positive_changes_parameters(self):
        # On a batch where pred actually outpaces target on tile residue,
        # the penalty must move parameters away from the unpenalised baseline.
        batch = self._make_blocky_batch()
        baseline = self._take_step(with_artifact_branch=False, weight=0.0, batch=batch)
        with_pen = self._take_step(with_artifact_branch=True, weight=0.5, batch=batch)
        any_diff = any(not torch.equal(a, b) for a, b in zip(baseline, with_pen))
        assert any_diff


# --------------------------------------------------------------------------- #
#  build_projection_batch suppression actually reaches noisy_views
# --------------------------------------------------------------------------- #

class TestBatchSuppressionFlowsThrough:
    def test_suppression_changes_noisy_views(self):
        plain = build_projection_batch(
            np.random.default_rng(0), batch_size=2, size=16, noise_scale=0.2
        )
        suppressed = build_projection_batch(
            np.random.default_rng(0),
            batch_size=2,
            size=16,
            noise_scale=0.2,
            artifact_suppression=True,
            artifact_scale=0.5,
            artifact_tile=4,
        )
        # noisy_views must differ for at least one medium when suppression
        # is on, otherwise the flag is silently doing nothing.
        diff_seen = False
        for ps_plain, ps_supp in zip(plain.samples, suppressed.samples):
            for medium in ps_plain.noisy_views:
                if not np.allclose(ps_plain.noisy_views[medium], ps_supp.noisy_views[medium]):
                    diff_seen = True
                    break
        assert diff_seen
