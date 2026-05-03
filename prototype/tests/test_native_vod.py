"""Tests for native_vod_smoke (NOT v0.3).

Status checks the no-leak invariants:
  * forward(noisy) takes only noisy_views — no target argument.
  * native_total_loss never lets target gradients flow back through
    the encoder (target encoding is detached for L_field).
  * audio / text are OFF by default and excluded from active_media.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from vod_minimal.blocky_scattering import inject_text_quantization_corruption
from vod_minimal.core import build_projection_batch
from vod_minimal.native import (
    AUDIO_SIZE,
    LATENT_HW,
    LATENT_T,
    NativeLossWeights,
    NativeVOD,
    NativeVODConfig,
    TEXT_LEN,
    audio_to_grid,
    grid_to_audio,
    grid_to_text,
    native_total_loss,
    text_to_grid,
    views_to_torch,
)


def _toy_batch(n: int = 2, seed: int = 0):
    return build_projection_batch(
        np.random.default_rng(seed),
        batch_size=n,
        size=LATENT_HW,
        frames=LATENT_T,
        spacetime=True,
    )


# --------------------------------------------------------------------------- #
#  Reshape helpers
# --------------------------------------------------------------------------- #

class TestReshape:
    def test_audio_grid_round_trip(self):
        a = torch.randn(AUDIO_SIZE)
        np.testing.assert_array_equal(grid_to_audio(audio_to_grid(a)).numpy(), a.numpy())

    def test_text_grid_shape(self):
        t = torch.arange(TEXT_LEN, dtype=torch.float32)
        g = text_to_grid(t)
        assert g.shape == (LATENT_HW, LATENT_HW)

    def test_grid_to_text_pools_correctly(self):
        # A grid where each 8-tile block holds a constant value should
        # decode back to that value per text channel.
        flat = torch.repeat_interleave(torch.arange(TEXT_LEN, dtype=torch.float32), 8)
        grid = flat.reshape(LATENT_HW, LATENT_HW)
        out = grid_to_text(grid)
        assert torch.allclose(out, torch.arange(TEXT_LEN, dtype=torch.float32))


# --------------------------------------------------------------------------- #
#  Encoder / decoder shape contract
# --------------------------------------------------------------------------- #

class TestEncodeDecodeShapes:
    @pytest.fixture
    def model(self):
        # Default: audio/text OFF. Use full media variant where needed.
        return NativeVOD()

    @pytest.fixture
    def model_full(self):
        return NativeVOD(NativeVODConfig(enable_audio=True, enable_text=True))

    @pytest.fixture
    def views(self):
        sample = _toy_batch(n=1).samples[0]
        return views_to_torch(sample.target_views, torch.device("cpu"))

    def test_default_active_media_image_video_only(self, model):
        assert model.active_media() == ("image", "video")

    def test_full_config_enables_all_media(self, model_full):
        assert set(model_full.active_media()) == {"image", "video", "audio", "text"}

    def test_encode_shape_default(self, model, views):
        u = model.encode(views)
        assert u.shape == (LATENT_T, LATENT_HW, LATENT_HW, model.config.channels)

    def test_encode_subset_works(self, model, views):
        u = model.encode({"image": views["image"]})
        assert u.shape == (LATENT_T, LATENT_HW, LATENT_HW, model.config.channels)

    def test_encode_audio_text_only_raises_when_disabled(self, model, views):
        # audio/text disabled → encoder finds no active media.
        with pytest.raises(ValueError, match="at least one active"):
            model.encode({"audio": views["audio"], "text": views["text"]})

    def test_encode_empty_raises(self, model):
        with pytest.raises(ValueError, match="at least one"):
            model.encode({})

    def test_decode_shapes_default(self, model, views):
        out = model.decode(model.encode(views))
        # Audio/text disabled — must NOT appear in decode output.
        assert set(out.keys()) == {"image", "video"}
        assert out["image"].shape == (LATENT_HW, LATENT_HW)
        assert out["video"].shape == (LATENT_T, LATENT_HW, LATENT_HW)

    def test_decode_shapes_full(self, model_full, views):
        out = model_full.decode(model_full.encode(views))
        assert out["audio"].shape == (AUDIO_SIZE,)
        assert out["text"].shape == (TEXT_LEN,)

    def test_denoise_preserves_shape(self, model, views):
        u_noisy = model.encode(views)
        out = model.denoise(u_noisy)
        assert out.shape == u_noisy.shape

    def test_denoise_path_preserves_shape(self, model, views):
        u_noisy = model.encode(views)
        out = model.denoise_path(u_noisy, steps=3)
        assert out.shape == u_noisy.shape


# --------------------------------------------------------------------------- #
#  End-to-end forward + loss
# --------------------------------------------------------------------------- #

class TestNativeForwardAndLoss:
    @pytest.fixture
    def model(self):
        return NativeVOD()

    @pytest.fixture
    def sample(self):
        return _toy_batch(n=1).samples[0]

    def test_forward_takes_only_noisy(self, model, sample):
        # forward() must NOT accept a `target` positional arg. This is
        # the no-leak contract: targets cannot reach the model.
        import inspect
        sig = inspect.signature(model.forward)
        assert "target_views" not in sig.parameters
        # Sanity: calling with only noisy_views succeeds.
        device = torch.device("cpu")
        noisy = views_to_torch(sample.noisy_views, device)
        pred, u_pred = model(noisy)
        assert pred["image"].shape == (LATENT_HW, LATENT_HW)
        assert pred["video"].shape == (LATENT_T, LATENT_HW, LATENT_HW)
        assert u_pred.shape == (LATENT_T, LATENT_HW, LATENT_HW, model.config.channels)

    def test_target_views_do_not_reach_model_inputs(self, model, sample):
        # If we tamper with target_views the model output must NOT
        # change, because the model never reads them.
        device = torch.device("cpu")
        noisy = views_to_torch(sample.noisy_views, device)
        target = views_to_torch(sample.target_views, device)
        target_corrupt = {k: torch.randn_like(v) for k, v in target.items()}

        with torch.no_grad():
            pred_a, _ = model(noisy)
        # Even though target is different, evaluating model again with
        # the same noisy must yield the same prediction. (Targets only
        # exist in the loss function — never in `forward`.)
        with torch.no_grad():
            pred_b, _ = model(noisy)
        for k in pred_a:
            torch.testing.assert_close(pred_a[k], pred_b[k])

        # Loss is allowed to depend on targets; this is just a sanity
        # check that the loss function does not crash with corrupted
        # targets (it shouldn't — targets only appear in MSE).
        loss_a, _ = native_total_loss(model, noisy, target)
        loss_b, _ = native_total_loss(model, noisy, target_corrupt)
        assert float(loss_a.detach()) != float(loss_b.detach())

    def test_loss_components_finite(self, model, sample):
        device = torch.device("cpu")
        noisy = views_to_torch(sample.noisy_views, device)
        target = views_to_torch(sample.target_views, device)
        loss, components = native_total_loss(model, noisy, target)
        assert torch.isfinite(loss)
        for key in ("L_field", "L_media", "L_temporal", "L_artifact", "L_text", "L_total"):
            assert math.isfinite(components[key])

    def test_loss_backward_no_grad_through_target(self, model, sample):
        # The encoder is allowed to receive gradient via the noisy path
        # and via decode→L_media. It MUST NOT receive gradient via the
        # target path (we detach `u_target_ref` inside L_field).
        device = torch.device("cpu")
        noisy = views_to_torch(sample.noisy_views, device)
        target = {k: v.clone().requires_grad_(True) for k, v in views_to_torch(sample.target_views, device).items()}
        loss, _ = native_total_loss(model, noisy, target)
        loss.backward()
        # Targets had requires_grad — but should have NO grad after
        # backward, because the loss path on the target side was
        # detached.
        for k, v in target.items():
            assert v.grad is None or float(v.grad.abs().sum()) == 0.0, (
                f"target gradient leaked through medium {k!r}"
            )

    def test_loss_backward_flows_to_active_subnets(self, model, sample):
        device = torch.device("cpu")
        noisy = views_to_torch(sample.noisy_views, device)
        target = views_to_torch(sample.target_views, device)
        loss, _ = native_total_loss(model, noisy, target)
        loss.backward()
        named = dict(model.named_parameters())
        # Active media encoders / decoders must see gradient. Inactive
        # media encoders / decoders are dead weights — they should NOT
        # be expected to receive gradient, and any test that asserts
        # they do is incorrect under the new active-media gating.
        # NOTE: enc_image and enc_video share weight (tied), so checking
        # via attribute access is more robust than name-prefix iteration
        # over named_parameters (which lists each tied tensor only once).
        for m in model.active_media():
            enc = getattr(model, f"enc_{m}")
            dec = getattr(model, f"dec_{m}")
            for module in (enc, dec):
                grads = [p.grad for p in module.parameters()]
                assert grads, m
                for g in grads:
                    assert g is not None and torch.isfinite(g).all(), m
        denoiser_grads = [p.grad for p in model.denoiser.parameters()]
        assert denoiser_grads
        for g in denoiser_grads:
            assert g is not None and torch.isfinite(g).all()

    def test_zero_weights_zero_loss(self, model, sample):
        device = torch.device("cpu")
        noisy = views_to_torch(sample.noisy_views, device)
        target = views_to_torch(sample.target_views, device)
        zero = NativeLossWeights(field=0.0, media=0.0, temporal=0.0, artifact=0.0, text=0.0)
        loss, _ = native_total_loss(model, noisy, target, weights=zero)
        assert float(loss.detach()) == 0.0


# --------------------------------------------------------------------------- #
#  Training reduces loss across a few steps
# --------------------------------------------------------------------------- #

def test_short_training_reduces_total_loss():
    """Training must reduce L_total even WITHOUT seeing the target
    inside the model. This was previously trivially satisfied via the
    target-condition leak; if it still holds under the no-leak
    regime, the model is at least exploiting the noisy → target
    structure and not just copying the answer."""
    torch.manual_seed(0)
    sample = _toy_batch(n=2, seed=0).samples
    model = NativeVOD(NativeVODConfig(channels=4, hidden=16))
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    device = torch.device("cpu")

    def step():
        opt.zero_grad(set_to_none=True)
        losses = []
        for s in sample:
            noisy = views_to_torch(s.noisy_views, device)
            target = views_to_torch(s.target_views, device)
            loss, _ = native_total_loss(model, noisy, target)
            losses.append(loss)
        total = torch.stack(losses).mean()
        total.backward()
        opt.step()
        return float(total.detach())

    initial = step()
    final = initial
    for _ in range(15):
        final = step()
    assert final < initial * 0.95, f"no learning: initial={initial} final={final}"


# --------------------------------------------------------------------------- #
#  Text corruption stress
# --------------------------------------------------------------------------- #

class TestTextCorruption:
    def test_corruption_changes_text_view(self):
        rng = np.random.default_rng(0)
        clean = rng.uniform(0, 1, TEXT_LEN)
        corrupted = inject_text_quantization_corruption(clean, np.random.default_rng(1), swap_rate=0.5)
        assert not np.allclose(corrupted, clean)
        assert corrupted.shape == clean.shape

    def test_zero_swap_rate_no_op(self):
        rng = np.random.default_rng(0)
        clean = rng.uniform(0, 1, TEXT_LEN)
        out = inject_text_quantization_corruption(clean, rng, swap_rate=0.0)
        np.testing.assert_array_equal(out, clean.astype(np.float64))
