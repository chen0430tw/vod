"""Contract tests locking in the baseline-recovery fixes (Section 12.11).

If any of these assertions break, the corresponding postmortem fix has
been silently undone. The intent is NOT to test new behavior, but to
catch future regressions when someone refactors `native.py`,
`projections.py`, or the experiment / core wiring.

Each test maps to a specific fix:
    - test_paired_denoising_protocol      → Section 12.11 Path-A #1 Task
    - test_image_projection_middle_frame  → Section 12.11 #3 Metric
    - test_denoiser_feat_multiscale_smooth → Section 12.11 #5 Impl-b
    - test_decode_image_uses_middle_frame → Section 12.11 #5 Impl-c
"""

from __future__ import annotations

import inspect

import numpy as np
import torch

from vod_minimal.core import build_projection_batch
from vod_minimal.experiment import make_sample
from vod_minimal.native import LATENT_HW, LATENT_T, NativeVOD, NativeVODConfig
from vod_minimal.projections import project_all
from vod_minimal.spacetime_chladni import chladni_spacetime_field, SpacetimeBoundary


# --------------------------------------------------------------------------- #
#  #1 Task: paired_denoising flag — noisy = target + N(0, σ), same field.
# --------------------------------------------------------------------------- #

class TestPairedDenoisingProtocol:
    def test_make_sample_accepts_paired_flag(self):
        # signature contract: paired_denoising must be a kw arg
        sig = inspect.signature(make_sample)
        assert "paired_denoising" in sig.parameters
        assert sig.parameters["paired_denoising"].default is False

    def test_paired_noisy_equals_target_plus_noise(self):
        # When paired=True, noisy_views[m] - target_views[m] should be
        # zero-mean Gaussian with variance ≈ noise_scale**2.
        rng = np.random.default_rng(0)
        sample = make_sample(rng, size=16, noise_scale=0.24, paired_denoising=True)
        diff = sample.noisy_views["image"] - sample.target_views["image"]
        var = float(np.var(diff))
        assert abs(var - 0.24 ** 2) / (0.24 ** 2) < 0.30, (
            f"paired_denoising noise var off: got {var:.4f}, expect {0.24**2:.4f}"
        )

    def test_paired_source_field_equals_target_field(self):
        # In paired mode, ground truth IS the target — Sample.source_field
        # and target_field must be the same array.
        rng = np.random.default_rng(0)
        sample = make_sample(rng, size=16, paired_denoising=True)
        np.testing.assert_array_equal(sample.source_field, sample.target_field)

    def test_legacy_independent_path_preserved(self):
        # Default paired_denoising=False must still produce two independent
        # Chladni fields. This locks the legacy conditional-gen toy in place.
        rng = np.random.default_rng(0)
        sample = make_sample(rng, size=16, paired_denoising=False)
        assert not np.allclose(sample.source_field, sample.target_field)

    def test_build_projection_batch_paired_spacetime(self):
        # Spacetime + paired must give noisy_views = target_views + noise.
        batch = build_projection_batch(
            np.random.default_rng(0), batch_size=2,
            size=LATENT_HW, frames=LATENT_T, spacetime=True,
            paired_denoising=True, noise_scale=0.24,
        )
        for s in batch.samples:
            for m in ("image", "video"):
                diff = s.noisy_views[m] - s.target_views[m]
                var = float(np.var(diff))
                assert abs(var - 0.24 ** 2) / (0.24 ** 2) < 0.40


# --------------------------------------------------------------------------- #
#  #3 Metric: image projection takes middle frame in spacetime mode.
# --------------------------------------------------------------------------- #

class TestImageProjectionMiddleFrame:
    def test_3d_spacetime_image_is_middle_frame(self):
        # Build a known 3-D field and verify project_all(...)["image"]
        # equals U[T//2], not U.mean(axis=0).
        field = chladni_spacetime_field(SpacetimeBoundary(size=16, frames=8))
        out = project_all(field, video_mode="3d")
        T = field.shape[0]
        np.testing.assert_array_equal(out["image"], field[T // 2])

    def test_audio_text_use_temporal_mean(self):
        # audio / text are time-aggregates: project_all(..., video_mode="3d")
        # routes them through field.mean(axis=0). Verify by computing the
        # expected outputs from the temporal mean and comparing.
        from vod_minimal.projections import project_audio, project_text
        field = chladni_spacetime_field(SpacetimeBoundary(size=16, frames=8))
        mean_field = field.mean(axis=0)
        out = project_all(field, video_mode="3d")
        np.testing.assert_array_equal(out["audio"], project_audio(mean_field))
        np.testing.assert_array_equal(out["text"], project_text(mean_field))


# --------------------------------------------------------------------------- #
#  #5 Impl-b: denoiser feature includes multi-scale spatial smooth.
# --------------------------------------------------------------------------- #

class TestDenoiserMultiscaleSmooth:
    def test_denoiser_input_dim_is_3C_plus_3(self):
        # feat_in should be 3*channels + 3
        # ([u_noisy, smooth3, smooth5, pos_t, pos_y, pos_x])
        # This is a contract on the LEGACY pointwise MLP backbone. The
        # default UNet backbone has the same feature inventory but its
        # input layer is a Conv2d, not a Linear — see
        # `test_unet_denoiser.py` for the parallel UNet shape contract.
        for c in (4, 8):
            model = NativeVOD(NativeVODConfig(channels=c, hidden=16, backbone="mlp"))
            first_linear = model.denoiser.net[0]
            assert isinstance(first_linear, torch.nn.Linear)
            assert first_linear.in_features == 3 * c + 3, (
                f"denoiser feat_in={first_linear.in_features}, expected {3*c+3}"
            )

    def test_denoise_output_shape_matches_input(self):
        # Sanity: even with multi-scale smooth feature, denoise() output
        # should match input shape (T, H, W, C). Run for both backbones
        # so the substrate contract holds either way.
        torch.manual_seed(0)
        for backbone in ("unet", "mlp"):
            model = NativeVOD(NativeVODConfig(channels=4, hidden=16, backbone=backbone))
            u = torch.randn(LATENT_T, LATENT_HW, LATENT_HW, 4)
            out = model.denoise(u)
            assert out.shape == u.shape, backbone


# --------------------------------------------------------------------------- #
#  #5 Impl-c: decode_image takes middle frame, not temporal mean.
# --------------------------------------------------------------------------- #

class TestDecodeImageMiddleFrame:
    def test_decode_image_equals_dec_image_of_middle_frame(self):
        torch.manual_seed(0)
        model = NativeVOD(NativeVODConfig(channels=4, hidden=16))
        # Build a U where each frame has a distinct constant value so
        # mean(t) and middle_frame produce different decode outputs.
        T, H, W, C = LATENT_T, LATENT_HW, LATENT_HW, 4
        u = torch.zeros(T, H, W, C)
        for t in range(T):
            u[t] = float(t) - T / 2.0  # frame t = const t-T/2
        out = model.decode(u)
        expected = model.dec_image(u[T // 2]).squeeze(-1)
        torch.testing.assert_close(out["image"], expected)

    def test_decode_video_unchanged_full_t(self):
        # Decode for video should still be per-frame, not collapsed.
        torch.manual_seed(0)
        model = NativeVOD(NativeVODConfig(channels=4, hidden=16))
        u = torch.randn(LATENT_T, LATENT_HW, LATENT_HW, 4)
        out = model.decode(u)
        assert out["video"].shape == (LATENT_T, LATENT_HW, LATENT_HW)
