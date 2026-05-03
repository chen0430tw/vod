"""Tests for the spatiotemporal upgrade.

Covers:
    - 3-D Chladni field generation and shape contract
    - project_video_3d slicing of U(t, y, x)
    - project_all video_mode = "auto" / "2d" / "3d" branches
    - temporal metrics on clean / flicker / drift videos
    - core.build_projection_batch(spacetime=True) populates spacetime fields
    - core compatibility: legacy (spacetime=False) batches unchanged
    - blocky_scattering temporal modes really change the targeted metric
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from vod_minimal.blocky_scattering import (
    build_blocky_scattering_batch,
    inject_temporal_blocky_drift,
    inject_temporal_flicker,
)
from vod_minimal.core import build_projection_batch
from vod_minimal.metrics import (
    cross_frame_consistency_score,
    frame_descriptor_drift,
    temporal_artifact_drift,
    temporal_metrics,
    temporal_smoothness,
)
from vod_minimal.projections import project_all, project_video, project_video_3d
from vod_minimal.spacetime_chladni import (
    SpacetimeBoundary,
    chladni_spacetime_field,
    random_chladni_spacetime_field,
)


# --------------------------------------------------------------------------- #
#  Spacetime field
# --------------------------------------------------------------------------- #

class TestSpacetimeField:
    def test_shape(self):
        field = chladni_spacetime_field(SpacetimeBoundary(size=32, frames=8))
        assert field.shape == (8, 32, 32)
        assert field.dtype == np.float64

    def test_normalised_amplitude(self):
        field = chladni_spacetime_field(SpacetimeBoundary(size=32, frames=8))
        assert np.max(np.abs(field)) <= 1.0 + 1e-9
        assert np.max(np.abs(field)) > 0.5  # not collapsed

    def test_random_field_reproducible(self):
        a = random_chladni_spacetime_field(np.random.default_rng(0), size=32, frames=8)
        b = random_chladni_spacetime_field(np.random.default_rng(0), size=32, frames=8)
        np.testing.assert_array_equal(a, b)

    def test_temporal_variation_present(self):
        # The whole point of a 3-D field: frames are NOT identical.
        f = chladni_spacetime_field(SpacetimeBoundary(size=32, frames=8))
        assert not np.allclose(f[0], f[-1])

    def test_invalid_dimensions_raise(self):
        with pytest.raises(ValueError, match="size"):
            chladni_spacetime_field(SpacetimeBoundary(size=1, frames=4))
        with pytest.raises(ValueError, match="frames"):
            chladni_spacetime_field(SpacetimeBoundary(size=32, frames=0))


# --------------------------------------------------------------------------- #
#  3-D projections
# --------------------------------------------------------------------------- #

class TestProjectVideo3D:
    def test_3d_input_returns_volume(self):
        vol = chladni_spacetime_field(SpacetimeBoundary(size=32, frames=10))
        out = project_video_3d(vol)
        assert out.shape == (10, 32, 32)
        np.testing.assert_array_equal(out, vol)

    def test_3d_input_resampled_frames(self):
        vol = chladni_spacetime_field(SpacetimeBoundary(size=32, frames=10))
        out = project_video_3d(vol, frames=5)
        assert out.shape == (5, 32, 32)

    def test_2d_input_falls_back_to_legacy(self):
        flat = np.zeros((32, 32))
        flat[8, :] = 1.0
        legacy = project_video(flat, frames=4)
        out = project_video_3d(flat, frames=4)
        np.testing.assert_array_equal(out, legacy)

    def test_invalid_dim_raises(self):
        with pytest.raises(ValueError, match="ndim"):
            project_video_3d(np.zeros((4, 4, 4, 4)))


class TestProjectAllModes:
    def test_auto_picks_3d_for_3d_field(self):
        vol = chladni_spacetime_field(SpacetimeBoundary(size=32, frames=6))
        out = project_all(vol)  # auto
        assert out["video"].shape == (6, 32, 32)
        # image / audio / text remain single instances built from the
        # temporal mean, so video has one extra axis.
        assert out["image"].shape == (32, 32)
        assert out["audio"].ndim == 1
        assert out["text"].ndim == 1

    def test_explicit_2d_on_3d_field_uses_first_frame(self):
        vol = chladni_spacetime_field(SpacetimeBoundary(size=32, frames=6))
        out_2d = project_all(vol, video_mode="2d")
        # Legacy mode takes frame 0 as the source image.
        np.testing.assert_array_equal(out_2d["image"], vol[0])

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="video_mode"):
            project_all(np.zeros((32, 32)), video_mode="banana")

    def test_2d_field_default_is_legacy(self):
        flat = np.linspace(0, 1, 32 * 32).reshape(32, 32)
        out = project_all(flat)  # auto → 2d
        legacy = project_video(flat)
        np.testing.assert_array_equal(out["video"], legacy)


# --------------------------------------------------------------------------- #
#  Temporal metrics
# --------------------------------------------------------------------------- #

class TestTemporalMetrics:
    def _clean_clip(self, frames=8, size=32):
        return chladni_spacetime_field(SpacetimeBoundary(size=size, frames=frames))

    def test_smoothness_lower_for_clean_than_flicker(self):
        clip = self._clean_clip()
        flickered = inject_temporal_flicker(clip, np.random.default_rng(0), strength=0.5)
        assert temporal_smoothness(clip) < temporal_smoothness(flickered)

    def test_artifact_drift_higher_for_blocky_drift(self):
        clip = self._clean_clip()
        drift = inject_temporal_blocky_drift(
            clip, np.random.default_rng(0), tile=8, strength=0.6, drift=2
        )
        assert temporal_artifact_drift(drift, tile=8) > temporal_artifact_drift(clip, tile=8)

    def test_consistency_score_in_unit_interval(self):
        clip = self._clean_clip()
        s = cross_frame_consistency_score(clip)
        assert 0.0 <= s <= 1.0

    def test_consistency_score_drops_under_flicker(self):
        clip = self._clean_clip()
        flickered = inject_temporal_flicker(clip, np.random.default_rng(0), strength=0.5)
        assert cross_frame_consistency_score(flickered) < cross_frame_consistency_score(clip)

    def test_descriptor_drift_finite_on_clean_clip(self):
        clip = self._clean_clip()
        d = frame_descriptor_drift(clip)
        assert math.isfinite(d)

    def test_single_frame_returns_nan(self):
        clip = np.zeros((1, 32, 32))
        for fn in (temporal_smoothness, frame_descriptor_drift, cross_frame_consistency_score):
            assert math.isnan(fn(clip))

    def test_temporal_metrics_aggregate_keys(self):
        clip = self._clean_clip()
        m = temporal_metrics({"video": clip})
        assert set(m.keys()) == {
            "temporal_smoothness",
            "frame_descriptor_drift",
            "temporal_artifact_drift",
            "cross_frame_consistency_score",
        }

    def test_temporal_metrics_no_video_returns_nan(self):
        m = temporal_metrics({"image": np.zeros((32, 32))})
        for v in m.values():
            assert math.isnan(v)


# --------------------------------------------------------------------------- #
#  core.build_projection_batch spacetime path
# --------------------------------------------------------------------------- #

class TestSpacetimeBatch:
    def test_legacy_batch_has_no_spacetime_fields(self):
        batch = build_projection_batch(np.random.default_rng(0), batch_size=2, size=32)
        for s in batch.samples:
            assert s.source_spacetime_field is None
            assert s.target_spacetime_field is None
            # video remains the legacy 2-D shape (10, 32, 32)
            assert s.noisy_views["video"].shape == (10, 32, 32)

    def test_spacetime_batch_populates_volumes(self):
        batch = build_projection_batch(
            np.random.default_rng(0), batch_size=2, size=32, spacetime=True, frames=6
        )
        for s in batch.samples:
            assert s.source_spacetime_field is not None
            assert s.target_spacetime_field is not None
            assert s.source_spacetime_field.shape == (6, 32, 32)
            # video projection comes straight from the volume → identical
            # frame count.
            assert s.noisy_views["video"].shape[0] == 6

    def test_spacetime_video_has_real_temporal_variation(self):
        # The legacy video projection is a roll of one image → all frames
        # are correlated copies. The spacetime projection should show
        # genuine temporal_smoothness > 0 even before noise is added.
        batch = build_projection_batch(
            np.random.default_rng(0), batch_size=1, size=32, spacetime=True, frames=8
        )
        s = batch.samples[0]
        # We measure smoothness on the *target* views to avoid the
        # additive noise inflating the number — the underlying volume
        # has temporal modes baked in.
        assert temporal_smoothness(s.target_views["video"]) > 0.0


# --------------------------------------------------------------------------- #
#  Temporal stress: each mode moves the right metric
# --------------------------------------------------------------------------- #

class TestTemporalStressBatches:
    def _batch(self, mode: str):
        return build_blocky_scattering_batch(
            np.random.default_rng(0),
            batch_size=2,
            size=32,
            noise_scale=0.24,
            artifact_strength=0.4,
            tile=8,
            spacetime=True,
            frames=8,
            temporal_mode=mode,
            flicker_strength=0.5,
            drift=2,
        )

    def test_flicker_breaks_temporal_smoothness(self):
        clean = self._batch("static")
        flicker = self._batch("flicker")
        clean_v = clean.samples[0].noisy_views["video"]
        flicker_v = flicker.samples[0].noisy_views["video"]
        assert temporal_smoothness(flicker_v) > temporal_smoothness(clean_v)

    def test_blocky_drift_raises_temporal_artifact_drift(self):
        static = self._batch("static")
        drift = self._batch("blocky_drift")
        static_v = static.samples[0].noisy_views["video"]
        drift_v = drift.samples[0].noisy_views["video"]
        assert temporal_artifact_drift(drift_v, tile=8) > temporal_artifact_drift(static_v, tile=8)

    def test_invalid_temporal_mode_raises(self):
        with pytest.raises(ValueError, match="temporal_mode"):
            build_blocky_scattering_batch(
                np.random.default_rng(0), batch_size=1, size=32,
                spacetime=True, frames=8, temporal_mode="banana",
            )
