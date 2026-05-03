"""Tests for VOD tile-residue artifact suppression."""

from __future__ import annotations

import math

import numpy as np
import pytest

from vod_minimal.artifacts import (
    D_AXIS,
    D_OC,
    oc_four_over_e,
    tile_residue,
)
from vod_minimal.core import build_projection_batch
from vod_minimal.metrics import mean_tile_residue
from vod_minimal.projections import add_noise


def _blocky_field(size: int = 64, tile: int = 8) -> np.ndarray:
    y = np.arange(size)[:, None] // tile
    x = np.arange(size)[None, :] // tile
    return ((x + y) % 2).astype(np.float64)


def test_orthogonal_compression_decay_matches_formula():
    assert D_OC == pytest.approx(4.0 / math.e)
    assert D_AXIS == pytest.approx(D_OC / 4.0)
    assert D_AXIS == pytest.approx(1.0 / math.e)


def test_tile_residue_detects_block_boundaries():
    blocky = _blocky_field(size=64, tile=8)
    smooth = np.linspace(0.0, 1.0, 64 * 64, dtype=np.float64).reshape(64, 64)

    assert tile_residue(blocky, tile=8) > 2.0
    assert tile_residue(blocky, tile=8) > tile_residue(smooth, tile=8)


def test_oc_four_over_e_is_deterministic_for_seed():
    view = _blocky_field(size=64, tile=8)
    out1 = oc_four_over_e(view, np.random.default_rng(430), beta=0.01, tile=8)
    out2 = oc_four_over_e(view, np.random.default_rng(430), beta=0.01, tile=8)

    np.testing.assert_array_equal(out1, out2)
    assert not np.allclose(out1, view)


def test_oc_four_over_e_respects_clean_views():
    view = np.zeros((32, 32), dtype=np.float64)
    out = oc_four_over_e(view, np.random.default_rng(0), beta=0.1, tile=8)
    np.testing.assert_array_equal(out, view)


def test_add_noise_keeps_default_behavior_when_disabled():
    view = _blocky_field(size=16, tile=4)
    rng1 = np.random.default_rng(1)
    rng2 = np.random.default_rng(1)

    plain = add_noise(view, rng1, scale=0.2)
    disabled = add_noise(view, rng2, scale=0.2, artifact_suppression=False)
    np.testing.assert_array_equal(plain, disabled)


def test_build_projection_batch_accepts_artifact_suppression():
    batch = build_projection_batch(
        np.random.default_rng(0),
        batch_size=2,
        size=16,
        artifact_suppression=True,
        artifact_scale=0.01,
        artifact_tile=4,
    )

    assert len(batch) == 2
    assert math.isfinite(mean_tile_residue(batch.samples[0].noisy_views, tile=4))
