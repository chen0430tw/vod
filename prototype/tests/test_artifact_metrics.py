"""Tests for the artifact-metrics layer.

Covers:
    - metrics.artifact_metrics returns the required keys
    - artifact_score is in [0, 1] for both blocky and smooth fields
    - blocky fields produce higher tile residue than smooth ones
    - core.evaluate_projection_error preserves its default 4-key contract
    - core.evaluate_projection_error gains the artifact_* keys when enabled
    - schema.canonical_metrics preserves artifact_* keys in checkpoint payloads
    - the train CLIs accept --artifact-metrics without crashing (smoke level)

Artifact metrics are evaluation-only diagnostics. None of these tests
exercise training; that contract is held by `feedback_no_detour_test_scripts`
and the trainers themselves never read artifact metrics into their loss.
"""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from vod_minimal.artifacts import tile_residue
from vod_minimal.core import (
    MEDIA,
    ProjectionBatch,
    build_projection_batch,
    evaluate_projection_error,
    make_numpy_rollout_fn,
    make_numpy_update_fn,
)
from vod_minimal.metrics import artifact_metrics
from vod_minimal.schema import canonical_metrics, checkpoint_payload


PROTOTYPE_ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #

def _blocky_field(size: int = 64, tile: int = 8) -> np.ndarray:
    y = np.arange(size)[:, None] // tile
    x = np.arange(size)[None, :] // tile
    return ((x + y) % 2).astype(np.float64)


def _smooth_field(size: int = 64) -> np.ndarray:
    return np.linspace(0.0, 1.0, size * size, dtype=np.float64).reshape(size, size)


# --------------------------------------------------------------------------- #
#  metrics.artifact_metrics
# --------------------------------------------------------------------------- #

class TestArtifactMetrics:
    def test_required_keys(self):
        views = {"image": _smooth_field()}
        out = artifact_metrics(views, tile=8)
        # Spatial main score + non-spatial diagnostic block. The
        # non-spatial keys are NaN here because there's no audio/text in
        # `views`, but they MUST always be present so callers and the
        # checkpoint schema never have to special-case absence.
        assert set(out.keys()) == {
            "mean_tile_residue",
            "max_tile_residue",
            "artifact_score",
            "non_spatial_mean_tile_residue",
            "non_spatial_max_tile_residue",
        }

    def test_image_only_yields_nan_non_spatial(self):
        out = artifact_metrics({"image": _smooth_field()}, tile=8)
        assert math.isnan(out["non_spatial_mean_tile_residue"])
        assert math.isnan(out["non_spatial_max_tile_residue"])

    def test_audio_text_only_yields_nan_spatial(self):
        # Pure non-spatial input: spatial main keys MUST be NaN — they
        # are the failure mode that does not exist in 1-D media.
        out = artifact_metrics({"audio": np.zeros(2048), "text": np.zeros(32)}, tile=8)
        assert math.isnan(out["mean_tile_residue"])
        assert math.isnan(out["max_tile_residue"])
        assert math.isnan(out["artifact_score"])
        # Non-spatial block is finite (audio/text both have computable
        # 1-D residues even if they aren't spatially meaningful).
        assert math.isfinite(out["non_spatial_mean_tile_residue"])
        assert math.isfinite(out["non_spatial_max_tile_residue"])

    def test_audio_text_do_not_dilute_spatial_score(self):
        # The whole point of the redesign: adding audio/text to a batch
        # of spatial views must NOT change the main artifact_score.
        rng = np.random.default_rng(0)
        audio = rng.standard_normal(2048)
        text = rng.standard_normal(32)
        spatial_only = artifact_metrics({"image": _blocky_field()}, tile=8)
        with_extras = artifact_metrics(
            {"image": _blocky_field(), "audio": audio, "text": text}, tile=8
        )
        for key in ("mean_tile_residue", "max_tile_residue", "artifact_score"):
            assert spatial_only[key] == pytest.approx(with_extras[key])

    def test_score_within_unit_interval_for_smooth_and_blocky(self):
        for views in [
            {"image": _smooth_field()},
            {"image": _blocky_field()},
            {"image": _smooth_field(), "video": _blocky_field()},
        ]:
            out = artifact_metrics(views, tile=8)
            assert 0.0 <= out["artifact_score"] <= 1.0

    def test_blocky_residue_exceeds_smooth(self):
        smooth_score = artifact_metrics({"image": _smooth_field()}, tile=8)
        blocky_score = artifact_metrics({"image": _blocky_field()}, tile=8)
        assert blocky_score["mean_tile_residue"] > smooth_score["mean_tile_residue"]
        # consistency with the underlying detector
        assert blocky_score["mean_tile_residue"] == pytest.approx(
            tile_residue(_blocky_field(), tile=8)
        )

    def test_score_decreases_as_residue_grows(self):
        smooth = artifact_metrics({"image": _smooth_field()}, tile=8)
        blocky = artifact_metrics({"image": _blocky_field()}, tile=8)
        assert blocky["artifact_score"] <= smooth["artifact_score"]

    def test_max_residue_is_at_least_mean(self):
        views = {"image": _smooth_field(), "video": _blocky_field()}
        out = artifact_metrics(views, tile=8)
        assert out["max_tile_residue"] >= out["mean_tile_residue"]

    def test_empty_views_returns_nan(self):
        out = artifact_metrics({}, tile=8)
        for key in (
            "mean_tile_residue",
            "max_tile_residue",
            "artifact_score",
            "non_spatial_mean_tile_residue",
            "non_spatial_max_tile_residue",
        ):
            assert math.isnan(out[key])


# --------------------------------------------------------------------------- #
#  core.evaluate_projection_error contract
# --------------------------------------------------------------------------- #

class TestEvaluateProjectionErrorArtifactOption:
    @pytest.fixture
    def small_batch(self) -> ProjectionBatch:
        rng = np.random.default_rng(430)
        return build_projection_batch(rng, batch_size=2, size=16, noise_scale=0.2)

    @pytest.fixture
    def numpy_rollout(self):
        return make_numpy_rollout_fn(make_numpy_update_fn(), steps=4)

    def test_default_keys_unchanged(self, small_batch, numpy_rollout):
        metrics = evaluate_projection_error(numpy_rollout, small_batch)
        assert set(metrics.keys()) == {
            "mean_before",
            "mean_after",
            "mean_improvement",
            "success_rate",
        }

    def test_enabled_keys_appended(self, small_batch, numpy_rollout):
        metrics = evaluate_projection_error(
            numpy_rollout, small_batch, include_artifact_metrics=True, artifact_tile=8
        )
        for required in (
            "mean_before",
            "mean_after",
            "mean_improvement",
            "success_rate",
            "artifact_before_mean_tile_residue",
            "artifact_after_mean_tile_residue",
            "artifact_after_score",
            "artifact_improvement",
            "non_spatial_artifact_before_mean_tile_residue",
            "non_spatial_artifact_after_mean_tile_residue",
        ):
            assert required in metrics, required

    def test_enabled_artifact_score_in_unit_interval(self, small_batch, numpy_rollout):
        metrics = evaluate_projection_error(
            numpy_rollout, small_batch, include_artifact_metrics=True
        )
        assert 0.0 <= metrics["artifact_after_score"] <= 1.0

    def test_empty_batch_returns_nan_artifact_keys(self, numpy_rollout):
        metrics = evaluate_projection_error(
            numpy_rollout, ProjectionBatch(samples=()), include_artifact_metrics=True
        )
        for key in (
            "artifact_before_mean_tile_residue",
            "artifact_after_mean_tile_residue",
            "artifact_after_score",
            "artifact_improvement",
            "non_spatial_artifact_before_mean_tile_residue",
            "non_spatial_artifact_after_mean_tile_residue",
        ):
            assert math.isnan(metrics[key])

    def test_default_invocation_does_not_compute_artifacts(self, small_batch, numpy_rollout):
        # Behavioural guarantee: opting in is required to surface artifact_*
        metrics = evaluate_projection_error(numpy_rollout, small_batch)
        for key in metrics:
            assert not key.startswith("artifact_")


# --------------------------------------------------------------------------- #
#  schema.canonical_metrics preserves artifact keys
# --------------------------------------------------------------------------- #

class TestSchemaPreservesArtifactKeys:
    def test_canonical_metrics_keeps_artifact_extras(self):
        raw = {
            "mean_before": 9.5,
            "mean_after": 1.2,
            "mean_improvement": 8.3,
            "success_rate": 1.0,
            "artifact_before_mean_tile_residue": 1.4,
            "artifact_after_mean_tile_residue": 1.1,
            "artifact_after_score": 0.91,
            "artifact_improvement": 0.3,
            "non_spatial_artifact_before_mean_tile_residue": 0.95,
            "non_spatial_artifact_after_mean_tile_residue": 0.93,
        }
        out = canonical_metrics(raw)
        for key, value in raw.items():
            assert out[key] == pytest.approx(value)

    def test_checkpoint_payload_includes_artifact_metrics(self):
        train = {"mean_before": 10.0, "mean_after": 2.0, "artifact_after_score": 0.8}
        test = {"mean_before": 9.5, "mean_after": 1.5, "artifact_improvement": 0.2}
        payload = checkpoint_payload(
            state_dict={"placeholder": 0},
            model_type="ToyModel",
            train_args={"epochs": 1},
            train_metrics=train,
            test_metrics=test,
        )
        assert payload["train_metrics"]["artifact_after_score"] == pytest.approx(0.8)
        assert payload["test_metrics"]["artifact_improvement"] == pytest.approx(0.2)


# --------------------------------------------------------------------------- #
#  Smoke: train CLIs accept --artifact-metrics
# --------------------------------------------------------------------------- #

class TestTrainCLISmoke:
    @pytest.mark.parametrize(
        "script,extra",
        [
            (
                "train_torch_prototype.py",
                ["--train-n", "2", "--test-n", "2", "--epochs", "1", "--steps", "1",
                 "--hidden", "8", "--cpu", "--save", ""],
            ),
            (
                "train_vdit_prototype.py",
                ["--train-n", "2", "--test-n", "2", "--epochs", "1", "--steps", "1",
                 "--hidden", "16", "--depth", "1", "--heads", "2", "--max-tokens", "64",
                 "--cpu", "--save", ""],
            ),
        ],
    )
    def test_help_includes_artifact_flags(self, script, extra):
        # Use --help so the smoke is sub-second and never trains anything.
        del extra  # only needed for parameter id
        out = subprocess.run(
            [sys.executable, str(PROTOTYPE_ROOT / script), "--help"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
            cwd=str(PROTOTYPE_ROOT),
        )
        assert "--artifact-metrics" in out.stdout
        assert "--artifact-tile" in out.stdout
