"""Pytest suite for `vod_minimal.core`.

Exercises the four simplified interfaces (build_projection_batch,
shared_update_rollout, projection_loss, evaluate_projection_error) plus the
ProjectionSample/ProjectionBatch dataclasses and the NumPy convenience
factories. Does not retrain anything — only verifies the contracts.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from vod_minimal.core import (
    MEDIA,
    ProjectionBatch,
    ProjectionSample,
    build_projection_batch,
    evaluate_projection_error,
    make_numpy_rollout_fn,
    make_numpy_update_fn,
    projection_loss,
    shared_update_rollout,
)


# --------------------------------------------------------------------------- #
#  Shared fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture
def small_batch() -> ProjectionBatch:
    """Tiny deterministic batch — keeps tests fast (size=16, n=2)."""
    rng = np.random.default_rng(430)
    return build_projection_batch(rng, batch_size=2, size=16, noise_scale=0.2)


@pytest.fixture
def numpy_update():
    return make_numpy_update_fn()


@pytest.fixture
def torch_update():
    """A trivially-trainable torch update_fn: scaled blend toward target."""

    alpha = torch.nn.Parameter(torch.tensor(0.3, dtype=torch.float32))

    def _step(current: torch.Tensor, target: torch.Tensor, medium: str) -> torch.Tensor:
        del medium
        return current + alpha * (target - current)

    _step.alpha = alpha  # expose for grad-flow assertions
    return _step


# --------------------------------------------------------------------------- #
#  build_projection_batch
# --------------------------------------------------------------------------- #

class TestBuildProjectionBatch:
    def test_returns_projection_batch(self, small_batch):
        assert isinstance(small_batch, ProjectionBatch)

    def test_length_matches_batch_size(self):
        rng = np.random.default_rng(0)
        batch = build_projection_batch(rng, batch_size=5, size=16)
        assert len(batch) == 5

    def test_each_sample_has_all_four_media(self, small_batch):
        for sample in small_batch:
            assert isinstance(sample, ProjectionSample)
            assert set(sample.noisy_views.keys()) == set(MEDIA)
            assert set(sample.target_views.keys()) == set(MEDIA)

    def test_view_shapes(self, small_batch):
        # image: (size, size); video: (10, size, size); audio: (2048,); text: (32,)
        sample = small_batch.samples[0]
        assert sample.noisy_views["image"].shape == (16, 16)
        assert sample.noisy_views["video"].shape == (10, 16, 16)
        assert sample.noisy_views["audio"].shape == (2048,)
        assert sample.noisy_views["text"].shape == (32,)
        for medium in MEDIA:
            assert sample.target_views[medium].shape == sample.noisy_views[medium].shape

    def test_source_and_target_field_shape(self, small_batch):
        for sample in small_batch:
            assert sample.source_field.shape == (16, 16)
            assert sample.target_field.shape == (16, 16)

    def test_noise_actually_added(self, small_batch):
        # noisy_views must differ from a re-projection of source — at least we
        # can check that noisy != target (target is a different field anyway).
        sample = small_batch.samples[0]
        assert not np.allclose(sample.noisy_views["image"], sample.target_views["image"])

    def test_reproducibility_same_seed(self):
        b1 = build_projection_batch(np.random.default_rng(7), batch_size=3, size=16)
        b2 = build_projection_batch(np.random.default_rng(7), batch_size=3, size=16)
        for s1, s2 in zip(b1.samples, b2.samples):
            for medium in MEDIA:
                np.testing.assert_array_equal(s1.noisy_views[medium], s2.noisy_views[medium])
                np.testing.assert_array_equal(s1.target_views[medium], s2.target_views[medium])

    def test_different_seed_produces_different_data(self):
        b1 = build_projection_batch(np.random.default_rng(1), batch_size=2, size=16)
        b2 = build_projection_batch(np.random.default_rng(2), batch_size=2, size=16)
        assert not np.allclose(b1.samples[0].noisy_views["image"], b2.samples[0].noisy_views["image"])

    def test_zero_batch_size_returns_empty(self):
        batch = build_projection_batch(np.random.default_rng(0), batch_size=0, size=16)
        assert len(batch) == 0
        assert tuple(batch) == ()

    def test_negative_batch_size_raises(self):
        with pytest.raises(ValueError, match="batch_size"):
            build_projection_batch(np.random.default_rng(0), batch_size=-1, size=16)

    def test_media_subset_filter(self):
        rng = np.random.default_rng(0)
        batch = build_projection_batch(rng, batch_size=2, size=16, media=("image", "audio"))
        assert batch.media == ("image", "audio")
        sample = batch.samples[0]
        assert set(sample.noisy_views.keys()) == {"image", "audio"}
        assert set(sample.target_views.keys()) == {"image", "audio"}


# --------------------------------------------------------------------------- #
#  shared_update_rollout
# --------------------------------------------------------------------------- #

class TestSharedUpdateRollout:
    def test_steps_zero_returns_input_unchanged(self, numpy_update):
        x = np.array([1.0, 2.0, 3.0])
        t = np.array([0.0, 0.0, 0.0])
        out = shared_update_rollout(numpy_update, x, t, "image", steps=0)
        np.testing.assert_array_equal(out, x)

    def test_steps_positive_changes_input(self, small_batch, numpy_update):
        sample = small_batch.samples[0]
        out = shared_update_rollout(
            numpy_update,
            sample.noisy_views["image"],
            sample.target_views["image"],
            "image",
            steps=4,
        )
        assert not np.allclose(out, sample.noisy_views["image"])

    def test_return_path_length(self, small_batch, numpy_update):
        sample = small_batch.samples[0]
        path = shared_update_rollout(
            numpy_update,
            sample.noisy_views["image"],
            sample.target_views["image"],
            "image",
            steps=5,
            return_path=True,
        )
        assert isinstance(path, list)
        assert len(path) == 6  # steps + 1

    def test_return_path_first_is_input(self, small_batch, numpy_update):
        sample = small_batch.samples[0]
        path = shared_update_rollout(
            numpy_update,
            sample.noisy_views["image"],
            sample.target_views["image"],
            "image",
            steps=3,
            return_path=True,
        )
        np.testing.assert_array_equal(path[0], sample.noisy_views["image"])

    def test_torch_input_returns_torch(self, torch_update):
        x = torch.zeros(8)
        t = torch.ones(8)
        out = shared_update_rollout(torch_update, x, t, "image", steps=3)
        assert torch.is_tensor(out)
        assert out.shape == x.shape

    def test_call_count_equals_steps(self, monkeypatch):
        calls = {"n": 0}

        def counting_update(current, target, medium):
            calls["n"] += 1
            return current

        x = np.zeros(4)
        shared_update_rollout(counting_update, x, x, "image", steps=7)
        assert calls["n"] == 7

    def test_negative_steps_raises(self, numpy_update):
        x = np.zeros(4)
        with pytest.raises(ValueError, match="steps"):
            shared_update_rollout(numpy_update, x, x, "image", steps=-1)

    def test_numpy_step_pulls_toward_target(self, numpy_update):
        # The default numpy_update hyperparameters are tuned for smooth
        # Chladni-like targets (the prototype's actual workload), not for
        # white-noise targets — random targets can overshoot. Use a real
        # synthetic Chladni field for the convergence assertion.
        from vod_minimal.chladni import random_chladni_field

        rng = np.random.default_rng(0)
        target = random_chladni_field(rng, size=16, n_modes=2)
        noisy = target + rng.standard_normal(target.shape) * 0.3
        before = np.linalg.norm(noisy - target)
        after = shared_update_rollout(numpy_update, noisy, target, "image", steps=12)
        after_err = np.linalg.norm(after - target)
        assert after_err < before


# --------------------------------------------------------------------------- #
#  projection_loss
# --------------------------------------------------------------------------- #

class TestProjectionLoss:
    def test_returns_scalar_tensor(self, small_batch, torch_update):
        loss = projection_loss(torch_update, small_batch, steps=2, device=torch.device("cpu"))
        assert torch.is_tensor(loss)
        assert loss.ndim == 0

    def test_loss_is_finite(self, small_batch, torch_update):
        loss = projection_loss(torch_update, small_batch, steps=2, device=torch.device("cpu"))
        assert torch.isfinite(loss)

    def test_normalize_changes_value(self, small_batch, torch_update):
        loss_norm = projection_loss(
            torch_update, small_batch, steps=2, device=torch.device("cpu"), normalize=True
        )
        loss_raw = projection_loss(
            torch_update, small_batch, steps=2, device=torch.device("cpu"), normalize=False
        )
        # Different scaling → different magnitude
        assert not torch.isclose(loss_norm, loss_raw)

    def test_gradient_flows_through_update_fn(self, small_batch, torch_update):
        loss = projection_loss(torch_update, small_batch, steps=2, device=torch.device("cpu"))
        loss.backward()
        assert torch_update.alpha.grad is not None
        assert torch.isfinite(torch_update.alpha.grad)
        assert torch_update.alpha.grad.abs().item() > 0  # actually responsive

    def test_empty_batch_returns_zero_no_crash(self, torch_update):
        empty = ProjectionBatch(samples=())
        loss = projection_loss(torch_update, empty, steps=1, device=torch.device("cpu"))
        assert torch.is_tensor(loss)
        assert float(loss) == 0.0

    def test_media_filter_subset(self, small_batch, torch_update):
        loss_all = projection_loss(
            torch_update, small_batch, steps=1, device=torch.device("cpu"), media=MEDIA
        )
        loss_image = projection_loss(
            torch_update, small_batch, steps=1, device=torch.device("cpu"), media=("image",)
        )
        # Image-only and all-media losses are typically different
        assert not torch.isclose(loss_all, loss_image)

    def test_empty_media_tuple_raises(self, small_batch, torch_update):
        with pytest.raises(ValueError, match="media"):
            projection_loss(torch_update, small_batch, steps=1, device=torch.device("cpu"), media=())


# --------------------------------------------------------------------------- #
#  evaluate_projection_error
# --------------------------------------------------------------------------- #

class TestEvaluateProjectionError:
    def test_returns_required_keys(self, small_batch, numpy_update):
        rollout_fn = make_numpy_rollout_fn(numpy_update, steps=4)
        metrics = evaluate_projection_error(rollout_fn, small_batch)
        assert set(metrics.keys()) == {
            "mean_before",
            "mean_after",
            "mean_improvement",
            "success_rate",
        }

    def test_metrics_are_finite(self, small_batch, numpy_update):
        rollout_fn = make_numpy_rollout_fn(numpy_update, steps=4)
        metrics = evaluate_projection_error(rollout_fn, small_batch)
        for key, value in metrics.items():
            assert math.isfinite(value), f"{key} is not finite: {value}"

    def test_identity_rollout_produces_zero_improvement(self, small_batch):
        # An identity rollout keeps noisy == noisy → improvement == 0,
        # success_rate == 0 (strict less-than fails), and after == before.
        def identity(noisy_views, target_views):
            return dict(noisy_views)

        metrics = evaluate_projection_error(identity, small_batch)
        assert metrics["mean_improvement"] == pytest.approx(0.0, abs=1e-6)
        assert metrics["mean_after"] == pytest.approx(metrics["mean_before"], abs=1e-6)
        assert metrics["success_rate"] == 0.0

    def test_real_rollout_improves_error(self, small_batch, numpy_update):
        rollout_fn = make_numpy_rollout_fn(numpy_update, steps=12)
        metrics = evaluate_projection_error(rollout_fn, small_batch)
        # Analytic step with reasonable hyperparameters should improve the error
        assert metrics["mean_after"] < metrics["mean_before"]
        assert metrics["mean_improvement"] > 0.0
        assert 0.0 <= metrics["success_rate"] <= 1.0

    def test_empty_batch_returns_nan(self, numpy_update):
        rollout_fn = make_numpy_rollout_fn(numpy_update, steps=1)
        metrics = evaluate_projection_error(rollout_fn, ProjectionBatch(samples=()))
        for key in ("mean_before", "mean_after", "mean_improvement", "success_rate"):
            assert math.isnan(metrics[key])


# --------------------------------------------------------------------------- #
#  ProjectionSample / ProjectionBatch dataclasses
# --------------------------------------------------------------------------- #

class TestProjectionContainers:
    def test_projection_sample_from_sample_roundtrip(self):
        from vod_minimal.experiment import make_sample

        rng = np.random.default_rng(0)
        sample = make_sample(rng, size=16)
        ps = ProjectionSample.from_sample(sample)
        np.testing.assert_array_equal(ps.source_field, sample.source_field)
        np.testing.assert_array_equal(ps.target_field, sample.target_field)
        assert set(ps.noisy_views.keys()) == set(sample.noisy_views.keys())
        for medium in sample.noisy_views:
            np.testing.assert_array_equal(ps.noisy_views[medium], sample.noisy_views[medium])

    def test_projection_batch_default_media(self):
        batch = ProjectionBatch(samples=())
        assert batch.media == MEDIA

    def test_projection_batch_iter(self, small_batch):
        # __iter__ should produce ProjectionSample instances
        items = list(small_batch)
        assert len(items) == len(small_batch)
        for s in items:
            assert isinstance(s, ProjectionSample)


# --------------------------------------------------------------------------- #
#  Convenience helpers
# --------------------------------------------------------------------------- #

class TestNumpyHelpers:
    def test_make_numpy_update_fn_matches_minimal_vod(self):
        # core's helper should produce the same single-step result as
        # model.MinimalVOD.update_path with steps=1.
        from vod_minimal.model import MinimalVOD

        rng = np.random.default_rng(42)
        x = rng.standard_normal((8, 8))
        t = rng.standard_normal((8, 8))

        core_step = make_numpy_update_fn(diffusion=0.55, reaction=0.18, step_size=0.9)
        core_out = core_step(x, t, "image")

        model_path = MinimalVOD(diffusion=0.55, reaction=0.18, step_size=0.9, steps=1).update_path(x, t)
        np.testing.assert_allclose(core_out, model_path[-1], rtol=1e-9, atol=1e-12)

    def test_make_numpy_rollout_fn_skips_missing_media(self, numpy_update):
        rng = np.random.default_rng(0)
        batch = build_projection_batch(rng, batch_size=1, size=16, media=("image", "audio"))
        rollout_fn = make_numpy_rollout_fn(numpy_update, steps=2, media=MEDIA)
        out = rollout_fn(batch.samples[0].noisy_views, batch.samples[0].target_views)
        # should only output the media that were present in the input
        assert set(out.keys()) == {"image", "audio"}

    def test_rollout_step_count_matches(self, numpy_update):
        # rollout_fn(steps=K) should equal K direct shared_update_rollout calls.
        rng = np.random.default_rng(0)
        batch = build_projection_batch(rng, batch_size=1, size=16)
        sample = batch.samples[0]
        K = 5
        rollout_fn = make_numpy_rollout_fn(numpy_update, steps=K)
        out = rollout_fn(sample.noisy_views, sample.target_views)
        for medium in MEDIA:
            expected = shared_update_rollout(
                numpy_update,
                sample.noisy_views[medium],
                sample.target_views[medium],
                medium,
                steps=K,
            )
            np.testing.assert_allclose(out[medium], expected, rtol=1e-9, atol=1e-12)


# --------------------------------------------------------------------------- #
#  End-to-end smoke
# --------------------------------------------------------------------------- #

def test_end_to_end_validation_pipeline_succeeds():
    """Same shape as run_core_validation.py but tiny — confirms the four
    interfaces compose correctly out of the box."""
    rng = np.random.default_rng(123)
    batch = build_projection_batch(rng, batch_size=4, size=16)
    update_fn = make_numpy_update_fn()
    rollout_fn = make_numpy_rollout_fn(update_fn, steps=8)
    metrics = evaluate_projection_error(rollout_fn, batch)
    assert metrics["mean_after"] < metrics["mean_before"]
    assert metrics["mean_improvement"] > 0.0
    assert metrics["success_rate"] > 0.0
