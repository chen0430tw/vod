"""Minimal prototype experiment runner."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np

from .chladni import random_chladni_field
from .metrics import mean_target_error, modular_shrinking_number
from .model import MinimalVOD
from .projections import add_noise, project_all


@dataclass(frozen=True)
class Sample:
    source_field: np.ndarray
    target_field: np.ndarray
    noisy_views: dict[str, np.ndarray]
    target_views: dict[str, np.ndarray]


def make_sample(
    rng: np.random.Generator,
    size: int = 64,
    noise_scale: float = 0.24,
    *,
    artifact_suppression: bool = False,
    artifact_scale: float | None = None,
    artifact_tile: int = 8,
    paired_denoising: bool = False,
) -> Sample:
    """
    Build one (source, target, noisy_views, target_views) sample.

    Two protocols:

      paired_denoising=False  (default, legacy):
        source and target are TWO INDEPENDENT random Chladni fields.
        noisy_views = projections(source) + noise.
        target_views = projections(target).
        Information-theoretically `noisy ⟂ target`, so any model trained
        with per-sample MSE saturates at zero baseline (E[target | noisy]
        = E[target] ≈ 0). Useful as a conditional-generation toy IF a
        non-leaking condition channel is also wired in; otherwise the
        task is unsolvable and `model_error == zero_baseline_error` is
        the protocol upper bound.

      paired_denoising=True:
        ONE underlying field U (sampled with target's mode count to keep
        eval reference consistent). target_views = projections(U).
        noisy_views = target_views + noise. This is classic supervised
        denoising — noisy and target share `U`, so the model has a
        deterministic information path to learn from.
    """
    if paired_denoising:
        # One field, used both as the target and as the corruption source.
        # n_modes matches the legacy `target` so that target var is
        # comparable to the legacy zero-baseline reading.
        ground_truth = random_chladni_field(rng, size=size, n_modes=2)
        target_views = project_all(ground_truth)
        noisy_views = {
            name: add_noise(
                view,
                rng,
                scale=noise_scale,
                artifact_suppression=artifact_suppression,
                artifact_scale=artifact_scale,
                artifact_tile=artifact_tile,
            )
            for name, view in target_views.items()
        }
        # source_field is set to ground_truth so downstream code that
        # inspects `sample.source_field` still gets a coherent array.
        return Sample(ground_truth, ground_truth, noisy_views, target_views)

    source = random_chladni_field(rng, size=size, n_modes=3)
    target = random_chladni_field(rng, size=size, n_modes=2)
    source_views = project_all(source)
    target_views = project_all(target)
    noisy_views = {
        name: add_noise(
            view,
            rng,
            scale=noise_scale,
            artifact_suppression=artifact_suppression,
            artifact_scale=artifact_scale,
            artifact_tile=artifact_tile,
        )
        for name, view in source_views.items()
    }
    return Sample(source, target, noisy_views, target_views)


def evaluate_model(model: MinimalVOD, samples: list[Sample]) -> dict[str, float]:
    before = []
    after = []
    msn_values = []

    for sample in samples:
        denoised, paths = model.denoise_views(sample.noisy_views, sample.target_views)
        before.append(mean_target_error(sample.noisy_views, sample.target_views))
        after.append(mean_target_error(denoised, sample.target_views))
        msn_values.extend(modular_shrinking_number(path) for path in paths.values())

    before_arr = np.array(before, dtype=np.float64)
    after_arr = np.array(after, dtype=np.float64)
    return {
        "mean_before": float(before_arr.mean()),
        "mean_after": float(after_arr.mean()),
        "mean_improvement": float((before_arr - after_arr).mean()),
        "success_rate": float(np.mean(after_arr < before_arr)),
        "mean_msn": float(np.mean(msn_values)),
    }


def grid_search(train_samples: list[Sample]) -> MinimalVOD:
    best_model = None
    best_score = float("inf")

    for diffusion, reaction, step_size in product(
        (0.25, 0.45, 0.65, 0.85),
        (0.08, 0.14, 0.22, 0.32),
        (0.55, 0.75, 0.95),
    ):
        model = MinimalVOD(diffusion=diffusion, reaction=reaction, step_size=step_size, steps=12)
        metrics = evaluate_model(model, train_samples)
        score = metrics["mean_after"]
        if score < best_score:
            best_score = score
            best_model = model

    assert best_model is not None
    return best_model


def run_experiment(seed: int = 430, train_n: int = 32, test_n: int = 32) -> tuple[MinimalVOD, dict[str, float], dict[str, float]]:
    rng = np.random.default_rng(seed)
    train = [make_sample(rng) for _ in range(train_n)]
    test = [make_sample(rng) for _ in range(test_n)]
    model = grid_search(train)
    return model, evaluate_model(model, train), evaluate_model(model, test)
