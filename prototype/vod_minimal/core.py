"""Simplified mathematical core for the VOD minimal prototype.

This module exposes the four interfaces that correspond to the simplified
algorithm set in `D:\\VOD\\docs\\vod_math_simplification.md`:

    1. build_projection_batch       — Algorithm A (synthetic projection batch)
    2. shared_update_rollout        — Algorithm B (K shared update steps)
    3. projection_loss              — Algorithm C (projection-space training loss)
    4. evaluate_projection_error    — Algorithm D (before/after evaluation)

Design notes
------------
- The four interfaces are *backend-agnostic with respect to the updater*. The
  caller passes an `update_fn(current, target, medium) -> next_view` callable.
  This lets the core be reused unchanged by:
      * the NumPy `MinimalVOD`               (model.py)
      * the PyTorch `SharedPointUpdater`     (torch_model.py)
      * the `TinyVDiT` skeleton              (vdit.py)
- Only `numpy` and `torch` are needed. No new external dependencies.
- This file intentionally keeps the minimal batch / rollout contract small.
  Distinctive mechanisms live in focused modules:
      * TTNM toy diagnostics             metrics.py
      * Binary-Twin text/logo coupling   binary_twin.py
      * 4/e orthogonal compression       artifacts.py
      * TPSR/AIMP physical checks        aimp.py
  Keeping them out of the core avoids hiding mechanism-specific claims inside
  generic data plumbing.
- This file does NOT depend on, import, or call OPU. OPU is a reference module
  copied from APT-Transformer and is intentionally outside the minimal core.

The mathematical statement reduces to:

    P_m(U_target) ~= Update_theta(P_m(U_noisy), c_m)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

import numpy as np
import torch
import torch.nn.functional as F

from .experiment import Sample, make_sample
from .metrics import artifact_metrics, mean_target_error


MEDIA: tuple[str, ...] = ("image", "video", "audio", "text")


# --------------------------------------------------------------------------- #
#  Data containers
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ProjectionSample:
    """One synthetic Chladni source/target pair with media projections.

    Mirrors `experiment.Sample`; kept as a separate dataclass so that the
    core namespace is self-contained and can be re-exported without
    leaking the experiment module's name.

    `source_field` / `target_field` carry the 2-D Chladni image used by
    the legacy projection path. When the batch is built in spacetime
    mode, `source_spacetime_field` / `target_spacetime_field` carry the
    full U(t, y, x) volumes the views were sliced from. Legacy callers
    can ignore those two — they default to None and are never required
    by `evaluate_projection_error` or the trainers.
    """

    source_field: np.ndarray
    target_field: np.ndarray
    noisy_views: dict[str, np.ndarray]
    target_views: dict[str, np.ndarray]
    source_spacetime_field: np.ndarray | None = None
    target_spacetime_field: np.ndarray | None = None

    @classmethod
    def from_sample(cls, sample: Sample) -> "ProjectionSample":
        return cls(
            source_field=sample.source_field,
            target_field=sample.target_field,
            noisy_views=dict(sample.noisy_views),
            target_views=dict(sample.target_views),
        )


@dataclass(frozen=True)
class ProjectionBatch:
    """A small batch of `ProjectionSample` for training or evaluation."""

    samples: tuple[ProjectionSample, ...]
    media: tuple[str, ...] = MEDIA

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self):
        return iter(self.samples)


# --------------------------------------------------------------------------- #
#  1. Build a synthetic projection batch  (Algorithm A)
# --------------------------------------------------------------------------- #

def build_projection_batch(
    rng: np.random.Generator,
    batch_size: int,
    *,
    size: int = 64,
    noise_scale: float = 0.24,
    media: tuple[str, ...] = MEDIA,
    artifact_suppression: bool = False,
    artifact_scale: float | None = None,
    artifact_tile: int = 8,
    spacetime: bool = False,
    frames: int = 10,
    paired_denoising: bool = False,
) -> ProjectionBatch:
    """Sample a batch of (source, target, noisy_views, target_views) tuples.

    Parameters
    ----------
    rng:          numpy generator for reproducibility
    batch_size:   number of samples in the batch
    size:         spatial side-length of the underlying Chladni field
    noise_scale:  std of additive Gaussian noise on the source projections
    media:        which media projections to keep (defaults to all four)
    artifact_suppression / artifact_scale / artifact_tile:
                 4/e Orthogonal Compression Noise on noisy_views.
    spacetime:    when True, build a 3-D U(t, y, x) field per sample and
                  project the video medium directly from the volume.
                  Image / audio / text use the temporal mean of the
                  volume so they remain single-frame projections of the
                  same underlying field.
    frames:       number of time slices when `spacetime=True`. Ignored
                  in 2-D mode (legacy `project_video` always emits 10).
    """

    if batch_size < 0:
        raise ValueError(f"batch_size must be non-negative, got {batch_size}")

    if spacetime:
        # Build 3-D fields directly here. We don't route through
        # experiment.make_sample because Sample only carries 2-D fields,
        # and ProjectionSample now exposes source_spacetime_field /
        # target_spacetime_field for downstream temporal analysis.
        from .projections import add_noise as _add_noise, project_all as _project_all
        from .spacetime_chladni import random_chladni_spacetime_field

        media_set = set(media)
        samples: list[ProjectionSample] = []
        for _ in range(batch_size):
            if paired_denoising:
                # ONE field; noisy is the target plus additive noise.
                target_3d = random_chladni_spacetime_field(rng, size=size, frames=frames, n_modes=2)
                source_3d = target_3d  # same field; kept as field on the sample for downstream eval
                target_views = _project_all(target_3d, video_mode="3d", frames=frames)
                source_views = target_views  # noisy is built from target views
            else:
                source_3d = random_chladni_spacetime_field(rng, size=size, frames=frames, n_modes=3)
                target_3d = random_chladni_spacetime_field(rng, size=size, frames=frames, n_modes=2)
                source_views = _project_all(source_3d, video_mode="3d", frames=frames)
                target_views = _project_all(target_3d, video_mode="3d", frames=frames)
            noisy_views = {
                name: _add_noise(
                    view,
                    rng,
                    scale=noise_scale,
                    artifact_suppression=artifact_suppression,
                    artifact_scale=artifact_scale,
                    artifact_tile=artifact_tile,
                )
                for name, view in source_views.items()
            }
            if media_set != set(MEDIA):
                noisy_views = {m: noisy_views[m] for m in media if m in noisy_views}
                target_views = {m: target_views[m] for m in media if m in target_views}
            samples.append(
                ProjectionSample(
                    source_field=source_3d.mean(axis=0),
                    target_field=target_3d.mean(axis=0),
                    noisy_views=noisy_views,
                    target_views=target_views,
                    source_spacetime_field=source_3d,
                    target_spacetime_field=target_3d,
                )
            )
        return ProjectionBatch(samples=tuple(samples), media=tuple(media))

    media_set = set(media)
    samples: list[ProjectionSample] = []
    for _ in range(batch_size):
        sample = make_sample(
            rng,
            size=size,
            noise_scale=noise_scale,
            artifact_suppression=artifact_suppression,
            artifact_scale=artifact_scale,
            artifact_tile=artifact_tile,
            paired_denoising=paired_denoising,
        )
        if media_set != set(MEDIA):
            noisy_views = {m: sample.noisy_views[m] for m in media if m in sample.noisy_views}
            target_views = {m: sample.target_views[m] for m in media if m in sample.target_views}
            samples.append(
                ProjectionSample(
                    source_field=sample.source_field,
                    target_field=sample.target_field,
                    noisy_views=noisy_views,
                    target_views=target_views,
                )
            )
        else:
            samples.append(ProjectionSample.from_sample(sample))

    return ProjectionBatch(samples=tuple(samples), media=tuple(media))


# --------------------------------------------------------------------------- #
#  2. Shared update rollout  (Algorithm B)
# --------------------------------------------------------------------------- #

UpdateFn = Callable[[Any, Any, str], Any]
"""(current_view, target_view, medium_name) -> next_view.

`current` and `target` are either both `np.ndarray` or both `torch.Tensor`.
The returned value matches their type and shape.
"""


def shared_update_rollout(
    update_fn: UpdateFn,
    noisy: Any,
    target: Any,
    medium: str,
    *,
    steps: int,
    return_path: bool = False,
) -> Any:
    """Apply the shared one-step updater K times.

    This is the engineering form of Algorithm B:

        X_{k+1, m} = update_fn(X_{k, m}, target_m, m)

    Returns the final view by default, or the full path (length steps + 1) when
    `return_path=True`.
    """

    if steps < 0:
        raise ValueError(f"steps must be non-negative, got {steps}")

    current = noisy
    if return_path:
        path = [current]
        for _ in range(steps):
            current = update_fn(current, target, medium)
            path.append(current)
        return path

    for _ in range(steps):
        current = update_fn(current, target, medium)
    return current


# --------------------------------------------------------------------------- #
#  3. Projection-space training loss  (Algorithm C)
# --------------------------------------------------------------------------- #

def projection_loss(
    update_fn: UpdateFn,
    batch: ProjectionBatch,
    *,
    steps: int,
    device: torch.device,
    media: tuple[str, ...] | None = None,
    normalize: bool = True,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Compute the projection-space training loss across the batch.

    Loss form:

        L = mean_{sample, m} MSE( rollout(noisy_m, target_m, m, K),  target_m )

    With `normalize=True` (default) each per-sample MSE is divided by the
    detached `target.pow(2).mean()` to keep media on comparable scales.

    `update_fn` MUST consume torch tensors and return torch tensors. The
    `device` argument controls where samples are placed.
    """

    selected = media if media is not None else batch.media
    if not selected:
        raise ValueError("media tuple must be non-empty")

    losses: list[torch.Tensor] = []
    for sample in batch.samples:
        for medium in selected:
            if medium not in sample.noisy_views:
                continue
            noisy_t = torch.from_numpy(sample.noisy_views[medium].astype(np.float32)).to(device)
            target_t = torch.from_numpy(sample.target_views[medium].astype(np.float32)).to(device)
            pred = shared_update_rollout(update_fn, noisy_t, target_t, medium, steps=steps)
            if normalize:
                denom = target_t.pow(2).mean().detach().clamp_min(eps)
                losses.append(F.mse_loss(pred, target_t) / denom)
            else:
                losses.append(F.mse_loss(pred, target_t))

    if not losses:
        # Defensive: no media matched. Return a zero tensor on the right device
        # so backward() is still legal.
        return torch.zeros((), device=device, requires_grad=False)
    return torch.stack(losses).mean()


# --------------------------------------------------------------------------- #
#  4. Evaluation  (Algorithm D)
# --------------------------------------------------------------------------- #

RolloutFn = Callable[[dict[str, np.ndarray], dict[str, np.ndarray]], dict[str, np.ndarray]]
"""(noisy_views, target_views) -> denoised_views.

A NumPy-side wrapper around the chosen updater. The wrapper is responsible for
moving tensors to/from device, calling `shared_update_rollout` for each medium,
and returning the final views as numpy arrays.
"""


def evaluate_projection_error(
    rollout_fn: RolloutFn,
    batch: ProjectionBatch,
    *,
    include_artifact_metrics: bool = False,
    artifact_tile: int = 8,
) -> dict[str, float]:
    """Standard before/after evaluation on a batch of `ProjectionSample`.

    Default returns four keys only:
        mean_before, mean_after, mean_improvement, success_rate

    `mean_target_error` is computed across ALL media (image, video, audio,
    text). Projection-space error is meaningful end-to-end on every
    medium, so the base block stays unchanged.

    When `include_artifact_metrics=True`, additional keys are appended.
    The artifact block is restricted to *spatial* media (image / video):
    tile residue is geometrically defined on the 2-D grid, so mixing
    audio waveforms or text channels into the main score would silently
    dilute a real spatial failure with unrelated 1-D noise. Audio / text
    are reported separately under `non_spatial_artifact_*` so they stay
    visible without polluting the main score:

        artifact_before_mean_tile_residue              spatial only
        artifact_after_mean_tile_residue               spatial only
        artifact_after_score                           spatial only
        artifact_improvement                           spatial only
        non_spatial_artifact_before_mean_tile_residue  audio/text
        non_spatial_artifact_after_mean_tile_residue   audio/text

    Artifact metrics are evaluation-only diagnostics. They never enter
    any training loss in this prototype.
    """

    if len(batch) == 0:
        nan = float("nan")
        empty = {
            "mean_before": nan,
            "mean_after": nan,
            "mean_improvement": nan,
            "success_rate": nan,
        }
        if include_artifact_metrics:
            empty.update(
                {
                    "artifact_before_mean_tile_residue": nan,
                    "artifact_after_mean_tile_residue": nan,
                    "artifact_after_score": nan,
                    "artifact_improvement": nan,
                    "non_spatial_artifact_before_mean_tile_residue": nan,
                    "non_spatial_artifact_after_mean_tile_residue": nan,
                }
            )
        return empty

    before: list[float] = []
    after: list[float] = []
    artifact_before_residue: list[float] = []
    artifact_after_residue: list[float] = []
    artifact_after_score_acc: list[float] = []
    non_spatial_before_residue: list[float] = []
    non_spatial_after_residue: list[float] = []
    for sample in batch.samples:
        denoised = rollout_fn(sample.noisy_views, sample.target_views)
        before.append(mean_target_error(sample.noisy_views, sample.target_views))
        after.append(mean_target_error(denoised, sample.target_views))
        if include_artifact_metrics:
            before_a = artifact_metrics(sample.noisy_views, tile=artifact_tile)
            after_a = artifact_metrics(denoised, tile=artifact_tile)
            artifact_before_residue.append(before_a["mean_tile_residue"])
            artifact_after_residue.append(after_a["mean_tile_residue"])
            artifact_after_score_acc.append(after_a["artifact_score"])
            non_spatial_before_residue.append(before_a["non_spatial_mean_tile_residue"])
            non_spatial_after_residue.append(after_a["non_spatial_mean_tile_residue"])

    before_arr = np.asarray(before, dtype=np.float64)
    after_arr = np.asarray(after, dtype=np.float64)
    result: dict[str, float] = {
        "mean_before": float(before_arr.mean()),
        "mean_after": float(after_arr.mean()),
        "mean_improvement": float((before_arr - after_arr).mean()),
        "success_rate": float(np.mean(after_arr < before_arr)),
    }
    if include_artifact_metrics:
        before_res = np.asarray(artifact_before_residue, dtype=np.float64)
        after_res = np.asarray(artifact_after_residue, dtype=np.float64)
        score = np.asarray(artifact_after_score_acc, dtype=np.float64)
        ns_before = np.asarray(non_spatial_before_residue, dtype=np.float64)
        ns_after = np.asarray(non_spatial_after_residue, dtype=np.float64)
        # nanmean lets a per-sample NaN (no spatial / no non-spatial
        # media) propagate gracefully instead of poisoning the aggregate.
        result.update(
            {
                "artifact_before_mean_tile_residue": float(np.nanmean(before_res)),
                "artifact_after_mean_tile_residue": float(np.nanmean(after_res)),
                "artifact_after_score": float(np.nanmean(score)),
                "artifact_improvement": float(np.nanmean(before_res - after_res)),
                "non_spatial_artifact_before_mean_tile_residue": float(np.nanmean(ns_before)),
                "non_spatial_artifact_after_mean_tile_residue": float(np.nanmean(ns_after)),
            }
        )
    return result


# --------------------------------------------------------------------------- #
#  Convenience: a NumPy update_fn matching the analytic MinimalVOD step.
#  Used by run_core_validation.py to demonstrate the four interfaces with no
#  trained parameters. The trained PyTorch and VDiT updaters provide their own
#  update_fn closures and reuse the same four core interfaces.
# --------------------------------------------------------------------------- #

def make_numpy_update_fn(
    *,
    diffusion: float = 0.55,
    reaction: float = 0.18,
    step_size: float = 0.9,
) -> UpdateFn:
    """Build a NumPy-only `update_fn` based on the analytic Minimal VOD step.

    This is the same one-step rule used inside `model.MinimalVOD.update_path`,
    extracted here so the core's interface contract is demonstrable without
    importing the model class.
    """
    from .projections import smooth as _smooth

    def _step(current: np.ndarray, target: np.ndarray, medium: str) -> np.ndarray:
        del medium  # the analytic step is media-agnostic by design
        cur = current.astype(np.float64, copy=False)
        tgt = target.astype(np.float64, copy=False)
        diffusivity = 0.15 + 0.85 * np.abs(tgt) / (np.max(np.abs(tgt)) + 1e-9)
        diffusion_term = diffusivity * (_smooth(cur) - cur)
        reaction_term = tgt - cur
        delta = diffusion * diffusion_term + reaction * reaction_term
        return cur + step_size * delta

    return _step


def make_numpy_rollout_fn(update_fn: UpdateFn, *, steps: int, media: Iterable[str] = MEDIA) -> RolloutFn:
    """Wrap a NumPy `update_fn` into a per-batch `RolloutFn`."""

    media_tuple = tuple(media)

    def _rollout(noisy_views: dict[str, np.ndarray], target_views: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for medium in media_tuple:
            if medium not in noisy_views:
                continue
            out[medium] = shared_update_rollout(
                update_fn,
                noisy_views[medium],
                target_views[medium],
                medium,
                steps=steps,
            )
        return out

    return _rollout
