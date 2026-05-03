"""Differentiable artifact-suppression operators for VOD training.

This module is the *training-side* counterpart of `vod_minimal.artifacts`,
which is NumPy-only and used for batch construction and offline metrics.

Why a separate file
-------------------
- Training needs gradients flowing through the tile-residue calculation.
- The NumPy implementation closes over boolean indexing patterns and
  discrete sums that aren't differentiable as written.
- Audio (1-D) and text (1-D, quantized) projections do not carry a
  spatial grid; tile residue is undefined for them. This file is image /
  video only and the loss helpers explicitly skip everything else.

Design choices
--------------
- `torch_tile_residue` mirrors `artifacts.tile_residue`
  exactly on image-shape input, with one floating-point caveat: the
  NumPy version separately concatenates and then averages, so order of
  operations may differ at the eps level. Tests assert the two agree to
  rtol=1e-5.
- `artifact_regularization_loss` is one-sided: it only penalizes when
  the predicted residue *exceeds* the target's. This avoids pushing the
  model toward synthetic over-smoothness. The target's residue is
  detached so the gradient never leaks back through the dataset.
- `artifact_train_loss` is the helper trainers call. It iterates the
  same (sample, medium) pairs as `core.projection_loss` but only for
  `media=("image", "video")` by default.

Orthogonal Compression Noise (4/e) is a *generation-side* constraint of
the VOD model, not a post-processing knob. The training trajectories
need to see both the suppressed input distribution
(`build_projection_batch(..., artifact_suppression=True)`) and an
optional differentiable penalty on the predicted view, so the learned
update rule cannot rely on coherent block contours leaking back in.
"""

from __future__ import annotations

from typing import Callable, Iterable

import numpy as np
import torch

from .artifacts import D_OC


SPATIAL_MEDIA: tuple[str, ...] = ("image", "video")
EPS = 1e-9


def torch_tile_residue(values: torch.Tensor, *, tile: int = 8, eps: float = EPS) -> torch.Tensor:
    """Differentiable tile-boundary residue score for image / video tensors.

    Returns a 0-d tensor. For inputs whose spatial side is at most `tile`
    or whose dimensionality is below 2, returns a 0-d zero tensor (no
    detectable boundary structure to penalize).
    """

    if tile <= 1:
        raise ValueError(f"tile must be greater than 1, got {tile}")
    if values.ndim < 2:
        return values.new_zeros(())

    h, w = values.shape[-2], values.shape[-1]
    if min(h, w) <= tile:
        return values.new_zeros(())

    # Collapse leading dims so dy / dx are batched [B, H-1, W] / [B, H, W-1].
    spatial = values.reshape(-1, h, w)

    dy = (spatial[:, 1:, :] - spatial[:, :-1, :]).abs()
    dx = (spatial[:, :, 1:] - spatial[:, :, :-1]).abs()

    # Reduction over EVERY adjacent pair (denominator).
    all_sum = dy.sum() + dx.sum()
    all_count = dy.numel() + dx.numel()
    all_mean = all_sum / all_count

    y_idx = torch.arange(dy.shape[1], device=values.device)
    x_idx = torch.arange(dx.shape[2], device=values.device)
    y_mask = (y_idx + 1) % tile == 0
    x_mask = (x_idx + 1) % tile == 0

    boundary_sum = values.new_zeros(())
    boundary_count = 0
    if y_mask.any():
        bd = dy.index_select(1, y_mask.nonzero(as_tuple=False).squeeze(-1))
        boundary_sum = boundary_sum + bd.sum()
        boundary_count += bd.numel()
    if x_mask.any():
        bd = dx.index_select(2, x_mask.nonzero(as_tuple=False).squeeze(-1))
        boundary_sum = boundary_sum + bd.sum()
        boundary_count += bd.numel()

    if boundary_count == 0:
        return values.new_zeros(())

    boundary_mean = boundary_sum / boundary_count
    return boundary_mean / (all_mean + eps)


RESIDUE_FLOOR: float = 0.0
"""Conservative gating floor for `artifact_regularization_loss`.

Default is now **0.0** so the loss tracks `target_r` directly: if
`target_r < 1.0` (clean Chladni training target), the model is still
incentivised to drive pred down to that level. The previous default
of 1.0 silently suppressed the loss whenever the model was already
below the geometric neutral point — including stress-data scenarios
where target_r is naturally < 1.0 and the loss thus contributed zero
gradient regardless of weight (see VOD_agent_postmortem.md §12.17).

Set explicitly to 1.0 (or any positive value) to recover the legacy
"don't push below the geometric neutral point" behaviour as a safety
margin on data where target_r is itself uncertain.
"""


def artifact_regularization_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    tile: int = 8,
    residue_floor: float = RESIDUE_FLOOR,
    strength: float = 1.0,
) -> torch.Tensor:
    """One-sided gated penalty for tile-residue inflation, with a 4/e strength dial.

        gated_floor = max(residue(target).detach(), residue_floor)
        L_artifact  = strength * relu( residue(pred) - gated_floor )

    Three knobs:
      * `relu(...)`            — never reward smoother-than-target;
      * `residue_floor`        — minimum threshold pred residue is
                                 pushed toward. Default 0.0: track target
                                 directly. Set >0 to refuse to push pred
                                 below an absolute residue level
                                 (paranoid over-smoothing guard).
      * `strength`             — linear multiplier on the penalty so the
                                 4/e regularisation can be dialled
                                 continuously between methods that share
                                 the same overall loss budget. Default
                                 1.0 = backward-compatible.

    Target residue is detached so the data distribution never receives
    a gradient. Returns a 0-d tensor (zero on shapes with no spatial
    grid: audio waveforms, text channel strings).
    """

    if pred.ndim < 2 or pred.shape[-1] <= tile or pred.shape[-2] <= tile:
        return pred.new_zeros(())
    if strength < 0:
        raise ValueError(f"strength must be non-negative, got {strength}")
    pred_r = torch_tile_residue(pred, tile=tile)
    target_r = torch_tile_residue(target, tile=tile).detach()
    floor = pred.new_tensor(float(residue_floor))
    gated_floor = torch.maximum(target_r, floor)
    return float(strength) * torch.relu(pred_r - gated_floor)


# Type alias for the update_fn used by core.shared_update_rollout.
_UpdateFn = Callable[[torch.Tensor, torch.Tensor, str], torch.Tensor]


def artifact_train_loss(
    update_fn: _UpdateFn,
    batch,
    *,
    steps: int,
    device: torch.device,
    tile: int = 8,
    media: Iterable[str] = SPATIAL_MEDIA,
) -> torch.Tensor:
    """Average artifact penalty across spatial-media rollouts in `batch`.

    Mirrors the iteration shape of `core.projection_loss` but:
      - restricts to image / video only (the spatial media)
      - uses the differentiable tile-residue penalty above
      - returns a 0-d tensor on the correct device

    Empty batches and non-spatial-only batches return a 0-d zero tensor
    (so callers can safely add it into a sum).
    """

    # Imported here to keep core.py and this module mutually independent.
    from .core import shared_update_rollout

    media_tuple = tuple(m for m in media if m in SPATIAL_MEDIA)
    if not media_tuple:
        return torch.zeros((), device=device)

    losses: list[torch.Tensor] = []
    for sample in batch.samples:
        for medium in media_tuple:
            if medium not in sample.noisy_views:
                continue
            noisy_t = torch.from_numpy(sample.noisy_views[medium].astype(np.float32)).to(device)
            target_t = torch.from_numpy(sample.target_views[medium].astype(np.float32)).to(device)
            pred = shared_update_rollout(update_fn, noisy_t, target_t, medium, steps=steps)
            losses.append(artifact_regularization_loss(pred, target_t, tile=tile))

    if not losses:
        return torch.zeros((), device=device)
    return torch.stack(losses).mean()


__all__ = [
    "D_OC",
    "RESIDUE_FLOOR",
    "SPATIAL_MEDIA",
    "artifact_regularization_loss",
    "artifact_train_loss",
    "torch_tile_residue",
]
