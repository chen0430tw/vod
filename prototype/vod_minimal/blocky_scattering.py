"""Synthetic blocky-scattering injection — a stress / diagnostic dataset.

This module is NOT part of the default training distribution. It exists
only to verify that the artifact-suppression stack
(`tile_residue`, `oc_four_over_e`, the differentiable
`artifact_regularization_loss`) actually responds to the failure mode it
was designed against: AI-renderer-style tile light spots and dirty block
contours.

What it produces
----------------
A multiplicative scatter pattern aligned to the GPU tile grid:

    1. Per-tile intensity:    a uniform random scalar per `tile × tile`
                              block (some blocks bright, some dim — like
                              uneven shader scheduling residue).
    2. Boundary proximity:    weight that peaks at the tile edges and
                              decays exponentially toward the interior.
                              This is where coherent block contours live.
    3. Boundary noise:        Gaussian noise scaled by boundary proximity
                              (the "tile light spot" component).
    4. Interior particle:     a much smaller Gaussian inside each tile,
                              shaped by `1 - boundary_proximity`.

The combined mask is multiplied by `strength` and added to the input view.
Higher rank arrays (e.g. video `(F, H, W)`) share the same 2-D mask along
all leading axes — frame-varying scatter is intentionally not modelled
here; that adds noise without changing the spatial diagnostic.

For 1-D inputs (audio waveforms, quantised text channels) tile residue is
not defined, so injection is a no-op rather than an error. Trainers that
loop over media never hit a special case.

Why "stress" and not "training"
-------------------------------
The 4/e Orthogonal Compression Decay is a *generation-side* constraint of
VOD. Stress data lets us measure whether the constraint actually fires
when there is a real coherent boundary to break. Once a real renderer
(SDXL, VDiT, GPU tile shader) feeds the prototype, the same metrics will
calibrate against its outputs — but stress data itself should never be
treated as a target style for the model.
"""

from __future__ import annotations

import numpy as np


SPATIAL_MEDIA: tuple[str, ...] = ("image", "video")
EPS = 1e-9


def _ensure_rng(rng: np.random.Generator | None) -> np.random.Generator:
    return rng if rng is not None else np.random.default_rng()


def blocky_scattering_mask(
    shape: tuple[int, ...],
    *,
    tile: int = 8,
    strength: float = 1.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate a tile-aligned scatter mask.

    Returns a `np.ndarray` of dtype float64 with the requested shape.

    Design:
      - For 1-D shapes the mask is all zeros (tile residue is undefined
        on a single-axis sequence).
      - For 2-D and higher shapes the mask is computed once on the last
        two axes and broadcast over all leading axes. The leading axes
        therefore share the same coherent grid pattern, which is exactly
        what we want to detect.

    `strength` scales the entire mask linearly; `tile` controls the grid
    period; `rng` controls reproducibility.
    """

    if tile <= 1:
        raise ValueError(f"tile must be greater than 1, got {tile}")
    if strength < 0:
        raise ValueError(f"strength must be non-negative, got {strength}")
    if len(shape) < 2:
        return np.zeros(shape, dtype=np.float64)

    rng = _ensure_rng(rng)
    h, w = shape[-2], shape[-1]
    if min(h, w) < tile:
        # Too small to carry a real tile pattern; return zero mask
        # (callers can still .add() this safely).
        return np.zeros(shape, dtype=np.float64)

    # Per-tile random brightness, replicated to pixel grid.
    nty = (h + tile - 1) // tile
    ntx = (w + tile - 1) // tile
    block_intensity = rng.uniform(0.3, 1.0, size=(nty, ntx))
    block_full = np.repeat(np.repeat(block_intensity, tile, axis=0), tile, axis=1)[:h, :w]

    # Distance to nearest tile boundary (0 at boundary, grows inward).
    y = np.arange(h)[:, None]
    x = np.arange(w)[None, :]
    y_dist = np.minimum(y % tile, (tile - 1) - (y % tile))
    x_dist = np.minimum(x % tile, (tile - 1) - (x % tile))
    edge_distance = np.minimum(y_dist, x_dist).astype(np.float64)
    boundary_proximity = np.exp(-edge_distance)  # 1.0 at boundary, ~0.37 one in, ~0.14 two in

    boundary_noise = rng.standard_normal((h, w)) * boundary_proximity
    interior_noise = rng.standard_normal((h, w)) * 0.3 * (1.0 - boundary_proximity)

    mask_2d = block_full * (boundary_noise + interior_noise) * strength

    if len(shape) == 2:
        return mask_2d
    # Broadcast 2-D mask across leading axes (frames / batch / etc.)
    out = np.broadcast_to(mask_2d, shape).copy()
    return out


def inject_blocky_scattering(
    view: np.ndarray,
    rng: np.random.Generator | None = None,
    *,
    tile: int = 8,
    strength: float = 0.2,
) -> np.ndarray:
    """Add a tile-aligned scatter mask to `view`.

    Returns float64. For 1-D inputs the function is a copy + cast no-op.
    """
    arr = view.astype(np.float64, copy=True)
    if arr.ndim < 2:
        return arr
    mask = blocky_scattering_mask(arr.shape, tile=tile, strength=strength, rng=rng)
    return arr + mask


def axial_contour_mask(
    shape: tuple[int, ...],
    *,
    tile: int = 8,
    strength: float = 1.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Tile-aligned contour stress whose dominant structure lies along the
    four canonical projection axes (vertical / horizontal / 2 diagonals).

    Spec-aligned counterpart of `blocky_scattering_mask`. While
    blocky_scattering_mask uses random hash-textured per-tile intensity,
    this mask injects coherent contour PATTERNS on tile boundaries
    along the four axes 4/e was designed to break:

      - vertical:    sharp jumps along columns at j ≡ -1 (mod q)
      - horizontal:  sharp jumps along rows at i ≡ -1 (mod q)
      - diagonals:   coherent jumps along i+j and i-j tile boundary lines

    Each axis gets its own random sign / amplitude per boundary line, so
    the mask is not deterministic — it remains a random stress field —
    but its dominant structure is genuinely 4-axis-aligned. This lets
    Claim 1 ablation actually exercise the failure mode 4/e claims to
    target, instead of generic random scatter.

    1-D inputs return zero (no spatial grid).
    """

    if tile <= 1:
        raise ValueError(f"tile must be greater than 1, got {tile}")
    if strength < 0:
        raise ValueError(f"strength must be non-negative, got {strength}")
    if len(shape) < 2:
        return np.zeros(shape, dtype=np.float64)

    rng = _ensure_rng(rng)
    h, w = shape[-2], shape[-1]
    if min(h, w) < tile:
        return np.zeros(shape, dtype=np.float64)

    mask = np.zeros((h, w), dtype=np.float64)

    # axis 1: vertical jumps (per column at i ≡ -1 mod q boundaries)
    n_h_lines = h // tile
    for line_idx in range(n_h_lines):
        i = line_idx * tile + (tile - 1)
        if i + 1 >= h:
            continue
        amp = rng.normal(0.0, strength)
        mask[i, :] += amp
        mask[i + 1, :] -= amp

    # axis 2: horizontal jumps (per row at j ≡ -1 mod q boundaries)
    n_v_lines = w // tile
    for line_idx in range(n_v_lines):
        j = line_idx * tile + (tile - 1)
        if j + 1 >= w:
            continue
        amp = rng.normal(0.0, strength)
        mask[:, j] += amp
        mask[:, j + 1] -= amp

    # axis 3: primary-diagonal jumps (along i+j ≡ -1 mod q lines)
    i_idx = np.arange(h)[:, None]
    j_idx = np.arange(w)[None, :]
    d1 = i_idx + j_idx
    diag1_amps = rng.normal(0.0, strength * 0.5, size=(h + w - 1,))
    # decay weights to keep mask bounded (only some diagonals selected)
    diag1_mod = ((d1 + 1) % tile == 0).astype(np.float64)
    mask += diag1_amps[d1] * diag1_mod

    # axis 4: secondary-diagonal jumps (along i-j ≡ -1 mod q lines)
    d2 = i_idx - j_idx + (w - 1)
    diag2_amps = rng.normal(0.0, strength * 0.5, size=(h + w - 1,))
    diag2_mod = ((d2 + 1) % tile == 0).astype(np.float64)
    mask += diag2_amps[d2] * diag2_mod

    if len(shape) == 2:
        return mask
    return np.broadcast_to(mask, shape).copy()


def inject_axial_contour(
    view: np.ndarray,
    rng: np.random.Generator | None = None,
    *,
    tile: int = 8,
    strength: float = 0.5,
) -> np.ndarray:
    """Add an axial-contour mask to `view`. 1-D no-op."""
    arr = view.astype(np.float64, copy=True)
    if arr.ndim < 2:
        return arr
    return arr + axial_contour_mask(arr.shape, tile=tile, strength=strength, rng=rng)


def inject_temporal_flicker(
    video: np.ndarray,
    rng: np.random.Generator | None = None,
    *,
    strength: float = 0.3,
) -> np.ndarray:
    """Add per-frame independent Gaussian noise to a (F, H, W) clip.

    Models the failure mode where each frame is denoised in isolation
    by an image-only generator: the spatial content is plausible but
    `temporal_smoothness` collapses because the noise field changes
    every frame. Acts on 3-D inputs only; 2-D / 1-D inputs are returned
    unchanged.
    """
    arr = np.asarray(video, dtype=np.float64)
    if arr.ndim != 3:
        return arr.copy()
    rng = _ensure_rng(rng)
    return arr + rng.standard_normal(arr.shape) * float(strength)


def inject_text_quantization_corruption(
    text_view: np.ndarray,
    rng: np.random.Generator | None = None,
    *,
    swap_rate: float = 0.3,
) -> np.ndarray:
    """Randomly perturb a fraction of text quantization levels.

    `project_text` produces values in [0, 1] quantised to 16 levels.
    This corruption picks `swap_rate` of those positions and replaces
    them with uniform [0, 1] noise — modelling the failure mode where
    a text channel hallucinates plausible-looking but semantically
    wrong characters.
    """
    rng = _ensure_rng(rng)
    arr = np.asarray(text_view, dtype=np.float64).copy()
    if arr.size == 0 or swap_rate <= 0.0:
        return arr
    swap_count = max(1, int(arr.size * swap_rate))
    indices = rng.choice(arr.size, swap_count, replace=False)
    arr.flat[indices] = rng.uniform(0.0, 1.0, swap_count)
    return arr


def inject_temporal_blocky_drift(
    video: np.ndarray,
    rng: np.random.Generator | None = None,
    *,
    tile: int = 8,
    strength: float = 0.2,
    drift: int = 1,
) -> np.ndarray:
    """Tile-aligned scatter that *shifts* between frames.

    Each frame receives its own blocky scatter mask and the mask is
    rolled by `drift` pixels per frame. This is the worst real-world
    case: visible block contours that wander frame to frame, so both
    `tile_residue` AND `temporal_artifact_drift` should fire.
    """
    arr = np.asarray(video, dtype=np.float64)
    if arr.ndim != 3:
        return arr.copy()
    rng = _ensure_rng(rng)

    out = arr.copy()
    base_mask = blocky_scattering_mask(arr.shape[1:], tile=tile, strength=strength, rng=rng)
    for t in range(arr.shape[0]):
        # Rolling the mask gives the contour a per-frame offset, which
        # is exactly what an unstable tile shader would produce.
        out[t] = arr[t] + np.roll(base_mask, shift=drift * t, axis=1)
    return out


def build_blocky_scattering_batch(
    rng: np.random.Generator,
    batch_size: int,
    *,
    size: int = 64,
    noise_scale: float = 0.24,
    artifact_strength: float = 0.2,
    tile: int = 8,
    media: tuple[str, ...] | None = None,
    spacetime: bool = False,
    frames: int = 10,
    temporal_mode: str = "static",
    flicker_strength: float = 0.3,
    drift: int = 1,
    paired_denoising: bool = False,
):
    """Build a `ProjectionBatch` whose spatial noisy_views carry blocky scatter.

    The base batch is built via `core.build_projection_batch`; spatial
    noisy_views (image / video) then receive a tile-aligned scatter
    mask. Audio / text are left untouched — tile residue is not defined
    for them.

    `temporal_mode` selects the kind of corruption applied to the video
    medium when `spacetime=True`:

        "static"       — same blocky_scattering_mask on every frame
                         (default; matches the 2-D mode behaviour)
        "flicker"      — per-frame independent noise overlaid on the
                         static blocky mask, breaks temporal_smoothness
        "blocky_drift" — blocky mask rolls between frames, breaks both
                         tile_residue and temporal_artifact_drift

    `temporal_mode` is only honoured when `spacetime=True`.
    """

    # Local import to keep module load fast and avoid circular imports.
    from .core import MEDIA, ProjectionBatch, ProjectionSample, build_projection_batch

    if media is None:
        media = MEDIA
    if temporal_mode not in {"static", "flicker", "blocky_drift"}:
        raise ValueError(
            f"temporal_mode must be 'static' / 'flicker' / 'blocky_drift', got {temporal_mode!r}"
        )

    base = build_projection_batch(
        rng,
        batch_size=batch_size,
        size=size,
        noise_scale=noise_scale,
        media=media,
        spacetime=spacetime,
        frames=frames,
        paired_denoising=paired_denoising,
    )

    spatial_targets = tuple(m for m in SPATIAL_MEDIA if m in media)
    new_samples = []
    for sample in base.samples:
        new_noisy = dict(sample.noisy_views)
        for medium in spatial_targets:
            if medium not in new_noisy:
                continue
            view = new_noisy[medium]
            if medium == "video" and spacetime and view.ndim == 3:
                if temporal_mode == "flicker":
                    view = inject_blocky_scattering(view, rng, tile=tile, strength=artifact_strength)
                    view = inject_temporal_flicker(view, rng, strength=flicker_strength)
                elif temporal_mode == "blocky_drift":
                    view = inject_temporal_blocky_drift(
                        view, rng, tile=tile, strength=artifact_strength, drift=drift
                    )
                else:  # "static"
                    view = inject_blocky_scattering(view, rng, tile=tile, strength=artifact_strength)
            else:
                view = inject_blocky_scattering(view, rng, tile=tile, strength=artifact_strength)
            new_noisy[medium] = view
        new_samples.append(
            ProjectionSample(
                source_field=sample.source_field,
                target_field=sample.target_field,
                noisy_views=new_noisy,
                target_views=dict(sample.target_views),
                source_spacetime_field=sample.source_spacetime_field,
                target_spacetime_field=sample.target_spacetime_field,
            )
        )
    return ProjectionBatch(samples=tuple(new_samples), media=base.media)


__all__ = [
    "SPATIAL_MEDIA",
    "axial_contour_mask",
    "blocky_scattering_mask",
    "build_blocky_scattering_batch",
    "inject_axial_contour",
    "inject_blocky_scattering",
    "inject_temporal_blocky_drift",
    "inject_temporal_flicker",
    "inject_text_quantization_corruption",
]
