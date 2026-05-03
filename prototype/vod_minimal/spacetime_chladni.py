"""3-D synthetic Chladni-like field over (t, y, x).

This is the spatiotemporal upgrade of `chladni.py`. The 2-D field
operates on a single image; the 3-D field operates on the canonical VOD
view of a generation: one shared latent U(t, y, x) that produces every
medium projection by slicing or aggregating the same volume.

Field equation
--------------
For temporal mode m_t with phase phi, spatial modes (m_x, m_y), weight w:

    U(t, y, x) = Σ_i w_i
                 * cos(2π m_t_i τ + phi_i)
                 * ( cos(π m_x_i x) cos(π m_y_i y)
                   - cos(π m_y_i x) cos(π m_x_i y) )

where x, y, τ are normalised to [0, 1]. The 2-D Chladni structure of
`chladni.py` is the τ = 0 cross-section of this volume; the temporal
factor introduces coherent low-frequency oscillation across frames so
the slices stay related (the alternative — independent frames per
timestep — would give white-noise videos).

Boundaries are deliberately simple (Dirichlet-like via the cos-cos
symmetry, same as the 2-D version). Real boundary maps come later.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


EPS = 1e-9


@dataclass(frozen=True)
class SpacetimeBoundary:
    """Boundary spec for U(t, y, x).

    `modes` is a tuple of (mode_x, mode_y, mode_t, weight, phase).
    """

    size: int = 64
    frames: int = 10
    modes: tuple[tuple[int, int, int, float, float], ...] = (
        (2, 3, 1, 0.7, 0.0),
        (5, 4, 2, 0.45, 0.6),
        (7, 2, 1, 0.25, 1.4),
    )


def chladni_spacetime_field(boundary: SpacetimeBoundary) -> np.ndarray:
    """Materialise U(t, y, x) for the given boundary.

    Returns a numpy array of shape (frames, size, size) with values
    normalised so max(|U|) = 1.
    """

    size = boundary.size
    frames = boundary.frames
    if size <= 1:
        raise ValueError(f"size must be > 1, got {size}")
    if frames <= 0:
        raise ValueError(f"frames must be > 0, got {frames}")

    t_axis = np.linspace(0.0, 1.0, frames, endpoint=False)
    y_grid, x_grid = np.mgrid[0:size, 0:size]
    x = x_grid / max(1, size - 1)
    y = y_grid / max(1, size - 1)

    field = np.zeros((frames, size, size), dtype=np.float64)
    for mx, my, mt, weight, phase in boundary.modes:
        spatial = (
            np.cos(np.pi * mx * x) * np.cos(np.pi * my * y)
            - np.cos(np.pi * my * x) * np.cos(np.pi * mx * y)
        )
        temporal = np.cos(2.0 * np.pi * mt * t_axis + phase)
        # broadcast (F,1,1) * (1,H,W) → (F,H,W)
        field += weight * temporal[:, None, None] * spatial[None, :, :]

    peak = float(np.max(np.abs(field)))
    return field / (peak + EPS)


def random_spacetime_boundary(
    rng: np.random.Generator,
    *,
    size: int = 64,
    frames: int = 10,
    n_modes: int = 3,
) -> SpacetimeBoundary:
    modes = []
    for _ in range(n_modes):
        mx = int(rng.integers(2, 9))
        my = int(rng.integers(2, 9))
        # Temporal modes intentionally low-frequency: 1–3 cycles across the
        # clip avoids per-frame chaos. This is what makes the resulting
        # video look coherent rather than flickery.
        mt = int(rng.integers(1, 4))
        weight = float(rng.uniform(0.15, 0.9))
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        modes.append((mx, my, mt, weight, phase))
    return SpacetimeBoundary(size=size, frames=frames, modes=tuple(modes))


def random_chladni_spacetime_field(
    rng: np.random.Generator,
    *,
    size: int = 64,
    frames: int = 10,
    n_modes: int = 3,
) -> np.ndarray:
    return chladni_spacetime_field(
        random_spacetime_boundary(rng, size=size, frames=frames, n_modes=n_modes)
    )


__all__ = [
    "SpacetimeBoundary",
    "chladni_spacetime_field",
    "random_chladni_spacetime_field",
    "random_spacetime_boundary",
]
