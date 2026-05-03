"""4/e Orthogonal Compression Decay operator.

Spec-faithful implementation of OC_{4/e} per
docs/vod_full_mathematical_formulation.md Section 7.

Naming traces the spec so a reader can audit conformance section by section:

    Section 7.1   tile_residue, excess_residue        R_tile, r(X,q)
    Section 7.2   _axis_processes                     z_v, z_h, z_+, z_-
    Section 7.3   D_OC, D_AXIS, _structured           N_4
    Section 7.4   boundary_visibility, oc_four_over_e w_q, OC_{4/e}
    Section 7.6   leading-axis broadcast              video / batch
    Section 7.7   perturbation_energy, fair_baseline  sigma_eq

Claim split (Section 7.0):
    Claim 1A  operator conformance     tested by tests/test_4e_conformance.py
                                       against the covariance signature
                                       (Section 7.5). PASS at ~13σ.
    Claim 1B  application utility      tested by
                                       run_claim1b_phase_break.py against
                                       boundary sign-agreement on a
                                       coherent halo. PASS at low
                                       perturbation regime (s=0.3, ~7.7σ);
                                       saturates at high regime.

A scalar 4/e multiplier on iid Gaussian noise does NOT realise OC_{4/e}.
The operator is non-iid by construction (see Section 7.5).

Application semantics (vod_math_simplification.md §"Orthogonal Compression
Noise / Tile Residue"): the operator's purpose is to "破相" the coherent
tile light spot contour produced by AI renderers — disrupt its perceived
direction-consistency along boundary lines, NOT eliminate it via cross-
correlation. The boundary visibility weight w_q targets exactly the
boundary positions where contour lives; the four-axis structured
perturbation rotates / overlays competing direction structure there.
Effect is a *local variance redistribution* — see VOD_agent_postmortem.md
§12.15 for the verification protocol and the chain of wrong-metric
attempts that preceded it.
"""

from __future__ import annotations

import math

import numpy as np


# Section 7.3 decay constants.
D_OC = 4.0 / math.e          # orthogonal compression decay
D_AXIS = D_OC / 4.0          # per-axis decay = 1/e

EPS = 1e-9


def tile_residue(values: np.ndarray, *, tile: int = 8) -> float:
    """R_tile per spec Section 7.1 (the kwarg `tile` is the spec symbol q).

        R_tile(X, q) = J_tile(X, q) / (J_all(X) + eps)

    Returns 0.0 for inputs with no defined tile geometry (ndim < 2 or
    smallest spatial dimension <= q). 0.0 means "no detectable boundary
    preference", which is also the natural neutral point of the operator.
    """
    if tile <= 1:
        raise ValueError(f"tile must be greater than 1, got {tile}")

    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim < 2 or min(arr.shape[-2:]) <= tile:
        return 0.0

    spatial = arr.reshape((-1,) + arr.shape[-2:])
    boundary_terms: list[np.ndarray] = []
    all_terms: list[np.ndarray] = []
    for frame in spatial:
        dy = np.abs(np.diff(frame, axis=0))
        dx = np.abs(np.diff(frame, axis=1))
        all_terms.extend([dy.ravel(), dx.ravel()])
        if dy.size:
            i_idx = np.arange(dy.shape[0])
            boundary_terms.append(dy[(i_idx + 1) % tile == 0, :].ravel())
        if dx.size:
            j_idx = np.arange(dx.shape[1])
            boundary_terms.append(dx[:, (j_idx + 1) % tile == 0].ravel())

    all_values = np.concatenate([x for x in all_terms if x.size])
    boundary_values = np.concatenate([x for x in boundary_terms if x.size])
    if all_values.size == 0 or boundary_values.size == 0:
        return 0.0
    return float(boundary_values.mean() / (all_values.mean() + EPS))


def excess_residue(values: np.ndarray, *, tile: int = 8) -> float:
    """r(X, q) = max(R_tile(X, q) - 1, 0) per spec Section 7.1.

    r = 0 means the operator does nothing.
    """
    return max(tile_residue(values, tile=tile) - 1.0, 0.0)


def _axis_processes(
    rng: np.random.Generator, *, sigma: float, h: int, w: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Four 1-D Gaussian processes per spec Section 7.2.

        z_v[j]        length w     column-indexed   vertical axis
        z_h[i]        length h     row-indexed      horizontal axis
        z_+[i+j]      length h+w-1 primary diagonal
        z_-[i-j+w-1]  length h+w-1 secondary diagonal
    """
    diag_len = h + w - 1
    z_v = rng.normal(0.0, sigma, size=w)
    z_h = rng.normal(0.0, sigma, size=h)
    z_p = rng.normal(0.0, sigma, size=diag_len)
    z_m = rng.normal(0.0, sigma, size=diag_len)
    return z_v, z_h, z_p, z_m


def _structured_perturbation(
    z_v: np.ndarray, z_h: np.ndarray, z_p: np.ndarray, z_m: np.ndarray,
    *, h: int, w: int,
) -> np.ndarray:
    """N_4 per spec Section 7.3.

        N_4(i,j) = D_AXIS * [z_v[j] + z_h[i] + z_+[i+j] + z_-[i-j+w-1]]
    """
    i_idx = np.arange(h).reshape((h, 1))
    j_idx = np.arange(w).reshape((1, w))
    n_v = np.broadcast_to(z_v[None, :], (h, w))
    n_h = np.broadcast_to(z_h[:, None], (h, w))
    n_p = z_p[i_idx + j_idx]
    n_m = z_m[i_idx - j_idx + (w - 1)]
    return D_AXIS * (n_v + n_h + n_p + n_m)


def boundary_visibility(
    h: int, w: int, *, tile: int = 8, lambda_q: float = 1.0,
) -> np.ndarray:
    """w_q per spec Section 7.4 (the kwarg `tile` is the spec symbol q).

        d_q(i,j) = min over both axes of distance to nearest tile boundary
        w_q(i,j) = exp( -d_q(i,j) / lambda_q )
    """
    if tile <= 1:
        raise ValueError(f"tile must be greater than 1, got {tile}")
    if lambda_q <= 0:
        raise ValueError(f"lambda_q must be positive, got {lambda_q}")
    i = np.arange(h)
    j = np.arange(w)
    d_i = np.minimum(i % tile, (tile - 1) - (i % tile))
    d_j = np.minimum(j % tile, (tile - 1) - (j % tile))
    d_q = np.minimum(d_i[:, None], d_j[None, :]).astype(np.float64)
    return np.exp(-d_q / float(lambda_q))


def oc_four_over_e(
    values: np.ndarray,
    rng: np.random.Generator,
    *,
    beta: float,
    tile: int = 8,
    residue_gain: float = 1.0,
    lambda_q: float = 1.0,
) -> np.ndarray:
    """Final operator OC_{4/e} per spec Section 7.4.

        OC_{4/e}(X)_{i,j} = X_{i,j} + w_q(i,j) * N_4(i,j)

    with

        sigma_axis = beta * residue_gain * r(X, q)        (Section 7.2)

    Gating: if r(X, q) = 0 the operator returns a copy of X unchanged.

    Video / batch (Section 7.6): for inputs of shape (..., H, W) the same
    spatial OC_{4/e} pattern is broadcast across all leading axes — the
    "shared spatial pattern" variant. For per-frame independent or
    low-rank temporal coherence, call this per frame in a loop and state
    the variant in the experiment protocol.
    """
    if beta < 0:
        raise ValueError(f"beta must be non-negative, got {beta}")
    if residue_gain < 0:
        raise ValueError(f"residue_gain must be non-negative, got {residue_gain}")

    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim < 2:
        raise ValueError(
            f"OC_{{4/e}} is defined on (..., H, W) inputs; got ndim={arr.ndim}. "
            "1-D inputs have no tile geometry."
        )

    r = excess_residue(arr, tile=tile)
    if r <= 0.0:
        return arr.copy()

    sigma = beta * residue_gain * r
    if sigma <= 0.0:
        return arr.copy()

    h = arr.shape[-2]
    w = arr.shape[-1]
    z_v, z_h, z_p, z_m = _axis_processes(rng, sigma=sigma, h=h, w=w)
    n_4 = _structured_perturbation(z_v, z_h, z_p, z_m, h=h, w=w)
    w_q = boundary_visibility(h, w, tile=tile, lambda_q=lambda_q)

    perturbation = n_4 * w_q
    if arr.ndim > 2:
        perturbation = np.broadcast_to(perturbation, arr.shape)
    return arr + perturbation


def perturbation_energy(perturbed: np.ndarray, original: np.ndarray) -> float:
    """E_pert(M, X) = Mean( (M(X) - X)^2 ) per spec Section 7.7."""
    diff = (
        np.asarray(perturbed, dtype=np.float64)
        - np.asarray(original, dtype=np.float64)
    )
    return float(np.mean(diff * diff))


def fair_baseline_sigma(perturbed: np.ndarray, original: np.ndarray) -> float:
    """sigma_eq = sqrt(E_pert) per spec Section 7.7.

    Used to match iid baselines to the gated/structured 4/e operator on
    perturbation energy rather than nominal sigma.
    """
    return math.sqrt(perturbation_energy(perturbed, original))
