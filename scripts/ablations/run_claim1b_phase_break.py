"""Claim 1B: 4/e application utility — phase-break of tile contour.

Tested per docs/vod_math_simplification.md §"Orthogonal Compression
Noise / Tile Residue":

    "用受控随机性消除不自然的相干边界...
     把块状粒子散射轮廓破相"

Operationalisation of "破相" (phase-break)
------------------------------------------
A coherent tile contour line has uniform jump magnitudes along its
extent: every pixel on a tile-boundary column shows roughly the same
boundary-jump strength, so the eye perceives a clean continuous line.

A phase-broken contour line has *variable* jump magnitudes along its
extent: some boundary pixels jump strongly, others barely jump, the
line stops looking like a uniform contour and starts looking like
irregular edge texture.

So the right metric is the COEFFICIENT OF VARIATION of jump magnitudes
*along each tile-boundary line*, averaged across all such lines:

    CV(line) = std(|jumps along line|) / mean(|jumps along line|)
    phase_break(image) = mean over all boundary lines of CV(line)

Higher = more phase-broken = less perceptually coherent contour.

Why 4/e should win at matched energy
------------------------------------
At matched perturbation energy E:
    iid per-pixel variance:  σ²_iid = E       (uniform over N pixels)
    4/e per-pixel variance:  σ²_4e(i,j) = (E·w_q²(i,j) / mean(w_q²))
        — concentrated where w_q ≈ 1 (boundaries), small in interior

On a boundary line every pixel has w_q ≈ 1 → σ²_4e at the boundary is
HIGHER than σ²_iid by factor 1/mean(w_q²). For tile=8, lambda_q=1
that ratio is ~3-4x.

So 4/e adds more local variance to boundary jumps than iid → CV of
boundary jumps under 4/e > CV under iid → MORE phase-break.

This is NOT a cross-correlation effect (E[⟨δ,P⟩] = 0 for any zero-
mean δ regardless of structure). It's a *local variance redistribution*
effect — the structure of 4/e doesn't reduce template detectability, it
redistributes per-pixel variance from interior to boundary, which
shows up perceptually as phase-broken contours.

Pre-registered hypothesis
-------------------------
At every strength s in {0.3, 0.5, 0.8} and on both [image] and [video]:
    gain = phase_break(4/e) - phase_break(best_iid)
    PASS iff gain >= 0.05 AND |gain| >= 3·SE(paired diff)

(Direction is reversed from the prior tests: HIGHER phase_break is the
goal here, so 4/e should be HIGHER than iid.)
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

sys.path.insert(0, "D:/VOD/prototype")

from vod_minimal.artifacts import oc_four_over_e
from vod_minimal.chladni import random_chladni_field


def coherent_tile_light_spot(h: int, w: int, *, tile: int = 8, strength: float = 0.5) -> np.ndarray:
    """A *coherent* tile light spot pattern, not random.

    A 1-pixel-wide bright halo at every tile-boundary row / column.
    The boundary row is i where (i+1) mod tile == 0 (the spec's
    boundary index). The next row is dark, so the boundary jump
    a[i+1, :] - a[i, :] = -strength with CONSISTENT sign along the
    entire row. Sign-agreement on the boundary line is ~1.0.

    Two-pixel halos would set both i and i+1 bright, making the spec
    boundary jump = 0 and gating the 4/e operator off. One pixel only.
    """
    out = np.zeros((h, w), dtype=np.float64)
    for i in range(tile - 1, h, tile):
        out[i, :] += strength
    for j in range(tile - 1, w, tile):
        out[:, j] += strength
    return out


# --------------------------------------------------------------------------- #
#  Suppressors
# --------------------------------------------------------------------------- #

def supp_gaussian(view, rng, sigma, tile):
    return view.astype(np.float64) + rng.normal(0.0, sigma, view.shape)


def supp_uniform(view, rng, sigma, tile):
    half = sigma * np.sqrt(3.0)
    return view.astype(np.float64) + rng.uniform(-half, half, view.shape)


def supp_fourover_e(view, rng, sigma, tile):
    return oc_four_over_e(view, rng, beta=sigma, tile=tile)


# --------------------------------------------------------------------------- #
#  Phase-break metric
# --------------------------------------------------------------------------- #

def _to_2d(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float64)
    while a.ndim > 2:
        a = a.mean(axis=0)
    return a


def boundary_sign_agreement(arr: np.ndarray, *, tile: int = 8, eps: float = 1e-9) -> float:
    """Fraction of consecutive same-sign boundary jumps along tile-boundary lines.

    A perceptually coherent contour line (clean tile light spot) has
    *direction-consistent* boundary jumps: every pixel along the
    boundary shows brightness changing in the same direction (always
    brighter on one side, or always darker). Sign-agreement → 1.0.

    A phase-broken contour has *sign-flipping* jumps along its length:
    some pixels brighter on right, some on left → eye no longer sees a
    coherent line. Sign-agreement → 0.5 (random).

    Returns mean over all tile-boundary lines.

    LOWER = more phase-broken (4/e should win here at matched energy).
    """
    a = _to_2d(arr)
    H, W = a.shape
    if min(H, W) <= tile:
        return 0.5

    fractions = []

    # Vertical tile-boundary columns: signed jump = a[:, j+1] - a[:, j]
    for j in range(tile - 1, W - 1, tile):
        signed = a[:, j + 1] - a[:, j]
        sig = signed[np.abs(signed) > eps]
        if sig.size < 2:
            continue
        signs = np.sign(sig)
        agree = float((signs[:-1] == signs[1:]).mean())
        fractions.append(agree)

    # Horizontal tile-boundary rows
    for i in range(tile - 1, H - 1, tile):
        signed = a[i + 1, :] - a[i, :]
        sig = signed[np.abs(signed) > eps]
        if sig.size < 2:
            continue
        signs = np.sign(sig)
        agree = float((signs[:-1] == signs[1:]).mean())
        fractions.append(agree)

    if not fractions:
        return 0.5
    return float(np.mean(fractions))


def perturbation_energy(after: np.ndarray, before: np.ndarray) -> float:
    return float(np.mean((after.astype(np.float64) - before.astype(np.float64)) ** 2))


# --------------------------------------------------------------------------- #
#  Single-seed experiment
# --------------------------------------------------------------------------- #

def run_one_seed(*, seed: int, size: int, tile: int, strength: float, beta: float) -> dict:
    rng = np.random.default_rng(seed)

    # Step 1: clean Chladni field (no tile structure)
    X_0 = random_chladni_field(rng, size=size, n_modes=3)

    # Step 2: deterministic COHERENT tile light spot — a clean grid of
    # constant-brightness halos. The sign of every boundary jump is the
    # same along each line, so without perturbation sign_agreement ~ 1.0.
    P = coherent_tile_light_spot(size, size, tile=tile, strength=1.0)

    # Step 3: composite — the simulated tile-corrupted image
    X = X_0 + strength * P

    # Step 4: 4/e first to set the energy budget
    rng_4e = np.random.default_rng(seed + 200_000)
    Y_4e = supp_fourover_e(X, rng_4e, beta, tile)
    E = perturbation_energy(Y_4e, X)
    sigma_eq = float(np.sqrt(E))

    # Step 5: matched-energy iid baselines
    rng_g = np.random.default_rng(seed + 300_000)
    Y_g = supp_gaussian(X, rng_g, sigma_eq, tile)
    rng_u = np.random.default_rng(seed + 400_000)
    Y_u = supp_uniform(X, rng_u, sigma_eq, tile)

    # Step 6: sign-agreement of boundary jumps (LOWER = more phase-broken)
    return {
        "X_baseline":  boundary_sign_agreement(X, tile=tile),
        "gaussian":    boundary_sign_agreement(Y_g, tile=tile),
        "uniform":     boundary_sign_agreement(Y_u, tile=tile),
        "fourover":    boundary_sign_agreement(Y_4e, tile=tile),
        "perturb_E":   E,
        "sigma_eq":    sigma_eq,
    }


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strengths", type=float, nargs="+", default=[0.3, 0.5, 0.8])
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--tile", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.02,
                        help="pre-registered minimum sign-agreement DROP under 4/e vs best iid")
    args = parser.parse_args()

    print("=" * 60)
    print("Claim 1B: 4/e Application Utility — Phase-Break")
    print("=" * 60)
    print(f"  strengths={args.strengths}  n_seeds={args.n_seeds}")
    print(f"  size={args.size}  tile={args.tile}  threshold={args.threshold}")
    print()
    print("  Hypothesis (pre-registered):")
    print("    boundary_sign_agreement(4/e) < boundary_sign_agreement(best iid)")
    print("    at matched perturbation energy.")
    print("  Lower sign-agreement = boundary jumps lose direction consistency = phase-broken.")
    print("  PASS iff gain (best_iid - 4/e) >= threshold AND |gain| >= 3·SE(paired diff)")

    overall_pass = True
    for strength in args.strengths:
        per_seed = [
            run_one_seed(
                seed=s, size=args.size, tile=args.tile,
                strength=strength, beta=strength,
            )
            for s in range(args.n_seeds)
        ]

        means = {}
        for key in ("X_baseline", "gaussian", "uniform", "fourover", "perturb_E"):
            means[key] = float(np.mean([r[key] for r in per_seed]))

        # Paired diff per seed: best_iid - 4/e (we want POSITIVE → 4/e lower)
        paired = [
            min(r["gaussian"], r["uniform"]) - r["fourover"]
            for r in per_seed
        ]
        paired_mean = float(np.mean(paired))
        paired_std = float(np.std(paired, ddof=1))
        se_mean = paired_std / np.sqrt(args.n_seeds)
        three_sig_se = 3.0 * se_mean

        score_pass = paired_mean >= args.threshold
        sig_pass = abs(paired_mean) >= three_sig_se
        verdict = score_pass and sig_pass
        if not verdict:
            overall_pass = False

        marker = "PASS" if verdict else "FAIL"
        print()
        print(f"  --- strength = {strength}  ({args.n_seeds} seeds) ---")
        print(f"    sign_agreement(X before perturb)        = {means['X_baseline']:.4f}")
        print(f"    sign_agreement(Y_gaussian)              = {means['gaussian']:.4f}")
        print(f"    sign_agreement(Y_uniform)               = {means['uniform']:.4f}")
        print(f"    sign_agreement(Y_4/e)                   = {means['fourover']:.4f}")
        print(f"    perturbation energy (matched)           = {means['perturb_E']:.5f}")
        print(f"    paired diff (best_iid - 4/e), mean      = {paired_mean:+.5f}")
        print(f"    paired diff std (ddof=1)                = {paired_std:.5f}")
        print(f"    threshold for PASS                      = {args.threshold:.5f}")
        print(f"    3-sigma SE of mean (significance band)  = {three_sig_se:.5f}")
        print(f"    score_pass={score_pass}  sig_pass={sig_pass}  -->  {marker}")

    print()
    print("=" * 60)
    print(f"Claim 1B (phase-break): {'PASS' if overall_pass else 'FAIL'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
