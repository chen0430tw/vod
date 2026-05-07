"""Claim 1B: 4/e application utility — template-detectability operationalisation.

Replaces the previous run_claim1b_application_utility.py metrics, which
were ill-designed: M1 (anisotropy CV) and M3 (FFT peak) measured
properties that the operator's own design (boundary-targeted w_q +
axial covariance signature) intentionally goes the OPPOSITE way on, by
construction. They could never PASS without violating Claim 1A.

This script tests the docstring claim directly:

    "The four directional processes overlay coherent stripes along
     each tile-boundary direction, washing out the shader's coherent
     block contour memory along its own direction."
                                       — vod_minimal/artifacts.py

Operationalisation
------------------
Setup per seed:
    1. Build a clean Chladni field X_0  (no tile structure)
    2. Build a deterministic tile pattern P (the "coherent block contour
       memory" — exactly the failure mode 4/e is designed against)
    3. Composite X = X_0 + s * P
    4. Run 4/e on X, measure perturbation energy E
    5. For iid baselines, set sigma_eq = sqrt(E) so all methods inject
       the same total energy
    6. For each method M, compute Y = M(X) and measure
           detectability(Y; P) = Pearson correlation between Y and P
       — how well the original tile pattern P can still be recovered
       from the perturbed image by template matching.

Pre-registered hypothesis
-------------------------
At matched perturbation energy E:
    detectability(Y_4e; P) < detectability(Y_iid_best; P)

Why this should hold (given operator works as spec says):
    iid spreads E uniformly across H*W pixels → low per-pixel sigma
    everywhere; P (which lives at boundary positions) is barely affected.
    4/e concentrates E at boundary positions via w_q → high per-pixel
    sigma exactly where P lives → P's signal-to-noise drops faster.

Threshold (pre-registered before running):
    gain = detectability(iid_best) - detectability(4/e)
    PASS iff gain >= 0.02 AND |gain| >= 3 * std(per-seed paired diff)

This is a calibrated threshold: a 0.02 drop in Pearson correlation
corresponds to ~10% reduction in template-matching SNR for typical
correlation values around 0.2-0.4.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

sys.path.insert(0, "D:/VOD/prototype")

from vod_minimal.artifacts import oc_four_over_e
from vod_minimal.blocky_scattering import blocky_scattering_mask
from vod_minimal.chladni import random_chladni_field


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
#  Detectability metric
# --------------------------------------------------------------------------- #

def template_detectability(image: np.ndarray, template: np.ndarray) -> float:
    """Pearson correlation between (image - mean) and (template - mean).

    Range: [-1, +1]. Higher absolute value = template more recoverable.
    We report the signed value; for our setup template is a non-negative
    pattern so positive correlation = pattern still visible.
    """
    img = (image - image.mean()).flatten()
    tmp = (template - template.mean()).flatten()
    norm_i = float(np.linalg.norm(img))
    norm_t = float(np.linalg.norm(tmp))
    if norm_i < 1e-12 or norm_t < 1e-12:
        return 0.0
    return float(img @ tmp / (norm_i * norm_t))


def perturbation_energy(after: np.ndarray, before: np.ndarray) -> float:
    return float(np.mean((after.astype(np.float64) - before.astype(np.float64)) ** 2))


# --------------------------------------------------------------------------- #
#  Single-seed experiment
# --------------------------------------------------------------------------- #

def run_one_seed(*, seed: int, size: int, tile: int, strength: float, beta: float) -> dict:
    rng = np.random.default_rng(seed)

    # Step 1: clean Chladni field (no tile structure)
    X_0 = random_chladni_field(rng, size=size, n_modes=3)

    # Step 2: deterministic tile pattern P (the contour memory)
    template_rng = np.random.default_rng(seed + 100_000)
    P = blocky_scattering_mask(
        (size, size), tile=tile, strength=1.0, rng=template_rng
    )

    # Step 3: composite — the simulated tile-corrupted output
    X = X_0 + strength * P

    # Step 4: measure 4/e perturbation energy first to set fair sigma
    rng_4e = np.random.default_rng(seed + 200_000)
    Y_4e = supp_fourover_e(X, rng_4e, beta, tile)
    E = perturbation_energy(Y_4e, X)
    sigma_eq = float(np.sqrt(E))

    # Step 5: run all methods at matched energy
    rng_g = np.random.default_rng(seed + 300_000)
    Y_g = supp_gaussian(X, rng_g, sigma_eq, tile)
    rng_u = np.random.default_rng(seed + 400_000)
    Y_u = supp_uniform(X, rng_u, sigma_eq, tile)

    # Step 6: detectability of P in each perturbed output
    return {
        "X_baseline":     template_detectability(X, P),         # before perturbation
        "gaussian":       template_detectability(Y_g, P),
        "uniform":        template_detectability(Y_u, P),
        "fourover":       template_detectability(Y_4e, P),
        "perturb_E":      E,
        "sigma_eq":       sigma_eq,
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
                        help="pre-registered minimum detectability drop for PASS")
    args = parser.parse_args()

    print("=" * 60)
    print("Claim 1B: 4/e Application Utility — Template Detectability")
    print("=" * 60)
    print(f"  strengths={args.strengths}  n_seeds={args.n_seeds}")
    print(f"  size={args.size}  tile={args.tile}  threshold={args.threshold}")
    print()
    print("  Hypothesis (pre-registered):")
    print("    detectability(4/e Y; P) < detectability(best iid Y; P)")
    print("    at matched perturbation energy.")
    print("  PASS iff gain >= threshold AND |gain| >= 3 * paired_std")

    overall_pass = True
    for strength in args.strengths:
        per_seed_results = [
            run_one_seed(
                seed=s, size=args.size, tile=args.tile,
                strength=strength, beta=strength,
            )
            for s in range(args.n_seeds)
        ]

        # Aggregate
        means = {}
        for key in ("X_baseline", "gaussian", "uniform", "fourover", "perturb_E"):
            vals = [r[key] for r in per_seed_results]
            means[key] = float(np.mean(vals))

        # Per-seed paired diff: best_iid - 4/e
        paired_diffs = [
            min(r["gaussian"], r["uniform"]) - r["fourover"]
            for r in per_seed_results
        ]
        paired_mean = float(np.mean(paired_diffs))
        paired_std = float(np.std(paired_diffs, ddof=1))
        three_sig = 3.0 * paired_std / np.sqrt(args.n_seeds)  # SE of mean
        # gain criterion compares to threshold
        score_pass = paired_mean >= args.threshold
        sig_pass = abs(paired_mean) >= three_sig
        verdict = score_pass and sig_pass
        if not verdict:
            overall_pass = False

        marker = "PASS" if verdict else "FAIL"
        print()
        print(f"  --- strength = {strength}  ({args.n_seeds} seeds) ---")
        print(f"    detectability(X_before_perturb)         = {means['X_baseline']:+.4f}")
        print(f"    detectability(Y_gaussian)               = {means['gaussian']:+.4f}")
        print(f"    detectability(Y_uniform)                = {means['uniform']:+.4f}")
        print(f"    detectability(Y_4/e)                    = {means['fourover']:+.4f}")
        print(f"    perturbation energy (matched)           = {means['perturb_E']:.5f}")
        print(f"    paired diff (best_iid - 4/e), mean      = {paired_mean:+.5f}")
        print(f"    paired diff std (ddof=1)                = {paired_std:.5f}")
        print(f"    threshold for PASS                      = {args.threshold:.5f}")
        print(f"    3-sigma SE of mean (significance band)  = {three_sig:.5f}")
        print(f"    score_pass={score_pass}  sig_pass={sig_pass}  -->  {marker}")

    print()
    print("=" * 60)
    print(f"Claim 1B (template detectability): {'PASS' if overall_pass else 'FAIL'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
