"""Claim 1B: 4/e application utility under spec-aligned metrics.

NOT a Claim 1A test. Claim 1A (operator conformance) is locked by
tests/test_4e_conformance.py against spec Section 7.5 and PASSES at
~13σ on the AXCOV signature. This script tests Claim 1B (application
utility): given that the operator IS spec-conformant, does it actually
*help* on metrics that correspond to its design intent?

Why three new metrics
---------------------
The previous Claim 1B test used `artifact_score = 1 / (1 + max(R_tile - 1, 0))`,
which collapses tile residue to a single mean over all neighbour pairs.
Spec 7.0 explicitly states this metric cannot test Claim 1A. It also
cannot test Claim 1B faithfully because mean tile residue is direction-
agnostic — it cannot distinguish "tile contour memory wiped along its own
direction" (4/e's claim from artifacts.py docstring) from "uniform iid
noise added everywhere".

The three metrics here each correspond to a property 4/e claims to
deliver:

    M1  Anisotropy redistribution
        4-axis residue coefficient-of-variation. Blocky originals have
        axis-aligned residue dominant (vertical / horizontal). 4/e's
        four-axis structured perturbation should equalise residue across
        axes; iid is direction-symmetric and won't redistribute it
        beyond raising the noise floor uniformly.

    M2  Tile-period autocorrelation suppression
        Original blocky has strong autocorrelation at lag = tile period.
        4/e's boundary-targeted perturbation specifically interferes at
        the boundary positions. iid is uncorrelated and only adds noise
        floor.

    M3  Tile-frequency FFT peak suppression
        2D FFT power at the axis-aligned tile-frequency peaks
        ((H/tile, 0) and (0, W/tile)). Same logic as M2 in the spectral
        domain — 4/e's structured perturbation injects coherent counter-
        signals at the same frequency; iid spreads energy uniformly
        across all frequencies.

Pre-registered hypothesis
-------------------------
For each metric, lower-after-perturbation = better disruption of the
original tile pattern. Define:

    gain(metric) = metric(best_iid_baseline) - metric(4/e)

Falsification: at every strength s in {0.3, 0.5, 0.8}, on both
[image] and [video] paths, AT LEAST ONE of the three metrics must
satisfy:

    gain >= threshold_metric                (effect size)
    AND
    |gain| >= 3 * std_across_seeds          (statistical significance)

Thresholds:
    M1 anisotropy CV reduction:  >= +0.10
    M2 autocorr reduction:       >= +0.05
    M3 FFT-peak reduction:       >= +0.10  (relative to original peak)

Result interpretation
---------------------
- If all metrics PASS at all strengths: 4/e has demonstrable application
  utility on the kind of contour-memory disruption it was designed for.
- If some metrics PASS and some FAIL: report honestly which metric
  detects the effect; 4/e helps on a subset of design-aligned tasks.
- If ALL metrics FAIL: Claim 1B is FALSIFIED for these metrics under
  this protocol. The operator is still spec-conformant (Claim 1A PASS),
  but no detectable application benefit emerges in this setup.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

sys.path.insert(0, "D:/VOD/prototype")

from vod_minimal.artifacts import oc_four_over_e
from vod_minimal.blocky_scattering import build_blocky_scattering_batch
from vod_minimal.core import build_projection_batch, ProjectionBatch, ProjectionSample


# --------------------------------------------------------------------------- #
#  Suppressors (matched to run_claim1_four_over_e_ablation.py)
# --------------------------------------------------------------------------- #

def supp_none(view, rng, sigma, tile):
    return view.astype(np.float64).copy()


def supp_gaussian(view, rng, sigma, tile):
    return view.astype(np.float64) + rng.normal(0.0, sigma, view.shape)


def supp_uniform(view, rng, sigma, tile):
    half = sigma * np.sqrt(3.0)
    return view.astype(np.float64) + rng.uniform(-half, half, view.shape)


def supp_fourover_e(view, rng, sigma, tile):
    return oc_four_over_e(view, rng, beta=sigma, tile=tile)


# --------------------------------------------------------------------------- #
#  Metrics
# --------------------------------------------------------------------------- #

def _to_2d(arr: np.ndarray) -> np.ndarray:
    """Collapse leading axes by mean so all metrics see a (H, W) frame."""
    a = np.asarray(arr, dtype=np.float64)
    while a.ndim > 2:
        a = a.mean(axis=0)
    return a


def axial_residue_split(arr: np.ndarray, *, tile: int = 8) -> dict:
    """Per-axis tile-boundary jump magnitudes — 4 separate values.

    For axes 1/2 the boundary is at every (i+1) mod tile == 0 row /
    column. For diagonals 3/4 the boundary is at every (i+j) mod tile ==
    0 / (i-j) mod tile == 0 line.
    """
    a = _to_2d(arr)
    H, W = a.shape
    if min(H, W) <= tile:
        return {"vert": 0.0, "horiz": 0.0, "diag1": 0.0, "diag2": 0.0}

    # Axis 1: vertical residue = jumps along columns at row-boundary positions
    # (boundary on i: (i+1) mod tile == 0)
    dy = np.abs(np.diff(a, axis=0))
    i_idx = np.arange(dy.shape[0])
    R_h = float(dy[(i_idx + 1) % tile == 0, :].mean())   # row-boundary jumps

    # Axis 2: horizontal residue = jumps along rows at column-boundary positions
    dx = np.abs(np.diff(a, axis=1))
    j_idx = np.arange(dx.shape[1])
    R_v = float(dx[:, (j_idx + 1) % tile == 0].mean())   # column-boundary jumps

    # Axis 3 / 4: diagonal residues — jumps along i+j = const / i-j = const
    # at tile-period diagonal boundaries.
    i_grid, j_grid = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    d1 = i_grid + j_grid                  # primary diagonal index
    d2 = i_grid - j_grid + (W - 1)        # secondary diagonal index (shifted)

    # For each diagonal index value, compute jumps along that diagonal.
    # We measure mean abs jump between consecutive cells on the diagonal,
    # restricted to "tile-period boundaries" along the diagonal.
    diag1_jumps = []
    for d in range(1, H + W - 2):
        mask = (d1 == d)
        if mask.sum() < 2:
            continue
        # Cells on this diagonal, ordered by i:
        ii = i_grid[mask]
        jj = j_grid[mask]
        order = np.argsort(ii)
        ii = ii[order]; jj = jj[order]
        vals = a[ii, jj]
        if vals.size >= 2 and (d % tile == 0):
            diag1_jumps.extend(np.abs(np.diff(vals)).tolist())
    R_d1 = float(np.mean(diag1_jumps)) if diag1_jumps else 0.0

    diag2_jumps = []
    for d in range(1, H + W - 2):
        mask = (d2 == d)
        if mask.sum() < 2:
            continue
        ii = i_grid[mask]
        jj = j_grid[mask]
        order = np.argsort(ii)
        ii = ii[order]; jj = jj[order]
        vals = a[ii, jj]
        if vals.size >= 2 and (d % tile == 0):
            diag2_jumps.extend(np.abs(np.diff(vals)).tolist())
    R_d2 = float(np.mean(diag2_jumps)) if diag2_jumps else 0.0

    return {"vert": R_v, "horiz": R_h, "diag1": R_d1, "diag2": R_d2}


def m1_anisotropy_cv(arr: np.ndarray, *, tile: int = 8) -> float:
    """Coefficient of variation across the four axial residues.

    High CV  = strongly anisotropic (axis-aligned dominates)
    Low  CV  = equalised across axes (4/e disruption goal)

    PASS direction: 4/e should LOWER this vs iid baselines.
    """
    parts = axial_residue_split(arr, tile=tile)
    vals = np.array([parts["vert"], parts["horiz"], parts["diag1"], parts["diag2"]])
    if vals.mean() < 1e-9:
        return 0.0
    return float(vals.std(ddof=0) / (vals.mean() + 1e-9))


def m2_tile_autocorr(arr: np.ndarray, *, tile: int = 8) -> float:
    """Mean autocorrelation at lag = tile, averaged over rows and columns.

    Original blocky pattern repeats with period `tile` so this is high.
    A perturbation that breaks tile-period coherence reduces it.

    PASS direction: 4/e should LOWER this vs iid baselines.
    """
    a = _to_2d(arr)
    a = a - a.mean()
    var = float(a.var())
    if var < 1e-12:
        return 0.0

    # Row-direction autocorrelation at lag = tile
    H, W = a.shape
    if W <= tile or H <= tile:
        return 0.0
    row_ac = float(np.mean(a[:, :W - tile] * a[:, tile:])) / var
    col_ac = float(np.mean(a[:H - tile, :] * a[tile:, :])) / var
    return float(0.5 * (abs(row_ac) + abs(col_ac)))


def m3_tile_freq_peak(arr: np.ndarray, *, tile: int = 8) -> float:
    """Relative power at the axis-aligned tile-frequency FFT peaks.

    Position in 2D rfft for a period-`tile` pattern: (H/tile, 0) and
    (0, W/tile). Returns peak_power / total_power so the metric is
    invariant to overall energy scale.

    PASS direction: 4/e should LOWER this vs iid baselines.
    """
    a = _to_2d(arr)
    H, W = a.shape
    if H % tile != 0 or W % tile != 0:
        return 0.0
    fft = np.fft.fft2(a - a.mean())
    power = np.abs(fft) ** 2
    total = float(power.sum())
    if total < 1e-12:
        return 0.0
    # Tile-frequency components in axis-aligned directions
    kh = H // tile
    kw = W // tile
    peak = float(power[kh, 0] + power[0, kw] + power[H - kh, 0] + power[0, W - kw])
    return peak / total


# --------------------------------------------------------------------------- #
#  Energy matching
# --------------------------------------------------------------------------- #

def perturbation_energy(after: np.ndarray, before: np.ndarray) -> float:
    return float(np.mean((after.astype(np.float64) - before.astype(np.float64)) ** 2))


def equal_energy_sigma(blocky: np.ndarray, rng_seed: int, *, beta: float, tile: int) -> float:
    """Run 4/e once to measure its E_pert, then return sqrt(E_pert) for iid baselines."""
    out = oc_four_over_e(blocky, np.random.default_rng(rng_seed), beta=beta, tile=tile)
    return float(np.sqrt(perturbation_energy(out, blocky)))


# --------------------------------------------------------------------------- #
#  Per-strength experiment
# --------------------------------------------------------------------------- #

def evaluate_one_seed(
    blocky_view: np.ndarray, *, beta: float, tile: int, seed: int,
) -> dict[str, dict[str, float]]:
    """Run 4 suppressors on `blocky_view`, return all metrics per method."""
    sigma_eq = equal_energy_sigma(blocky_view, seed, beta=beta, tile=tile)
    methods = {
        "none":     ("supp_none",     supp_none,     beta),     # beta unused for none
        "gaussian": ("supp_gaussian", supp_gaussian, sigma_eq),
        "uniform":  ("supp_uniform",  supp_uniform,  sigma_eq),
        "fourover": ("supp_fourover", supp_fourover_e, beta),
    }
    out = {}
    for key, (_name, fn, s) in methods.items():
        rng = np.random.default_rng(seed)
        result = fn(blocky_view, rng, s, tile)
        out[key] = {
            "M1_anisotropy_cv":  m1_anisotropy_cv(result, tile=tile),
            "M2_tile_autocorr":  m2_tile_autocorr(result, tile=tile),
            "M3_tile_freq_peak": m3_tile_freq_peak(result, tile=tile),
            "perturb_E":         perturbation_energy(result, blocky_view),
        }
    return out


def aggregate_seeds(per_seed_results: list[dict]) -> dict:
    """Mean and std for each (method, metric) across seeds."""
    agg = {}
    methods = ["none", "gaussian", "uniform", "fourover"]
    metrics = ["M1_anisotropy_cv", "M2_tile_autocorr", "M3_tile_freq_peak", "perturb_E"]
    for method in methods:
        agg[method] = {}
        for metric in metrics:
            vals = [s[method][metric] for s in per_seed_results]
            agg[method][metric] = {
                "mean": float(np.mean(vals)),
                "std":  float(np.std(vals, ddof=0)),
            }
    return agg


# --------------------------------------------------------------------------- #
#  Verdict
# --------------------------------------------------------------------------- #

THRESHOLDS = {
    "M1_anisotropy_cv":  0.10,
    "M2_tile_autocorr":  0.05,
    "M3_tile_freq_peak": 0.10,
}


def per_metric_verdict(agg: dict, *, paired_diffs: dict | None = None) -> dict:
    """For each metric: gain (best_iid - 4/e), threshold check, 3sig check.

    If `paired_diffs` is provided, the 3σ band is computed from the
    paired (per-seed iid - per-seed 4/e) distribution, which removes
    between-input variance and gives a much sharper test of whether the
    4/e effect is consistent within seeds.
    """
    out = {}
    for metric, threshold in THRESHOLDS.items():
        iid_gauss = agg["gaussian"][metric]["mean"]
        iid_unif  = agg["uniform"][metric]["mean"]
        fourover  = agg["fourover"][metric]["mean"]

        best_iid = min(iid_gauss, iid_unif)
        gain = best_iid - fourover

        if paired_diffs is not None and metric in paired_diffs:
            paired = paired_diffs[metric]
            paired_mean = float(np.mean(paired))
            paired_std = float(np.std(paired, ddof=1)) if len(paired) > 1 else 0.0
            three_sig = 3.0 * paired_std
            sig_pass = abs(paired_mean) >= three_sig
        else:
            std_4e = agg["fourover"][metric]["std"]
            std_iid = min(agg["gaussian"][metric]["std"], agg["uniform"][metric]["std"])
            pooled_std = float(np.sqrt(std_4e ** 2 + std_iid ** 2))
            three_sig = 3.0 * pooled_std
            paired_mean = gain
            sig_pass = abs(gain) >= three_sig

        score_pass = gain >= threshold
        verdict = score_pass and sig_pass

        out[metric] = {
            "iid_gaussian":   iid_gauss,
            "iid_uniform":    iid_unif,
            "best_iid":       best_iid,
            "fourover":       fourover,
            "gain":           gain,
            "paired_mean":    paired_mean,
            "threshold":      threshold,
            "three_sig":      three_sig,
            "score_pass":     score_pass,
            "sig_pass":       sig_pass,
            "verdict":        verdict,
        }
    return out


def print_block(strength: float, medium: str, agg: dict, verdict: dict) -> None:
    print(f"\n  [{medium}]  strength = {strength}")
    for method in ("none", "gaussian", "uniform", "fourover"):
        m = agg[method]
        print(
            f"    {method:10s}"
            f"  M1_aniso={m['M1_anisotropy_cv']['mean']:.4f}"
            f"  M2_autocorr={m['M2_tile_autocorr']['mean']:.4f}"
            f"  M3_FFTpeak={m['M3_tile_freq_peak']['mean']:.4f}"
            f"  E={m['perturb_E']['mean']:.4f}"
        )
    for metric, v in verdict.items():
        marker = "PASS" if v["verdict"] else "FAIL"
        print(
            f"    --> {metric:20s}"
            f"  paired_diff={v['paired_mean']:+.5f}"
            f"  unpaired_gain={v['gain']:+.4f}"
            f"  thr={v['threshold']:.4f}"
            f"  3sig={v['three_sig']:.5f}"
            f"  --> {marker}"
        )


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strengths", type=float, nargs="+", default=[0.3, 0.5, 0.8])
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--tile", type=int, default=8)
    args = parser.parse_args()

    print("=" * 60)
    print("Claim 1B: 4/e Application Utility (spec-aligned metrics)")
    print("=" * 60)
    print(f"  strengths={args.strengths}  n_seeds={args.n_seeds}")
    print(f"  size={args.size}  tile={args.tile}  batch_size={args.batch_size}")
    print(f"  thresholds:  M1>={THRESHOLDS['M1_anisotropy_cv']:.2f}"
          f"  M2>={THRESHOLDS['M2_tile_autocorr']:.2f}"
          f"  M3>={THRESHOLDS['M3_tile_freq_peak']:.2f}")
    print("\n  Metric direction: 4/e should be LOWER than best iid baseline")
    print("  (lower = more disruption of original tile pattern)")

    per_strength_verdicts = {}
    for strength in args.strengths:
        per_medium_verdicts = {}
        for medium in ("image", "video"):
            per_seed = []
            for seed in range(args.n_seeds):
                # Build a blocky batch deterministically per seed
                blocky_batch = build_blocky_scattering_batch(
                    np.random.default_rng(seed),
                    batch_size=args.batch_size,
                    size=args.size,
                    tile=args.tile,
                    artifact_strength=strength,
                )
                # Aggregate metric over batch samples
                seed_metrics = {"none": [], "gaussian": [], "uniform": [], "fourover": []}
                for sample in blocky_batch.samples:
                    if medium not in sample.noisy_views:
                        continue
                    view = sample.noisy_views[medium]
                    one = evaluate_one_seed(view, beta=strength, tile=args.tile, seed=seed)
                    for method in seed_metrics:
                        seed_metrics[method].append(one[method])
                # Average within seed
                avg = {}
                for method, results_list in seed_metrics.items():
                    avg[method] = {}
                    for m_key in ("M1_anisotropy_cv", "M2_tile_autocorr", "M3_tile_freq_peak", "perturb_E"):
                        avg[method][m_key] = float(np.mean([r[m_key] for r in results_list]))
                per_seed.append(avg)

            # Aggregate across seeds
            agg = {}
            for method in ("none", "gaussian", "uniform", "fourover"):
                agg[method] = {}
                for m_key in ("M1_anisotropy_cv", "M2_tile_autocorr", "M3_tile_freq_peak", "perturb_E"):
                    vals = [s[method][m_key] for s in per_seed]
                    agg[method][m_key] = {
                        "mean": float(np.mean(vals)),
                        "std":  float(np.std(vals, ddof=0)),
                    }

            # Paired diff: per-seed (best_iid - 4/e) — removes between-input variance.
            paired_diffs = {}
            for m_key in ("M1_anisotropy_cv", "M2_tile_autocorr", "M3_tile_freq_peak"):
                diffs = []
                for s in per_seed:
                    best_iid_val = min(s["gaussian"][m_key], s["uniform"][m_key])
                    diffs.append(best_iid_val - s["fourover"][m_key])
                paired_diffs[m_key] = diffs

            verdict = per_metric_verdict(agg, paired_diffs=paired_diffs)
            print_block(strength, medium, agg, verdict)
            per_medium_verdicts[medium] = verdict
        per_strength_verdicts[strength] = per_medium_verdicts

    # Final tally
    print("\n" + "=" * 60)
    print("FINAL VERDICT")
    print("=" * 60)
    print("PASS criterion: at every (strength, medium) AT LEAST ONE metric must PASS")
    print()

    overall_pass = True
    for strength, medium_dict in per_strength_verdicts.items():
        for medium, verdict in medium_dict.items():
            metric_passes = {m: v["verdict"] for m, v in verdict.items()}
            any_pass = any(metric_passes.values())
            label = "PASS" if any_pass else "FAIL"
            passing = [m for m, p in metric_passes.items() if p]
            failing = [m for m, p in metric_passes.items() if not p]
            print(f"  s={strength}  [{medium}]:  {label}    "
                  f"pass={passing if passing else '∅'}  fail={failing if failing else '∅'}")
            if not any_pass:
                overall_pass = False

    print()
    print(f"Claim 1B: {'PASS' if overall_pass else 'FAIL'}")


if __name__ == "__main__":
    main()
