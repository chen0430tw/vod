"""Claim 1: 4/e Orthogonal Compression Decay vs gaussian / uniform.

NOT a model test. Pure data-side ablation:
    blocky view --[4 suppressors]--> 4 outputs --> compare metrics

Hypothesis (pre-registered in postmortem Section 7 / 11):
    oc_four_over_e reduces tile residue significantly more per unit
    structural damage than gaussian iid or uniform iid noise of
    perturbation-energy-matched strength.

Key fairness fix (postmortem 12.7):
    4/e is gated — on clean regions sigma_effective = 0. Matching three
    methods on nominal sigma would let 4/e win for free in low-residue
    areas. Instead, match on actual perturbation energy:
        E_pert = mean((after - before) ** 2)
    Run 4/e first, measure its E_pert, then size gaussian/uniform's
    sigma_eq = sqrt(E_pert) so all three methods inject the same total
    energy.

Falsification (pre-registered):
    Claim PASSES iff at every strength s in {0.3, 0.5, 0.8}:
        score_gain = artifact_score(4/e) - max(score_gauss, score_uni)
        score_gain >= +0.05
      AND
        |edge_ratio(4/e) - 1.0| <= |edge_ratio(best_other) - 1.0| + 0.05
    Multi-seed std must also be reported (postmortem 12.4):
        each effect must be >= 3 * baseline_std to count as signal.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

sys.path.insert(0, "D:/VOD/prototype")

from vod_minimal.artifacts import oc_four_over_e, tile_residue
from vod_minimal.blocky_scattering import (
    build_blocky_scattering_batch,
    inject_axial_contour,
)
from vod_minimal.core import build_projection_batch, ProjectionBatch, ProjectionSample
from vod_minimal.metrics import artifact_metrics


# --------------------------------------------------------------------------- #
#  Suppressors (gaussian / uniform / 4/e). All take view + rng + sigma.
# --------------------------------------------------------------------------- #

def supp_none(view, rng, sigma, tile):
    return view.astype(np.float64).copy()


def supp_gaussian(view, rng, sigma, tile):
    return view.astype(np.float64) + rng.normal(0.0, sigma, view.shape)


def supp_uniform(view, rng, sigma, tile):
    # uniform variance-matched: U(-σ√3, +σ√3) has variance σ²
    half = sigma * np.sqrt(3.0)
    return view.astype(np.float64) + rng.uniform(-half, half, view.shape)


def supp_fourover_e(view, rng, sigma, tile):
    return oc_four_over_e(view, rng, beta=sigma, tile=tile)


# --------------------------------------------------------------------------- #
#  Metrics
# --------------------------------------------------------------------------- #

def axial_covariance(perturbation: np.ndarray, *, max_shift: int = 8) -> dict:
    """Spec 7.5 covariance signature: a non-iid 4/e perturbation must have
    non-zero covariance along the four projection axes. iid gaussian has
    zero covariance for any non-zero shift.

    Returns mean |Cov| over `k=1..max_shift` for each of the four axes.
    """
    arr = perturbation.astype(np.float64)
    if arr.ndim < 2:
        return {"vert": 0.0, "horiz": 0.0, "diag1": 0.0, "diag2": 0.0}
    if arr.ndim == 3:
        arr = arr.mean(axis=0)
    H, W = arr.shape
    arr_centered = arr - arr.mean()
    var = float(arr_centered.var())
    if var < 1e-12:
        return {"vert": 0.0, "horiz": 0.0, "diag1": 0.0, "diag2": 0.0}

    def cov_at(dy, dx):
        if dy >= H or dx >= W:
            return 0.0
        a = arr_centered[: H - dy, : W - dx] if dx >= 0 else arr_centered[: H - dy, -dx:]
        if dx >= 0:
            b = arr_centered[dy:, dx:]
        else:
            b = arr_centered[dy:, : W + dx]
        return float(np.mean(a * b))

    out = {"vert": 0.0, "horiz": 0.0, "diag1": 0.0, "diag2": 0.0}
    n = 0
    for k in range(1, min(max_shift, min(H, W) - 1) + 1):
        out["horiz"] += abs(cov_at(0, k))     # row coherence (along x)
        out["vert"]  += abs(cov_at(k, 0))     # column coherence (along y)
        out["diag1"] += abs(cov_at(k, k))     # primary diagonal i+j
        out["diag2"] += abs(cov_at(k, -k))    # secondary diagonal i-j
        n += 1
    if n > 0:
        for key in out:
            out[key] /= n
    # Normalise by the perturbation's own variance so values are
    # comparable across different sigma magnitudes. Result ∈ [0, 1].
    for key in out:
        out[key] = out[key] / var
    return out


def edge_energy(view: np.ndarray) -> float:
    """Sobel-like (L1) edge magnitude on last 2 spatial axes.

    For (H, W) returns mean(|dy|) + mean(|dx|).
    For (T, H, W) averages over time first.
    """
    arr = view.astype(np.float64)
    if arr.ndim < 2:
        return 0.0
    if arr.ndim == 3:
        arr = arr.mean(axis=0)
    dy = np.abs(np.diff(arr, axis=0))
    dx = np.abs(np.diff(arr, axis=1))
    return float(dy.mean() + dx.mean())


def measure(suppressed, blocky, clean, *, medium, tile) -> dict:
    """Per-view metrics under one suppressor."""
    if medium == "image":
        am = artifact_metrics({"image": suppressed}, tile=tile)
    else:
        am = artifact_metrics({"video": suppressed}, tile=tile)
    e_after = edge_energy(suppressed)
    e_before = edge_energy(blocky)
    edge_ratio = e_after / e_before if e_before > 1e-9 else 1.0

    # Spec 7.5 covariance signature: measure axial coherence of the
    # PERTURBATION (suppressed - blocky), NOT of the suppressed image.
    # 4/e's claim is that its perturbation has non-zero 4-axis cov.
    perturbation = suppressed - blocky
    axcov = axial_covariance(perturbation)
    axial_total = sum(axcov.values()) / 4.0  # average over 4 axes

    return {
        "artifact_score": am["artifact_score"],
        "mean_tile_residue": am["mean_tile_residue"],
        "max_tile_residue": am["max_tile_residue"],
        "l2_vs_clean": float(np.mean((suppressed - clean) ** 2)),
        "edge_ratio": edge_ratio,
        "perturb_energy": float(np.mean((suppressed - blocky) ** 2)),
        "axcov_vert": axcov["vert"],
        "axcov_horiz": axcov["horiz"],
        "axcov_diag1": axcov["diag1"],
        "axcov_diag2": axcov["diag2"],
        "axcov_mean": axial_total,
    }


# --------------------------------------------------------------------------- #
#  Single-seed run: 4 suppressors x batch x media
# --------------------------------------------------------------------------- #

def _build_axial_contour_batch(seed, batch_size, size, frames, strength, tile):
    """Build a stress batch where the corruption pattern is axial-contour
    (sharp jumps along 4 canonical directions on tile boundaries). This
    is the spec-aligned stress mode for testing 4/e — the failure mode
    4/e was designed against."""
    base = build_projection_batch(
        np.random.default_rng(seed), batch_size=batch_size, size=size,
        frames=frames, spacetime=True,
    )
    rng = np.random.default_rng(seed)
    new_samples = []
    for s in base.samples:
        new_noisy = dict(s.noisy_views)
        for medium in ("image", "video"):
            if medium in new_noisy:
                new_noisy[medium] = inject_axial_contour(
                    new_noisy[medium], rng, tile=tile, strength=strength
                )
        new_samples.append(
            ProjectionSample(
                source_field=s.source_field,
                target_field=s.target_field,
                noisy_views=new_noisy,
                target_views=dict(s.target_views),
                source_spacetime_field=s.source_spacetime_field,
                target_spacetime_field=s.target_spacetime_field,
            )
        )
    return ProjectionBatch(samples=tuple(new_samples), media=base.media)


def run_one_seed(seed, strength, sigma, *, batch_size, size, frames, tile,
                 stress_mode="blocky"):
    """Returns {method: {medium: list[per-view-metrics]}}.

    stress_mode:
      'blocky' = original blocky_scattering_mask (random hash + boundary
                 noise; not aligned with 4/e's 4-axis design)
      'axial'  = axial_contour_mask (sharp jumps along 4 canonical axes
                 on tile boundaries; spec-aligned stress for 4/e)
    """
    clean = build_projection_batch(
        np.random.default_rng(seed), batch_size=batch_size, size=size,
        frames=frames, spacetime=True
    )
    if stress_mode == "axial":
        blocky = _build_axial_contour_batch(
            seed, batch_size, size, frames, strength, tile,
        )
    else:
        blocky = build_blocky_scattering_batch(
            np.random.default_rng(seed), batch_size=batch_size, size=size,
            frames=frames, artifact_strength=strength, tile=tile,
            spacetime=True, temporal_mode="static",
        )

    results = {"none": {"image": [], "video": []},
               "gaussian": {"image": [], "video": []},
               "uniform": {"image": [], "video": []},
               "fourover_e": {"image": [], "video": []}}

    for idx, (ci, bi) in enumerate(zip(clean.samples, blocky.samples)):
        for medium in ("image", "video"):
            clean_v = ci.target_views[medium]
            blocky_v = bi.noisy_views[medium]

            # Run 4/e FIRST and measure actual perturbation energy.
            # Then size gaussian/uniform sigma_eq = sqrt(E_pert) so the
            # three methods inject the same total energy.
            rng_fe = np.random.default_rng(seed + 1000 + idx)
            fe_out = supp_fourover_e(blocky_v, rng_fe, sigma, tile)
            e_fe = float(np.mean((fe_out - blocky_v) ** 2))
            # Fairness: if 4/e didn't fire (gated → e_fe == 0), gaussian
            # / uniform must also not fire. Using nominal sigma here
            # would let baselines burn extra energy that 4/e didn't.
            sigma_eq = float(np.sqrt(e_fe))

            rng_g = np.random.default_rng(seed + 2000 + idx)
            rng_u = np.random.default_rng(seed + 3000 + idx)

            results["none"][medium].append(
                measure(blocky_v, blocky_v, clean_v, medium=medium, tile=tile)
            )
            results["gaussian"][medium].append(
                measure(supp_gaussian(blocky_v, rng_g, sigma_eq, tile),
                        blocky_v, clean_v, medium=medium, tile=tile)
            )
            results["uniform"][medium].append(
                measure(supp_uniform(blocky_v, rng_u, sigma_eq, tile),
                        blocky_v, clean_v, medium=medium, tile=tile)
            )
            results["fourover_e"][medium].append(
                measure(fe_out, blocky_v, clean_v, medium=medium, tile=tile)
            )

    return results


def aggregate_method(per_view_list, key):
    return float(np.mean([m[key] for m in per_view_list]))


def std_method(per_view_list, key):
    return float(np.std([m[key] for m in per_view_list], ddof=0))


# --------------------------------------------------------------------------- #
#  Strength-level verdict
# --------------------------------------------------------------------------- #

def evaluate_strength(seed_runs, *, score_threshold=0.05, edge_threshold=0.05,
                      sigma_multiplier=3.0):
    """Aggregate over seeds, output per-medium verdict per strength."""
    out = {}
    for medium in ("image", "video"):
        # Collect per-view metrics across all seeds for each method.
        flat = {m: [] for m in ("none", "gaussian", "uniform", "fourover_e")}
        for run in seed_runs:
            for m in flat:
                flat[m].extend(run[m][medium])

        agg = {m: {k: aggregate_method(flat[m], k)
                   for k in ("artifact_score", "mean_tile_residue",
                             "max_tile_residue", "l2_vs_clean", "edge_ratio",
                             "perturb_energy", "axcov_vert", "axcov_horiz",
                             "axcov_diag1", "axcov_diag2", "axcov_mean")}
               for m in flat}
        std = {m: {k: std_method(flat[m], k)
                   for k in ("artifact_score", "edge_ratio", "axcov_mean")}
               for m in flat}

        # Falsification.
        fe_score = agg["fourover_e"]["artifact_score"]
        best_other_score = max(agg["gaussian"]["artifact_score"],
                               agg["uniform"]["artifact_score"])
        score_gain = fe_score - best_other_score

        fe_edge_dev = abs(agg["fourover_e"]["edge_ratio"] - 1.0)
        others_edge_dev = [abs(agg["gaussian"]["edge_ratio"] - 1.0),
                            abs(agg["uniform"]["edge_ratio"] - 1.0)]
        best_other_edge_dev = min(others_edge_dev)

        # Std-based effect-size check (postmortem 12.4).
        baseline_std = max(std["gaussian"]["artifact_score"],
                           std["uniform"]["artifact_score"])
        effect_significant = score_gain >= sigma_multiplier * baseline_std

        score_pass = score_gain >= score_threshold
        edge_pass = fe_edge_dev <= best_other_edge_dev + edge_threshold

        verdict = "PASS" if (score_pass and edge_pass and effect_significant) else "FAIL"

        out[medium] = {
            "agg": agg, "std": std,
            "score_gain": score_gain,
            "score_threshold": score_threshold,
            "score_pass": score_pass,
            "fe_edge_dev": fe_edge_dev,
            "best_other_edge_dev": best_other_edge_dev,
            "edge_pass": edge_pass,
            "baseline_std": baseline_std,
            "effect_significant": effect_significant,
            "verdict": verdict,
        }
    return out


# --------------------------------------------------------------------------- #
#  Reporting
# --------------------------------------------------------------------------- #

def print_strength_report(strength, results):
    print(f"\n=== strength = {strength} ===")
    for medium in ("image", "video"):
        r = results[medium]
        print(f"\n  [{medium}]")
        for m in ("none", "gaussian", "uniform", "fourover_e"):
            a = r["agg"][m]
            print(f"    {m:<11}  artifact_score={a['artifact_score']:.4f}  "
                  f"mean_residue={a['mean_tile_residue']:.4f}  "
                  f"perturb_E={a['perturb_energy']:.4f}  "
                  f"axcov_mean={a['axcov_mean']:.4f}")
            print(f"                axcov per-axis: "
                  f"v={a['axcov_vert']:.3f}  h={a['axcov_horiz']:.3f}  "
                  f"d1={a['axcov_diag1']:.3f}  d2={a['axcov_diag2']:.3f}")
        print(f"    score_gain (4/e - best_other) = {r['score_gain']:+.4f}    "
              f"threshold = {r['score_threshold']:.4f}    "
              f"3-sigma required = {3 * r['baseline_std']:.4f}")
        print(f"    edge_dev (4/e) = {r['fe_edge_dev']:.4f}    "
              f"best_other_edge_dev = {r['best_other_edge_dev']:.4f}")
        # Spec 7.5 covariance signature comparison
        fe_axcov = r["agg"]["fourover_e"]["axcov_mean"]
        gauss_axcov = r["agg"]["gaussian"]["axcov_mean"]
        unif_axcov = r["agg"]["uniform"]["axcov_mean"]
        axcov_gain = fe_axcov - max(gauss_axcov, unif_axcov)
        axcov_std = r["std"]["gaussian"]["axcov_mean"]
        print(f"    AXCOV signature: 4/e={fe_axcov:.4f}  gauss={gauss_axcov:.4f}  "
              f"unif={unif_axcov:.4f}  gain={axcov_gain:+.4f}  3sig={3*axcov_std:.4f}")
        print(f"    score_pass={r['score_pass']}   "
              f"edge_pass={r['edge_pass']}   "
              f"effect_sig={r['effect_significant']}   "
              f"--> artifact_score verdict: {r['verdict']}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--size", type=int, default=32)
    p.add_argument("--frames", type=int, default=8)
    p.add_argument("--tile", type=int, default=4)
    p.add_argument("--strengths", type=float, nargs="+",
                   default=[0.3, 0.5, 0.8])
    p.add_argument("--sigma", type=float, default=0.24,
                   help="nominal sigma for 4/e (gaussian/uniform get matched)")
    p.add_argument("--score-threshold", type=float, default=0.05)
    p.add_argument("--edge-threshold", type=float, default=0.05)
    p.add_argument("--sigma-multiplier", type=float, default=3.0,
                   help="effect must be ≥ N × baseline_std to count as signal")
    p.add_argument("--stress-mode", choices=("blocky", "axial"), default="blocky",
                   help="blocky = random hash mask; axial = spec-aligned 4-axis "
                        "contour mask (designed to exercise 4/e's target failure mode)")
    args = p.parse_args()

    print("Claim 1 - 4/e Orthogonal Compression Decay ablation")
    print(f"seeds={args.seeds}, batch={args.batch_size}, size={args.size}, "
          f"frames={args.frames}, tile={args.tile}, sigma={args.sigma}, "
          f"stress_mode={args.stress_mode}")
    print(f"strengths = {args.strengths}")
    print(f"falsification: score_gain >= {args.score_threshold}  AND  "
          f"edge_dev within +{args.edge_threshold}  AND  "
          f"effect >= {args.sigma_multiplier}*baseline_std  per strength per medium")

    overall = {"image": [], "video": []}

    for strength in args.strengths:
        seed_runs = []
        for seed_i in range(args.seeds):
            seed_runs.append(run_one_seed(
                seed=430 + seed_i, strength=strength, sigma=args.sigma,
                batch_size=args.batch_size, size=args.size, frames=args.frames,
                tile=args.tile, stress_mode=args.stress_mode,
            ))
        r = evaluate_strength(
            seed_runs, score_threshold=args.score_threshold,
            edge_threshold=args.edge_threshold,
            sigma_multiplier=args.sigma_multiplier,
        )
        print_strength_report(strength, r)
        for medium in ("image", "video"):
            overall[medium].append((strength, r[medium]["verdict"]))

    print("\n" + "=" * 60)
    print("FINAL VERDICT (all strengths must PASS for the claim to PASS)")
    print("=" * 60)
    final_pass = True
    for medium in ("image", "video"):
        verdicts = overall[medium]
        all_pass = all(v == "PASS" for _, v in verdicts)
        final_pass = final_pass and all_pass
        per_str = " | ".join(f"s={s}: {v}" for s, v in verdicts)
        tag = "PASS" if all_pass else "FAIL"
        print(f"  [{medium}]  {per_str}   →   {tag}")
    print()
    print(f"Claim 1: {'PASS' if final_pass else 'FAIL'}")


if __name__ == "__main__":
    main()
