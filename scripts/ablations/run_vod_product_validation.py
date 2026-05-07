"""VOD minimal product validation — single-script closed loop.

Runs all 6 product validations in one pass and emits:
    prototype/vod_product_validation_result.json   (machine-readable)
    prototype/vod_product_validation_report.md     (human-readable)

Sections (all PASS criteria pre-registered, NO post-hoc threshold tuning):

    2.1 Baseline Recovery        — model_error < noisy AND <= 0.95 * zero
    2.2 4/e Phase-Break          — sign_agreement(4/e) <= sign_agreement(iid_best) - 0.02
    2.3 Binary-Twin Gradient     — BT update improves symbol_accuracy more than MSE
    2.4 TPSR/AIMP Consistency    — consistent score > corrupted > random
    2.5 TTNM Flicker Diagnostic  — temporal_smoothness(flicker) > clean
    2.6 MSN Monitoring           — report only, no PASS/FAIL

Reference instructions: this script realises the spec laid out in
the 2026-05-01 Codex+user product roadmap. All metrics correspond to
their spec section in vod_full_mathematical_formulation.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path("D:/VOD/prototype")
sys.path.insert(0, str(PROJECT_ROOT))

from vod_minimal.aimp import (
    aimp_tpsr_metrics,
    synthesize_tpsr_measurements,
    TPSRMeasurement,
)
from vod_minimal.artifacts import oc_four_over_e
from vod_minimal.binary_twin import (
    binary_twin_torch_loss,
    encode_symbols,
    symbol_accuracy,
)
from vod_minimal.blocky_scattering import inject_temporal_flicker
from vod_minimal.chladni import random_chladni_field
from vod_minimal.experiment import evaluate_model, make_sample
from vod_minimal.metrics import temporal_smoothness
from vod_minimal.model import MinimalVOD


# --------------------------------------------------------------------------- #
#  Common helpers
# --------------------------------------------------------------------------- #

def perturbation_energy(after: np.ndarray, before: np.ndarray) -> float:
    return float(np.mean((after.astype(np.float64) - before.astype(np.float64)) ** 2))


def coherent_tile_light_spot(h: int, w: int, *, tile: int = 8, strength: float = 0.5) -> np.ndarray:
    """1-pixel-wide bright halo at every tile boundary; sign-consistent jumps."""
    out = np.zeros((h, w), dtype=np.float64)
    for i in range(tile - 1, h, tile):
        out[i, :] += strength
    for j in range(tile - 1, w, tile):
        out[:, j] += strength
    return out


def boundary_sign_agreement(arr: np.ndarray, *, tile: int = 8, eps: float = 1e-9) -> float:
    a = np.asarray(arr, dtype=np.float64)
    while a.ndim > 2:
        a = a.mean(axis=0)
    H, W = a.shape
    if min(H, W) <= tile:
        return 0.5
    fractions = []
    for j in range(tile - 1, W - 1, tile):
        signed = a[:, j + 1] - a[:, j]
        sig = signed[np.abs(signed) > eps]
        if sig.size < 2:
            continue
        signs = np.sign(sig)
        fractions.append(float((signs[:-1] == signs[1:]).mean()))
    for i in range(tile - 1, H - 1, tile):
        signed = a[i + 1, :] - a[i, :]
        sig = signed[np.abs(signed) > eps]
        if sig.size < 2:
            continue
        signs = np.sign(sig)
        fractions.append(float((signs[:-1] == signs[1:]).mean()))
    if not fractions:
        return 0.5
    return float(np.mean(fractions))


# --------------------------------------------------------------------------- #
#  2.1 Baseline Recovery
# --------------------------------------------------------------------------- #

def run_baseline_recovery(*, n_samples: int = 64, seed: int = 430) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    samples = [make_sample(rng) for _ in range(n_samples)]

    # Use default MinimalVOD config (the prototype's reference update rule).
    model = MinimalVOD()

    noisy_errors = []
    model_errors = []
    zero_errors = []

    from vod_minimal.metrics import mean_target_error

    for sample in samples:
        # Noisy baseline: error of the noisy input vs target
        noisy_errors.append(mean_target_error(sample.noisy_views, sample.target_views))

        # Zero baseline: pred = zeros (worst informative baseline)
        zero_views = {name: np.zeros_like(v) for name, v in sample.noisy_views.items()}
        zero_errors.append(mean_target_error(zero_views, sample.target_views))

        # Model
        denoised, _paths = model.denoise_views(sample.noisy_views, sample.target_views)
        model_errors.append(mean_target_error(denoised, sample.target_views))

    n_arr = np.array(noisy_errors)
    m_arr = np.array(model_errors)
    z_arr = np.array(zero_errors)

    success_rate = float(np.mean(m_arr < n_arr))
    pass_vs_noisy = float(m_arr.mean()) < float(n_arr.mean())
    pass_vs_zero = float(m_arr.mean()) <= 0.95 * float(z_arr.mean())
    verdict = pass_vs_noisy and pass_vs_zero

    return {
        "n_samples": n_samples,
        "noisy_baseline_mean_error": float(n_arr.mean()),
        "zero_baseline_mean_error": float(z_arr.mean()),
        "model_mean_error": float(m_arr.mean()),
        "improvement_vs_noisy": float(n_arr.mean() - m_arr.mean()),
        "improvement_vs_zero": float(z_arr.mean() - m_arr.mean()),
        "success_rate": success_rate,
        "pass_vs_noisy": pass_vs_noisy,
        "pass_vs_zero_threshold_0.95": pass_vs_zero,
        "verdict": "PASS" if verdict else "FAIL",
    }


# --------------------------------------------------------------------------- #
#  2.2 4/e Phase-Break
# --------------------------------------------------------------------------- #

def run_phase_break(*, n_seeds: int = 100, size: int = 64, tile: int = 8,
                     strength: float = 0.3, threshold: float = 0.02) -> dict[str, Any]:
    diffs = []
    means = {"none": [], "gaussian": [], "uniform": [], "fourover": []}
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        X_0 = random_chladni_field(rng, size=size, n_modes=3)
        P = coherent_tile_light_spot(size, size, tile=tile, strength=1.0)
        X = X_0 + strength * P

        rng_4e = np.random.default_rng(seed + 200_000)
        Y_4e = oc_four_over_e(X, rng_4e, beta=strength, tile=tile)
        E = perturbation_energy(Y_4e, X)
        sigma_eq = float(np.sqrt(E))

        rng_g = np.random.default_rng(seed + 300_000)
        Y_g = X + rng_g.normal(0.0, sigma_eq, X.shape)
        rng_u = np.random.default_rng(seed + 400_000)
        Y_u = X + rng_u.uniform(-sigma_eq * np.sqrt(3.0), sigma_eq * np.sqrt(3.0), X.shape)

        sa_4e = boundary_sign_agreement(Y_4e, tile=tile)
        sa_g  = boundary_sign_agreement(Y_g,  tile=tile)
        sa_u  = boundary_sign_agreement(Y_u,  tile=tile)
        sa_n  = boundary_sign_agreement(X,    tile=tile)

        means["none"].append(sa_n)
        means["gaussian"].append(sa_g)
        means["uniform"].append(sa_u)
        means["fourover"].append(sa_4e)
        diffs.append(min(sa_g, sa_u) - sa_4e)

    diffs_arr = np.array(diffs)
    paired_mean = float(diffs_arr.mean())
    paired_std = float(diffs_arr.std(ddof=1))
    se = paired_std / np.sqrt(n_seeds)
    three_sig = 3.0 * se

    score_pass = paired_mean >= threshold
    sig_pass = abs(paired_mean) >= three_sig
    verdict = score_pass and sig_pass

    return {
        "n_seeds": n_seeds,
        "strength": strength,
        "tile": tile,
        "threshold": threshold,
        "sign_agreement_baseline_X": float(np.mean(means["none"])),
        "sign_agreement_gaussian": float(np.mean(means["gaussian"])),
        "sign_agreement_uniform": float(np.mean(means["uniform"])),
        "sign_agreement_fourover": float(np.mean(means["fourover"])),
        "paired_diff_mean": paired_mean,
        "paired_diff_std": paired_std,
        "three_sigma_SE": three_sig,
        "score_pass": score_pass,
        "sig_pass": sig_pass,
        "verdict": "PASS" if verdict else "FAIL",
    }


# --------------------------------------------------------------------------- #
#  2.3 Binary-Twin one-step gradient direction
# --------------------------------------------------------------------------- #

def run_binary_twin_gradient(*, n_seeds: int = 50, length: int = 32,
                              levels: int = 16, lr: float = 0.05) -> dict[str, Any]:
    """One-step gradient direction test: BT vs MSE-only on quantized text.

    Pre-registered (per the product roadmap):
      BT update improves symbol_accuracy or symbol_hamming more than
      MSE-only WITHOUT increasing continuous_mse-to-target beyond a
      registered tolerance (BT must not drift continuous values away
      from the underlying clean target while chasing symbols).

    Operational metric for "continuous_mse": MSE(pred_after, target) —
    distance from target, not displacement from corrupted input. A
    BT update that pushes pred toward the correct quantized symbol
    necessarily moves pred CLOSER to target in continuous space too,
    so this metric is correctly anchored to the application goal.
    """
    bt_acc_gain = []
    mse_acc_gain = []
    bt_to_target = []
    mse_to_target = []
    corrupt_to_target = []

    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        target_symbols = rng.integers(0, levels, size=length)
        target = target_symbols.astype(np.float64) / float(levels - 1)

        corruption = rng.normal(0.0, 0.4 / (levels - 1), size=length)
        corrupt = np.clip(target + corruption, 0.0, 1.0)

        baseline_acc = symbol_accuracy(corrupt, target, levels=levels)
        corrupt_to_target.append(float(np.mean((corrupt - target) ** 2)))

        pred_mse = torch.tensor(corrupt, dtype=torch.float32, requires_grad=True)
        target_t = torch.tensor(target, dtype=torch.float32)
        loss_mse = F.mse_loss(pred_mse, target_t)
        grad_mse, = torch.autograd.grad(loss_mse, pred_mse)
        new_mse = (pred_mse.detach() - lr * grad_mse).clamp(0.0, 1.0).numpy().astype(np.float64)

        pred_bt = torch.tensor(corrupt, dtype=torch.float32, requires_grad=True)
        loss_bt = binary_twin_torch_loss(pred_bt, target_t, levels=levels)
        grad_bt, = torch.autograd.grad(loss_bt, pred_bt)
        new_bt = (pred_bt.detach() - lr * grad_bt).clamp(0.0, 1.0).numpy().astype(np.float64)

        acc_mse = symbol_accuracy(new_mse, target, levels=levels)
        acc_bt = symbol_accuracy(new_bt, target, levels=levels)

        bt_acc_gain.append(acc_bt - baseline_acc)
        mse_acc_gain.append(acc_mse - baseline_acc)
        bt_to_target.append(float(np.mean((new_bt - target) ** 2)))
        mse_to_target.append(float(np.mean((new_mse - target) ** 2)))

    bt_arr = np.array(bt_acc_gain)
    mse_arr = np.array(mse_acc_gain)
    paired = bt_arr - mse_arr
    paired_mean = float(paired.mean())
    paired_std = float(paired.std(ddof=1))
    se = paired_std / np.sqrt(n_seeds) if n_seeds > 1 else 0.0
    three_sig = 3.0 * se

    # Tolerance: BT distance-to-target <= 1.5 * MSE distance-to-target.
    # Pure MSE on this single step is approximately optimal for L2-to-target,
    # so 1.5x is generous but still rules out catastrophic drift.
    tolerance = 1.5
    bt_to_target_mean = float(np.mean(bt_to_target))
    mse_to_target_mean = float(np.mean(mse_to_target))
    bt_continuous_ok = bt_to_target_mean <= tolerance * mse_to_target_mean
    score_pass = paired_mean > 0
    sig_pass = abs(paired_mean) >= three_sig
    verdict = score_pass and sig_pass and bt_continuous_ok

    return {
        "n_seeds": n_seeds,
        "length": length,
        "levels": levels,
        "lr": lr,
        "corrupt_to_target_mean": float(np.mean(corrupt_to_target)),
        "mse_only_acc_gain_mean": float(mse_arr.mean()),
        "binary_twin_acc_gain_mean": float(bt_arr.mean()),
        "paired_diff_BT_minus_MSE_mean": paired_mean,
        "paired_diff_std": paired_std,
        "three_sigma_SE": three_sig,
        "binary_twin_distance_to_target_mean": bt_to_target_mean,
        "mse_only_distance_to_target_mean": mse_to_target_mean,
        "continuous_tolerance_factor": tolerance,
        "continuous_within_tolerance": bool(bt_continuous_ok),
        "score_pass": score_pass,
        "sig_pass": sig_pass,
        "verdict": "PASS" if verdict else "FAIL",
    }


# --------------------------------------------------------------------------- #
#  2.4 TPSR/AIMP Physical Consistency
# --------------------------------------------------------------------------- #

def run_tpsr_consistency(*, n_seeds: int = 50, n_frames: int = 6,
                          brightness_error: float = 2.0) -> dict[str, Any]:
    consistent_scores = []
    corrupted_scores = []
    random_scores = []
    consistent_kcv = []
    corrupted_kcv = []

    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        distances = np.linspace(1.0, 2.5, n_frames)

        consistent = synthesize_tpsr_measurements(distances, brightness_error=1.0)
        corrupted  = synthesize_tpsr_measurements(distances, brightness_error=brightness_error)

        random_meas = [
            TPSRMeasurement(
                highlight_energy=float(rng.uniform(0.05, 5.0)),
                highlight_area=float(rng.uniform(5.0, 80.0)),
                light_diopter=1.0,
                gamma=4.0,
            )
            for _ in range(n_frames)
        ]

        m_consistent = aimp_tpsr_metrics(consistent)
        m_corrupted = aimp_tpsr_metrics(corrupted)
        m_random = aimp_tpsr_metrics(random_meas)

        consistent_scores.append(m_consistent["tpsr_consistency_score"])
        corrupted_scores.append(m_corrupted["tpsr_consistency_score"])
        random_scores.append(m_random["tpsr_consistency_score"])
        consistent_kcv.append(m_consistent["tpsr_k_cv"])
        corrupted_kcv.append(m_corrupted["tpsr_k_cv"])

    cs_arr = np.array(consistent_scores)
    co_arr = np.array(corrupted_scores)
    ra_arr = np.array(random_scores)

    pass_cons_vs_corr = bool(cs_arr.mean() > co_arr.mean())
    pass_cons_vs_rand = bool(cs_arr.mean() > ra_arr.mean())
    pass_kcv = bool(np.mean(consistent_kcv) < np.mean(corrupted_kcv))
    verdict = pass_cons_vs_corr and pass_cons_vs_rand and pass_kcv

    return {
        "n_seeds": n_seeds,
        "n_frames": n_frames,
        "brightness_error_for_corrupted": brightness_error,
        "consistent_score_mean": float(cs_arr.mean()),
        "corrupted_score_mean": float(co_arr.mean()),
        "random_score_mean": float(ra_arr.mean()),
        "consistent_K_cv_mean": float(np.mean(consistent_kcv)),
        "corrupted_K_cv_mean": float(np.mean(corrupted_kcv)),
        "pass_consistent_gt_corrupted": pass_cons_vs_corr,
        "pass_consistent_gt_random": pass_cons_vs_rand,
        "pass_Kcv_consistent_lt_corrupted": pass_kcv,
        "verdict": "PASS" if verdict else "FAIL",
    }


# --------------------------------------------------------------------------- #
#  2.5 TTNM Flicker Diagnostic
# --------------------------------------------------------------------------- #

def build_clean_video(rng: np.random.Generator, *, size: int = 32, frames: int = 8) -> np.ndarray:
    a = random_chladni_field(rng, size=size, n_modes=3)
    b = random_chladni_field(rng, size=size, n_modes=3)
    out = np.empty((frames, size, size), dtype=np.float64)
    for t in range(frames):
        alpha = 0.5 - 0.5 * np.cos(np.pi * t / max(frames - 1, 1))
        out[t] = (1 - alpha) * a + alpha * b
    return out


def run_ttnm_flicker(*, n_seeds: int = 100, size: int = 32, frames: int = 8,
                      flicker_strength: float = 0.3) -> dict[str, Any]:
    diffs = []
    sign_positive = 0
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        clean = build_clean_video(rng, size=size, frames=frames)
        flicker_rng = np.random.default_rng(seed + 1_000_000)
        flicker = inject_temporal_flicker(clean, flicker_rng, strength=flicker_strength)
        d = float(temporal_smoothness(flicker) - temporal_smoothness(clean))
        diffs.append(d)
        if d > 0:
            sign_positive += 1

    diffs_arr = np.array(diffs)
    mean_d = float(diffs_arr.mean())
    std_d = float(diffs_arr.std(ddof=1))
    se_d = std_d / np.sqrt(n_seeds)
    three_sig = 3.0 * se_d
    sign_frac = sign_positive / n_seeds

    score_pass = mean_d > 0 and sign_frac >= 0.95
    sig_pass = mean_d >= three_sig
    verdict = score_pass and sig_pass

    return {
        "n_seeds": n_seeds,
        "flicker_strength": flicker_strength,
        "paired_diff_mean": mean_d,
        "paired_diff_std": std_d,
        "three_sigma_SE": three_sig,
        "sign_positive_fraction": sign_frac,
        "score_pass": score_pass,
        "sig_pass": sig_pass,
        "verdict": "PASS" if verdict else "FAIL",
    }


# --------------------------------------------------------------------------- #
#  2.6 MSN Monitoring (no PASS/FAIL)
# --------------------------------------------------------------------------- #

def run_msn_monitoring(*, n_configs: int = 30, n_samples: int = 30,
                        seed: int = 430) -> dict[str, Any]:
    sample_rng = np.random.default_rng(seed + 1)
    samples = [make_sample(sample_rng) for _ in range(n_samples)]

    config_rng = np.random.default_rng(seed + 2)
    msn_vals = []
    err_vals = []
    for _ in range(n_configs):
        cfg = MinimalVOD(
            diffusion=float(config_rng.uniform(0.05, 1.5)),
            reaction=float(config_rng.uniform(0.02, 0.8)),
            step_size=float(config_rng.uniform(0.1, 1.5)),
            steps=int(config_rng.integers(4, 25)),
        )
        m = evaluate_model(cfg, samples)
        msn_vals.append(m["mean_msn"])
        err_vals.append(m["mean_after"])

    msn_arr = np.array(msn_vals)
    err_arr = np.array(err_vals)

    n = len(msn_arr)
    if n >= 3:
        xm = msn_arr - msn_arr.mean()
        ym = err_arr - err_arr.mean()
        denom = float(np.sqrt((xm * xm).sum() * (ym * ym).sum()))
        r = float((xm * ym).sum() / denom) if denom > 1e-12 else 0.0
    else:
        r = float("nan")

    return {
        "n_configs": n_configs,
        "n_samples": n_samples,
        "msn_min": float(msn_arr.min()),
        "msn_max": float(msn_arr.max()),
        "msn_mean": float(msn_arr.mean()),
        "error_min": float(err_arr.min()),
        "error_max": float(err_arr.max()),
        "error_mean": float(err_arr.mean()),
        "pearson_r_msn_error": r,
        "verdict": "MONITORING_ONLY",
        "note": "Path-stability diagnostic only. Not a model-selection hard criterion.",
    }


# --------------------------------------------------------------------------- #
#  Orchestration
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT))
    parser.add_argument("--baseline-n", type=int, default=64)
    parser.add_argument("--phase-break-seeds", type=int, default=100)
    parser.add_argument("--bt-seeds", type=int, default=50)
    parser.add_argument("--tpsr-seeds", type=int, default=50)
    parser.add_argument("--ttnm-seeds", type=int, default=100)
    parser.add_argument("--msn-configs", type=int, default=30)
    args = parser.parse_args()

    print("=" * 60)
    print("VOD Minimal Product Validation")
    print("=" * 60)

    results: dict[str, Any] = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
    }

    print("\n[2.1] Baseline Recovery ...")
    results["baseline_recovery"] = run_baseline_recovery(n_samples=args.baseline_n)
    print(f"     verdict: {results['baseline_recovery']['verdict']}")

    print("\n[2.2] 4/e Phase-Break ...")
    results["four_over_e"] = run_phase_break(n_seeds=args.phase_break_seeds)
    print(f"     verdict: {results['four_over_e']['verdict']}")

    print("\n[2.3] Binary-Twin Gradient Direction ...")
    results["binary_twin"] = run_binary_twin_gradient(n_seeds=args.bt_seeds)
    print(f"     verdict: {results['binary_twin']['verdict']}")

    print("\n[2.4] TPSR/AIMP Physical Consistency ...")
    results["tpsr_aimp"] = run_tpsr_consistency(n_seeds=args.tpsr_seeds)
    print(f"     verdict: {results['tpsr_aimp']['verdict']}")

    print("\n[2.5] TTNM Flicker Diagnostic ...")
    results["ttnm"] = run_ttnm_flicker(n_seeds=args.ttnm_seeds)
    print(f"     verdict: {results['ttnm']['verdict']}")

    print("\n[2.6] MSN Monitoring ...")
    results["msn"] = run_msn_monitoring(n_configs=args.msn_configs)
    print(f"     verdict: {results['msn']['verdict']}  (r={results['msn']['pearson_r_msn_error']:+.3f})")

    # Overall verdict — count hard PASS/FAIL only (MSN excluded)
    hard_sections = ["baseline_recovery", "four_over_e", "binary_twin", "tpsr_aimp", "ttnm"]
    hard_pass = sum(1 for k in hard_sections if results[k]["verdict"] == "PASS")
    overall = "PASS" if hard_pass == len(hard_sections) else f"PARTIAL ({hard_pass}/{len(hard_sections)})"
    results["overall_verdict"] = overall
    results["hard_sections_passed"] = hard_pass
    results["hard_sections_total"] = len(hard_sections)

    # Write JSON
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "vod_product_validation_result.json"

    def _coerce(o):
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"not JSON serializable: {type(o).__name__}")

    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=_coerce), encoding="utf-8")

    # Write Markdown report
    md_path = out_dir / "vod_product_validation_report.md"
    md_path.write_text(_format_report(results), encoding="utf-8")

    print()
    print("=" * 60)
    print(f"Overall: {overall}  (5 hard sections + 1 monitoring)")
    print(f"JSON:    {json_path}")
    print(f"Report:  {md_path}")
    print("=" * 60)


def _format_report(results: dict[str, Any]) -> str:
    def _coerce(o):
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"not JSON serializable: {type(o).__name__}")

    def v(section: str) -> str:
        return results[section]["verdict"]

    lines = [
        "# VOD Minimal Product Validation Report",
        "",
        f"- Date: {results['date']}",
        f"- Python: {results['python']}",
        f"- Torch: {results['torch']}",
        f"- NumPy: {results['numpy']}",
        f"- Overall: **{results['overall_verdict']}** ({results['hard_sections_passed']}/{results['hard_sections_total']} hard sections PASS)",
        "",
        "## Summary table",
        "",
        "| Section | Spec | Verdict | Notes |",
        "|---|---|---|---|",
        f"| 2.1 Baseline Recovery | denoising protocol | **{v('baseline_recovery')}** | model<noisy AND <=0.95×zero |",
        f"| 2.2 4/e Phase-Break | §7 (Operator) | **{v('four_over_e')}** | sign-agreement on coherent halo, s=0.3 |",
        f"| 2.3 Binary-Twin Gradient | §4 (Symbol coupling) | **{v('binary_twin')}** | minimal — one-step gradient direction test |",
        f"| 2.4 TPSR/AIMP Consistency | §8 (Physical card) | **{v('tpsr_aimp')}** | metric layer only — full AIMP controller NOT done |",
        f"| 2.5 TTNM Flicker | §6 (Temporal stability) | **{v('ttnm')}** | toy temporal_smoothness only — full graph NOT done |",
        f"| 2.6 MSN Monitoring | §5 (Path stability) | **{v('msn')}** | path-stability diagnostic only |",
        "",
    ]

    lines += ["## What passed", ""]
    for section in ("baseline_recovery", "four_over_e", "binary_twin", "tpsr_aimp", "ttnm"):
        if results[section]["verdict"] == "PASS":
            lines.append(f"- **{section}**: PASS")

    lines += ["", "## What failed", ""]
    failed_any = False
    for section in ("baseline_recovery", "four_over_e", "binary_twin", "tpsr_aimp", "ttnm"):
        if results[section]["verdict"] != "PASS":
            failed_any = True
            lines.append(f"- **{section}**: {results[section]['verdict']}")
    if not failed_any:
        lines.append("- (none)")

    lines += [
        "",
        "## What is only minimal (NOT full spec)",
        "",
        "- **Binary-Twin**: minimal CE+reconstruction coupling via `binary_twin_torch_loss`. "
        "Solves toy quantized-text channel only. Full OCR / logo grounding NOT implemented.",
        "- **TPSR/AIMP**: metric layer only (K, Uij, consistency_score). "
        "Full Field Card / Perspective Card scene controller NOT implemented.",
        "- **TTNM**: toy temporal_smoothness diagnostic. "
        "Full §6 tropical-graph soft-min update NOT implemented.",
        "- **MSN**: simplified path-stability diagnostic. "
        "Full §5 continuous + discrete + pair-coupling NOT implemented.",
        "",
        "## What is still NOT implemented",
        "",
        "- Full TTNM tropical graph (G_t = (N, E, W) with hard / soft tropical updates)",
        "- Full MSN three-layer normalization (continuous + discrete + pair coupling)",
        "- Full Binary-Twin OCR / logo / region-conflict resolver",
        "- Full AIMP controller (Field Card scheduler, perspective constraint, scene loop)",
        "- Linear regression calibration head",
        "- Full mode-eigenvalue regularizer",
        "",
        "## Next product step",
        "",
        "1. If section 2.1 PASS: VOD has a verified minimal denoising loop. "
        "   Next is to wire the four-PASS distinctives (4/e + Binary-Twin + TPSR + TTNM) into "
        "   the native_vod training pipeline as composite loss, then train one short run "
        "   on the Chladni dataset and verify the *training-time* effect of each layer is non-zero.",
        "2. If section 2.1 FAIL: do NOT add new mechanisms. First diagnose protocol / "
        "   data pairing / metric / leakage / implementation. Then re-run.",
        "3. Hypothesis validation plan in `docs/vod_hypothesis_validation_plan.md` covers "
        "   open hypotheses H0–H5 — pick one and execute its 'Next minimal implementation' line.",
        "",
        "## Detail blocks",
        "",
    ]

    for section in ("baseline_recovery", "four_over_e", "binary_twin", "tpsr_aimp", "ttnm", "msn"):
        lines.append(f"### {section}")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(results[section], indent=2, ensure_ascii=False, default=_coerce))
        lines.append("```")
        lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    main()
