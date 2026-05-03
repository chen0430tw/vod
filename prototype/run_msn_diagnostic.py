"""MSN application utility test.

Spec claim (vod_full_mathematical_formulation.md §5.3):
    "Low MSN means stable convergence. High MSN means the model is
     jumping between inconsistent continuous/discrete interpretations."

Application interpretation: MSN should predict which model configuration
will produce worse final outputs — i.e., MSN value (computed online from
the iteration path) should correlate with eventual mean_target_error
(measured against ground truth).

If this holds, MSN is useful as an *online diagnostic* — you can rank
models by stability without needing held-out labels.

Operationalisation
------------------
1. Sample K random MinimalVOD configurations over a wide hyperparameter
   range (some configs deliberately bad: high reaction, tiny step,
   huge diffusion).
2. For each config, denoise N validation samples, collect
       per-config mean MSN  (averaged across samples and media)
       per-config mean error (mean_target_error)
3. Compute Pearson correlation between the two across configs.

Pre-registered thresholds
-------------------------
PASS iff:
    Pearson correlation(MSN, error) >= 0.4
    AND p-value < 0.01

(Spec language is "high MSN = bad", so the predicted sign is positive.)

If correlation passes but with low effect size (< 0.4) or weak
significance, MSN is "directionally correct but not actionable" — record
honestly, do not promote to PASS.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

sys.path.insert(0, "D:/VOD/prototype")

from vod_minimal.experiment import make_sample, evaluate_model
from vod_minimal.model import MinimalVOD


THRESHOLD_CORR = 0.4
THRESHOLD_PVALUE = 0.01


def random_config(rng: np.random.Generator) -> MinimalVOD:
    """Sample a MinimalVOD config across a deliberately wide range.

    The range covers stable configs (moderate diffusion + reaction) and
    obviously bad ones (extreme reaction with tiny step size, etc.) so
    that mean_target_error has variance to correlate against.
    """
    return MinimalVOD(
        diffusion=float(rng.uniform(0.05, 1.5)),
        reaction=float(rng.uniform(0.02, 0.8)),
        step_size=float(rng.uniform(0.1, 1.5)),
        steps=int(rng.integers(4, 25)),
    )


def evaluate_config(model: MinimalVOD, samples: list, *, master_rng: np.random.Generator) -> tuple[float, float]:
    """Return (mean_MSN, mean_error) for this config across `samples`."""
    metrics = evaluate_model(model, samples)
    return metrics["mean_msn"], metrics["mean_after"]


def pearson_pvalue(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Pearson r + Fisher-z + two-sided p-value (normal approx, no scipy)."""
    n = x.size
    xm = x - x.mean()
    ym = y - y.mean()
    denom = float(np.sqrt((xm * xm).sum() * (ym * ym).sum()))
    if denom < 1e-12:
        return 0.0, 0.0, 1.0
    r = float((xm * ym).sum() / denom)
    if abs(r) >= 0.999999:
        return r, float("inf"), 0.0
    z = float(np.arctanh(r) * np.sqrt(max(n - 3, 1)))
    from math import erfc, sqrt
    p = float(erfc(abs(z) / sqrt(2)))
    return r, z, p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-configs", type=int, default=40)
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=430)
    args = parser.parse_args()

    print("=" * 60)
    print("MSN Application Utility Test")
    print("=" * 60)
    print(f"  n_configs={args.n_configs}  n_samples={args.n_samples}  seed={args.seed}")
    print(f"  Hypothesis (pre-registered):")
    print(f"    Pearson r(MSN, mean_error) >= {THRESHOLD_CORR}")
    print(f"    p-value < {THRESHOLD_PVALUE}")
    print()

    master_rng = np.random.default_rng(args.seed)

    # Build sample set ONCE so all configs see same inputs
    sample_rng = np.random.default_rng(args.seed + 1)
    samples = [make_sample(sample_rng) for _ in range(args.n_samples)]

    # Random configs
    config_rng = np.random.default_rng(args.seed + 2)
    msn_vals = []
    err_vals = []

    for k in range(args.n_configs):
        cfg = random_config(config_rng)
        msn, err = evaluate_config(cfg, samples, master_rng=master_rng)
        msn_vals.append(msn)
        err_vals.append(err)
        print(f"  config {k+1:2d}/{args.n_configs}:  D={cfg.diffusion:.3f}  R={cfg.reaction:.3f}  "
              f"step={cfg.step_size:.3f}  K={cfg.steps:2d}  -->  MSN={msn:.4f}  err={err:.4f}")

    msn_arr = np.array(msn_vals)
    err_arr = np.array(err_vals)

    r, z, p = pearson_pvalue(msn_arr, err_arr)
    score_pass = r >= THRESHOLD_CORR
    sig_pass = p < THRESHOLD_PVALUE
    verdict = score_pass and sig_pass

    print()
    print("-" * 60)
    print(f"  MSN range:    {msn_arr.min():.4f} .. {msn_arr.max():.4f}  (mean {msn_arr.mean():.4f})")
    print(f"  error range:  {err_arr.min():.4f} .. {err_arr.max():.4f}  (mean {err_arr.mean():.4f})")
    print(f"  Pearson r:    {r:+.4f}  (predicted direction: positive)")
    print(f"  Fisher z:     {z:+.4f}")
    print(f"  p-value:      {p:.6g}")
    print(f"  threshold:    r >= {THRESHOLD_CORR}  AND  p < {THRESHOLD_PVALUE}")
    print(f"  score_pass={score_pass}  sig_pass={sig_pass}")
    print()
    print(f"MSN diagnostic: {'PASS' if verdict else 'FAIL'}")


if __name__ == "__main__":
    main()
