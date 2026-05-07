"""TTNM application utility test.

Spec claim (vod_full_mathematical_formulation.md §6):
    "TTNM provides the temporal stability logic"

Simplified prototype form (vod_math_simplification.md §4):
    L_temporal = mean_t || Y_{t+1} - warp_or_shift(Y_t) ||
    or toy fallback: mean_t |Y_{t+1} - Y_t|

The differentiable form actually used in native.py:
    L_temporal = relu(temporal_smoothness(pred) - temporal_smoothness(target).detach())
where temporal_smoothness = mean abs frame-to-frame diff.

Application interpretation: the simplified TTNM diagnostic
(temporal_smoothness) should reliably distinguish *coherent* temporal
motion from *flicker* (per-frame independent noise). If it can't make
that distinction, using it as a training signal is meaningless — the
model would receive identical penalty for genuine motion vs random
flicker.

Operationalisation (paired test)
--------------------------------
For each seed:
    1. Build a clean spacetime Chladni field — smooth temporal mode,
       so frame-to-frame difference is small and structured.
    2. Build the flickered version by adding per-frame independent
       Gaussian noise on top of the same clean field.
    3. Compute temporal_smoothness on both.
    4. paired_diff = smoothness(flickered) - smoothness(clean).

Pre-registered hypothesis
-------------------------
PASS iff:
    paired_diff > 0 in >= 95% of seeds (sign consistency)
    mean(paired_diff) > 0
    |mean(paired_diff)| >= 3 * SE(paired_diff)

A trivial-PASS protection: the test compares clean vs noise-added at
the SAME underlying field. If temporal_smoothness can't even win this,
the simplified TTNM has no application utility.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

sys.path.insert(0, "D:/VOD/prototype")

from vod_minimal.blocky_scattering import inject_temporal_flicker
from vod_minimal.chladni import random_chladni_field
from vod_minimal.metrics import temporal_smoothness
from vod_minimal.projections import project_video_3d


def build_clean_video(rng: np.random.Generator, *, size: int = 32, frames: int = 8) -> np.ndarray:
    """Smooth-in-time spacetime Chladni field, projected to (frames, H, W)."""
    # Build a 2D field, then synthesize a smooth temporal mode by
    # interpolating between two related Chladni fields.
    a = random_chladni_field(rng, size=size, n_modes=3)
    b = random_chladni_field(rng, size=size, n_modes=3)
    # Smooth blend across frames (cosine taper)
    out = np.empty((frames, size, size), dtype=np.float64)
    for t in range(frames):
        alpha = 0.5 - 0.5 * np.cos(np.pi * t / max(frames - 1, 1))
        out[t] = (1 - alpha) * a + alpha * b
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=200)
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--flicker-strength", type=float, default=0.3)
    args = parser.parse_args()

    print("=" * 60)
    print("TTNM Application Utility Test")
    print("=" * 60)
    print(f"  n_seeds={args.n_seeds}  size={args.size}  frames={args.frames}")
    print(f"  flicker_strength={args.flicker_strength}")
    print(f"  Hypothesis (pre-registered):")
    print(f"    paired_diff = smoothness(flicker) - smoothness(clean) > 0")
    print(f"    in >= 95% of seeds")
    print(f"    AND mean(paired_diff) >= 3 * SE")
    print()

    diffs = []
    sign_positive = 0
    for seed in range(args.n_seeds):
        rng = np.random.default_rng(seed)
        clean = build_clean_video(rng, size=args.size, frames=args.frames)
        flicker_rng = np.random.default_rng(seed + 1_000_000)
        flicker = inject_temporal_flicker(clean, flicker_rng, strength=args.flicker_strength)

        s_clean = temporal_smoothness(clean)
        s_flicker = temporal_smoothness(flicker)
        d = float(s_flicker - s_clean)
        diffs.append(d)
        if d > 0:
            sign_positive += 1

    diffs_arr = np.array(diffs)
    mean_d = float(diffs_arr.mean())
    std_d = float(diffs_arr.std(ddof=1))
    se_d = std_d / np.sqrt(args.n_seeds)
    three_sig = 3.0 * se_d
    sign_frac = sign_positive / args.n_seeds

    score_pass = mean_d > 0 and sign_frac >= 0.95
    sig_pass = mean_d >= three_sig
    verdict = score_pass and sig_pass

    print(f"  mean smoothness(clean)             = {float(np.mean([temporal_smoothness(build_clean_video(np.random.default_rng(s), size=args.size, frames=args.frames)) for s in range(min(args.n_seeds, 20))])):.4f} (sample mean over 20)")
    print(f"  mean paired_diff                    = {mean_d:+.5f}")
    print(f"  std paired_diff (ddof=1)            = {std_d:.5f}")
    print(f"  SE of mean                          = {se_d:.5f}")
    print(f"  3-sigma SE of mean                  = {three_sig:.5f}")
    print(f"  sign positive fraction              = {sign_frac:.4f}  ({sign_positive}/{args.n_seeds})")
    print(f"  score_pass={score_pass}  sig_pass={sig_pass}")
    print()
    print(f"TTNM diagnostic: {'PASS' if verdict else 'FAIL'}")


if __name__ == "__main__":
    main()
