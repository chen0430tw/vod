"""Validate the simplified VOD minimal core (`vod_minimal/core.py`).

This script exercises the four core interfaces end-to-end with the analytic
NumPy updater (no training, no PyTorch). If `core.py` is wired correctly the
output should report a strong drop from `mean_before` to `mean_after` and a
`success_rate` close to 1.0 — matching the historical numbers in
`docs/vod_minimal_prototype_result.txt`.

Usage:
    py -3.13 D:\\VOD\\prototype\\run_core_validation.py
    py -3.13 D:\\VOD\\prototype\\run_core_validation.py --train-n 16 --test-n 16 --steps 12
"""

from __future__ import annotations

import argparse

import numpy as np

from vod_minimal.core import (
    MEDIA,
    build_projection_batch,
    evaluate_projection_error,
    make_numpy_rollout_fn,
    make_numpy_update_fn,
)


def _print_metrics(title: str, metrics: dict[str, float]) -> None:
    print(title)
    for key in ("mean_before", "mean_after", "mean_improvement", "success_rate"):
        print(f"  {key:<18} {metrics[key]:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the simplified VOD minimal core.")
    parser.add_argument("--seed", type=int, default=430)
    parser.add_argument("--train-n", type=int, default=16)
    parser.add_argument("--test-n", type=int, default=16)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--noise-scale", type=float, default=0.24)
    parser.add_argument("--diffusion", type=float, default=0.55)
    parser.add_argument("--reaction", type=float, default=0.18)
    parser.add_argument("--step-size", type=float, default=0.9)
    args = parser.parse_args()

    train_rng = np.random.default_rng(args.seed)
    test_rng = np.random.default_rng(args.seed + 1)

    train_batch = build_projection_batch(
        train_rng,
        batch_size=args.train_n,
        size=args.size,
        noise_scale=args.noise_scale,
    )
    test_batch = build_projection_batch(
        test_rng,
        batch_size=args.test_n,
        size=args.size,
        noise_scale=args.noise_scale,
    )

    update_fn = make_numpy_update_fn(
        diffusion=args.diffusion,
        reaction=args.reaction,
        step_size=args.step_size,
    )
    rollout_fn = make_numpy_rollout_fn(update_fn, steps=args.steps, media=MEDIA)

    train_metrics = evaluate_projection_error(rollout_fn, train_batch)
    test_metrics = evaluate_projection_error(rollout_fn, test_batch)

    print("VOD Minimal Core Validation")
    print("===========================")
    print(f"size       = {args.size}")
    print(f"steps      = {args.steps}")
    print(f"noise      = {args.noise_scale:.4f}")
    print(f"diffusion  = {args.diffusion:.4f}")
    print(f"reaction   = {args.reaction:.4f}")
    print(f"step_size  = {args.step_size:.4f}")
    print(f"train_n    = {args.train_n}")
    print(f"test_n     = {args.test_n}")
    print()
    _print_metrics("Train metrics", train_metrics)
    print()
    _print_metrics("Test metrics", test_metrics)


if __name__ == "__main__":
    main()
