"""Run the Minimal VOD Prototype Order."""

from __future__ import annotations

import argparse

from vod_minimal.experiment import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the minimal VOD Chladni prototype.")
    parser.add_argument("--seed", type=int, default=430)
    parser.add_argument("--train-n", type=int, default=32)
    parser.add_argument("--test-n", type=int, default=32)
    args = parser.parse_args()

    model, train_metrics, test_metrics = run_experiment(
        seed=args.seed,
        train_n=args.train_n,
        test_n=args.test_n,
    )

    print("Minimal VOD Prototype")
    print("=====================")
    print(f"model.diffusion = {model.diffusion:.4f}")
    print(f"model.reaction  = {model.reaction:.4f}")
    print(f"model.step_size = {model.step_size:.4f}")
    print(f"model.steps     = {model.steps}")
    print()

    print("Train metrics")
    for key, value in train_metrics.items():
        print(f"  {key:<18} {value:.6f}")
    print()

    print("Test metrics")
    for key, value in test_metrics.items():
        print(f"  {key:<18} {value:.6f}")


if __name__ == "__main__":
    main()
