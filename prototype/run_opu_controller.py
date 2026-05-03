"""Run OPU as a VOD prototype controller over checkpoint metrics."""

from __future__ import annotations

import argparse

import torch

from vod_minimal.opu_adapter import (
    VODControlState,
    checkpoint_metrics,
    quality_from_metrics,
    suggest_from_metrics,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Suggest VOD runtime knobs from checkpoint metrics via OPU.")
    parser.add_argument("--checkpoint", default="D:\\VOD\\prototype\\checkpoints\\tiny_vdit.pt")
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--step-size", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--quality-strength", type=float, default=1.0)
    parser.add_argument("--hot-pressure", type=float, default=0.0)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    metrics = checkpoint_metrics(checkpoint, split=args.split)
    state = VODControlState(
        steps=args.steps,
        step_size=args.step_size,
        max_tokens=args.max_tokens,
        quality_strength=args.quality_strength,
    )
    suggestion = suggest_from_metrics(metrics, state, hot_pressure=args.hot_pressure)

    print("VOD OPU Controller")
    print("==================")
    print(f"checkpoint = {args.checkpoint}")
    print(f"split      = {args.split}")
    print(f"model_type = {checkpoint.get('model_type', '<unknown>')}")
    print(f"quality    = {quality_from_metrics(metrics):.6f}")
    print()
    print("Input metrics")
    for key in sorted(metrics):
        print(f"  {key:<18} {metrics[key]:.6f}")
    print()
    print("Actions")
    if suggestion.actions:
        for action in suggestion.actions:
            print(f"  {action}")
    else:
        print("  <none>")
    print()
    print("Suggested controls")
    print(f"  steps            {suggestion.steps}")
    print(f"  step_size        {suggestion.step_size:.6f}")
    print(f"  max_tokens       {suggestion.max_tokens}")
    print(f"  quality_strength {suggestion.quality_strength:.6f}")


if __name__ == "__main__":
    main()
