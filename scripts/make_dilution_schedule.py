"""Generate a curriculum-style dilution schedule for HVBD anchor + dataset mix.

Per `HVBD_VOD_Claude_Work_Guide.md` §8:

    0 – 5k steps:   100% anchor
    5k – 20k steps: 50% anchor / 50% real dataset
    20k+ steps:     10% anchor / 90% real dataset
    late stage:     0–5% anchor / 95–100% real dataset

This is a curriculum.json-style hot-swap config. The trainer reads the
current global step and looks up the anchor probability from the
schedule.

Usage:
    py -3.13 scripts/make_dilution_schedule.py \
        --out configs/curriculum_default.json \
        --total-steps 60000

Output is a JSON list of (step_threshold, anchor_prob, dataset_prob)
triples. Trainer picks the highest-step entry whose threshold ≤ current
step.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="configs/curriculum_default.json")
    p.add_argument("--total-steps", type=int, default=60000)
    p.add_argument("--phase-a-end", type=int, default=5000,
                   help="end of 100% anchor")
    p.add_argument("--phase-b-end", type=int, default=20000,
                   help="end of 50/50 anchor/data")
    p.add_argument("--late-anchor-prob", type=float, default=0.05)
    p.add_argument("--mid-anchor-prob", type=float, default=0.10)
    args = p.parse_args()

    schedule = [
        # (step_threshold, anchor_prob, dataset_prob)
        {"step": 0, "anchor_prob": 1.00, "dataset_prob": 0.00,
         "label": "Phase A: 100% anchor (planting prior)"},
        {"step": args.phase_a_end, "anchor_prob": 0.50, "dataset_prob": 0.50,
         "label": "Phase B: 50/50 anchor + dataset"},
        {"step": args.phase_b_end, "anchor_prob": args.mid_anchor_prob,
         "dataset_prob": 1.0 - args.mid_anchor_prob,
         "label": "Phase C: 10% anchor gravity + 90% dataset"},
        {"step": int(args.total_steps * 0.85), "anchor_prob": args.late_anchor_prob,
         "dataset_prob": 1.0 - args.late_anchor_prob,
         "label": "Late: 5% anchor gravity, mostly dataset"},
    ]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump({
            "version": "curriculum.v1",
            "total_steps": args.total_steps,
            "schedule": schedule,
            "notes": (
                "Trainer picks the highest-step entry whose "
                "step <= current_step. anchor_prob is the probability that "
                "any minibatch sample comes from HVBD cells; "
                "dataset_prob is the probability it comes from real dataset "
                "(CIFAR/ImageNet/etc.). Anchor_prob + dataset_prob = 1."
            ),
        }, f, indent=2, ensure_ascii=False)
    print(f"[curriculum] wrote {out}")
    print(json.dumps(schedule, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    sys.exit(main())
