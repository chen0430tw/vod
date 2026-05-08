"""Per-experiment metrics stub for VOD/LDM comparison.

Per `HVBD_VOD_Claude_Work_Guide.md` §9 acceptance criteria:

Static HVBD verdict — whether trained_sample is closer to ref than
no-anchor baseline; whether multi-domain diversity is achieved;
whether mosaic/grid artifact appeared (failure mode).

VOD vs LDM verdict — training time, params/storage, multi-domain
coverage, cross-domain composition, out-of-anchor recall, per-domain
quality, whether domain routing is required.

Smoke metrics (cheap, no FID/CLIPScore — guide §7 explicitly says no
heavy FID dependency in first round).

Usage:
    py -3.13 scripts/eval_samples.py \
        --generated-dir experiments/static_anchor_ablation/vod_100m_hvbd/generated \
        --train-ref-dir data/hvbd_static/cells/anchor_core_v1 \
        --report-out experiments/static_anchor_ablation/vod_100m_hvbd/eval.json
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def load_image_set(d: Path, max_n: int = 64) -> np.ndarray:
    paths = sorted(d.glob("*.png"))[:max_n]
    if not paths:
        return np.zeros((0, 0, 0, 3), dtype=np.float32)
    imgs = []
    for p in paths:
        img = Image.open(p).convert("RGB").resize((64, 64), Image.BILINEAR)
        imgs.append(np.asarray(img, dtype=np.float32) / 127.5 - 1.0)
    return np.stack(imgs, axis=0)


def descriptor(arr: np.ndarray) -> dict:
    """6-dim aggregate: mean, std, finite_ratio, amp_range, entropy, unique_color_count."""
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "finite_ratio": 0.0,
                "amp_range": 0.0, "entropy": 0.0, "unique_colors": 0}
    finite = float(np.isfinite(arr).all(axis=(1, 2, 3)).mean())
    flat = arr.reshape(arr.shape[0], -1)
    mean = float(arr.mean())
    std = float(arr.std())
    amp_range = float(flat.max(axis=1).mean() - flat.min(axis=1).mean())
    # crude entropy via 32-bin histogram per image then mean
    ents = []
    for i in range(arr.shape[0]):
        h, _ = np.histogram(arr[i].ravel(), bins=32)
        p = h / max(1, h.sum())
        p = p[p > 0]
        ents.append(float(-(p * np.log2(p)).sum()))
    return {
        "mean": mean,
        "std": std,
        "finite_ratio": finite,
        "amp_range": amp_range,
        "entropy": float(np.mean(ents)),
        "unique_colors": int(np.unique((arr * 127 + 128).clip(0, 255).astype(np.uint8).reshape(-1, 3),
                                        axis=0).shape[0]),
    }


def diversity_pixel_std(arr: np.ndarray) -> float:
    if arr.shape[0] < 2:
        return 0.0
    n = arr.shape[0]
    diffs = []
    for i in range(min(n, 16)):
        for j in range(i + 1, min(n, 16)):
            diffs.append(float(np.mean((arr[i] - arr[j]) ** 2)))
    return float(np.mean(diffs)) if diffs else 0.0


def grid_artifact_score(arr: np.ndarray) -> float:
    """Detect mosaic/grid layout leakage. Heuristic: if many samples
    contain consistent vertical/horizontal lines at fixed pixel
    spacings, the substrate has learned the grid geometry. Returns
    a 0-1 score where higher = more grid-like."""
    if arr.shape[0] == 0:
        return 0.0
    score = 0.0
    for i in range(min(arr.shape[0], 8)):
        # convert to grayscale-ish
        g = arr[i].mean(axis=-1)
        # row gradient mean — high uniformity across columns = grid line
        row_grad = np.abs(np.diff(g, axis=1)).mean(axis=1)
        col_grad = np.abs(np.diff(g, axis=0)).mean(axis=0)
        # variance of these gradients — low variance = grid present
        # convert to 0-1: 1 = strong grid, 0 = no grid
        v = 1.0 / (1.0 + 10 * (row_grad.std() + col_grad.std()))
        score += v
    return float(score / min(arr.shape[0], 8))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--generated-dir", required=True)
    p.add_argument("--train-ref-dir", required=True)
    p.add_argument("--report-out", required=True)
    p.add_argument("--max-n", type=int, default=64)
    args = p.parse_args()

    gen = load_image_set(Path(args.generated_dir), args.max_n)
    ref = load_image_set(Path(args.train_ref_dir), args.max_n)

    gen_d = descriptor(gen)
    ref_d = descriptor(ref)

    gen_diversity = diversity_pixel_std(gen)
    ref_diversity = diversity_pixel_std(ref)

    grid_score = grid_artifact_score(gen)

    # crude descriptor distance (Euclidean over 6 dims)
    keys = ["mean", "std", "finite_ratio", "amp_range", "entropy"]
    desc_dist = float(np.linalg.norm(
        np.array([gen_d[k] for k in keys]) -
        np.array([ref_d[k] for k in keys])
    ))

    payload = {
        "generated_dir": args.generated_dir,
        "train_ref_dir": args.train_ref_dir,
        "n_generated": int(gen.shape[0]),
        "n_ref": int(ref.shape[0]),
        "descriptor_generated": gen_d,
        "descriptor_ref": ref_d,
        "descriptor_distance_to_ref": desc_dist,
        "diversity_generated": gen_diversity,
        "diversity_ref": ref_diversity,
        "grid_artifact_score": grid_score,
        "warnings": [],
    }
    if grid_score > 0.5:
        payload["warnings"].append(
            f"high grid_artifact_score={grid_score:.3f} — substrate may "
            f"have learned 14×16 grid layout from anchor; reduce "
            f"anchor_prob / increase random crop / rotate / scale aug."
        )
    if gen_diversity < 0.05:
        payload["warnings"].append(
            f"low diversity={gen_diversity:.3f} — possible mode collapse"
        )

    out = Path(args.report_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    sys.exit(main())
