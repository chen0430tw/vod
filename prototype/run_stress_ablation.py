"""5-distinctive leave-one-out stress ablation for NativeVOD.

Reproduces the §12.19 substrate-composite verification under the new
default backbone (UNet). Each row ablates one of the 5 spec
distinctives by zeroing its loss weight; the substrate should still
beat both trivial baselines on the headline metric, and each
leave-one-out should leave a fingerprint on its target metric.

Usage (local CPU smoke):
    py -3.13 D:\\VOD\\prototype\\run_stress_ablation.py --epochs 50 \
        --train-n 8 --test-n 4 --cpu

Usage (cluster GPU, full):
    python3 run_stress_ablation.py --epochs 200 --train-n 16 --test-n 8 \
        --backbone unet --flicker-strength 0.3
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from vod_minimal.blocky_scattering import build_blocky_scattering_batch
from vod_minimal.metrics import temporal_metrics, artifact_metrics
from vod_minimal.native import (
    LATENT_HW,
    LATENT_T,
    NativeLossWeights,
    NativeVOD,
    NativeVODConfig,
    native_total_loss,
    views_to_numpy,
    views_to_torch,
)
from train_native_vod import evaluate


# Five leave-one-out + bare configurations. Each row maps NativeLossWeights
# field name -> weight. The "full" config uses all five distinctives at
# the §12.19 weights (each at 0.1 except field=0.5, media=1.0).
_FULL = dict(
    field=0.5, media=1.0,
    temporal=0.1, artifact=0.1, binary_twin_pixel=0.1, aimp=0.1, text=0.0,
)

_CONFIGS = {
    "full":           _FULL,
    "no_artifact":    {**_FULL, "artifact": 0.0},
    "no_aimp":        {**_FULL, "aimp": 0.0},
    "no_temporal":    {**_FULL, "temporal": 0.0},
    "no_binary_twin": {**_FULL, "binary_twin_pixel": 0.0},
    "bare":           {**_FULL, "artifact": 0.0, "aimp": 0.0,
                       "temporal": 0.0, "binary_twin_pixel": 0.0},
}


def _build_train(rng_seed: int, n: int, flicker_strength: float):
    return build_blocky_scattering_batch(
        np.random.default_rng(rng_seed),
        batch_size=n, size=LATENT_HW, frames=LATENT_T,
        artifact_strength=0.6, tile=4,
        spacetime=True, temporal_mode="flicker",
        flicker_strength=flicker_strength,
        paired_denoising=True,
    )


def _build_test(rng_seed: int, n: int, flicker_strength: float):
    return build_blocky_scattering_batch(
        np.random.default_rng(rng_seed + 17),
        batch_size=n, size=LATENT_HW, frames=LATENT_T,
        artifact_strength=0.6, tile=4,
        spacetime=True, temporal_mode="flicker",
        flicker_strength=flicker_strength,
        paired_denoising=True,
    )


def _train_one(model: NativeVOD, batch, weights: NativeLossWeights, *,
               epochs: int, lr: float, device: torch.device) -> None:
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        losses = []
        for sample in batch.samples:
            noisy = views_to_torch(sample.noisy_views, device)
            target = views_to_torch(sample.target_views, device)
            loss, _ = native_total_loss(model, noisy, target, weights=weights)
            losses.append(loss)
        torch.stack(losses).mean().backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()


@torch.no_grad()
def _per_distinctive_metrics(model: NativeVOD, batch, device: torch.device) -> dict[str, float]:
    """Compute the per-distinctive readouts on the model's predictions.

    pred_video_tile_residue: tile-residue on predicted video (4/e + AIMP target)
    pred_video_smoothness:   abs frame-diff on predicted video (TTNM target)
    """
    model.eval()
    residues = []
    smooths = []
    for sample in batch.samples:
        noisy = views_to_torch(sample.noisy_views, device)
        pred, _ = model.forward(noisy)
        pred_np = views_to_numpy(pred)
        if "video" in pred_np:
            v = np.asarray(pred_np["video"])
            ar = artifact_metrics({"video": v}, tile=4)
            tm = temporal_metrics({"video": v})
            residues.append(float(ar.get("mean_tile_residue", float("nan"))))
            smooths.append(float(tm.get("temporal_smoothness", float("nan"))))
    return {
        "pred_video_tile_residue": float(np.nanmean(residues)) if residues else float("nan"),
        "pred_video_smoothness":   float(np.nanmean(smooths))  if smooths else float("nan"),
    }


def _run_one(label: str, weights_dict: dict, *, args, device: torch.device,
             train_batch, test_batch) -> dict[str, float]:
    torch.manual_seed(args.seed)
    config = NativeVODConfig(
        channels=args.channels,
        hidden=args.hidden,
        denoise_steps=args.steps,
        backbone=args.backbone,
    )
    model = NativeVOD(config).to(device)
    weights = NativeLossWeights(**weights_dict)
    t0 = time.time()
    _train_one(model, train_batch, weights, epochs=args.epochs, lr=args.lr, device=device)
    train_secs = time.time() - t0

    metrics = evaluate(model, test_batch, device)
    distinctives = _per_distinctive_metrics(model, test_batch, device)
    return {
        "label": label,
        "train_secs": train_secs,
        "imp_noisy": metrics.get("mean_improvement_over_noisy", float("nan")),
        "imp_zero":  metrics.get("mean_improvement_over_zero",  float("nan")),
        "model_image_error": metrics.get("model_image_error", float("nan")),
        "model_video_error": metrics.get("model_video_error", float("nan")),
        **distinctives,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="5-distinctive leave-one-out stress ablation.")
    p.add_argument("--seed", type=int, default=430)
    p.add_argument("--train-n", type=int, default=16)
    p.add_argument("--test-n", type=int, default=8)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--channels", type=int, default=4)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--backbone", choices=["unet", "mlp"], default="unet")
    p.add_argument("--flicker-strength", type=float, default=0.3)
    p.add_argument("--out-json", type=str, default="")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    train_batch = _build_train(args.seed, args.train_n, args.flicker_strength)
    test_batch  = _build_test(args.seed, args.test_n,  args.flicker_strength)

    print(f"backbone={args.backbone} device={device} epochs={args.epochs} "
          f"channels={args.channels} hidden={args.hidden} steps={args.steps}")
    print(f"flicker_strength={args.flicker_strength} train_n={args.train_n} test_n={args.test_n}")
    n_params = sum(p_.numel() for p_ in NativeVOD(NativeVODConfig(
        channels=args.channels, hidden=args.hidden, denoise_steps=args.steps,
        backbone=args.backbone)).parameters())
    print(f"NativeVOD total params: {n_params}")
    print()

    rows: list[dict] = []
    for label, w in _CONFIGS.items():
        row = _run_one(label, w, args=args, device=device,
                       train_batch=train_batch, test_batch=test_batch)
        rows.append(row)
        print(
            f"{row['label']:<16} "
            f"imp_noisy={row['imp_noisy']:+.4f} "
            f"imp_zero={row['imp_zero']:+.4f} "
            f"vid_residue={row['pred_video_tile_residue']:.4f} "
            f"vid_smooth={row['pred_video_smoothness']:.4f} "
            f"train={row['train_secs']:.1f}s"
        )

    # Fingerprint table — diff vs full.
    full = next(r for r in rows if r["label"] == "full")
    print()
    print("Leave-one-out fingerprint (delta vs full):")
    print(f"{'config':<16} {'d_imp_noisy':>12} {'d_residue':>12} {'d_smooth':>12}")
    for r in rows:
        if r["label"] == "full":
            continue
        print(
            f"{r['label']:<16} "
            f"{r['imp_noisy'] - full['imp_noisy']:+12.4f} "
            f"{r['pred_video_tile_residue'] - full['pred_video_tile_residue']:+12.4f} "
            f"{r['pred_video_smoothness']   - full['pred_video_smoothness']  :+12.4f}"
        )

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"\nWrote results to {args.out_json}")


if __name__ == "__main__":
    main()
