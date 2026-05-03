"""Train native_vod_smoke (NOT v0.3).

This is the smoke prototype after the target-leak removal. The model
sees only noisy views; targets enter solely through the loss.

Default media: image + video. Audio / text are experimental and OFF
unless explicitly enabled — see flags below.

Usage:
    py -3.13 D:\\VOD\\prototype\\train_native_vod.py --train-n 8 --test-n 4 --epochs 20
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from vod_minimal.core import build_projection_batch
from vod_minimal.metrics import temporal_metrics
from vod_minimal.metrics import artifact_metrics
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
from vod_minimal.schema import checkpoint_payload, print_metrics_block


CORE_CONTRACT_VERSION = "vod-minimal-core-v1"


def build_dataset(seed: int, n: int, *, paired_denoising: bool = False):
    """Build a 3-D spacetime batch sized to the native latent grid."""
    rng = np.random.default_rng(seed)
    return build_projection_batch(
        rng,
        batch_size=n,
        size=LATENT_HW,
        frames=LATENT_T,
        spacetime=True,
        paired_denoising=paired_denoising,
    )


def _per_medium_mse(pred_views, target_views, media):
    out: dict[str, float] = {}
    for k in media:
        if k in pred_views and k in target_views:
            out[k] = float(np.mean((pred_views[k] - target_views[k]) ** 2))
    return out


def _zero_baseline(target_views, media):
    """Trivial baseline: predict zeros. Error = mean(target ** 2)."""
    return {
        k: float(np.mean(np.asarray(target_views[k]) ** 2))
        for k in media
        if k in target_views
    }


def _noisy_baseline(noisy_views, target_views, media):
    """Trivial baseline: predict the noisy input unchanged."""
    return {
        k: float(np.mean((np.asarray(noisy_views[k]) - np.asarray(target_views[k])) ** 2))
        for k in media
        if k in target_views and k in noisy_views
    }


@torch.no_grad()
def evaluate(model: NativeVOD, batch, device: torch.device) -> dict[str, float]:
    """Per-medium MSE for model + zero baseline + noisy baseline.

    Returns a flat dict keyed by `<source>_<medium>_error`, plus
    aggregate `*_improvement_over_*` numbers. A model that has not
    learned anything will produce model errors >= noisy-baseline
    errors; this evaluator surfaces that fact instead of hiding it.
    """
    model.eval()
    media = model.active_media()

    sample_records: list[dict[str, dict[str, float]]] = []

    for sample in batch.samples:
        noisy_t = views_to_torch(sample.noisy_views, device)
        pred, _ = model.forward(noisy_t)
        pred_np = views_to_numpy(pred)

        record = {
            "model": _per_medium_mse(pred_np, sample.target_views, media),
            "noisy": _noisy_baseline(sample.noisy_views, sample.target_views, media),
            "zero": _zero_baseline(sample.target_views, media),
        }
        sample_records.append(record)

    # Aggregate per (source, medium).
    metrics: dict[str, float] = {}
    for source in ("model", "noisy", "zero"):
        for k in media:
            vals = [r[source].get(k) for r in sample_records if k in r[source]]
            if vals:
                metrics[f"{source}_{k}_error"] = float(np.mean(vals))

    # Improvements (positive = model beats baseline).
    for k in media:
        m_key = f"model_{k}_error"
        if m_key not in metrics:
            continue
        if f"noisy_{k}_error" in metrics:
            metrics[f"improvement_over_noisy_{k}"] = (
                metrics[f"noisy_{k}_error"] - metrics[m_key]
            )
        if f"zero_{k}_error" in metrics:
            metrics[f"improvement_over_zero_{k}"] = (
                metrics[f"zero_{k}_error"] - metrics[m_key]
            )

    # Mean improvement across active media — for a single overall summary.
    noisy_imps = [metrics[f"improvement_over_noisy_{k}"] for k in media if f"improvement_over_noisy_{k}" in metrics]
    zero_imps = [metrics[f"improvement_over_zero_{k}"] for k in media if f"improvement_over_zero_{k}" in metrics]
    if noisy_imps:
        metrics["mean_improvement_over_noisy"] = float(np.mean(noisy_imps))
    if zero_imps:
        metrics["mean_improvement_over_zero"] = float(np.mean(zero_imps))

    return metrics


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    torch.manual_seed(args.seed)

    train_batch = build_dataset(args.seed, args.train_n, paired_denoising=args.paired_denoising)
    test_batch = build_dataset(args.seed + 1, args.test_n, paired_denoising=args.paired_denoising)

    config = NativeVODConfig(
        channels=args.channels,
        hidden=args.hidden,
        denoise_steps=args.steps,
        enable_audio=args.enable_audio,
        enable_text=args.enable_text,
        backbone=args.backbone,
    )
    model = NativeVOD(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    weights = NativeLossWeights(
        field=args.weight_field,
        media=args.weight_media,
        temporal=args.weight_temporal,
        artifact=args.weight_artifact,
        text=args.weight_text,
        binary_twin_pixel=args.weight_binary_twin_pixel,
        aimp=args.weight_aimp,
    )

    last_components: dict[str, float] = {}

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        sample_losses = []
        last_components = {}
        for sample in train_batch.samples:
            noisy = views_to_torch(sample.noisy_views, device)
            target = views_to_torch(sample.target_views, device)
            loss, components = native_total_loss(model, noisy, target, weights=weights)
            sample_losses.append(loss)
            last_components = components
        total = torch.stack(sample_losses).mean()
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            test_metrics = evaluate(model, test_batch, device)
            imp_noisy = test_metrics.get("mean_improvement_over_noisy", float("nan"))
            imp_zero = test_metrics.get("mean_improvement_over_zero", float("nan"))
            print(
                f"epoch={epoch:04d} "
                f"L_total={float(total.detach()):.4f} "
                f"L_field={last_components['L_field']:.4f} "
                f"L_media={last_components['L_media']:.4f} "
                f"L_temp={last_components['L_temporal']:.4f} "
                f"L_art={last_components['L_artifact']:.4f} "
                f"L_btpx={last_components.get('L_binary_twin_pixel', 0.0):.6f} "
                f"L_aimp={last_components.get('L_aimp', 0.0):.6f} "
                f"imp_vs_noisy={imp_noisy:+.4f} "
                f"imp_vs_zero={imp_zero:+.4f}"
            )

    train_metrics = evaluate(model, train_batch, device)
    test_metrics = evaluate(model, test_batch, device)

    if args.save:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            checkpoint_payload(
                state_dict=model.state_dict(),
                model_type="NativeVOD",
                config={**config.__dict__, "weights": weights.asdict()},
                train_args=vars(args),
                train_metrics=train_metrics,
                test_metrics=test_metrics,
            ),
            save_path,
        )

    print()
    print("native_vod_smoke (not v0.3)")
    print("===========================")
    print(f"device        = {device}")
    print(f"active media  = {model.active_media()}")
    print(f"channels      = {args.channels}, hidden = {args.hidden}, steps = {args.steps}")
    print()
    print_metrics_block("Train metrics", train_metrics)
    print()
    print_metrics_block("Test metrics", test_metrics)
    print()
    imp_noisy = test_metrics.get("mean_improvement_over_noisy", float("nan"))
    imp_zero = test_metrics.get("mean_improvement_over_zero", float("nan"))
    if imp_noisy <= 0:
        print(f"VERDICT: model does NOT beat the noisy-baseline (Δ={imp_noisy:+.4f}). "
              f"Treat all numbers above as smoke only.")
    elif imp_zero <= 0:
        print(f"VERDICT: model beats noisy (Δ={imp_noisy:+.4f}) but loses to zero "
              f"(Δ={imp_zero:+.4f}). Marginal — do not interpret as learned dynamics.")
    else:
        print(f"VERDICT: model beats both baselines on average "
              f"(noisy Δ={imp_noisy:+.4f}, zero Δ={imp_zero:+.4f}).")


def main() -> None:
    p = argparse.ArgumentParser(description="Train the native unified VOD generator.")
    p.add_argument("--seed", type=int, default=430)
    p.add_argument("--train-n", type=int, default=8)
    p.add_argument("--test-n", type=int, default=4)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--channels", type=int, default=4)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--save", default="D:\\VOD\\prototype\\checkpoints\\native_vod.pt")
    p.add_argument("--weight-field", type=float, default=0.5)
    p.add_argument("--weight-media", type=float, default=1.0)
    p.add_argument("--weight-temporal", type=float, default=0.1)
    p.add_argument("--weight-artifact", type=float, default=0.1)
    p.add_argument("--weight-text", type=float, default=0.3)
    p.add_argument("--weight-binary-twin-pixel", type=float, default=0.0,
                   help="Per-pixel Binary-Twin loss weight on image+video. "
                        "Default 0 = backward-compat ablation off.")
    p.add_argument("--weight-aimp", type=float, default=0.0,
                   help="TPSR/AIMP video consistency loss weight. "
                        "Default 0 = backward-compat ablation off.")
    p.add_argument(
        "--paired-denoising",
        action="store_true",
        help="use paired denoising protocol: target_views = projections(U), "
             "noisy_views = target_views + N(0, noise_scale). Default off "
             "preserves the legacy independent source/target conditional toy.",
    )
    p.add_argument("--enable-audio", action="store_true",
                   help="(experimental) include the audio reshape adapter; numbers are "
                        "not meaningful media quality")
    p.add_argument("--enable-text", action="store_true",
                   help="(experimental) include the text reshape adapter; not a real "
                        "discrete loss")
    p.add_argument("--backbone", choices=["unet", "mlp"], default="unet",
                   help="Field denoiser backbone. 'unet' (default) uses the small "
                        "3-level spatial UNet with a 1-D temporal mix at the bottleneck. "
                        "'mlp' uses the legacy pointwise MLP — kept for ablation.")
    train(p.parse_args())


if __name__ == "__main__":
    main()
