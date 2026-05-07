"""Validate native_vod_smoke under the no-leak protocol.

Hard rules followed by this script:
    1. The model never sees target_views (forward(noisy) only).
    2. Every regime is reported alongside two trivial baselines:
          - zero baseline   (predict zeros)
          - noisy baseline  (predict the noisy input)
       A model that does not beat the noisy baseline has not learned
       anything; the script states that verdict explicitly per regime.
    3. Stress regimes are reported as `degradation_vs_clean`
       (positive = stress is harder than clean; negative = stress is
       easier, which is itself a red flag).

Active media: image + video by default. Audio / text are experimental
reshape adapters and excluded unless explicitly enabled.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from vod_minimal.blocky_scattering import (
    build_blocky_scattering_batch,
    inject_text_quantization_corruption,
)
from vod_minimal.core import (
    ProjectionBatch,
    ProjectionSample,
    build_projection_batch,
)
from vod_minimal.native import (
    LATENT_HW,
    LATENT_T,
    NativeLossWeights,
    NativeVOD,
    NativeVODConfig,
    native_total_loss,
    views_to_torch,
)
from train_native_vod import build_dataset, evaluate


def _text_corrupted_batch(rng: np.random.Generator, n: int, swap_rate: float) -> ProjectionBatch:
    base = build_projection_batch(
        rng, batch_size=n, size=LATENT_HW, frames=LATENT_T, spacetime=True
    )
    new_samples = []
    for sample in base.samples:
        new_noisy = dict(sample.noisy_views)
        new_noisy["text"] = inject_text_quantization_corruption(
            new_noisy["text"], rng, swap_rate=swap_rate
        )
        new_samples.append(
            ProjectionSample(
                source_field=sample.source_field,
                target_field=sample.target_field,
                noisy_views=new_noisy,
                target_views=dict(sample.target_views),
                source_spacetime_field=sample.source_spacetime_field,
                target_spacetime_field=sample.target_spacetime_field,
            )
        )
    return ProjectionBatch(samples=tuple(new_samples), media=base.media)


def _train_quick(model: NativeVOD, batch: ProjectionBatch, *, epochs: int, lr: float, device: torch.device) -> None:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    weights = NativeLossWeights()
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        sample_losses = []
        for sample in batch.samples:
            noisy = views_to_torch(sample.noisy_views, device)
            target = views_to_torch(sample.target_views, device)
            loss, _ = native_total_loss(model, noisy, target, weights=weights)
            sample_losses.append(loss)
        torch.stack(sample_losses).mean().backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()


def _print_row(label: str, m: dict[str, float], media: tuple[str, ...]) -> None:
    parts = [f"{label:<22}"]
    for k in media:
        zero = m.get(f"zero_{k}_error", float("nan"))
        noisy = m.get(f"noisy_{k}_error", float("nan"))
        model = m.get(f"model_{k}_error", float("nan"))
        parts.append(f"{k}: zero={zero:.3f} noisy={noisy:.3f} model={model:.3f}")
    print("  " + "   ".join(parts))


def _print_summary(label: str, m: dict[str, float]) -> None:
    imp_noisy = m.get("mean_improvement_over_noisy", float("nan"))
    imp_zero = m.get("mean_improvement_over_zero", float("nan"))
    verdict = ""
    if imp_noisy <= 0:
        verdict = "  <-- model does NOT beat noisy baseline"
    elif imp_zero <= 0:
        verdict = "  <-- model beats noisy but loses to zero"
    print(
        f"  {label:<22}  imp_vs_noisy={imp_noisy:+.4f}  "
        f"imp_vs_zero={imp_zero:+.4f}{verdict}"
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Validate native_vod_smoke against trivial baselines.")
    p.add_argument("--seed", type=int, default=430)
    p.add_argument("--train-n", type=int, default=8)
    p.add_argument("--test-n", type=int, default=4)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--channels", type=int, default=4)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--blocky-strength", type=float, default=0.4)
    p.add_argument("--flicker-strength", type=float, default=0.4)
    p.add_argument("--text-swap-rate", type=float, default=0.3)
    p.add_argument("--enable-audio", action="store_true")
    p.add_argument("--enable-text", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    torch.manual_seed(args.seed)

    print("native_vod_smoke (NOT v0.3) — no-leak validation")
    print(f"device={device}, train_n={args.train_n}, epochs={args.epochs}, "
          f"channels={args.channels}, hidden={args.hidden}, steps={args.steps}")
    print()

    config = NativeVODConfig(
        channels=args.channels,
        hidden=args.hidden,
        denoise_steps=args.steps,
        enable_audio=args.enable_audio,
        enable_text=args.enable_text,
    )
    model = NativeVOD(config).to(device)
    media = model.active_media()

    train_batch = build_dataset(args.seed, args.train_n)
    _train_quick(model, train_batch, epochs=args.epochs, lr=args.lr, device=device)

    rng_seed = args.seed + 17
    clean = build_projection_batch(
        np.random.default_rng(rng_seed),
        batch_size=args.test_n, size=LATENT_HW, frames=LATENT_T, spacetime=True,
    )
    blocky = build_blocky_scattering_batch(
        np.random.default_rng(rng_seed),
        batch_size=args.test_n, size=LATENT_HW, frames=LATENT_T,
        artifact_strength=args.blocky_strength, tile=4,
        spacetime=True, temporal_mode="static",
    )
    flicker = build_blocky_scattering_batch(
        np.random.default_rng(rng_seed),
        batch_size=args.test_n, size=LATENT_HW, frames=LATENT_T,
        artifact_strength=args.blocky_strength, tile=4,
        spacetime=True, temporal_mode="flicker",
        flicker_strength=args.flicker_strength,
    )
    text_corrupt = _text_corrupted_batch(
        np.random.default_rng(rng_seed),
        n=args.test_n, swap_rate=args.text_swap_rate,
    )

    eval_clean = evaluate(model, clean, device)
    eval_blocky = evaluate(model, blocky, device)
    eval_flicker = evaluate(model, flicker, device)
    eval_text = evaluate(model, text_corrupt, device)

    print(f"Active media: {media}")
    print()
    print("Per-medium MSE (lower is better):")
    _print_row("clean",            eval_clean,   media)
    _print_row("blocky scatter",   eval_blocky,  media)
    _print_row("temporal flicker", eval_flicker, media)
    _print_row("text corruption",  eval_text,    media)
    print()
    print("Mean improvement vs trivial baselines:")
    _print_summary("clean",            eval_clean)
    _print_summary("blocky scatter",   eval_blocky)
    _print_summary("temporal flicker", eval_flicker)
    _print_summary("text corruption",  eval_text)
    print()
    print("Stress degradation vs clean (model_error_stress - model_error_clean):")
    for k in media:
        ce = eval_clean.get(f"model_{k}_error")
        if ce is None:
            continue
        for label, ev in (
            ("blocky scatter",   eval_blocky),
            ("temporal flicker", eval_flicker),
            ("text corruption",  eval_text),
        ):
            se = ev.get(f"model_{k}_error")
            if se is None:
                continue
            print(f"  {label:<22}  {k}: delta={se - ce:+.4f}")


if __name__ == "__main__":
    main()
