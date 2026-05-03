"""Stress-test the 4/e Orthogonal Compression Noise stack.

Default behaviour
-----------------
Builds three batches:

    clean       — standard `core.build_projection_batch`
    blocky      — clean + tile-aligned scatter injected on image/video
    suppressed  — blocky run through `oc_four_over_e` per view

and prints the artifact metrics for each. This is a non-training smoke
check: it runs in well under a second and never touches an optimizer.

Optional --train-smoke
----------------------
Trains a tiny SharedPointUpdater on the blocky batch for `--epochs`
steps, once with `--artifact-loss-weight 0` and once with the requested
weight, so we can read whether the differentiable penalty actually
moves the artifact diagnostics. This still keeps under a few seconds at
default sizes.

Usage:
    py -3.13 D:\\VOD\\prototype\\run_artifact_stress.py
    py -3.13 D:\\VOD\\prototype\\run_artifact_stress.py --train-smoke
"""

from __future__ import annotations

import argparse
import copy

import numpy as np
import torch

from vod_minimal.artifacts import oc_four_over_e
from vod_minimal.blocky_scattering import (
    SPATIAL_MEDIA,
    build_blocky_scattering_batch,
)
from vod_minimal.core import (
    MEDIA,
    ProjectionBatch,
    ProjectionSample,
    build_projection_batch,
    evaluate_projection_error,
    projection_loss,
    shared_update_rollout,
)
from vod_minimal.blocky_scattering import build_blocky_scattering_batch
from vod_minimal.metrics import (
    artifact_metrics,
    mean_target_error,
    temporal_metrics,
)
from vod_minimal.torch_artifacts import artifact_train_loss
from vod_minimal.torch_model import SharedPointUpdater, to_tensor


def _aggregate_metrics(batch: ProjectionBatch, *, tile: int) -> dict[str, float]:
    """Aggregate spatial-only artifact metrics + non-spatial diagnostic.

    The main `artifact_score` / `mean_tile_residue` / `max_tile_residue`
    only see image and video. Audio + text are reported separately as
    `non_spatial_*` so the spatial signal is not diluted.
    """
    score_acc: list[float] = []
    mean_acc: list[float] = []
    max_acc: list[float] = []
    ns_mean_acc: list[float] = []
    ns_max_acc: list[float] = []
    for sample in batch.samples:
        m = artifact_metrics(sample.noisy_views, tile=tile)
        score_acc.append(m["artifact_score"])
        mean_acc.append(m["mean_tile_residue"])
        max_acc.append(m["max_tile_residue"])
        ns_mean_acc.append(m["non_spatial_mean_tile_residue"])
        ns_max_acc.append(m["non_spatial_max_tile_residue"])
    return {
        "artifact_score": float(np.nanmean(score_acc)),
        "mean_tile_residue": float(np.nanmean(mean_acc)),
        "max_tile_residue": float(np.nanmax(max_acc)),
        "non_spatial_mean_tile_residue": float(np.nanmean(ns_mean_acc)),
        "non_spatial_max_tile_residue": float(np.nanmax(ns_max_acc)),
        "mean_target_error": float(
            np.mean(
                [mean_target_error(s.noisy_views, s.target_views) for s in batch.samples]
            )
        ),
    }


def _suppressed_batch(batch: ProjectionBatch, rng: np.random.Generator, *, tile: int, scale: float) -> ProjectionBatch:
    out_samples = []
    for sample in batch.samples:
        new_noisy = dict(sample.noisy_views)
        for medium in SPATIAL_MEDIA:
            if medium in new_noisy:
                new_noisy[medium] = oc_four_over_e(
                    new_noisy[medium], rng, beta=scale, tile=tile
                )
        out_samples.append(
            ProjectionSample(
                source_field=sample.source_field,
                target_field=sample.target_field,
                noisy_views=new_noisy,
                target_views=dict(sample.target_views),
            )
        )
    return ProjectionBatch(samples=tuple(out_samples), media=batch.media)


def _print_block(title: str, metrics: dict[str, float]) -> None:
    print(title)
    print("  spatial (image + video):")
    for key in ("artifact_score", "mean_tile_residue", "max_tile_residue"):
        print(f"    {key:<24} {metrics[key]:.6f}")
    print("  non-spatial (audio + text, informational only):")
    for key in ("non_spatial_mean_tile_residue", "non_spatial_max_tile_residue"):
        print(f"    {key:<24} {metrics[key]:.6f}")
    print("  end-to-end:")
    print(f"    {'mean_target_error':<24} {metrics['mean_target_error']:.6f}")


def _run_train_smoke(
    blocky_batch: ProjectionBatch,
    *,
    epochs: int,
    steps: int,
    weight: float,
    tile: int,
    seed: int,
    hidden: int,
) -> dict[str, float]:
    """Tiny in-process training loop on the blocky batch.

    Returns a metrics dict including the final artifact_score and
    mean_after, plus the final raw loss components.
    """
    torch.manual_seed(seed)
    device = torch.device("cpu")
    model = SharedPointUpdater(hidden=hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

    def update_fn(current, target, medium):
        del medium
        return model.forward_step(current, target)

    last_proj = 0.0
    last_art = 0.0
    for _ in range(epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        proj = projection_loss(update_fn, blocky_batch, steps=steps, device=device)
        loss = proj
        if weight > 0:
            art = artifact_train_loss(
                update_fn, blocky_batch, steps=steps, device=device, tile=tile
            )
            loss = loss + weight * art
            last_art = float(art.detach())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        last_proj = float(proj.detach())

    # Final eval on the same blocky batch
    model.eval()
    with torch.no_grad():
        def _rollout(noisy_views, target_views):
            denoised = {}
            for medium in MEDIA:
                noisy = to_tensor(noisy_views[medium], device)
                target = to_tensor(target_views[medium], device)
                pred = shared_update_rollout(update_fn, noisy, target, medium, steps=steps)
                denoised[medium] = pred.detach().cpu().numpy()
            return denoised

        em = evaluate_projection_error(
            _rollout, blocky_batch, include_artifact_metrics=True, artifact_tile=tile
        )

    em["projection_loss"] = last_proj
    em["artifact_loss"] = last_art
    return em


def _aggregate_temporal(batch: ProjectionBatch, *, tile: int) -> dict[str, float]:
    """Collect temporal diagnostics across a batch of (F,H,W) videos."""
    smooth_acc: list[float] = []
    drift_acc: list[float] = []
    art_drift_acc: list[float] = []
    consist_acc: list[float] = []
    for sample in batch.samples:
        m = temporal_metrics(sample.noisy_views, tile=tile)
        smooth_acc.append(m["temporal_smoothness"])
        drift_acc.append(m["frame_descriptor_drift"])
        art_drift_acc.append(m["temporal_artifact_drift"])
        consist_acc.append(m["cross_frame_consistency_score"])
    return {
        "temporal_smoothness": float(np.nanmean(smooth_acc)),
        "frame_descriptor_drift": float(np.nanmean(drift_acc)),
        "temporal_artifact_drift": float(np.nanmean(art_drift_acc)),
        "cross_frame_consistency_score": float(np.nanmean(consist_acc)),
    }


def _print_temporal_block(title: str, m: dict[str, float]) -> None:
    print(title)
    for key in (
        "temporal_smoothness",
        "frame_descriptor_drift",
        "temporal_artifact_drift",
        "cross_frame_consistency_score",
    ):
        print(f"    {key:<32} {m[key]:.6f}")


def main() -> None:
    p = argparse.ArgumentParser(description="Stress-test 4/e artifact suppression on blocky scattering data.")
    p.add_argument("--seed", type=int, default=430)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--size", type=int, default=32)
    p.add_argument("--tile", type=int, default=8)
    p.add_argument("--strength", type=float, default=0.25)
    p.add_argument("--noise-scale", type=float, default=0.24)
    p.add_argument("--artifact-scale", type=float, default=None,
                   help="scale for the suppression noise; defaults to noise_scale")
    p.add_argument("--train-smoke", action="store_true",
                   help="also run a short training loop comparing baseline vs --artifact-loss-weight")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--steps", type=int, default=2)
    p.add_argument("--hidden", type=int, default=16)
    p.add_argument("--artifact-loss-weight", type=float, default=0.2)
    p.add_argument("--temporal-stress", action="store_true",
                   help="instead of the spatial stress pipeline, build a 3-D U(t,y,x) "
                        "batch and compare clean / flicker / blocky-drift videos via "
                        "temporal metrics. Disables the spatial-suppression block.")
    p.add_argument("--frames", type=int, default=10)
    p.add_argument("--flicker-strength", type=float, default=0.5)
    p.add_argument("--drift", type=int, default=1)
    args = p.parse_args()

    if args.temporal_stress:
        return _run_temporal_stress(args)

    print("Artifact stress configuration")
    print(f"  batch_size={args.batch_size}, size={args.size}, tile={args.tile}, "
          f"strength={args.strength}, noise={args.noise_scale}")
    print()

    # Build clean and blocky batches with the same seed so the underlying
    # Chladni source/target fields are identical — only the noisy_views differ.
    clean = build_projection_batch(
        np.random.default_rng(args.seed),
        batch_size=args.batch_size,
        size=args.size,
        noise_scale=args.noise_scale,
    )
    blocky = build_blocky_scattering_batch(
        np.random.default_rng(args.seed),
        batch_size=args.batch_size,
        size=args.size,
        noise_scale=args.noise_scale,
        artifact_strength=args.strength,
        tile=args.tile,
    )
    suppressed = _suppressed_batch(
        blocky,
        np.random.default_rng(args.seed + 1),
        tile=args.tile,
        scale=args.artifact_scale if args.artifact_scale is not None else args.noise_scale,
    )

    clean_m = _aggregate_metrics(clean, tile=args.tile)
    blocky_m = _aggregate_metrics(blocky, tile=args.tile)
    suppressed_m = _aggregate_metrics(suppressed, tile=args.tile)

    _print_block("Clean (Chladni only)", clean_m)
    print()
    _print_block("Blocky (clean + tile scatter injected)", blocky_m)
    print()
    _print_block("Suppressed (blocky + 4/e Orthogonal Compression Noise)", suppressed_m)
    print()
    print("Improvement (suppressed vs blocky)")
    print(f"  artifact_score    +{suppressed_m['artifact_score'] - blocky_m['artifact_score']:+.6f}")
    print(f"  mean_tile_residue {suppressed_m['mean_tile_residue'] - blocky_m['mean_tile_residue']:+.6f}")
    print(f"  max_tile_residue  {suppressed_m['max_tile_residue'] - blocky_m['max_tile_residue']:+.6f}")
    print(f"  mean_target_error {suppressed_m['mean_target_error'] - blocky_m['mean_target_error']:+.6f}")

    if not args.train_smoke:
        return

    print()
    print("Train smoke (SharedPointUpdater on the blocky batch)")
    print(f"  epochs={args.epochs} steps={args.steps} hidden={args.hidden} "
          f"weight={args.artifact_loss_weight}")

    base = _run_train_smoke(
        blocky,
        epochs=args.epochs,
        steps=args.steps,
        weight=0.0,
        tile=args.tile,
        seed=args.seed,
        hidden=args.hidden,
    )
    pen = _run_train_smoke(
        blocky,
        epochs=args.epochs,
        steps=args.steps,
        weight=args.artifact_loss_weight,
        tile=args.tile,
        seed=args.seed,
        hidden=args.hidden,
    )

    print()
    print("Baseline (--artifact-loss-weight 0)")
    print(f"  projection_loss             {base['projection_loss']:.6f}")
    print(f"  mean_after                  {base['mean_after']:.6f}")
    print(f"  artifact_after_score        {base['artifact_after_score']:.6f}")
    print(f"  artifact_after_residue      {base['artifact_after_mean_tile_residue']:.6f}")
    print()
    print(f"With penalty (--artifact-loss-weight {args.artifact_loss_weight})")
    print(f"  projection_loss             {pen['projection_loss']:.6f}")
    print(f"  artifact_loss               {pen['artifact_loss']:.6f}")
    print(f"  mean_after                  {pen['mean_after']:.6f}")
    print(f"  artifact_after_score        {pen['artifact_after_score']:.6f}")
    print(f"  artifact_after_residue      {pen['artifact_after_mean_tile_residue']:.6f}")
    print()
    print("Delta (penalty - baseline)")
    print(f"  artifact_after_score        {pen['artifact_after_score'] - base['artifact_after_score']:+.6f}")
    print(f"  artifact_after_residue      {pen['artifact_after_mean_tile_residue'] - base['artifact_after_mean_tile_residue']:+.6f}")
    print(f"  mean_after                  {pen['mean_after'] - base['mean_after']:+.6f}")


def _run_temporal_stress(args) -> None:
    """3-D spacetime stress: clean vs flicker vs blocky-drift videos.

    Uses a true U(t,y,x) field for every sample so the temporal axis is
    not a 2-D shortcut. Reports four temporal metrics per regime.
    """
    print("Temporal stress configuration")
    print(f"  batch_size={args.batch_size}, size={args.size}, tile={args.tile}, "
          f"frames={args.frames}, strength={args.strength}, "
          f"flicker={args.flicker_strength}, drift={args.drift}")
    print()

    clean = build_blocky_scattering_batch(
        np.random.default_rng(args.seed),
        batch_size=args.batch_size,
        size=args.size,
        noise_scale=args.noise_scale,
        artifact_strength=0.0,
        tile=args.tile,
        spacetime=True,
        frames=args.frames,
        temporal_mode="static",
    )
    flicker = build_blocky_scattering_batch(
        np.random.default_rng(args.seed),
        batch_size=args.batch_size,
        size=args.size,
        noise_scale=args.noise_scale,
        artifact_strength=args.strength,
        tile=args.tile,
        spacetime=True,
        frames=args.frames,
        temporal_mode="flicker",
        flicker_strength=args.flicker_strength,
    )
    drift = build_blocky_scattering_batch(
        np.random.default_rng(args.seed),
        batch_size=args.batch_size,
        size=args.size,
        noise_scale=args.noise_scale,
        artifact_strength=args.strength,
        tile=args.tile,
        spacetime=True,
        frames=args.frames,
        temporal_mode="blocky_drift",
        drift=args.drift,
    )

    for title, batch in (
        ("Clean 3-D video (no scatter, no flicker)", clean),
        ("Temporal flicker (per-frame independent noise)", flicker),
        ("Blocky drift (rolling tile mask across frames)", drift),
    ):
        spatial = _aggregate_metrics(batch, tile=args.tile)
        temporal = _aggregate_temporal(batch, tile=args.tile)
        print(title)
        print("  spatial (image + video):")
        for key in ("artifact_score", "mean_tile_residue", "max_tile_residue"):
            print(f"    {key:<32} {spatial[key]:.6f}")
        print("  temporal (video):")
        for key in (
            "temporal_smoothness",
            "frame_descriptor_drift",
            "temporal_artifact_drift",
            "cross_frame_consistency_score",
        ):
            print(f"    {key:<32} {temporal[key]:.6f}")
        print()


if __name__ == "__main__":
    main()
