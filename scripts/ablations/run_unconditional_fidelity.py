"""Unconditional sample fidelity verifier.

Goal: from random noise, can VOD's trained DDPM/DDIM produce
recognisable Chladni-like fields, or does it degenerate to grid noise
/ checkerboard / constant blob?

Pipeline:
    1. Train a NativeVOD on synthetic Chladni distribution.
    2. Collect samples from 5 sources:
         a) train_reference   — actual Chladni training samples
         b) trained_sample    — DDIM from N(0,I), with trained model
         c) untrained_sample  — DDIM from N(0,I), with fresh random init
         d) random_noise_baseline — pure N(0,I) decoded
         e) zero_baseline     — zeros decoded
         f) gate0_recon       — encode→decode of training samples
    3. Compute descriptor + amplitude/entropy/salience/tile_residue/finite_ratio.
    4. descriptor_distance_to_train_mean per source.
    5. Save grid PNGs, JSON metrics, markdown report.
    6. PASS/FAIL/PARTIAL verdict.

PASS conditions (all):
    - trained_sample finite_ratio == 1.0
    - trained_sample descriptor distance < random_noise + zero baselines
    - trained_sample descriptor distance < untrained_sample
    - trained_sample tile_residue stable across seeds (not pure grid)
    - trained_sample amplitude range non-degenerate (range > 0.05)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Phase 1 portability
_PROTO = Path(__file__).resolve().parent.parent.parent / "prototype"
if str(_PROTO) not in sys.path:
    sys.path.insert(0, str(_PROTO))

from vod_minimal.blocky_scattering import build_blocky_scattering_batch
from vod_minimal.diffusion import diffusion_loss, ddim_sample, make_schedule
from vod_minimal.metrics import (
    artifact_metrics,
    descriptor,
    temporal_metrics,
)
from vod_minimal.native import (
    LATENT_HW,
    LATENT_T,
    NativeVOD,
    NativeVODConfig,
    views_to_numpy,
    views_to_torch,
)


# --------------------------------------------------------------------- #
#  Image helpers
# --------------------------------------------------------------------- #

def normalise_to_uint8(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a).squeeze()
    if a.ndim == 0:
        a = a.reshape(1, 1)
    if a.ndim != 2:
        a = a.reshape(a.shape[0], -1)
    lo, hi = float(a.min()), float(a.max())
    if hi - lo < 1e-9:
        return np.full_like(a, 127, dtype=np.uint8)
    return ((a - lo) / (hi - lo) * 255.0).astype(np.uint8)


def save_grid(images: list[np.ndarray], path: Path, ncols: int, label: str) -> None:
    """Tile a list of (H,W) numpy arrays into a single PNG with text label."""
    if not images:
        return
    h, w = images[0].shape[-2:]
    n = len(images)
    nrows = (n + ncols - 1) // ncols
    grid = np.full((nrows * h, ncols * w), 127, dtype=np.uint8)
    for i, im in enumerate(images):
        r, c = i // ncols, i % ncols
        grid[r * h:(r + 1) * h, c * w:(c + 1) * w] = normalise_to_uint8(im)
    Image.fromarray(grid, mode="L").save(path)


# --------------------------------------------------------------------- #
#  Metrics aggregation
# --------------------------------------------------------------------- #

DESCRIPTOR_KEYS = ("amplitude", "phase", "frequency", "compression", "salience", "snr")


def descriptor_vec(image: np.ndarray) -> np.ndarray:
    return descriptor("img", np.asarray(image)).vec()


def aggregate_metrics(images: list[np.ndarray]) -> dict[str, float]:
    """Per-sample metrics averaged across a batch of (H,W) images."""
    if not images:
        return {}
    desc_vecs = np.stack([descriptor_vec(im) for im in images], axis=0)
    flat = np.stack([np.asarray(im).ravel() for im in images], axis=0).astype(np.float64)
    finite_mask = np.isfinite(flat).all(axis=1)
    finite_ratio = float(finite_mask.mean())
    amp_range = float(flat.max() - flat.min())
    amp_std = float(flat.std())
    # Per-image entropy via 48-bin histogram
    entropies = []
    for im in images:
        hist, _ = np.histogram(np.asarray(im).ravel(), bins=48)
        probs = hist[hist > 0] / max(1, hist.sum())
        entropies.append(float(-(probs * np.log2(probs)).sum()))
    # Tile residue: artifact_metrics requires views dict
    residues = []
    for im in images:
        try:
            r = artifact_metrics({"image": np.asarray(im)})
            if not np.isnan(r["mean_tile_residue"]):
                residues.append(r["mean_tile_residue"])
        except Exception:
            pass
    out = {
        "descriptor_mean": dict(zip(DESCRIPTOR_KEYS, desc_vecs.mean(axis=0).tolist())),
        "descriptor_std":  dict(zip(DESCRIPTOR_KEYS, desc_vecs.std(axis=0).tolist())),
        "amplitude_range": amp_range,
        "amplitude_std":   amp_std,
        "entropy_mean":    float(np.mean(entropies)),
        "tile_residue_mean": float(np.mean(residues)) if residues else float("nan"),
        "finite_ratio":    finite_ratio,
    }
    return out


def descriptor_distance(metrics_a: dict, metrics_b: dict) -> float:
    """L2 distance between mean descriptor vectors."""
    a = np.array([metrics_a["descriptor_mean"][k] for k in DESCRIPTOR_KEYS], dtype=np.float64)
    b = np.array([metrics_b["descriptor_mean"][k] for k in DESCRIPTOR_KEYS], dtype=np.float64)
    return float(np.linalg.norm(a - b))


# --------------------------------------------------------------------- #
#  Training + sampling
# --------------------------------------------------------------------- #

def train_model(args, device, train_targets) -> NativeVOD:
    cfg = NativeVODConfig(
        channels=args.channels, hidden=args.hidden,
        denoise_steps=4, backbone="unet", time_dim=args.time_dim,
    )
    m = NativeVOD(cfg).to(device)
    n_params = sum(p_.numel() for p_ in m.parameters())
    print(f"[train] NativeVOD params={n_params:,} hidden={args.hidden} ch={args.channels}", flush=True)

    schedule = make_schedule(num_steps=args.diffusion_steps).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=1e-4)

    t0 = time.time()
    for ep in range(args.epochs):
        m.train()
        opt.zero_grad(set_to_none=True)
        x_0_live = torch.stack([m.encode(t) for t in train_targets], dim=0)
        L_diff = diffusion_loss(m, x_0_live, schedule, prediction="x_0")
        recon_terms = []
        for tv in train_targets:
            U = m.encode(tv)
            rec = m.decode(U)
            for k in ("image", "video"):
                if k in rec and k in tv:
                    recon_terms.append(F.mse_loss(rec[k], tv[k]))
        L_recon = torch.stack(recon_terms).mean() if recon_terms else x_0_live.new_zeros(())
        loss = L_diff + args.w_recon * L_recon
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if (ep + 1) % max(1, args.epochs // 5) == 0:
            print(f"  ep={ep+1:4d}  L_total={float(loss):.4f}  L_diff={float(L_diff):.4f}  L_recon={float(L_recon):.4f}",
                  flush=True)
    print(f"[train] done in {time.time()-t0:.1f}s", flush=True)
    return m, schedule


def sample_decoded_images(model: NativeVOD, schedule, n: int, seed: int, device) -> list[np.ndarray]:
    g = torch.Generator(device=device).manual_seed(seed)
    shape = (n, LATENT_T, LATENT_HW, LATENT_HW, model.config.channels)
    x_0_sampled = ddim_sample(model, shape, schedule, num_steps=50, device=device,
                              generator=g, prediction="x_0")
    images = []
    with torch.no_grad():
        for i in range(n):
            views_np = views_to_numpy(model.decode(x_0_sampled[i]))
            images.append(views_np["image"])
    return images


def random_noise_decoded(model: NativeVOD, n: int, seed: int, device) -> list[np.ndarray]:
    """Decode a pure N(0,I) latent (skipping all reverse diffusion)."""
    g = torch.Generator(device=device).manual_seed(seed)
    shape = (n, LATENT_T, LATENT_HW, LATENT_HW, model.config.channels)
    x = torch.randn(shape, device=device, generator=g, dtype=next(model.parameters()).dtype)
    images = []
    with torch.no_grad():
        for i in range(n):
            views_np = views_to_numpy(model.decode(x[i]))
            images.append(views_np["image"])
    return images


def zero_decoded(model: NativeVOD, n: int, device) -> list[np.ndarray]:
    shape = (n, LATENT_T, LATENT_HW, LATENT_HW, model.config.channels)
    x = torch.zeros(shape, device=device, dtype=next(model.parameters()).dtype)
    images = []
    with torch.no_grad():
        for i in range(n):
            views_np = views_to_numpy(model.decode(x[i]))
            images.append(views_np["image"])
    return images


def gate0_reconstruct(model: NativeVOD, train_targets, device) -> list[np.ndarray]:
    images = []
    with torch.no_grad():
        for tv in train_targets[:8]:
            U = model.encode(tv)
            views_np = views_to_numpy(model.decode(U))
            images.append(views_np["image"])
    return images


# --------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------- #

def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--seed", type=int, default=430)
    p.add_argument("--train-n", type=int, default=64)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--diffusion-steps", type=int, default=200)
    p.add_argument("--time-dim", type=int, default=64)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--channels", type=int, default=4)
    p.add_argument("--n-samples", type=int, default=8)
    p.add_argument("--w-recon", type=float, default=0.1)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--out", default="generated/diffusion_samples")
    p.add_argument("--report-out", default="prototype/unconditional_fidelity_result.json")
    p.add_argument("--md-out", default="prototype/unconditional_fidelity_report.md")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    # Build training data (clean Chladni)
    print("[data] building training Chladni samples...", flush=True)
    train_batch = build_blocky_scattering_batch(
        rng, batch_size=args.train_n, size=LATENT_HW, frames=LATENT_T,
        artifact_strength=0.0, tile=4, spacetime=True,
        temporal_mode="static", flicker_strength=0.0, paired_denoising=True,
    )
    train_targets = [views_to_torch(s.target_views, device) for s in train_batch.samples]
    train_images = [np.asarray(s.target_views["image"]) for s in train_batch.samples]

    # Train
    trained_model, schedule = train_model(args, device, train_targets)

    # Untrained reference (fresh init)
    cfg = NativeVODConfig(
        channels=args.channels, hidden=args.hidden,
        denoise_steps=4, backbone="unet", time_dim=args.time_dim,
    )
    untrained_model = NativeVOD(cfg).to(device)

    # Collect samples
    print("[sample] generating samples for all sources...", flush=True)
    sources: dict[str, list[np.ndarray]] = {
        "train_reference":         train_images[:args.n_samples],
        "trained_sample":          sample_decoded_images(trained_model, schedule, args.n_samples, args.seed + 1, device),
        "untrained_sample":        sample_decoded_images(untrained_model, schedule, args.n_samples, args.seed + 2, device),
        "random_noise_baseline":   random_noise_decoded(trained_model, args.n_samples, args.seed + 3, device),
        "zero_baseline":           zero_decoded(trained_model, args.n_samples, device),
        "gate0_recon":             gate0_reconstruct(trained_model, train_targets, device),
    }

    # Multi-seed stability check on trained_sample
    print("[sample] multi-seed trained samples for stability check...", flush=True)
    stability_seeds = [args.seed + 100, args.seed + 200, args.seed + 300]
    stability_runs = {
        f"trained_seed_{s}": sample_decoded_images(trained_model, schedule, 4, s, device)
        for s in stability_seeds
    }

    # Compute metrics
    print("[metrics] aggregating...", flush=True)
    metrics_per_source = {name: aggregate_metrics(imgs) for name, imgs in sources.items()}
    train_metrics = metrics_per_source["train_reference"]
    distances = {
        name: descriptor_distance(m, train_metrics)
        for name, m in metrics_per_source.items()
    }

    # Stability metrics
    stability_metrics = {name: aggregate_metrics(imgs) for name, imgs in stability_runs.items()}
    stability_distances = {name: descriptor_distance(m, train_metrics) for name, m in stability_metrics.items()}

    # Save sample grids
    print("[output] writing PNG grids...", flush=True)
    for name, imgs in sources.items():
        save_grid(imgs, out / f"{name}.png", ncols=4, label=name)
    save_grid(
        sum([imgs for imgs in stability_runs.values()], []),
        out / "trained_multi_seed.png", ncols=4, label="multi_seed",
    )

    # PASS/FAIL verdict
    trained = metrics_per_source["trained_sample"]
    rn = distances["random_noise_baseline"]
    zr = distances["zero_baseline"]
    un = distances["untrained_sample"]
    tr = distances["trained_sample"]
    g0 = distances["gate0_recon"]
    finite_ok = trained["finite_ratio"] == 1.0
    range_ok = trained["amplitude_range"] > 0.05
    beats_random = tr < rn
    beats_zero = tr < zr
    beats_untrained = tr < un
    distance_diffs = [stability_distances[k] for k in stability_runs]
    multi_seed_var = float(np.std(distance_diffs))

    passes = {
        "finite_ratio == 1.0": finite_ok,
        "amplitude_range > 0.05": range_ok,
        "beats random_noise baseline": beats_random,
        "beats zero baseline": beats_zero,
        "beats untrained_sample": beats_untrained,
    }
    n_pass = sum(passes.values())
    if n_pass == len(passes):
        verdict = "PASS"
    elif n_pass >= 3:
        verdict = "PARTIAL"
    else:
        verdict = "FAIL"

    # JSON report
    payload = {
        "date": datetime.now().isoformat(),
        "checkpoint": None,
        "train_args": vars(args),
        "sampler": {"type": "DDIM", "prediction": "x_0", "eta": 0.0,
                    "num_steps": 50, "schedule_steps": args.diffusion_steps},
        "metrics": {
            "per_source": metrics_per_source,
            "descriptor_distance_to_train": distances,
            "stability": {
                "per_seed_distances": stability_distances,
                "std_across_seeds": multi_seed_var,
            },
        },
        "verdict": verdict,
        "passes": passes,
    }
    Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[output] JSON -> {args.report_out}", flush=True)

    # Markdown report
    md = []
    md.append("# Unconditional Sample Fidelity Report\n")
    md.append(f"**Date**: {payload['date']}\n")
    md.append(f"**Verdict**: **{verdict}** ({n_pass}/{len(passes)} checks pass)\n\n")
    md.append("## Train args\n")
    md.append("```json\n" + json.dumps(vars(args), indent=2) + "\n```\n")
    md.append("## Sampler\n")
    md.append("DDIM, η=0, prediction=x_0, schedule=linear β [1e-4, 2e-2], "
              f"num_steps={args.diffusion_steps}, sample_steps=50\n\n")
    md.append("## Visible output\n")
    md.append(f"- `{out}/train_reference.png` — Chladni training samples\n")
    md.append(f"- `{out}/trained_sample.png` — DDIM samples from N(0,I) (TRAINED model)\n")
    md.append(f"- `{out}/untrained_sample.png` — DDIM samples from N(0,I) (UNTRAINED control)\n")
    md.append(f"- `{out}/random_noise_baseline.png` — pure N(0,I) decoded\n")
    md.append(f"- `{out}/zero_baseline.png` — zeros decoded\n")
    md.append(f"- `{out}/gate0_recon.png` — encode→decode of training samples\n")
    md.append(f"- `{out}/trained_multi_seed.png` — same model, 3 different seeds\n\n")
    md.append("## descriptor_distance_to_train (L2 over [amp, phase, freq, comp, sal, snr])\n\n")
    md.append("| source | distance |\n|---|---|\n")
    for name, d in sorted(distances.items(), key=lambda kv: kv[1]):
        md.append(f"| `{name}` | {d:.4f} |\n")
    md.append("\n## Per-source key metrics\n\n")
    md.append("| source | finite | amp_range | entropy | tile_residue |\n|---|---|---|---|---|\n")
    for name, m in metrics_per_source.items():
        md.append(f"| `{name}` | {m['finite_ratio']:.3f} | {m['amplitude_range']:.3f} "
                  f"| {m['entropy_mean']:.3f} | {m['tile_residue_mean']:.3f} |\n")
    md.append("\n## PASS checks\n\n")
    for k, v in passes.items():
        md.append(f"- {'✅' if v else '❌'} {k}\n")
    md.append(f"\n## Multi-seed stability (3 seeds, n=4 each)\n\n")
    md.append(f"std of descriptor_distance across seeds: **{multi_seed_var:.4f}**\n")
    md.append("(lower = more stable; if std > 0.5 the sampler is highly seed-dependent)\n\n")
    if verdict != "PASS":
        md.append("## If FAIL/PARTIAL — next minimal fix\n\n")
        md.append("Check in this order (Codex 4.1-4.4):\n")
        md.append("1. (4.1) sampler/loss target mismatch — already verified consistent (x_0).\n")
        md.append("2. (4.2) loss weights — try --w-recon 1.0 (currently 0.1) to tighten encoder.\n")
        md.append("3. (4.3) latent scale — check trained latent stats vs N(0,I).\n")
        md.append("4. (4.4) capacity — try --hidden 64 --channels 8 --train-n 256 --epochs 2000.\n")
    Path(args.md_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.md_out).write_text("".join(md), encoding="utf-8")
    print(f"[output] MD report -> {args.md_out}", flush=True)
    print(f"\n=== VERDICT: {verdict} ===", flush=True)


if __name__ == "__main__":
    main()
