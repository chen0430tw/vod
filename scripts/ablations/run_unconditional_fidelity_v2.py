"""Unconditional sample fidelity v2.

Two fixes vs v1, motivated by ASCII inspection of raw data and latent
stats dump:

  Fix A — video target = image.broadcast(T)
      `temporal_mode='static'` only controls noisy_views corruption, not
      target_views. The Chladni spacetime field has mt=2 modes that on
      frames=8 alias into a 4-period flicker. We force every video frame
      to equal the corresponding still image, killing temporal variation
      in the training target.

  Fix B — latent normalization (EMA-tracked μ, σ)
      Encoder output has std ≈ 0.226 << 1.0 vs DDIM init N(0,I). The
      reverse process is geometrically off-manifold. We measure latent
      mean/std with EMA across training, normalize x_0 before diffusion
      loss, sample in normalized space, then unnormalize before decode.

Both fixes are in this trainer script only — no changes to vod_minimal/
or to the Chladni field substrate.
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

_PROTO = Path(__file__).resolve().parent.parent.parent / "prototype"
if str(_PROTO) not in sys.path:
    sys.path.insert(0, str(_PROTO))

from vod_minimal.blocky_scattering import build_blocky_scattering_batch
from vod_minimal.diffusion import diffusion_loss, ddim_sample, make_schedule, q_sample
from vod_minimal.metrics import artifact_metrics, descriptor
from vod_minimal.native import (
    LATENT_HW, LATENT_T, NativeVOD, NativeVODConfig,
    views_to_numpy, views_to_torch,
)


DESCRIPTOR_KEYS = ("amplitude", "phase", "frequency", "compression", "salience", "snr")


# --------------------------------------------------------------------- #
#  Image helpers
# --------------------------------------------------------------------- #

def normalise_to_uint8(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a).squeeze()
    if a.ndim != 2:
        a = a.reshape(a.shape[0], -1)
    lo, hi = float(a.min()), float(a.max())
    if hi - lo < 1e-9:
        return np.full_like(a, 127, dtype=np.uint8)
    return ((a - lo) / (hi - lo) * 255.0).astype(np.uint8)


def save_grid(images: list[np.ndarray], path: Path, ncols: int) -> None:
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
#  Metrics
# --------------------------------------------------------------------- #

def descriptor_vec(image: np.ndarray) -> np.ndarray:
    return descriptor("img", np.asarray(image)).vec()


def aggregate_metrics(images: list[np.ndarray]) -> dict[str, float]:
    if not images:
        return {}
    desc_vecs = np.stack([descriptor_vec(im) for im in images], axis=0)
    flat = np.stack([np.asarray(im).ravel() for im in images], axis=0).astype(np.float64)
    finite_ratio = float(np.isfinite(flat).all(axis=1).mean())
    amp_range = float(flat.max() - flat.min())
    entropies = []
    for im in images:
        hist, _ = np.histogram(np.asarray(im).ravel(), bins=48)
        probs = hist[hist > 0] / max(1, hist.sum())
        entropies.append(float(-(probs * np.log2(probs)).sum()))
    residues = []
    for im in images:
        try:
            r = artifact_metrics({"image": np.asarray(im)})
            if not np.isnan(r["mean_tile_residue"]):
                residues.append(r["mean_tile_residue"])
        except Exception:
            pass
    return {
        "descriptor_mean": dict(zip(DESCRIPTOR_KEYS, desc_vecs.mean(axis=0).tolist())),
        "amplitude_range": amp_range,
        "entropy_mean": float(np.mean(entropies)),
        "tile_residue_mean": float(np.mean(residues)) if residues else float("nan"),
        "finite_ratio": finite_ratio,
    }


def descriptor_distance(a: dict, b: dict) -> float:
    av = np.array([a["descriptor_mean"][k] for k in DESCRIPTOR_KEYS], dtype=np.float64)
    bv = np.array([b["descriptor_mean"][k] for k in DESCRIPTOR_KEYS], dtype=np.float64)
    return float(np.linalg.norm(av - bv))


# --------------------------------------------------------------------- #
#  Fix A: enforce video target = image broadcast across T
# --------------------------------------------------------------------- #

def force_video_static(target_views: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out = dict(target_views)
    if "image" in out and "video" in out:
        img = out["image"]  # (H, W)
        out["video"] = img.unsqueeze(0).expand(LATENT_T, *img.shape).contiguous()
    return out


# --------------------------------------------------------------------- #
#  Fix B: latent normalization with EMA stats
# --------------------------------------------------------------------- #

class LatentStats:
    """Tracks running mean/std of encoder output, EMA-updated each epoch."""

    def __init__(self, ema_decay: float = 0.99):
        self.mean = 0.0
        self.std = 1.0
        self.decay = ema_decay
        self.initialized = False

    def update(self, x: torch.Tensor) -> None:
        cur_mean = float(x.detach().mean().item())
        cur_std = float(x.detach().std().item()) + 1e-8
        if not self.initialized:
            self.mean, self.std = cur_mean, cur_std
            self.initialized = True
        else:
            self.mean = self.decay * self.mean + (1 - self.decay) * cur_mean
            self.std = self.decay * self.std + (1 - self.decay) * cur_std

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std + self.mean


# --------------------------------------------------------------------- #
#  Training + sampling
# --------------------------------------------------------------------- #

def train_model(args, device, train_targets):
    cfg = NativeVODConfig(
        channels=args.channels, hidden=args.hidden,
        denoise_steps=4, backbone="unet", time_dim=args.time_dim,
    )
    m = NativeVOD(cfg).to(device)
    n_params = sum(p_.numel() for p_ in m.parameters())
    print(f"[train] params={n_params:,} hidden={args.hidden} ch={args.channels}", flush=True)

    schedule = make_schedule(num_steps=args.diffusion_steps).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=1e-4)
    stats = LatentStats(ema_decay=0.99)

    t0 = time.time()
    for ep in range(args.epochs):
        m.train()
        opt.zero_grad(set_to_none=True)
        x_0_live = torch.stack([m.encode(t) for t in train_targets], dim=0)
        stats.update(x_0_live)
        x_0_norm = stats.normalize(x_0_live)
        L_diff = diffusion_loss(m, x_0_norm, schedule, prediction="x_0")
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
        if (ep + 1) % max(1, args.epochs // 10) == 0:
            print(f"  ep={ep+1:4d}  L_diff={float(L_diff.detach()):.4f}  "
                  f"L_recon={float(L_recon.detach()):.4f}  "
                  f"latent_μ={stats.mean:+.3f}  latent_σ={stats.std:.3f}", flush=True)
    print(f"[train] done in {time.time()-t0:.1f}s   final latent stats μ={stats.mean:+.4f} σ={stats.std:.4f}", flush=True)
    return m, schedule, stats


def sample_decoded_images(model, schedule, stats, n, seed, device):
    g = torch.Generator(device=device).manual_seed(seed)
    shape = (n, LATENT_T, LATENT_HW, LATENT_HW, model.config.channels)
    # DDIM in normalized space
    x_norm = ddim_sample(model, shape, schedule, num_steps=50,
                         device=device, generator=g, prediction="x_0")
    # unnormalize back to encoder space before decode
    x_unnorm = stats.unnormalize(x_norm)
    images = []
    with torch.no_grad():
        for i in range(n):
            views_np = views_to_numpy(model.decode(x_unnorm[i]))
            images.append(views_np["image"])
    return images


def random_noise_decoded(model, n, seed, device):
    g = torch.Generator(device=device).manual_seed(seed)
    shape = (n, LATENT_T, LATENT_HW, LATENT_HW, model.config.channels)
    x = torch.randn(shape, device=device, generator=g, dtype=next(model.parameters()).dtype)
    images = []
    with torch.no_grad():
        for i in range(n):
            images.append(views_to_numpy(model.decode(x[i]))["image"])
    return images


def zero_decoded(model, n, device):
    shape = (n, LATENT_T, LATENT_HW, LATENT_HW, model.config.channels)
    x = torch.zeros(shape, device=device, dtype=next(model.parameters()).dtype)
    images = []
    with torch.no_grad():
        for i in range(n):
            images.append(views_to_numpy(model.decode(x[i]))["image"])
    return images


def gate0_reconstruct(model, train_targets, device):
    images = []
    with torch.no_grad():
        for tv in train_targets[:8]:
            U = model.encode(tv)
            images.append(views_to_numpy(model.decode(U))["image"])
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
    p.add_argument("--w-recon", type=float, default=1.0)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--out", default="generated/diffusion_samples_v2")
    p.add_argument("--report-out", default="prototype/unconditional_fidelity_result_v2.json")
    p.add_argument("--md-out", default="prototype/unconditional_fidelity_report_v2.md")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    print("[data] building Chladni samples + Fix A (force video=image.broadcast)...", flush=True)
    batch = build_blocky_scattering_batch(
        rng, batch_size=args.train_n, size=LATENT_HW, frames=LATENT_T,
        artifact_strength=0.0, tile=4, spacetime=True,
        temporal_mode="static", flicker_strength=0.0, paired_denoising=True,
    )
    train_targets = [force_video_static(views_to_torch(s.target_views, device))
                     for s in batch.samples]
    train_images = [np.asarray(s.target_views["image"]) for s in batch.samples]

    # Verify Fix A worked
    sample_video = train_targets[0]["video"]
    diff = (sample_video[1:] - sample_video[:-1]).abs().max().item()
    print(f"[verify] forced-static video: max |frame[t+1]-frame[t]| = {diff:.6f} (should be 0)", flush=True)

    trained_model, schedule, stats = train_model(args, device, train_targets)
    cfg = NativeVODConfig(
        channels=args.channels, hidden=args.hidden,
        denoise_steps=4, backbone="unet", time_dim=args.time_dim,
    )
    untrained_model = NativeVOD(cfg).to(device)
    # untrained model uses identity stats (μ=0, σ=1) since no training happened
    untrained_stats = LatentStats()
    untrained_stats.mean, untrained_stats.std = 0.0, 1.0
    untrained_stats.initialized = True

    print("[sample] generating samples for all sources...", flush=True)
    sources = {
        "train_reference":         train_images[:args.n_samples],
        "trained_sample":          sample_decoded_images(trained_model, schedule, stats, args.n_samples, args.seed + 1, device),
        "untrained_sample":        sample_decoded_images(untrained_model, schedule, untrained_stats, args.n_samples, args.seed + 2, device),
        "random_noise_baseline":   random_noise_decoded(trained_model, args.n_samples, args.seed + 3, device),
        "zero_baseline":           zero_decoded(trained_model, args.n_samples, device),
        "gate0_recon":             gate0_reconstruct(trained_model, train_targets, device),
    }

    print("[sample] multi-seed stability...", flush=True)
    stability_seeds = [args.seed + 100, args.seed + 200, args.seed + 300]
    stability_runs = {
        f"trained_seed_{s}": sample_decoded_images(trained_model, schedule, stats, 4, s, device)
        for s in stability_seeds
    }

    print("[metrics] aggregating...", flush=True)
    metrics_per_source = {name: aggregate_metrics(imgs) for name, imgs in sources.items()}
    train_metrics = metrics_per_source["train_reference"]
    distances = {name: descriptor_distance(m, train_metrics) for name, m in metrics_per_source.items()}
    stability_metrics = {name: aggregate_metrics(imgs) for name, imgs in stability_runs.items()}
    stability_distances = {name: descriptor_distance(m, train_metrics) for name, m in stability_metrics.items()}

    print("[output] writing PNG grids...", flush=True)
    for name, imgs in sources.items():
        save_grid(imgs, out / f"{name}.png", ncols=4)
    save_grid(sum([imgs for imgs in stability_runs.values()], []),
              out / "trained_multi_seed.png", ncols=4)

    trained = metrics_per_source["trained_sample"]
    rn, zr, un, tr = (distances[k] for k in
                      ("random_noise_baseline", "zero_baseline", "untrained_sample", "trained_sample"))
    finite_ok = trained["finite_ratio"] == 1.0
    range_ok = trained["amplitude_range"] > 0.05
    beats = {"random": tr < rn, "zero": tr < zr, "untrained": tr < un}
    multi_seed_var = float(np.std([stability_distances[k] for k in stability_runs]))
    passes = {
        "finite_ratio == 1.0": finite_ok,
        "amplitude_range > 0.05": range_ok,
        "beats random_noise baseline": beats["random"],
        "beats zero baseline": beats["zero"],
        "beats untrained_sample": beats["untrained"],
    }
    n_pass = sum(passes.values())
    verdict = "PASS" if n_pass == len(passes) else ("PARTIAL" if n_pass >= 3 else "FAIL")

    payload = {
        "date": datetime.now().isoformat(),
        "version": "v2: force video=image broadcast + latent normalization",
        "train_args": vars(args),
        "sampler": {"type": "DDIM", "prediction": "x_0", "eta": 0.0,
                    "num_steps": 50, "schedule_steps": args.diffusion_steps},
        "latent_stats": {"mean": stats.mean, "std": stats.std},
        "metrics": {
            "per_source": metrics_per_source,
            "descriptor_distance_to_train": distances,
            "stability": {"per_seed_distances": stability_distances,
                          "std_across_seeds": multi_seed_var},
        },
        "verdict": verdict,
        "passes": passes,
    }
    Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[output] JSON -> {args.report_out}", flush=True)

    md = [
        f"# Unconditional Sample Fidelity v2 Report\n\n",
        f"**Date**: {payload['date']}\n",
        f"**Version**: {payload['version']}\n",
        f"**Verdict**: **{verdict}** ({n_pass}/{len(passes)} checks pass)\n\n",
        "## Train args\n```json\n" + json.dumps(vars(args), indent=2) + "\n```\n\n",
        f"## Latent stats (Fix B)\nμ={stats.mean:+.4f}  σ={stats.std:.4f}\n\n",
        "## descriptor_distance_to_train\n\n",
        "| source | distance |\n|---|---|\n",
        *[f"| `{name}` | {d:.4f} |\n" for name, d in sorted(distances.items(), key=lambda kv: kv[1])],
        "\n## Per-source key metrics\n\n",
        "| source | finite | amp_range | entropy | tile_residue |\n|---|---|---|---|---|\n",
        *[f"| `{name}` | {m['finite_ratio']:.3f} | {m['amplitude_range']:.3f} | "
          f"{m['entropy_mean']:.3f} | {m['tile_residue_mean']:.3f} |\n"
          for name, m in metrics_per_source.items()],
        "\n## PASS checks\n\n",
        *[f"- {'PASS' if v else 'FAIL'}  {k}\n" for k, v in passes.items()],
        f"\n## Multi-seed stability\n\nstd of distance across 3 seeds: **{multi_seed_var:.4f}**\n",
    ]
    Path(args.md_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.md_out).write_text("".join(md), encoding="utf-8")
    print(f"[output] MD report -> {args.md_out}", flush=True)
    print(f"\n=== VERDICT: {verdict} ===", flush=True)


if __name__ == "__main__":
    main()
