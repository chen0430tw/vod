"""Unconditional sample fidelity v3 — GPU-friendly batched training.

Three fixes vs v1:

  Fix A — video target = image.broadcast(T)
      `temporal_mode='static'` does not control target_views; the
      Chladni spacetime field has aliasing modes (mt=2 on frames=8) that
      give a 4-period flicker in raw video targets. We force every frame
      to equal the still image, killing temporal variation.

  Fix B — latent normalization (EMA-tracked μ, σ)
      Encoder output has std ≈ 0.22 << 1.0; DDIM init N(0,I) is
      off-manifold. We normalize x_0 in diffusion loss, sample in
      normalized space, unnormalize before decode.

  Fix C (NEW in v3) — batched training, no per-sample Python loop
      v1/v2 ran `[m.encode(t) for t in targets]` every step — 256 small
      ops with GPU sync between each, leaving H100 at ~55% utilization
      and 7GB/80GB memory. v3 pre-stacks targets into a single batched
      dict and uses helper functions that bypass NativeVOD.encode (which
      only handles single-sample dicts) to do one batched encode / one
      batched decode per step.

All three fixes live in this trainer script — no changes to vod_minimal/.
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
from vod_minimal.diffusion import diffusion_loss, ddim_sample, make_schedule
from vod_minimal.metrics import artifact_metrics, descriptor
from vod_minimal.native import (
    LATENT_HW, LATENT_T, NativeVOD, NativeVODConfig,
    views_to_torch,
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
#  Batched encode/decode (Fix C)
#  Bypasses NativeVOD.encode which only accepts single-sample dicts.
#  Uses the same nn.Linear weights, just calls them on batched tensors.
# --------------------------------------------------------------------- #

def batched_encode(model: NativeVOD, batched_views: dict[str, torch.Tensor]) -> torch.Tensor:
    """Encode a dict of batched media views to a single (B, T, H, W, C) latent.

    batched_views format:
        image: (B, H, W)
        video: (B, T, H, W)
    """
    contributions = []
    if "image" in batched_views:
        img = batched_views["image"]                                    # (B, H, W)
        u = model.enc_image(img.unsqueeze(-1))                          # (B, H, W, C)
        u = u.unsqueeze(1).expand(-1, LATENT_T, -1, -1, -1).contiguous()  # (B, T, H, W, C)
        contributions.append(u)
    if "video" in batched_views:
        vid = batched_views["video"]                                    # (B, T, H, W)
        u = model.enc_video(vid.unsqueeze(-1))                          # (B, T, H, W, C)
        contributions.append(u)
    return torch.stack(contributions, dim=0).mean(dim=0)


def batched_decode(model: NativeVOD, U_batched: torch.Tensor) -> dict[str, torch.Tensor]:
    """Decode (B, T, H, W, C) latent back to media views.

    Mirrors NativeVOD.decode: image takes the middle T-slice, video uses all T.
    """
    out = {}
    mid = U_batched.shape[1] // 2
    out["image"] = model.dec_image(U_batched[:, mid]).squeeze(-1)       # (B, H, W)
    out["video"] = model.dec_video(U_batched).squeeze(-1)               # (B, T, H, W)
    return out


# --------------------------------------------------------------------- #
#  Latent normalization (Fix B)
# --------------------------------------------------------------------- #

class LatentStats:
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
#  Metrics
# --------------------------------------------------------------------- #

def aggregate_metrics(images: list[np.ndarray]) -> dict:
    if not images:
        return {}
    desc_vecs = np.stack([descriptor("img", np.asarray(im)).vec() for im in images], axis=0)
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
    av = np.array([a["descriptor_mean"][k] for k in DESCRIPTOR_KEYS])
    bv = np.array([b["descriptor_mean"][k] for k in DESCRIPTOR_KEYS])
    return float(np.linalg.norm(av - bv))


# --------------------------------------------------------------------- #
#  Main pipeline pieces
# --------------------------------------------------------------------- #

def build_batched_targets(args, device, rng):
    batch = build_blocky_scattering_batch(
        rng, batch_size=args.train_n, size=LATENT_HW, frames=LATENT_T,
        artifact_strength=0.0, tile=4, spacetime=True,
        temporal_mode="static", flicker_strength=0.0, paired_denoising=True,
    )
    images_np = [np.asarray(s.target_views["image"]) for s in batch.samples]
    images_t = torch.stack(
        [torch.as_tensor(im, dtype=torch.float32) for im in images_np], dim=0
    ).to(device)                                                # (B, H, W)
    # Fix A: force video = image broadcast (B, T, H, W)
    videos_t = images_t.unsqueeze(1).expand(-1, LATENT_T, -1, -1).contiguous()
    return {"image": images_t, "video": videos_t}, images_np


def train_model(args, device, batched_targets):
    cfg = NativeVODConfig(
        channels=args.channels, hidden=args.hidden,
        denoise_steps=4, backbone="unet", time_dim=args.time_dim,
    )
    m = NativeVOD(cfg).to(device)
    n_params = sum(p_.numel() for p_ in m.parameters())
    print(f"[train] params={n_params:,} hidden={args.hidden} ch={args.channels} train_n={args.train_n}",
          flush=True)

    schedule = make_schedule(num_steps=args.diffusion_steps).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=1e-4)
    stats = LatentStats(ema_decay=0.99)

    t0 = time.time()
    for ep in range(args.epochs):
        m.train()
        opt.zero_grad(set_to_none=True)
        x_0_live = batched_encode(m, batched_targets)               # (B, T, H, W, C)
        stats.update(x_0_live)
        x_0_norm = stats.normalize(x_0_live)
        L_diff = diffusion_loss(m, x_0_norm, schedule, prediction="x_0")

        # Recon: decode encoded latent back to media, compare with batched targets
        rec = batched_decode(m, x_0_live)
        L_recon_img = F.mse_loss(rec["image"], batched_targets["image"])
        L_recon_vid = F.mse_loss(rec["video"], batched_targets["video"])
        L_recon = (L_recon_img + L_recon_vid) / 2

        loss = L_diff + args.w_recon * L_recon
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if (ep + 1) % max(1, args.epochs // 10) == 0:
            print(f"  ep={ep+1:4d}  L_diff={float(L_diff.detach()):.4f}  "
                  f"L_recon={float(L_recon.detach()):.4f}  "
                  f"latent_μ={stats.mean:+.3f}  latent_σ={stats.std:.3f}", flush=True)
    elapsed = time.time() - t0
    print(f"[train] done in {elapsed:.1f}s   "
          f"({elapsed*1000/args.epochs:.1f} ms/epoch)   "
          f"final stats μ={stats.mean:+.4f} σ={stats.std:.4f}", flush=True)
    return m, schedule, stats


def sample_decoded_images(model, schedule, stats, n, seed, device):
    g = torch.Generator(device=device).manual_seed(seed)
    shape = (n, LATENT_T, LATENT_HW, LATENT_HW, model.config.channels)
    x_norm = ddim_sample(model, shape, schedule, num_steps=50,
                         device=device, generator=g, prediction="x_0")
    x_unnorm = stats.unnormalize(x_norm)
    with torch.no_grad():
        decoded = batched_decode(model, x_unnorm)
    return [decoded["image"][i].detach().cpu().numpy() for i in range(n)]


def random_noise_decoded(model, n, seed, device):
    g = torch.Generator(device=device).manual_seed(seed)
    shape = (n, LATENT_T, LATENT_HW, LATENT_HW, model.config.channels)
    x = torch.randn(shape, device=device, generator=g, dtype=next(model.parameters()).dtype)
    with torch.no_grad():
        decoded = batched_decode(model, x)
    return [decoded["image"][i].detach().cpu().numpy() for i in range(n)]


def zero_decoded(model, n, device):
    shape = (n, LATENT_T, LATENT_HW, LATENT_HW, model.config.channels)
    x = torch.zeros(shape, device=device, dtype=next(model.parameters()).dtype)
    with torch.no_grad():
        decoded = batched_decode(model, x)
    return [decoded["image"][i].detach().cpu().numpy() for i in range(n)]


def gate0_reconstruct(model, batched_targets, n, device):
    sub = {k: v[:n] for k, v in batched_targets.items()}
    with torch.no_grad():
        U = batched_encode(model, sub)
        decoded = batched_decode(model, U)
    return [decoded["image"][i].detach().cpu().numpy() for i in range(n)]


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
    p.add_argument("--out", default="generated/diffusion_samples_v3")
    p.add_argument("--report-out", default="prototype/unconditional_fidelity_result_v3.json")
    p.add_argument("--md-out", default="prototype/unconditional_fidelity_report_v3.md")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    print(f"[device] {device}", flush=True)
    print("[data] building batched Chladni targets + Fix A (video=image.broadcast)...",
          flush=True)
    batched_targets, train_images = build_batched_targets(args, device, rng)

    # Verify Fix A
    diff = (batched_targets["video"][:, 1:] - batched_targets["video"][:, :-1]).abs().max().item()
    print(f"[verify] forced-static video: max |frame[t+1]-frame[t]| = {diff:.6f}",
          flush=True)
    print(f"[verify] batched_targets shapes: "
          f"image={tuple(batched_targets['image'].shape)} video={tuple(batched_targets['video'].shape)}",
          flush=True)

    trained_model, schedule, stats = train_model(args, device, batched_targets)
    cfg = NativeVODConfig(
        channels=args.channels, hidden=args.hidden,
        denoise_steps=4, backbone="unet", time_dim=args.time_dim,
    )
    untrained_model = NativeVOD(cfg).to(device)
    untrained_stats = LatentStats()
    untrained_stats.mean, untrained_stats.std = 0.0, 1.0
    untrained_stats.initialized = True

    print("[sample] generating samples for all sources...", flush=True)
    sources = {
        "train_reference":         train_images[:args.n_samples],
        "trained_sample":          sample_decoded_images(trained_model, schedule, stats,
                                                        args.n_samples, args.seed + 1, device),
        "untrained_sample":        sample_decoded_images(untrained_model, schedule, untrained_stats,
                                                        args.n_samples, args.seed + 2, device),
        "random_noise_baseline":   random_noise_decoded(trained_model, args.n_samples,
                                                       args.seed + 3, device),
        "zero_baseline":           zero_decoded(trained_model, args.n_samples, device),
        "gate0_recon":             gate0_reconstruct(trained_model, batched_targets, 8, device),
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
    multi_seed_var = float(np.std([stability_distances[k] for k in stability_runs]))
    passes = {
        "finite_ratio == 1.0": finite_ok,
        "amplitude_range > 0.05": range_ok,
        "beats random_noise baseline": tr < rn,
        "beats zero baseline": tr < zr,
        "beats untrained_sample": tr < un,
    }
    n_pass = sum(passes.values())
    verdict = "PASS" if n_pass == len(passes) else ("PARTIAL" if n_pass >= 3 else "FAIL")

    payload = {
        "date": datetime.now().isoformat(),
        "version": "v3: Fix A (video static) + Fix B (latent norm) + Fix C (batched training)",
        "device": str(device),
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
        "# Unconditional Sample Fidelity v3 Report\n\n",
        f"**Date**: {payload['date']}\n",
        f"**Version**: {payload['version']}\n",
        f"**Device**: {device}\n",
        f"**Verdict**: **{verdict}** ({n_pass}/{len(passes)} checks pass)\n\n",
        "## Train args\n```json\n" + json.dumps(vars(args), indent=2) + "\n```\n\n",
        f"## Latent stats (Fix B)\nμ={stats.mean:+.4f}  σ={stats.std:.4f}\n\n",
        "## descriptor_distance_to_train\n\n",
        "| source | distance |\n|---|---|\n",
        *[f"| `{name}` | {d:.4f} |\n" for name, d in sorted(distances.items(), key=lambda kv: kv[1])],
        "\n## Per-source metrics\n\n",
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
