"""DDPM training + DDIM sampling on VOD substrate.

Pipeline:
    1. Build NativeVOD with time_dim > 0 (time-conditioned denoiser)
    2. Train with diffusion_loss (x_0 prediction at random t) + L_recon
    3. Sample new images via DDIM from N(0, I)
    4. Save PNGs

This is the "true generate" mode — model produces from random noise,
not recall from a perturbed clean input.
"""
from __future__ import annotations
import argparse, time
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
from PIL import Image

import sys
sys.path.insert(0, "D:/VOD/prototype")

from vod_minimal.blocky_scattering import build_blocky_scattering_batch
from vod_minimal.native import (
    LATENT_HW, LATENT_T, NativeVOD, NativeVODConfig,
    views_to_torch, views_to_numpy,
)
from vod_minimal.diffusion import make_schedule, diffusion_loss, ddim_sample


def to_png(a, p):
    a = np.asarray(a).squeeze()
    if a.ndim != 2:
        a = a.reshape(a.shape[0], -1)
    a = (a - a.min()) / (a.max() - a.min() + 1e-9) * 255
    Image.fromarray(a.astype(np.uint8), 'L').save(p)


def save_video_grid(video, p, ncols=4):
    v = np.asarray(video).squeeze()
    T, H, W = v.shape
    nrows = (T + ncols - 1) // ncols
    grid = np.zeros((nrows * H, ncols * W), dtype=np.uint8)
    for t in range(T):
        r, c = t // ncols, t % ncols
        a = v[t]
        a = (a - a.min()) / (a.max() - a.min() + 1e-9) * 255
        grid[r*H:(r+1)*H, c*W:(c+1)*W] = a.astype(np.uint8)
    Image.fromarray(grid, mode='L').save(p)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=430)
    p.add_argument("--train-n", type=int, default=64)
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--diffusion-steps", type=int, default=200)
    p.add_argument("--time-dim", type=int, default=64)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--channels", type=int, default=4)
    p.add_argument("--ddim-steps", type=int, default=50)
    p.add_argument("--n-samples", type=int, default=6)
    p.add_argument("--w-recon", type=float, default=0.1)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--out", default="D:/VOD/prototype/generated/diffusion")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    schedule = make_schedule(num_steps=args.diffusion_steps).to(device)

    # Training data: clean Chladni fields (no blocky / flicker — generation
    # task wants clean target distribution, not stress-corrupted).
    rng = np.random.default_rng(args.seed)
    train = build_blocky_scattering_batch(
        rng, batch_size=args.train_n, size=LATENT_HW, frames=LATENT_T,
        artifact_strength=0.0, tile=4, spacetime=True,
        temporal_mode='static', flicker_strength=0.0, paired_denoising=True,
    )
    # Encode all targets to latent x_0 once
    cfg = NativeVODConfig(channels=args.channels, hidden=args.hidden,
                           denoise_steps=4, backbone='unet', time_dim=args.time_dim)
    m = NativeVOD(cfg).to(device)
    n_params = sum(p_.numel() for p_ in m.parameters())
    print(f"NativeVOD(time_dim={args.time_dim}) params: {n_params:,}", flush=True)
    print(f"diffusion_steps={args.diffusion_steps} ddim_steps={args.ddim_steps}", flush=True)
    print(f"training {args.epochs} epochs on {args.train_n} clean samples...", flush=True)

    # Pre-encode training targets to latent x_0 batch (B, T, H, W, C)
    with torch.no_grad():
        x_0_list = []
        for s in train.samples:
            t = views_to_torch(s.target_views, device)
            x_0_list.append(m.encode(t))
        x_0_batch = torch.stack(x_0_list, dim=0)  # (B, T, H, W, C)

    opt = torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=1e-4)
    t0 = time.time()
    for ep in range(args.epochs):
        m.train()
        opt.zero_grad(set_to_none=True)

        # Re-encode each step (encoder is also being trained for L_recon)
        targets = [views_to_torch(s.target_views, device) for s in train.samples]
        x_0_live = torch.stack([m.encode(t) for t in targets], dim=0)

        # Diffusion loss on latent
        L_diff = diffusion_loss(m, x_0_live, schedule, prediction="x_0")

        # L_recon: encode→decode identity on clean targets
        recon_terms = []
        for tv in targets:
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
            print(f"  ep={ep+1:4d}  L_total={float(loss):.4f}  "
                  f"L_diff={float(L_diff):.4f}  L_recon={float(L_recon):.4f}", flush=True)
    print(f"trained in {time.time()-t0:.1f}s", flush=True)

    # Sampling
    m.eval()
    print()
    print(f"sampling {args.n_samples} new images via DDIM ({args.ddim_steps} steps)...", flush=True)
    sample_shape = (args.n_samples, LATENT_T, LATENT_HW, LATENT_HW, args.channels)
    g = torch.Generator(device=device).manual_seed(args.seed + 1)
    t0 = time.time()
    x_0_sampled = ddim_sample(m, sample_shape, schedule, num_steps=args.ddim_steps,
                              device=device, generator=g, prediction="x_0")
    print(f"sampled in {time.time()-t0:.1f}s, shape {tuple(x_0_sampled.shape)}, "
          f"range [{float(x_0_sampled.min()):.3f}, {float(x_0_sampled.max()):.3f}]", flush=True)

    # Decode each sample
    with torch.no_grad():
        for i in range(args.n_samples):
            U = x_0_sampled[i]
            views = m.decode(U)
            views_np = views_to_numpy(views)
            sd = out / f"sample_{i:02d}"; sd.mkdir(exist_ok=True)
            if "image" in views_np:
                to_png(views_np["image"], sd / "image.png")
            if "video" in views_np:
                save_video_grid(views_np["video"], sd / "video_grid.png")
            print(f"  sample {i}: {sd}/", flush=True)

    print(f"\ndone. inspect {out}/sample_*/")


if __name__ == "__main__":
    main()
