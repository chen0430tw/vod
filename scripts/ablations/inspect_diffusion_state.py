"""Dump actual tensor stats at every stage of the diffusion train + sample
pipeline. No assumptions, no narrative, just numbers.

Stages dumped:
  1. raw target images (Chladni)
  2. encode(target) — what the diffusion is asked to produce
  3. q_sample(x_0, t) for t in [0, 50, 100, 150, 199]
  4. DDIM x_T initial noise
  5. model.denoise(x, t) prediction at each ddim step (head/middle/tail)
  6. final x_0_pred
  7. decode(x_0_pred)

For each tensor: shape, dtype, mean, std, min, max, |x|.mean.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_PROTO = Path(__file__).resolve().parent.parent.parent / "prototype"
if str(_PROTO) not in sys.path:
    sys.path.insert(0, str(_PROTO))

from vod_minimal.blocky_scattering import build_blocky_scattering_batch
from vod_minimal.diffusion import diffusion_loss, make_schedule, q_sample
from vod_minimal.native import (
    LATENT_HW, LATENT_T, NativeVOD, NativeVODConfig,
    views_to_torch,
)


def stats(name: str, t: torch.Tensor):
    t_f = t.detach().float().cpu()
    shape_str = str(tuple(t.shape))
    print(f"  {name:40s}  shape={shape_str:22s}  "
          f"mean={t_f.mean().item():+.4f}  std={t_f.std().item():.4f}  "
          f"min={t_f.min().item():+.4f}  max={t_f.max().item():+.4f}  "
          f"|x|.mean={t_f.abs().mean().item():.4f}")


def main():
    device = torch.device("cpu")
    torch.manual_seed(430)
    rng = np.random.default_rng(430)

    print("=" * 100)
    print("STAGE 1: build training data")
    print("=" * 100)
    batch = build_blocky_scattering_batch(
        rng, batch_size=8, size=LATENT_HW, frames=LATENT_T,
        artifact_strength=0.0, tile=4, spacetime=True,
        temporal_mode="static", flicker_strength=0.0, paired_denoising=True,
    )
    targets = [views_to_torch(s.target_views, device) for s in batch.samples]
    raw_image = torch.stack([t["image"] for t in targets], dim=0)
    raw_video = torch.stack([t["video"] for t in targets], dim=0)
    stats("raw_target image", raw_image)
    stats("raw_target video", raw_video)

    print()
    print("=" * 100)
    print("STAGE 2: encode (untrained model)")
    print("=" * 100)
    cfg = NativeVODConfig(channels=4, hidden=32, denoise_steps=4,
                          backbone="unet", time_dim=64)
    m = NativeVOD(cfg).to(device)
    with torch.no_grad():
        x_0_untrained = torch.stack([m.encode(t) for t in targets], dim=0)
    stats("encode(target) UNTRAINED", x_0_untrained)

    print()
    print("=" * 100)
    print("STAGE 3: train short (100 epochs, w_recon=1.0) and re-encode")
    print("=" * 100)
    schedule = make_schedule(num_steps=200).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-4)
    for ep in range(100):
        m.train()
        opt.zero_grad(set_to_none=True)
        x_0_live = torch.stack([m.encode(t) for t in targets], dim=0)
        L_diff = diffusion_loss(m, x_0_live, schedule, prediction="x_0")
        recon_terms = []
        for tv in targets:
            U = m.encode(tv)
            rec = m.decode(U)
            for k in ("image", "video"):
                if k in rec and k in tv:
                    recon_terms.append(F.mse_loss(rec[k], tv[k]))
        L_recon = torch.stack(recon_terms).mean()
        loss = L_diff + 1.0 * L_recon
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if (ep + 1) % 25 == 0:
            print(f"  ep={ep+1:3d}  L_diff={float(L_diff):.4f}  L_recon={float(L_recon):.4f}")

    m.eval()
    with torch.no_grad():
        x_0_trained = torch.stack([m.encode(t) for t in targets], dim=0)
    stats("encode(target) TRAINED", x_0_trained)

    print()
    print("=" * 100)
    print("STAGE 4: forward q_sample(x_0, t) — what model is trained to denoise")
    print("=" * 100)
    with torch.no_grad():
        for t_val in [0, 50, 100, 150, 199]:
            t_tensor = torch.full((8,), t_val, dtype=torch.long, device=device)
            x_t = q_sample(x_0_trained, t_tensor, schedule)
            a_bar = float(schedule.alphas_cumprod[t_val])
            stats(f"q_sample(x_0, t={t_val}) [α_bar={a_bar:.4f}]", x_t)

    print()
    print("=" * 100)
    print("STAGE 5: DDIM reverse — actual x_t at each step + model prediction")
    print("=" * 100)
    shape = (4, LATENT_T, LATENT_HW, LATENT_HW, 4)
    g = torch.Generator(device=device).manual_seed(431)
    x = torch.randn(shape, device=device, generator=g)
    stats("x_T (DDIM init, randn)", x)

    timesteps = torch.linspace(199, 0, 50, dtype=torch.long, device=device)
    with torch.no_grad():
        for i, t in enumerate(timesteps):
            t_batch = t.expand(shape[0])
            pred = m.denoise(x, t=t_batch)  # x_0 prediction
            a_t = schedule.alphas_cumprod[t]
            eps_pred = (x - torch.sqrt(a_t) * pred) / torch.sqrt(torch.clamp(1 - a_t, min=1e-9))

            if i in (0, 1, 5, 12, 25, 37, 48, 49):
                print(f"\n  --- DDIM step {i:2d}/50  t={int(t):3d}  α_bar={float(a_t):.4f} ---")
                stats(f"  x_t (input to denoise)", x)
                stats(f"  pred (x_0_pred from denoise)", pred)
                stats(f"  eps_pred (computed)", eps_pred)

            if i < len(timesteps) - 1:
                t_next = timesteps[i + 1]
                a_next = schedule.alphas_cumprod[t_next]
                x = torch.sqrt(a_next) * pred + torch.sqrt(torch.clamp(1 - a_next, min=0)) * eps_pred
            else:
                x = pred

    print()
    print("=" * 100)
    print("STAGE 6: final x_0 and decoded image")
    print("=" * 100)
    stats("final x_0 from DDIM", x)
    with torch.no_grad():
        final_decode = m.decode(x[0])
    stats("decode(x_0_final)['image']", final_decode["image"])

    print()
    print("=" * 100)
    print("COMPARISON: trained x_0 stats VS DDIM x_T (the mismatch hypothesis)")
    print("=" * 100)
    stats("encode(target) TRAINED  (target distribution)", x_0_trained)
    stats("randn(shape)            (DDIM init = N(0,I))", torch.randn_like(x_0_trained, generator=g))
    print()
    print("If trained x_0 std << 1.0, DDIM N(0,I) start is OFF-MANIFOLD.")
    print("If trained x_0 std ≈ 1.0, DDIM start is fine — bug is elsewhere.")


if __name__ == "__main__":
    main()
