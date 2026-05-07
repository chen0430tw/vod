"""Gate 0 verification: train with L_recon + L_clean_noop, then test:
  1. decode(encode(x)) ≈ x   (round-trip identity)
  2. denoise_path(encode(clean)) ≈ encode(clean)  (no-op stability)
  3. Visible PNG output looks like Chladni
"""
from __future__ import annotations
import argparse, time
from pathlib import Path
import numpy as np, torch
from PIL import Image

import sys
sys.path.insert(0, "D:/VOD/prototype")

from vod_minimal.blocky_scattering import build_blocky_scattering_batch
from vod_minimal.native import (
    LATENT_HW, LATENT_T, NativeLossWeights, NativeVOD, NativeVODConfig,
    native_total_loss, views_to_torch, views_to_numpy,
)


def to_png(a, p):
    a = np.asarray(a).squeeze()
    a = (a - a.min()) / (a.max() - a.min() + 1e-9) * 255
    Image.fromarray(a.astype(np.uint8), 'L').save(p)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=430)
    p.add_argument("--train-n", type=int, default=16)
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--w-recon", type=float, default=1.0)
    p.add_argument("--w-clean-noop", type=float, default=1.0)
    p.add_argument("--w-distinctive-scale", type=float, default=0.03)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--out", default="D:/VOD/prototype/generated/gate0")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    train = build_blocky_scattering_batch(
        np.random.default_rng(args.seed), batch_size=args.train_n,
        size=LATENT_HW, frames=LATENT_T,
        artifact_strength=0.6, tile=4, spacetime=True,
        temporal_mode='flicker', flicker_strength=0.3, paired_denoising=True,
    )

    cfg = NativeVODConfig(channels=4, hidden=32, denoise_steps=4, backbone='unet')
    m = NativeVOD(cfg).to(device)
    s = args.w_distinctive_scale
    weights = NativeLossWeights(
        field=0.5, media=1.0, text=0.0,
        temporal=0.1*s, artifact=0.1*s, binary_twin_pixel=0.1*s, aimp=0.1*s,
        recon=args.w_recon, clean_noop=args.w_clean_noop,
    )
    print(f"weights: recon={args.w_recon} clean_noop={args.w_clean_noop} "
          f"distinctive_scale={s}", flush=True)
    opt = torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=1e-4)
    print(f"training {args.epochs} epochs...", flush=True)
    t0 = time.time()
    for ep in range(args.epochs):
        m.train(); opt.zero_grad(set_to_none=True)
        L, components = [], None
        for samp in train.samples:
            loss, comp = native_total_loss(
                m, views_to_torch(samp.noisy_views, device),
                views_to_torch(samp.target_views, device),
                weights=weights,
            )
            L.append(loss); components = comp
        torch.stack(L).mean().backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if (ep + 1) % max(1, args.epochs // 8) == 0:
            print(f"  ep={ep+1:4d}  L_total={components['L_total']:.4f}  "
                  f"L_recon={components['L_recon']:.4f}  "
                  f"L_clean_noop={components['L_clean_noop']:.4f}", flush=True)
    print(f"trained in {time.time()-t0:.1f}s", flush=True)

    # Gate 0 tests
    test = build_blocky_scattering_batch(
        np.random.default_rng(args.seed + 31), batch_size=4, size=LATENT_HW,
        frames=LATENT_T, artifact_strength=0.0, tile=4, spacetime=True,
        temporal_mode='static', flicker_strength=0.0, paired_denoising=True,
    )

    m.eval()
    print()
    print("=" * 60)
    print("Gate 0a — round-trip identity: decode(encode(x)) ≈ x")
    print("=" * 60)
    with torch.no_grad():
        for i, samp in enumerate(test.samples):
            t = views_to_torch(samp.target_views, device)
            U = m.encode(t)
            rec = m.decode(U)
            rec_np = views_to_numpy(rec)
            for k in ('image', 'video'):
                if k in rec_np and k in samp.target_views:
                    orig = np.asarray(samp.target_views[k])
                    r = np.asarray(rec_np[k])
                    mse = float(np.mean((orig - r) ** 2))
                    print(f"  sample {i} {k}: orig=[{orig.min():.3f},{orig.max():.3f}] "
                          f"rec=[{r.min():.3f},{r.max():.3f}] mse={mse:.4f}")

    print()
    print("=" * 60)
    print("Gate 0b — clean no-op: denoise_path(encode(clean)) ≈ encode(clean)")
    print("=" * 60)
    with torch.no_grad():
        for i, samp in enumerate(test.samples):
            t = views_to_torch(samp.target_views, device)
            U = m.encode(t)
            U_pred = m.denoise_path(U, steps=8)
            mse = float(((U - U_pred) ** 2).mean())
            print(f"  sample {i}: U range [{float(U.min()):.3f},{float(U.max()):.3f}]  "
                  f"U_pred range [{float(U_pred.min()):.3f},{float(U_pred.max()):.3f}]  "
                  f"latent_mse={mse:.4f}  finite={torch.isfinite(U_pred).all().item()}")

    print()
    print("=" * 60)
    print("Gate 0c — visible reconstruction")
    print("=" * 60)
    with torch.no_grad():
        for i, samp in enumerate(test.samples):
            t = views_to_torch(samp.target_views, device)
            U = m.encode(t)
            U_pred = m.denoise_path(U, steps=8)
            full = m.decode(U_pred)
            full_np = views_to_numpy(full)
            sd = out / f"sample_{i:02d}"; sd.mkdir(exist_ok=True)
            to_png(samp.target_views['image'], sd / 'orig.png')
            to_png(full_np['image'], sd / 'pipeline.png')
            U_decoded_only = m.decode(U)
            to_png(views_to_numpy(U_decoded_only)['image'], sd / 'recon_no_denoise.png')
            print(f"  sample {i} → {sd}/  (orig.png, recon_no_denoise.png, pipeline.png)")

    print()
    print(f"done. inspect {out}/sample_*/")


if __name__ == "__main__":
    main()
