"""Preflight gate for VOD-v16-real-image-smoke.

Per the project instruction §4: hard gates must PASS before any GPU
launch. Failure → report minimal fix, do not run training.

Hard gates:
  G1  finite data
  G2  image range in [-1, 1] (or documented other range)
  G3  Gate 0 reconstruction smoke: decode(encode(x)) not collapsed
  G4  latent std sane (close to v11 baseline ~0.5, or documented)
  G5  encoded channel std min > 0.05
  G6  encoded channel correlation max < 0.95, mean < 0.5

Soft warnings:
  S1  raw feature basis correlation
  S2  DCT feature corr after lift on raw images

Soft warnings DO NOT block GPU launch — only G1-G6 do.

Usage:
    py -3.13 scripts/preflight_real_image.py \
        --dataset cifar10 --image-size 32 --train-n 64
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS / "ablations"))
sys.path.insert(0, str(_THIS.parent / "prototype"))


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dataset", choices=["cifar10", "local"], default="cifar10")
    p.add_argument("--data-dir", type=str, default="D:/VOD/data/real_images")
    p.add_argument("--data-cache-dir", type=str, default=None)
    p.add_argument("--image-size", type=int, default=32)
    p.add_argument("--train-n", type=int, default=128)
    p.add_argument("--channels", type=int, default=8)
    p.add_argument("--seed", type=int, default=430)
    args = p.parse_args()

    # Configure native module BEFORE imports that use it
    os.environ["VOD_IMAGE_SIZE"] = str(args.image_size)
    os.environ["STATIC_T"] = "1"
    import vod_minimal.native as _nm
    _nm.LATENT_T = 1
    _nm.LATENT_HW = int(args.image_size)
    _nm.AUDIO_SIZE = _nm.LATENT_T * _nm.LATENT_HW * _nm.LATENT_HW

    # Import v16's helpers and the real_image_smoke loader
    sys.path.insert(0, str(_THIS / "ablations"))
    from run_real_image_smoke import (
        FieldLift, FieldLiftedEncoder,
        _load_cifar10_grayscale, _load_local_imagefolder,
        batched_encode,
    )
    from vod_minimal.native import NativeVOD, NativeVODConfig

    # Dummy "args" for the loader
    class _A: pass
    a = _A()
    a.train_n = args.train_n
    a.image_size = args.image_size
    a.dataset = args.dataset
    a.data_dir = args.data_dir
    a.data_cache_dir = args.data_cache_dir

    rng = np.random.default_rng(args.seed)
    print("=== preflight_real_image: 6 hard gates + 2 soft warnings ===\n")

    # =================================================================== #
    # G1: load data (finite)
    # =================================================================== #
    print("[G1] finite data")
    try:
        if args.dataset == "cifar10":
            try:
                images_np = _load_cifar10_grayscale(a, rng)
            except Exception as e:
                print(f"  HF cifar10 unreachable: {type(e).__name__}: {e}")
                images_np = _load_local_imagefolder(a, rng)
        else:
            images_np = _load_local_imagefolder(a, rng)
    except Exception as e:
        print(f"  FAIL: data load error {type(e).__name__}: {e}")
        return 1
    finite = bool(np.isfinite(images_np).all())
    print(f"  shape={images_np.shape}  finite={finite}")
    g1 = finite
    print(f"  {'PASS' if g1 else 'FAIL'}\n")

    # =================================================================== #
    # G2: image range [-1, 1] (we normalize in loader)
    # =================================================================== #
    print("[G2] image range [-1, 1]")
    lo, hi = float(images_np.min()), float(images_np.max())
    print(f"  range=[{lo:.3f}, {hi:.3f}]")
    g2 = (lo >= -1.05) and (hi <= 1.05)
    print(f"  {'PASS' if g2 else 'FAIL'}\n")

    # =================================================================== #
    # G3: Gate 0 reconstruction smoke (encode → decode round-trip)
    # =================================================================== #
    print("[G3] Gate-0 reconstruction smoke (40 steps L2)")
    cfg = NativeVODConfig(channels=args.channels, hidden=16,
                          denoise_steps=4, backbone="unet", time_dim=8)
    m = NativeVOD(cfg)
    m.enc_image = FieldLiftedEncoder(channels=args.channels)
    m.enc_video = FieldLiftedEncoder(channels=args.channels)
    images_t = torch.from_numpy(images_np)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-3)
    for step in range(40):
        opt.zero_grad()
        outs = []
        for i in range(images_t.shape[0]):
            outs.append(m._encode_image(images_t[i]))
        Lat = torch.stack(outs, dim=0).float()
        rec = []
        for i in range(Lat.shape[0]):
            rec.append(m.dec_image(Lat[i, 0]).squeeze(-1))
        rec = torch.stack(rec, dim=0)
        loss = ((rec - images_t) ** 2).mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        recon_amp = float(rec.max() - rec.min())
        image_amp = float(images_t.max() - images_t.min())
    ratio = recon_amp / max(image_amp, 1e-6)
    print(f"  recon_amp={recon_amp:.3f}  image_amp={image_amp:.3f}  ratio={ratio:.3f}")
    print(f"  threshold: ratio >= 0.5")
    g3 = ratio >= 0.5
    print(f"  {'PASS' if g3 else 'FAIL'}\n")

    # =================================================================== #
    # G4: latent std sane
    # =================================================================== #
    print("[G4] latent std sanity")
    with torch.no_grad():
        outs = []
        for i in range(images_t.shape[0]):
            outs.append(m._encode_image(images_t[i]))
        L_ref = torch.stack(outs, dim=0).float()
    L_std_global = float(L_ref.std())
    print(f"  global latent std = {L_std_global:.4f}")
    print(f"  threshold: 0.05 < std < 5.0")
    g4 = (0.05 < L_std_global < 5.0)
    print(f"  {'PASS' if g4 else 'FAIL'}\n")

    # =================================================================== #
    # G5: encoded channel std min > 0.05
    # =================================================================== #
    print("[G5] per-channel std min > 0.05")
    per_ch_std = L_ref.std(dim=(0, 1, 2, 3)).numpy()
    print(f"  per-ch std: {per_ch_std.round(3).tolist()}")
    print(f"  min = {per_ch_std.min():.4f}  threshold: > 0.05")
    g5 = float(per_ch_std.min()) > 0.05
    print(f"  {'PASS' if g5 else 'FAIL'}\n")

    # =================================================================== #
    # G6: channel correlation max < 0.95, mean < 0.5
    # =================================================================== #
    print("[G6] channel correlation")
    L_flat = L_ref.reshape(-1, args.channels).numpy()
    if L_flat.shape[0] > 1:
        cmat = np.corrcoef(L_flat.T)
        off = cmat[np.triu_indices_from(cmat, k=1)]
        cmax = float(np.abs(off).max())
        cmean = float(np.abs(off).mean())
    else:
        cmax = cmean = 0.0
    print(f"  max |corr|={cmax:.4f}  mean |corr|={cmean:.4f}")
    print(f"  thresholds: max<0.95 AND mean<0.5")
    g6 = (cmax < 0.95) and (cmean < 0.5)
    print(f"  {'PASS' if g6 else 'FAIL'}\n")

    # =================================================================== #
    # Soft warnings
    # =================================================================== #
    print("[S1/S2] DCT feature self-correlation on raw images (soft warn)")
    lift = FieldLift(K=args.channels)
    feats = lift(images_t.unsqueeze(-1))
    f_flat = feats.reshape(-1, args.channels).numpy()
    fcorr = np.corrcoef(f_flat.T)
    f_off = fcorr[np.triu_indices_from(fcorr, k=1)]
    fmax = float(np.abs(f_off).max())
    fmean = float(np.abs(f_off).mean())
    print(f"  raw DCT feature max |corr|={fmax:.4f}  mean={fmean:.4f}")
    print(f"  ideal: max<0.1 mean<0.03 — soft warn only, not block\n")

    # =================================================================== #
    print("=== SUMMARY ===")
    gates = [
        ("G1 finite data", g1),
        ("G2 image range [-1,1]", g2),
        ("G3 Gate-0 recon", g3),
        ("G4 latent std sane", g4),
        ("G5 ch std min>0.05", g5),
        ("G6 ch corr ok", g6),
    ]
    for name, ok in gates:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    all_pass = all(ok for _, ok in gates)
    print(f"\n{'ALL HARD GATES PASS — OK to launch GPU' if all_pass else 'HARD GATE FAIL — DO NOT LAUNCH GPU'}")
    if fmax >= 0.1 or fmean >= 0.03:
        print(f"(soft warning: DCT lift basis is suboptimal on this data; "
              f"GPU launch still allowed if hard gates passed.)")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
