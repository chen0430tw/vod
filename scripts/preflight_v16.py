"""Full preflight gate for v16 (per Codex: don't use training as checker).

Four CPU-second-level gates. Run BEFORE every GPU launch.

Gate 1: DCT basis self-correlation on raw train images
        max feature corr < 0.1
        mean feature corr < 0.03

Gate 2: encoded channel correlation (after FieldLift + scale/bias init)
        max channel corr < 0.7
        mean channel corr < 0.4

Gate 3: latent scale ≈ v11's empirical std or explicit recalibration
        |latent.std() - 0.5| < 0.3   (v11 baseline ~0.5)

Gate 4: Gate 0 reconstruction smoke
        decode(encode(image)) amp_range >= 0.5 × image amp_range
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import torch

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS / "ablations"))
sys.path.insert(0, str(_THIS.parent / "prototype"))

import vod_minimal.native as _native_mod
_native_mod.LATENT_T = 1
_native_mod.AUDIO_SIZE = (
    _native_mod.LATENT_T * _native_mod.LATENT_HW * _native_mod.LATENT_HW
)

from run_unconditional_fidelity_v16 import (
    FieldLift, FieldLiftedEncoder,
)
from vod_minimal.blocky_scattering import build_blocky_scattering_batch
from vod_minimal.native import LATENT_HW, LATENT_T, NativeVOD, NativeVODConfig


def main():
    print("=== v16 PREFLIGHT (Codex 4-gate) ===\n")
    rng = np.random.default_rng(123)
    batch = build_blocky_scattering_batch(
        rng, batch_size=64, size=LATENT_HW, frames=LATENT_T,
        artifact_strength=0.0, tile=4, spacetime=True,
        temporal_mode="static", flicker_strength=0.0, paired_denoising=True,
    )
    images_np = [np.asarray(s.target_views["image"], dtype=np.float32) for s in batch.samples]
    images_t = torch.stack([torch.as_tensor(im) for im in images_np], dim=0)
    print(f"data: image batch shape={tuple(images_t.shape)}, "
          f"std={float(images_t.std()):.3f}, range=[{float(images_t.min()):.2f}, {float(images_t.max()):.2f}]")

    # =========================================================== #
    # GATE 1: DCT feature self-correlation on raw images
    # =========================================================== #
    print("\n[Gate 1] DCT feature basis self-correlation (on raw images)")
    lift = FieldLift(K=8)
    feats = lift(images_t.unsqueeze(-1))         # (B, H, W, 8)
    feat_flat = feats.reshape(-1, 8).numpy()
    feat_corr = np.corrcoef(feat_flat.T)
    feat_off = feat_corr[np.triu_indices_from(feat_corr, k=1)]
    feat_max = float(np.abs(feat_off).max())
    feat_mean = float(np.abs(feat_off).mean())
    print(f"  max |corr|={feat_max:.4f}  mean |corr|={feat_mean:.4f}")
    print(f"  threshold: max<0.1, mean<0.03")
    g1 = (feat_max < 0.1) and (feat_mean < 0.03)
    print(f"  {'✓ PASS' if g1 else '✗ FAIL'}")

    # =========================================================== #
    # GATE 2: encoded channel correlation (FieldLift + scale/bias init)
    # =========================================================== #
    print("\n[Gate 2] Encoded channel correlation (init weights)")
    enc = FieldLiftedEncoder(channels=8)
    with torch.no_grad():
        L = enc(images_t.unsqueeze(-1))           # (B, H, W, 8)
    L_flat = L.reshape(-1, 8).numpy()
    L_corr = np.corrcoef(L_flat.T)
    L_off = L_corr[np.triu_indices_from(L_corr, k=1)]
    L_max = float(np.abs(L_off).max())
    L_mean = float(np.abs(L_off).mean())
    print(f"  max |corr|={L_max:.4f}  mean |corr|={L_mean:.4f}")
    print(f"  threshold: max<0.7, mean<0.4")
    g2 = (L_max < 0.7) and (L_mean < 0.4)
    print(f"  {'✓ PASS' if g2 else '✗ FAIL'}")

    # =========================================================== #
    # GATE 3: latent scale matches v11 empirical std (~0.5)
    # =========================================================== #
    print("\n[Gate 3] Latent scale calibration")
    L_std = float(L.std())
    print(f"  latent std={L_std:.3f}")
    print(f"  threshold: |std - 0.5| < 0.3  (v11 baseline ~0.5)")
    g3 = abs(L_std - 0.5) < 0.3
    print(f"  {'✓ PASS' if g3 else '✗ FAIL'}")

    # =========================================================== #
    # GATE 4: Gate-0 reconstruction smoke
    # =========================================================== #
    print("\n[Gate 4] Gate-0 reconstruction (encode+decode)")
    cfg = NativeVODConfig(channels=8, hidden=16, denoise_steps=4,
                          backbone="unet", time_dim=8)
    m = NativeVOD(cfg)
    m.enc_image = enc
    m.enc_video = FieldLiftedEncoder(channels=8)
    # quick L_recon training to set dec_image
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-3)
    for step in range(80):
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
        recon_amp_range = float(rec.max() - rec.min())
        image_amp_range = float(images_t.max() - images_t.min())
    ratio = recon_amp_range / max(image_amp_range, 1e-6)
    print(f"  recon amp_range={recon_amp_range:.3f}, image amp_range={image_amp_range:.3f}")
    print(f"  ratio={ratio:.3f}  threshold: >= 0.5")
    g4 = ratio >= 0.5
    print(f"  {'✓ PASS' if g4 else '✗ FAIL'}")

    # =========================================================== #
    print("\n=== SUMMARY ===")
    gates = [("DCT self-corr", g1), ("ch corr", g2),
             ("latent scale", g3), ("Gate-0 amp", g4)]
    for name, ok in gates:
        print(f"  {'✓' if ok else '✗'} {name}")
    all_pass = all(ok for _, ok in gates)
    print(f"\n{'ALL PREFLIGHT PASS — OK to launch GPU' if all_pass else 'PREFLIGHT FAIL — DO NOT LAUNCH GPU (or kill running)'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
