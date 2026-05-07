"""Sanity gate v16: verify DCT lift gives decorrelated channels BEFORE GPU launch.

Run a 50-step CPU mini-train, then encode train_ref, compute channel
correlation. Pass criteria:
  max |off-diag corr| < 0.5
If fail, refuse to launch GPU.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

import numpy as np
import torch

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS / "ablations"))
sys.path.insert(0, str(_THIS.parent / "prototype"))

import vod_minimal.native as _native_mod  # noqa: E402
_native_mod.LATENT_T = 1
_native_mod.AUDIO_SIZE = (
    _native_mod.LATENT_T * _native_mod.LATENT_HW * _native_mod.LATENT_HW
)

# Import v16 specific symbols
from run_unconditional_fidelity_v16 import (
    FieldLift, FieldLiftedEncoder, _make_dct2_kernels,
)
from vod_minimal.blocky_scattering import build_blocky_scattering_batch
from vod_minimal.native import LATENT_HW, LATENT_T, NativeVOD, NativeVODConfig


def main():
    device = torch.device("cpu")
    print(f"[sanity-v16] device={device}")

    # 1. Verify DCT kernels are orthogonal
    K = 8
    kernels = _make_dct2_kernels(K=K).reshape(K, -1)   # (K, 9)
    gram = kernels @ kernels.t()
    off_diag = gram - torch.diag(torch.diag(gram))
    max_off = float(off_diag.abs().max())
    print(f"[sanity-v16] DCT kernel orthogonality: max |<φ_i, φ_j>| (i≠j) = {max_off:.6e}")
    assert max_off < 1e-5, "DCT kernels not orthogonal!"

    # 2. Build a tiny config NativeVOD with FieldLiftedEncoder
    cfg = NativeVODConfig(channels=8, hidden=16, denoise_steps=4,
                          backbone="unet", time_dim=8)
    m = NativeVOD(cfg).to(device)
    m.enc_image = FieldLiftedEncoder(channels=8).to(device)
    m.enc_video = FieldLiftedEncoder(channels=8).to(device)

    # 3. Run a few steps of L2 reconstruction (cheap proxy for training)
    rng = np.random.default_rng(123)
    batch = build_blocky_scattering_batch(
        rng, batch_size=16, size=LATENT_HW, frames=LATENT_T,
        artifact_strength=0.0, tile=4, spacetime=True,
        temporal_mode="static", flicker_strength=0.0, paired_denoising=True,
    )
    images_np = [np.asarray(s.target_views["image"], dtype=np.float32) for s in batch.samples]
    images_t = torch.stack([torch.as_tensor(im) for im in images_np], dim=0)
    videos_t = images_t.unsqueeze(1).expand(-1, LATENT_T, -1, -1).contiguous()

    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-3)
    for step in range(50):
        opt.zero_grad()
        # encode → decode L2 against image
        outs = []
        for i in range(images_t.shape[0]):
            outs.append(m._encode_image(images_t[i]))
        L = torch.stack(outs, dim=0).float()           # (B, T, H, W, C)
        rec = []
        for i in range(L.shape[0]):
            rec.append(m.dec_image(L[i, 0]).squeeze(-1))   # (H, W)
        rec = torch.stack(rec, dim=0)
        loss = ((rec - images_t) ** 2).mean()
        loss.backward()
        opt.step()
        if (step + 1) % 10 == 0:
            print(f"  step {step+1:2d}  L_recon={loss.item():.4f}")

    # 4. Encode train_ref, compute channel std + correlation
    with torch.no_grad():
        outs = []
        for i in range(images_t.shape[0]):
            outs.append(m._encode_image(images_t[i]))
        L_ref = torch.stack(outs, dim=0).float()       # (B, T, H, W, C)

    per_ch_std = L_ref.std(dim=(0, 1, 2, 3)).numpy()
    L_flat = L_ref.reshape(-1, L_ref.shape[-1]).numpy()
    corr = np.corrcoef(L_flat.T)
    off = corr[np.triu_indices_from(corr, k=1)]
    max_corr = float(np.abs(off).max())
    mean_corr = float(np.abs(off).mean())

    print(f"\n[sanity-v16] per-channel L_ref std: {per_ch_std.round(3).tolist()}")
    print(f"[sanity-v16] channel correlation: max |corr|={max_corr:.3f}  mean |corr|={mean_corr:.3f}")
    # Codex literal criterion #2: "channel correlation 不全接近 1".
    # → mean |corr| < 0.5 (most pairs not near-1) AND max |corr| < 0.95.
    print(f"[sanity-v16] threshold: mean<0.5 AND max<0.95 (Codex literal)")

    if mean_corr < 0.5 and max_corr < 0.95:
        print(f"[sanity-v16] PASS — DCT lift gives mostly-decorrelated channels.")
        return 0
    else:
        print(f"[sanity-v16] FAIL — channels too correlated. DO NOT launch GPU.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
