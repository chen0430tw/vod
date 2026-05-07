"""Forensic: where does v10 lose the Chladni blocky structure?

Pipeline check:
  encoder(train_ref) → L_ref       (real latent)
  ddim_sample(noise) → L_sample    (sampled latent)
  decoder(L_ref)     → img_recon   (should be sharp blocky)
  decoder(L_sample)  → img_sample  (known: noise texture)

Then compare L_ref vs L_sample distribution per channel × spatial position.

Output:
  tensorearch_reports/v10_forensic.json     numerical
  tensorearch_reports/v10_forensic_*.png    visualizations
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

_PROTO = Path(__file__).resolve().parent.parent / "prototype"
if str(_PROTO) not in sys.path:
    sys.path.insert(0, str(_PROTO))

# Match v10 monkeypatch (T=1)
import vod_minimal.native as _native_mod
_native_mod.LATENT_T = 1
_native_mod.AUDIO_SIZE = (
    _native_mod.LATENT_T * _native_mod.LATENT_HW * _native_mod.LATENT_HW
)

from vod_minimal.blocky_scattering import build_blocky_scattering_batch
from vod_minimal.diffusion import diffusion_loss, ddim_sample, make_schedule
from vod_minimal.native import (
    LATENT_HW, LATENT_T, NativeVOD, NativeVODConfig,
)


# Reuse the helpers from the v10 trainer.
def batched_encode(m, batched_targets):
    out = []
    for k in batched_targets:
        if not hasattr(m, f"_encode_{k}"):
            continue
        for i in range(batched_targets[k].shape[0]):
            x = batched_targets[k][i]
            out.append(getattr(m, f"_encode_{k}")(x))
        break
    return torch.stack(out, dim=0)


def batched_decode(m, U):
    image_outs, video_outs = [], []
    for i in range(U.shape[0]):
        u = U[i]
        # native U → (T, H, W, C). decode_image squeezes channel + averages T.
        # We replicate the path: dec_image at first frame.
        image_outs.append(m.dec_image(u[0]).squeeze(-1))    # (H, W)
        video_outs.append(m.dec_video(u).squeeze(-1))        # (T, H, W)
    return {
        "image": torch.stack(image_outs, dim=0),
        "video": torch.stack(video_outs, dim=0),
    }


def save_grid(arrs, path, ncols=4, scale=8):
    """arrs: list of 2D float arrays in [-1, 1] roughly. Normalize per-array."""
    n = len(arrs)
    nrows = (n + ncols - 1) // ncols
    H, W = arrs[0].shape
    grid = np.ones((nrows * H, ncols * W), dtype=np.float32) * 0.5
    for i, a in enumerate(arrs):
        a = np.asarray(a, dtype=np.float32)
        lo, hi = a.min(), a.max()
        if hi > lo:
            a = (a - lo) / (hi - lo)
        else:
            a = np.zeros_like(a)
        r, c = i // ncols, i % ncols
        grid[r * H:(r + 1) * H, c * W:(c + 1) * W] = a
    img = (grid * 255).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(img, mode="L").resize(
        (img.shape[1] * scale, img.shape[0] * scale), Image.NEAREST
    )
    img.save(path)


def main():
    ckpt_path = "runs/v10_ckpt/v8_latest.pt"
    out_dir = Path("tensorearch_reports")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    # Load ckpt
    ck = torch.load(ckpt_path, map_location=device)
    args_dict = ck["args"]
    print(f"[ckpt] ep={ck['ep']}  args.lr={args_dict['lr']}  hidden={args_dict['hidden']}")

    cfg = NativeVODConfig(
        channels=args_dict["channels"], hidden=args_dict["hidden"],
        denoise_steps=4, backbone="unet", time_dim=args_dict["time_dim"],
    )
    m = NativeVOD(cfg).to(device)
    m.load_state_dict(ck["model"])
    m.eval()

    # latent stats from training
    class StatsBundle:
        pass
    stats = StatsBundle()
    stats.mean = ck["stats_mean"]
    stats.std = ck["stats_std"]
    print(f"[stats] mean={stats.mean:.4f}  std={stats.std:.4f}")

    schedule = make_schedule(num_steps=args_dict["diffusion_steps"]).to(device)

    # ---------------------------------------------------------------------- #
    #  Build a few real Chladni training samples
    # ---------------------------------------------------------------------- #
    rng = np.random.default_rng(args_dict["seed"])
    batch = build_blocky_scattering_batch(
        rng, batch_size=8, size=LATENT_HW, frames=LATENT_T,
        artifact_strength=0.0, tile=4, spacetime=True,
        temporal_mode="static", flicker_strength=0.0, paired_denoising=True,
    )
    images_np = [np.asarray(s.target_views["image"], dtype=np.float32) for s in batch.samples]
    images_t = torch.stack([torch.as_tensor(im) for im in images_np], dim=0).to(device)
    videos_t = images_t.unsqueeze(1).expand(-1, LATENT_T, -1, -1).contiguous()
    targets = {"image": images_t, "video": videos_t}
    print(f"[data] image.shape={images_t.shape}")

    # ---------------------------------------------------------------------- #
    #  STEP 1 — encoder(train_ref) → L_ref
    # ---------------------------------------------------------------------- #
    with torch.no_grad():
        L_ref = batched_encode(m, targets).float()  # (B, T, H, W, C)
    print(f"[L_ref] shape={tuple(L_ref.shape)}")
    print(f"[L_ref] mean={L_ref.mean():+.4f}  std={L_ref.std():.4f}  "
          f"min={L_ref.min():+.3f}  max={L_ref.max():+.3f}")

    # ---------------------------------------------------------------------- #
    #  STEP 2 — ddim_sample(noise) → L_sample (in normalised space, then unnorm)
    # ---------------------------------------------------------------------- #
    g = torch.Generator(device=device).manual_seed(args_dict["seed"] + 1)
    shape = (8, LATENT_T, LATENT_HW, LATENT_HW, cfg.channels)
    with torch.no_grad():
        L_sample_norm = ddim_sample(m, shape, schedule, num_steps=50,
                                     device=device, generator=g, prediction="epsilon")
        # Unnormalize (training applied μ/σ normalize)
        L_sample = L_sample_norm * stats.std + stats.mean
    print(f"[L_sample] shape={tuple(L_sample.shape)}")
    print(f"[L_sample] mean={L_sample.mean():+.4f}  std={L_sample.std():.4f}  "
          f"min={L_sample.min():+.3f}  max={L_sample.max():+.3f}")

    # ---------------------------------------------------------------------- #
    #  STEP 3 — decode L_ref → img_recon  (should be sharp, gate0_recon=0.017)
    # ---------------------------------------------------------------------- #
    with torch.no_grad():
        rec_ref = batched_decode(m, L_ref)
    img_recon = rec_ref["image"].float().cpu().numpy()  # (B, H, W)
    print(f"[img_recon] mean={img_recon.mean():+.4f}  range=[{img_recon.min():+.3f}, {img_recon.max():+.3f}]")

    # ---------------------------------------------------------------------- #
    #  STEP 4 — decode L_sample → img_sample (known: noise)
    # ---------------------------------------------------------------------- #
    with torch.no_grad():
        rec_sample = batched_decode(m, L_sample)
    img_sample = rec_sample["image"].float().cpu().numpy()
    print(f"[img_sample] mean={img_sample.mean():+.4f}  range=[{img_sample.min():+.3f}, {img_sample.max():+.3f}]")

    # ---------------------------------------------------------------------- #
    #  STEP 5 — quantitative L_ref vs L_sample comparison
    # ---------------------------------------------------------------------- #
    L_ref_np = L_ref.float().cpu().numpy()        # (B, T, H, W, C)
    L_sample_np = L_sample.float().cpu().numpy()
    train_imgs = images_t.cpu().numpy()

    # Per-channel statistics
    ref_per_ch = {
        "mean":  L_ref_np.mean(axis=(0, 1, 2, 3)).tolist(),    # (C,)
        "std":   L_ref_np.std(axis=(0, 1, 2, 3)).tolist(),
        "min":   L_ref_np.min(axis=(0, 1, 2, 3)).tolist(),
        "max":   L_ref_np.max(axis=(0, 1, 2, 3)).tolist(),
    }
    smp_per_ch = {
        "mean":  L_sample_np.mean(axis=(0, 1, 2, 3)).tolist(),
        "std":   L_sample_np.std(axis=(0, 1, 2, 3)).tolist(),
        "min":   L_sample_np.min(axis=(0, 1, 2, 3)).tolist(),
        "max":   L_sample_np.max(axis=(0, 1, 2, 3)).tolist(),
    }

    # Spatial coverage difference: per-(H,W) cell variance of latent across
    # samples — does sampling cover same spatial structure as encoded refs?
    spatial_var_ref = L_ref_np.var(axis=(0, 1, -1))      # (H, W)
    spatial_var_smp = L_sample_np.var(axis=(0, 1, -1))

    # KL-ish: per-channel, fit Gaussian and compute KL(P_sample || P_ref)
    def gauss_kl(p_mu, p_sd, q_mu, q_sd, eps=1e-6):
        return (np.log(q_sd / max(p_sd, eps)) +
                (p_sd**2 + (p_mu - q_mu)**2) / (2 * q_sd**2 + eps) - 0.5)

    kl_per_ch = []
    for c in range(cfg.channels):
        kl = gauss_kl(smp_per_ch["mean"][c], smp_per_ch["std"][c],
                      ref_per_ch["mean"][c], ref_per_ch["std"][c])
        kl_per_ch.append(float(kl))

    payload = {
        "ckpt": ckpt_path,
        "ckpt_ep": int(ck["ep"]),
        "L_ref": {
            "shape": list(L_ref.shape),
            "global_mean": float(L_ref.mean()),
            "global_std":  float(L_ref.std()),
            "per_channel": ref_per_ch,
        },
        "L_sample": {
            "shape": list(L_sample.shape),
            "global_mean": float(L_sample.mean()),
            "global_std":  float(L_sample.std()),
            "per_channel": smp_per_ch,
        },
        "L_diff": {
            "mean_shift": [smp_per_ch["mean"][c] - ref_per_ch["mean"][c]
                           for c in range(cfg.channels)],
            "std_ratio":  [smp_per_ch["std"][c] / max(ref_per_ch["std"][c], 1e-6)
                           for c in range(cfg.channels)],
            "kl_per_channel": kl_per_ch,
            "kl_total": float(sum(kl_per_ch)),
        },
        "spatial_var_ref_mean": float(spatial_var_ref.mean()),
        "spatial_var_smp_mean": float(spatial_var_smp.mean()),
        "img_recon_stats": {
            "mean": float(img_recon.mean()),
            "std":  float(img_recon.std()),
            "range": [float(img_recon.min()), float(img_recon.max())],
        },
        "img_sample_stats": {
            "mean": float(img_sample.mean()),
            "std":  float(img_sample.std()),
            "range": [float(img_sample.min()), float(img_sample.max())],
        },
        "train_img_stats": {
            "mean": float(train_imgs.mean()),
            "std":  float(train_imgs.std()),
            "range": [float(train_imgs.min()), float(train_imgs.max())],
        },
    }
    out_json = out_dir / "v10_forensic.json"
    out_json.write_text(json.dumps(payload, indent=2))
    print(f"[output] {out_json}")

    # ---------------------------------------------------------------------- #
    #  Visualizations
    # ---------------------------------------------------------------------- #
    save_grid(list(train_imgs[:4]), out_dir / "v10_forensic_train.png", ncols=4, scale=8)
    save_grid(list(img_recon[:4]),  out_dir / "v10_forensic_recon_from_Lref.png", ncols=4, scale=8)
    save_grid(list(img_sample[:4]), out_dir / "v10_forensic_recon_from_Lsample.png", ncols=4, scale=8)

    # Per-channel mean heatmap (across H,W) for first sample, T=0
    ref_ch_grid = L_ref_np[0, 0, ..., :].transpose(2, 0, 1)        # (C, H, W)
    smp_ch_grid = L_sample_np[0, 0, ..., :].transpose(2, 0, 1)
    save_grid(list(ref_ch_grid), out_dir / "v10_forensic_Lref_channels.png", ncols=4, scale=8)
    save_grid(list(smp_ch_grid), out_dir / "v10_forensic_Lsample_channels.png", ncols=4, scale=8)

    print("\n=== KEY DIAGNOSTIC ===")
    print(f"L_ref vs L_sample mean shift: max abs = {max(abs(x) for x in payload['L_diff']['mean_shift']):.4f}")
    print(f"L_ref vs L_sample std ratio:  range = [{min(payload['L_diff']['std_ratio']):.3f}, {max(payload['L_diff']['std_ratio']):.3f}]")
    print(f"KL(sample||ref) per channel:  total = {payload['L_diff']['kl_total']:.4f}")
    print(f"img_recon range  {payload['img_recon_stats']['range']}  vs train {payload['train_img_stats']['range']}")
    print(f"img_sample range {payload['img_sample_stats']['range']}  vs train {payload['train_img_stats']['range']}")


if __name__ == "__main__":
    main()
