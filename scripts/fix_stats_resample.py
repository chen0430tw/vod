"""Verify hypothesis #1: stats EMA mismatch causes sample noise.

Three resample paths from v10 ckpt:
  A. ORIG stats EMA (mean=0.07, std=0.46)         baseline
  B. REFIT global stats from real L_ref            test #1 (scalar)
  C. REFIT per-channel stats from real L_ref       test #1 (vector)

If C improves dramatically over A → stats EMA is root cause.
If A == C → DDIM convergence is root cause (#2).
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

_PROTO = Path(__file__).resolve().parent.parent / "prototype"
if str(_PROTO) not in sys.path:
    sys.path.insert(0, str(_PROTO))

# Match v10 monkeypatch
import vod_minimal.native as _native_mod
_native_mod.LATENT_T = 1
_native_mod.AUDIO_SIZE = (
    _native_mod.LATENT_T * _native_mod.LATENT_HW * _native_mod.LATENT_HW
)

from vod_minimal.blocky_scattering import build_blocky_scattering_batch
from vod_minimal.diffusion import ddim_sample, make_schedule
from vod_minimal.native import LATENT_HW, LATENT_T, NativeVOD, NativeVODConfig


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
    image_outs = []
    for i in range(U.shape[0]):
        u = U[i]
        image_outs.append(m.dec_image(u[0]).squeeze(-1))
    return torch.stack(image_outs, dim=0)


def save_grid(arrs, path, ncols=4, scale=8):
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


def descriptor_distance(imgs, train_imgs):
    """Quick visual distance: pixel L2 + amp range delta."""
    pix_l2 = float(np.mean([
        np.linalg.norm(imgs[i] - train_imgs[i % len(train_imgs)])
        for i in range(len(imgs))
    ]))
    amp_delta = abs(
        (imgs.max() - imgs.min()) - (train_imgs.max() - train_imgs.min())
    )
    return pix_l2, float(amp_delta)


def main():
    ckpt_path = "runs/v10_ckpt/v8_latest.pt"
    out_dir = Path("tensorearch_reports")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    ck = torch.load(ckpt_path, map_location=device)
    args_dict = ck["args"]
    cfg = NativeVODConfig(
        channels=args_dict["channels"], hidden=args_dict["hidden"],
        denoise_steps=4, backbone="unet", time_dim=args_dict["time_dim"],
    )
    m = NativeVOD(cfg).to(device)
    m.load_state_dict(ck["model"])
    m.eval()

    orig_mean = ck["stats_mean"]
    orig_std = ck["stats_std"]
    print(f"[stats orig EMA] mean={orig_mean:.4f}  std={orig_std:.4f}")

    schedule = make_schedule(num_steps=args_dict["diffusion_steps"]).to(device)

    # ---- collect a large L_ref pool to refit stats accurately ---------- #
    rng = np.random.default_rng(args_dict["seed"])
    batch = build_blocky_scattering_batch(
        rng, batch_size=512, size=LATENT_HW, frames=LATENT_T,
        artifact_strength=0.0, tile=4, spacetime=True,
        temporal_mode="static", flicker_strength=0.0, paired_denoising=True,
    )
    images_np = [np.asarray(s.target_views["image"], dtype=np.float32) for s in batch.samples]
    images_t = torch.stack([torch.as_tensor(im) for im in images_np], dim=0).to(device)
    videos_t = images_t.unsqueeze(1).expand(-1, LATENT_T, -1, -1).contiguous()

    with torch.no_grad():
        L_ref = batched_encode(m, {"image": images_t, "video": videos_t}).float()
    print(f"[L_ref] shape={tuple(L_ref.shape)}  mean={L_ref.mean():+.4f}  std={L_ref.std():.4f}")

    # ---- refit stats ---------------------------------------------------- #
    real_global_mean = float(L_ref.mean())
    real_global_std = float(L_ref.std())
    real_ch_mean = L_ref.mean(dim=(0, 1, 2, 3))     # (C,)
    real_ch_std = L_ref.std(dim=(0, 1, 2, 3))       # (C,)
    print(f"[refit global] mean={real_global_mean:+.4f}  std={real_global_std:.4f}")
    print(f"[refit per-ch mean] {real_ch_mean.cpu().numpy().round(3).tolist()}")
    print(f"[refit per-ch std]  {real_ch_std.cpu().numpy().round(3).tolist()}")
    print(f"[delta global] mean shift={real_global_mean - orig_mean:+.4f}  "
          f"std ratio={real_global_std / orig_std:.3f}")

    # ---- generate samples three ways ------------------------------------ #
    n_samples = 8
    shape = (n_samples, LATENT_T, LATENT_HW, LATENT_HW, cfg.channels)

    def sample_with(stats_mean, stats_std, seed):
        g = torch.Generator(device=device).manual_seed(seed)
        with torch.no_grad():
            x_norm = ddim_sample(m, shape, schedule, num_steps=50,
                                  device=device, generator=g, prediction="epsilon")
            # Broadcast: stats_mean/std can be scalar or (C,) tensor
            if isinstance(stats_mean, torch.Tensor) and stats_mean.ndim > 0:
                # per-channel: shape (C,) broadcast over (B, T, H, W, C)
                L = x_norm * stats_std.to(device) + stats_mean.to(device)
            else:
                L = x_norm * float(stats_std) + float(stats_mean)
            img = batched_decode(m, L)
        return L.float().cpu().numpy(), img.float().cpu().numpy()

    seed = args_dict["seed"] + 1
    print(f"\n--- A. ORIG EMA stats ---")
    L_A, img_A = sample_with(orig_mean, orig_std, seed)
    print(f"  L mean={L_A.mean():+.4f} std={L_A.std():.4f}  "
          f"img range=[{img_A.min():+.3f}, {img_A.max():+.3f}]")

    print(f"--- B. REFIT global stats ---")
    L_B, img_B = sample_with(real_global_mean, real_global_std, seed)
    print(f"  L mean={L_B.mean():+.4f} std={L_B.std():.4f}  "
          f"img range=[{img_B.min():+.3f}, {img_B.max():+.3f}]")

    print(f"--- C. REFIT per-channel stats ---")
    L_C, img_C = sample_with(real_ch_mean, real_ch_std, seed)
    print(f"  L mean={L_C.mean():+.4f} std={L_C.std():.4f}  "
          f"img range=[{img_C.min():+.3f}, {img_C.max():+.3f}]")

    # ---- compare to train_ref ------------------------------------------- #
    train_imgs = images_t[:n_samples].cpu().numpy()
    pa, ada = descriptor_distance(img_A, train_imgs)
    pb, adb = descriptor_distance(img_B, train_imgs)
    pc, adc = descriptor_distance(img_C, train_imgs)
    print(f"\n--- vs train_ref distance ---")
    print(f"A (orig)         pix_L2={pa:.3f}  amp_range_delta={ada:.3f}")
    print(f"B (global refit) pix_L2={pb:.3f}  amp_range_delta={adb:.3f}")
    print(f"C (per-ch refit) pix_L2={pc:.3f}  amp_range_delta={adc:.3f}")

    # ---- save grids ----------------------------------------------------- #
    save_grid(list(train_imgs[:4]), out_dir / "v10_resample_train.png", ncols=4)
    save_grid(list(img_A[:4]),       out_dir / "v10_resample_A_orig_ema.png", ncols=4)
    save_grid(list(img_B[:4]),       out_dir / "v10_resample_B_global_refit.png", ncols=4)
    save_grid(list(img_C[:4]),       out_dir / "v10_resample_C_perch_refit.png", ncols=4)

    payload = {
        "ckpt": ckpt_path,
        "stats_orig":   {"mean": orig_mean, "std": orig_std},
        "stats_refit_global": {"mean": real_global_mean, "std": real_global_std},
        "stats_refit_perch":  {
            "mean": real_ch_mean.cpu().numpy().tolist(),
            "std":  real_ch_std.cpu().numpy().tolist(),
        },
        "verdict_per_strategy": {
            "A_orig":          {"pix_L2": pa, "amp_range_delta": ada},
            "B_global_refit":  {"pix_L2": pb, "amp_range_delta": adb},
            "C_perch_refit":   {"pix_L2": pc, "amp_range_delta": adc},
        },
        "interpretation": (
            "B≈A → stats EMA not the issue, root cause is DDIM convergence (#2). "
            "C<<A → per-channel std mismatch is real, refit fixes it. "
            "C→0 → all problem was stats mismatch."
        ),
    }
    out_json = out_dir / "v10_resample_test.json"
    out_json.write_text(json.dumps(payload, indent=2))
    print(f"\n[output] {out_json}")


if __name__ == "__main__":
    main()
