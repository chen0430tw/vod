"""Forensic round 2: diffusion model internals.

Three deeper probes:
  1. Per-timestep noise prediction error
     For t = [1, 5, 10, 25, 50, 100, 150, 199], add noise to L_ref_norm,
     ask model to predict noise. Plot error vs t. Reveals which steps
     the model learned and which are useless.

  2. DDIM trajectory at intermediate steps
     Sample 50-step trajectory; save x at t = 199, 150, 100, 50, 25, 10, 5, 0
     Compare against q_sample(L_ref_norm, t) expected at same t.
     Trajectory deviates somewhere — locate.

  3. Channel-wise prediction error
     Same noise-pred error, broken down per channel.
     Reveals if some channels are systematically failed.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

_PROTO = Path(__file__).resolve().parent.parent / "prototype"
if str(_PROTO) not in sys.path:
    sys.path.insert(0, str(_PROTO))

import vod_minimal.native as _native_mod
_native_mod.LATENT_T = 1
_native_mod.AUDIO_SIZE = (
    _native_mod.LATENT_T * _native_mod.LATENT_HW * _native_mod.LATENT_HW
)

from vod_minimal.blocky_scattering import build_blocky_scattering_batch
from vod_minimal.diffusion import (
    q_sample, ddim_sample, make_schedule,
)
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
    schedule = make_schedule(num_steps=args_dict["diffusion_steps"]).to(device)
    stats_mean = ck["stats_mean"]
    stats_std = ck["stats_std"]
    print(f"[stats] EMA mean={stats_mean:.4f} std={stats_std:.4f}")

    # Build a batch of train_ref for encoding
    rng = np.random.default_rng(args_dict["seed"])
    batch = build_blocky_scattering_batch(
        rng, batch_size=64, size=LATENT_HW, frames=LATENT_T,
        artifact_strength=0.0, tile=4, spacetime=True,
        temporal_mode="static", flicker_strength=0.0, paired_denoising=True,
    )
    images_np = [np.asarray(s.target_views["image"], dtype=np.float32) for s in batch.samples]
    images_t = torch.stack([torch.as_tensor(im) for im in images_np], dim=0).to(device)
    videos_t = images_t.unsqueeze(1).expand(-1, LATENT_T, -1, -1).contiguous()
    with torch.no_grad():
        L_ref = batched_encode(m, {"image": images_t, "video": videos_t}).float()
    L_ref_norm = (L_ref - stats_mean) / stats_std        # what training saw
    print(f"[L_ref] shape={tuple(L_ref.shape)}  norm mean={L_ref_norm.mean():+.4f} std={L_ref_norm.std():.4f}")

    # ====================================================================== #
    #  PROBE 1: per-timestep noise prediction error
    # ====================================================================== #
    probe_t = [1, 5, 10, 25, 50, 75, 100, 125, 150, 175, 199]
    probe_errs = []
    probe_per_ch = []
    for t_int in probe_t:
        t = torch.full((L_ref_norm.shape[0],), t_int, device=device, dtype=torch.long)
        noise = torch.randn_like(L_ref_norm)
        x_t = q_sample(L_ref_norm, t, schedule, noise=noise)
        with torch.no_grad():
            # match training dtype (bf16 amp during train, but eval fp32 here)
            pred = m.denoise(x_t.to(next(m.parameters()).dtype), t=t).float()
        err = F.mse_loss(pred, noise).item()
        # Per-channel: average MSE over (B, T, H, W)
        per_ch = ((pred - noise) ** 2).mean(dim=(0, 1, 2, 3)).cpu().numpy().tolist()
        probe_errs.append(err)
        probe_per_ch.append(per_ch)
        print(f"  t={t_int:3d}  total_MSE={err:.4f}  per_ch_max={max(per_ch):.4f}  per_ch_min={min(per_ch):.4f}")

    # ====================================================================== #
    #  PROBE 2: DDIM trajectory snapshots
    # ====================================================================== #
    print("\n--- DDIM 50-step trajectory ---")
    save_at = [199, 175, 150, 100, 50, 25, 10, 5, 0]
    g = torch.Generator(device=device).manual_seed(args_dict["seed"] + 1)
    shape = (8, LATENT_T, LATENT_HW, LATENT_HW, cfg.channels)
    x = torch.randn(shape, device=device, dtype=next(m.parameters()).dtype, generator=g)
    timesteps = torch.linspace(
        schedule.num_steps - 1, 0, 50, dtype=torch.long, device=device,
    )
    traj_snaps = {}
    with torch.no_grad():
        for i in range(50):
            t = timesteps[i]
            t_batch = t.expand(shape[0])
            pred = m.denoise(x, t=t_batch)
            # epsilon prediction (matches v10 args)
            eps_pred = pred
            a_t = schedule.alphas_cumprod[t]
            x_0_pred = (x - torch.sqrt(1 - a_t) * eps_pred) / torch.sqrt(torch.clamp(a_t, min=1e-9))
            if int(t.item()) in save_at:
                traj_snaps[int(t.item())] = {
                    "x":  x.float().cpu().numpy().copy(),
                    "x_0_pred": x_0_pred.float().cpu().numpy().copy(),
                }
            if i < 49:
                t_next = timesteps[i + 1]
                a_next = schedule.alphas_cumprod[t_next]
                x = torch.sqrt(a_next) * x_0_pred + torch.sqrt(torch.clamp(1 - a_next, min=0)) * eps_pred
            else:
                x = x_0_pred
                if 0 in save_at:
                    traj_snaps[0] = {
                        "x": x.float().cpu().numpy().copy(),
                        "x_0_pred": x_0_pred.float().cpu().numpy().copy(),
                    }

    print(f"  trajectory snapshot timesteps: {sorted(traj_snaps.keys(), reverse=True)}")

    # Compare trajectory x at each t vs q_sample(L_ref_norm, t) — what x SHOULD look like
    traj_compare = []
    for t_int in sorted(save_at, reverse=True):
        if t_int not in traj_snaps:
            continue
        t = torch.full((L_ref_norm.shape[0],), t_int, device=device, dtype=torch.long)
        noise = torch.randn_like(L_ref_norm)  # fresh noise per t
        expected_x_t = q_sample(L_ref_norm, t, schedule, noise=noise)
        actual_x = torch.from_numpy(traj_snaps[t_int]["x"]).to(device).float()
        # mean / std comparison
        traj_compare.append({
            "t": t_int,
            "actual_mean": float(actual_x.mean()),
            "actual_std": float(actual_x.std()),
            "expected_mean": float(expected_x_t.mean()),
            "expected_std": float(expected_x_t.std()),
            "actual_per_ch_std": actual_x.std(dim=(0, 1, 2, 3)).cpu().numpy().tolist(),
            "expected_per_ch_std": expected_x_t.std(dim=(0, 1, 2, 3)).cpu().numpy().tolist(),
        })
        print(f"  t={t_int:3d}  actual μ={float(actual_x.mean()):+.3f} σ={float(actual_x.std()):.3f}  "
              f"expected μ={float(expected_x_t.mean()):+.3f} σ={float(expected_x_t.std()):.3f}")

    # ====================================================================== #
    #  PROBE 3: channel-wise gate0 round-trip — which channels carry signal?
    # ====================================================================== #
    print("\n--- L_ref channel-wise stats ---")
    L_ref_per_ch_std = L_ref.std(dim=(0, 1, 2, 3)).cpu().numpy()
    L_ref_per_ch_mean = L_ref.mean(dim=(0, 1, 2, 3)).cpu().numpy()
    print(f"  per-ch mean: {L_ref_per_ch_mean.round(3).tolist()}")
    print(f"  per-ch std:  {L_ref_per_ch_std.round(3).tolist()}")
    print(f"  channels with std<0.1 (collapsed?): {[i for i,s in enumerate(L_ref_per_ch_std) if s<0.1]}")

    # Save trajectory image grids for visual inspection
    for t_int, snap in traj_snaps.items():
        # show ch0 of first 4 samples at this timestep
        ch0_imgs = [snap["x"][i, 0, ..., 0] for i in range(4)]
        save_grid(ch0_imgs, out_dir / f"v10_traj_t{t_int:03d}_x_ch0.png", ncols=4)

    # Save report
    payload = {
        "ckpt": ckpt_path,
        "ckpt_ep": int(ck["ep"]),
        "stats_ema": {"mean": stats_mean, "std": stats_std},
        "L_ref_per_ch_mean": L_ref_per_ch_mean.tolist(),
        "L_ref_per_ch_std": L_ref_per_ch_std.tolist(),
        "probe1_noise_pred_error": {
            "timesteps": probe_t,
            "total_MSE": probe_errs,
            "per_channel_MSE": probe_per_ch,
        },
        "probe2_ddim_trajectory": traj_compare,
    }
    out_json = out_dir / "v10_diffusion_forensic.json"
    out_json.write_text(json.dumps(payload, indent=2))
    print(f"\n[output] {out_json}")

    print("\n=== DIAGNOSTIC SUMMARY ===")
    print(f"Noise pred error:  best at t={probe_t[probe_errs.index(min(probe_errs))]} "
          f"(MSE={min(probe_errs):.3f})  worst at t={probe_t[probe_errs.index(max(probe_errs))]} "
          f"(MSE={max(probe_errs):.3f})")
    expected_mse_baseline = 1.0  # if model predicts 0 noise → MSE = E[ε²] = 1
    print(f"  random-prediction baseline = 1.0 (ε~N(0,I))")
    print(f"  any t with MSE > 1.0 → model is WORSE than predicting zero noise")


if __name__ == "__main__":
    main()
