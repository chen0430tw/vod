"""Forensic v11 — verify Fix P/Q/R worked.

Three probes mirror v10 forensic but adapted for v11 (fixed scaling +
zero-terminal-SNR + v-prediction):

  1. Per-t v-prediction error (training objective)
     Compare with v10 epsilon-pred error in same t bins.
     Layer 3 fix verified if small-t MSE drops well below 36.

  2. Layer 1: per-channel L_ref std
     v10 had ch1 std=0.085 collapsed.
     v11 with weak decoder regularizer should pull ch1 std up.

  3. Effective scaling factor — does Fix P actually normalize σ=1?
     Compute L_ref * scaling, check std.
"""
from __future__ import annotations
import dataclasses
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


def rescale_schedule_zero_snr(schedule):
    a = schedule.alphas_cumprod.clone()
    a0 = a[0]; a_last = a[-1]
    a = (a - a_last) / (a0 - a_last) * a0
    a = torch.clamp(a, min=0.0, max=1.0)
    return dataclasses.replace(schedule, alphas_cumprod=a)


def v_target(x_0, t, schedule, noise):
    a_bar = schedule.alphas_cumprod[t]
    a_bar = a_bar.view(-1, *([1] * (x_0.ndim - 1)))
    alpha = torch.sqrt(a_bar)
    sigma = torch.sqrt(torch.clamp(1.0 - a_bar, min=0.0))
    return alpha * noise - sigma * x_0


def main():
    ckpt_path = "runs/v12_ckpt/v8_latest.pt"
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

    # Reproduce v11's Fix Q schedule
    schedule = make_schedule(num_steps=args_dict["diffusion_steps"]).to(device)
    schedule = rescale_schedule_zero_snr(schedule)
    print(f"[schedule] α_bar[0]={float(schedule.alphas_cumprod[0]):.4f}  "
          f"α_bar[-1]={float(schedule.alphas_cumprod[-1]):.6f}")

    # ---- Encode train_ref pool ----------------------------------------- #
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

    # ---- Probe 3 first: Fix P verification ---------------------------- #
    L_ref_per_ch_std = L_ref.std(dim=(0, 1, 2, 3)).cpu().numpy()
    L_ref_per_ch_mean = L_ref.mean(dim=(0, 1, 2, 3)).cpu().numpy()
    L_ref_global_std = float(L_ref.std())

    # Fix P: scaling = 1 / global_std (computed at training start)
    scaling = 1.0 / max(L_ref_global_std, 1e-6)
    L_ref_norm = L_ref * scaling
    L_ref_norm_global_std = float(L_ref_norm.std())
    print(f"\n[Fix P verify]")
    print(f"  L_ref global std: {L_ref_global_std:.4f}")
    print(f"  scaling factor:   {scaling:.4f}")
    print(f"  L_ref_norm std:   {L_ref_norm_global_std:.4f}  (target: 1.0)")
    print(f"  → {'OK' if abs(L_ref_norm_global_std - 1.0) < 0.05 else 'STILL OFF'}")

    # ---- Probe 2: Layer 1 channel-wise std ------------------------------ #
    print(f"\n[Layer 1 / Fix R verify - per-channel L_ref std]")
    print(f"  per-ch std:  {L_ref_per_ch_std.round(3).tolist()}")
    print(f"  per-ch mean: {L_ref_per_ch_mean.round(3).tolist()}")
    collapsed = [i for i, s in enumerate(L_ref_per_ch_std) if s < 0.1]
    print(f"  channels collapsed (std<0.1): {collapsed}")
    print(f"  v10 had ch1 std=0.085 (collapsed). v11 ch1 std={L_ref_per_ch_std[1]:.3f}")

    # ---- Probe 1: per-t v-prediction error ------------------------------ #
    probe_t = [1, 5, 10, 25, 50, 75, 100, 125, 150, 175, 199]
    probe_errs = []
    print(f"\n[Layer 3 / Fix Q verify - per-t v-prediction error]")
    for t_int in probe_t:
        t = torch.full((L_ref_norm.shape[0],), t_int, device=device, dtype=torch.long)
        noise = torch.randn_like(L_ref_norm)
        x_t = q_sample(L_ref_norm, t, schedule, noise=noise)
        target = v_target(L_ref_norm, t, schedule, noise)
        with torch.no_grad():
            pred = m.denoise(x_t.to(next(m.parameters()).dtype), t=t).float()
        err = F.mse_loss(pred, target).item()
        probe_errs.append(err)
        print(f"  t={t_int:3d}  v-MSE={err:.4f}")

    # Reference: v10 epsilon-MSE
    v10_eps_mse = {1: 36.5, 5: 37.3, 10: 36.1, 25: 30.0, 50: 19.3, 75: 4.6,
                   100: 2.0, 125: 1.1, 150: 0.66, 175: 0.39, 199: 0.24}

    print(f"\n[v11 v-MSE vs v10 ε-MSE]")
    print(f"  {'t':>4}  {'v11 v-MSE':>10}  {'v10 ε-MSE':>10}  improvement")
    for i, t_int in enumerate(probe_t):
        v11_mse = probe_errs[i]
        v10_mse = v10_eps_mse[t_int]
        ratio = v10_mse / max(v11_mse, 1e-6)
        print(f"  {t_int:>4}  {v11_mse:>10.4f}  {v10_mse:>10.4f}  {ratio:>6.1f}× better")

    # ---- save ----------------------------------------------------------- #
    payload = {
        "ckpt": ckpt_path,
        "ckpt_ep": int(ck["ep"]),
        "fix_P_scaling": {
            "L_ref_global_std": L_ref_global_std,
            "scaling": scaling,
            "L_ref_norm_std_after_scale": L_ref_norm_global_std,
            "verdict": "PASS" if abs(L_ref_norm_global_std - 1.0) < 0.05 else "FAIL",
        },
        "fix_R_per_ch_std": L_ref_per_ch_std.tolist(),
        "fix_R_per_ch_mean": L_ref_per_ch_mean.tolist(),
        "fix_R_collapsed_channels": collapsed,
        "fix_R_ch1_std_v10_was_0.085": float(L_ref_per_ch_std[1]),
        "fix_Q_v_pred_error": dict(zip(probe_t, probe_errs)),
        "fix_Q_v10_eps_pred_error_baseline": v10_eps_mse,
    }
    out_json = out_dir / "v12_diffusion_forensic.json"
    out_json.write_text(json.dumps(payload, indent=2))
    print(f"\n[output] {out_json}")


if __name__ == "__main__":
    main()
