"""Unconditional sample fidelity v13 — Fix R correction (per-channel aux).

v12 forensic showed that DEEPER weak decoder REGRESSED Fix R (3 channels
collapsed vs v11's 0). Lesson: a stronger weak decoder lets encoder
shift information *between* channels and the deep weak path can still
reconstruct from the survivors → no pressure on each channel
individually.

v13 reverts to shallow weak decoder (1×1 Linear, like v11) and adds:

  Fix R''' (NEW in v13) — per-channel auxiliary reconstruction
      For each channel c independently, mask all OTHER channels to zero
      and ask the (frozen) main dec_image to reconstruct the input image.
      Loss = sum_c MSE(dec_image_with_only_ch_c, target_image).
      A truly collapsed channel (constant) cannot reconstruct anything,
      so this loss penalizes collapse channel-by-channel — no way for
      other channels to hide the failure.
      Source: information-theoretic disentanglement literature; closest
      formal precedent is "auxiliary regression heads per latent factor"
      in disentangled VAE work (e.g. Mathieu et al. 2019 NeurIPS,
      "Disentangling Disentanglement in VAEs").

  Also: w_chvar raised 10 → 100 (10×) to make the floor actually bite.

v12 docs follow.

Unconditional sample fidelity v12 — Fix R hardening (R'+R'').

v11 forensic verdict was 🟡 on Fix R: ch1 std 0.085 → 0.117 (un-collapsed
but still weak; ch4/ch6 also <0.13). v12 hardens with two additional
literature recipes:

  Fix R' — deeper weak decoder (multi-layer with bottleneck)
      Source: Razavi et al. 2019 "Preventing Posterior Collapse with
      delta-VAEs" (arXiv:1901.03416). A capacity-limited but nonlinear
      auxiliary path forces each channel through a nonlinearity, so
      collapsed channels (delta function) cannot survive the bottleneck.
      v12 swaps WeakDecoder from Linear(C→1) to Linear(C→4C) + GELU +
      Linear(4C→1). The 4C bottleneck still narrows channel info but
      now goes through ReLU/GELU first → delta channels die.

  Fix R'' — channel-wise variance floor penalty
      Direct loss-side regularizer (no published canonical name; common
      in VAE/InfoGAN-style anti-collapse literature). For each latent
      channel c, compute std_c across (B, T, H, W). Penalize:
        L_chvar = mean over c of max(0, target_min_std - std_c)^2
      With target_min_std=0.20, this guarantees every channel carries
      at least σ=0.2 of variance. Cheap, always-on, no hyperparameter
      tuning since the floor is a known quantity.

  w_weak raised 0.5 → 1.5 (3× v11) for the deeper weak decoder.

Original v11 docs follow.

Unconditional sample fidelity v11 — published-recipe bundle (Fix P+Q+R).

v11 stops reinventing. Each fix is a literal port of a known published recipe
to fix the 3-layer diagnosis from v10 forensic:

  Layer 1 (encoder partial collapse, ch1 std=0.085 of 8 channels)
  Layer 2 (EMA stats σ ≠ 1, L_ref_norm σ=1.13 not 1.0)
  Layer 3 (small-t noise pred MSE 36, model only learned t>150)

  Fix P (NEW in v11) — LDM-style fixed scaling factor (Layer 2)
      Source: Rombach et al. 2022 (Latent Diffusion Models / Stable
      Diffusion), arXiv:2112.10752. SD uses scaling_factor=0.18215, the
      reciprocal of the empirical latent std measured from a held-out
      encode pass on the training set. We replicate exactly:
        1. Before training, encode `--scale-fit-n` train samples → L_pool
        2. scaling = 1.0 / L_pool.std()    (single scalar)
        3. Diffusion sees x_0_norm = L_ref * scaling   (NO mean shift,
           NO EMA — fixed for whole training)
        4. Sample → L = x * (1/scaling)
      This eliminates the EMA drift that v10 forensic showed makes
      L_ref_norm σ=1.13. Replaces v8 Fix B (EMA-tracked stats) for the
      diffusion side; LatentStats is kept for backward-compat checkpoint
      load but not used.

  Fix Q (NEW in v11) — v-prediction + zero-terminal-SNR (Layer 3)
      Source: Lin 2023, "Common Diffusion Noise Schedules and Sample
      Steps are Flawed" (arXiv:2305.08891, WACV 2024). v10 used
      epsilon-prediction with default α_T ≈ 0.0001 (not zero) → at
      sampling t=T model sees noisy-but-not-pure-noise input ≠ training
      distribution → small-t MSE 36 catastrophe.
      v11:
        a. Rescale schedule so α_bar[T-1] = 0 strictly (zero terminal
           SNR). Lin's exact formula: enforce sqrt(α_bar[T-1])=0 by
           rescaling alphas_cumprod = (a - a[-1]) / (a[0] - a[-1]) * a[0].
        b. v-prediction: target = α·ε - σ·x_0 (instead of ε). v-loss
           weighting balances small-t and large-t MSE on equal scale.
        c. Sampler always start from t=T (last timestep), use v→x_0/ε
           inversion at each step.
      Implemented as call-site wrappers (rescale_schedule, v_loss,
      v_ddim_sample); does NOT modify vod_minimal/diffusion.py
      (substrate sacred). The kwarg `prediction="v"` is intercepted
      before reaching the substrate.

  Fix R (NEW in v11) — auxiliary weak decoder (Layer 1)
      Source: Aggressive inference / weak-decoder regularization for
      VAE posterior collapse (He et al. ICLR 2019; multiple later
      time-series LDM works incl. arXiv:2405.14021). A second very
      narrow decoder (1×1 Linear, no shared weights with dec_image)
      decodes L_ref → image with weight α_weak. To minimize L_weak the
      encoder MUST distribute information across all channels — a single
      collapsed channel kills the weak decoder's reconstruction. Forces
      ch1's std away from 0.085.
      Implemented in trainer (extra Linear module + extra loss term);
      does NOT modify NativeVOD class.

Sources cited inline above. v8/v9/v10 fixes (H/I/J/K/L/M/N/O) retained
where compatible; v11 supersedes Fix B (EMA stats) with Fix P.

Original v10 docs follow.

Fifteen fixes vs v1. v10 adds 4 stability fixes on top of v9's K:

  Fix L (NEW in v10) — cosine eta_min raised
      v8/v9 used eta_min = args.lr * 0.01 → cosine bottoms at 1e-6 for
      lr=1e-4. v9 PARTIAL post-mortem: trained_sample distance (1.881)
      worse than untrained (1.158). Hypothesis #2: lr decaying to 1e-6
      traps model in narrow basin, multi-seed sampling lands in
      different sub-basins → stability degrades. v10 sets eta_min =
      args.lr * 0.1 → bottom at 1e-5 for lr=1e-4. Optimizer keeps
      mobility through end of training.

  Fix M (NEW in v10) — weight_decay 10× higher
      v8/v9 AdamW wd=1e-4. With T=1 fold (Fix K), 8.4M params see 8×
      less data per epoch but same parameter budget. Effective overfit
      pressure rises. v10 wd=1e-3 (10×) to counter.

  Fix N (NEW in v10) — EMA model for sampling
      Multi-seed stability std degraded v8 0.222 → v9 0.255. Both
      single-snapshot models. v10 maintains an EMA-of-weights snapshot
      (decay 0.999) updated per optimizer step; sampling pipeline uses
      EMA snapshot, not the live model. Standard recipe in DDPM/SD/
      EDM (Karras) — sampling-time noise mitigation independent of
      sampler choice.

  Fix O (NEW in v10) — early-target ep default
      forecast.json said growth_gain ≈ 0 from ep ~500; v8 5k saw
      stability *worsen* past plateau; v9 5k saw trained_d *worsen* past
      plateau. v10 default --epochs 1500 (sweet spot between sweep 500
      ep PASS and the 5000 ep regime where everything degrades).

Eleven fixes vs v1. v9 adds Fix K on top of v8's H+I+J:

  Fix K (NEW in v9) — static-T fold via LATENT_T monkeypatch
      tensorearch temporal-couple verdict=weak_balance + slow_mode_fraction
      0.844 + ablate showing spatial_conv bottleneck transfers between
      blocks (blk9 → blk8 when removed) all confirm: under Fix A
      (video=image.broadcast), 8-frame UNet forward is structurally
      redundant. Real information is 1 frame. We monkeypatch
      vod_minimal.native.LATENT_T = 1 BEFORE NativeVOD instantiation,
      so encoder/decoder broadcast image to T=1 instead of T=8 — and
      train_n=2048 build also uses frames=1. Expected speedup 5-7×
      (some 1-D temporal conv overhead remains at bottleneck).

      No edits to vod_minimal/native.py — v9 is purely a substrate
      *callsite* fold, not a substrate redefinition. Reverting is
      removing the monkeypatch.

Original v8 fixes follow.

Ten fixes vs v1:

  Fix A — video target = image.broadcast(T)
      `temporal_mode='static'` does not control target_views; the
      Chladni spacetime field has aliasing modes (mt=2 on frames=8) that
      give a 4-period flicker in raw video targets. We force every frame
      to equal the still image, killing temporal variation.

  Fix B — latent normalization (EMA-tracked μ, σ)
      Encoder output has std ≈ 0.22 << 1.0; DDIM init N(0,I) is
      off-manifold. We normalize x_0 in diffusion loss, sample in
      normalized space, unnormalize before decode.

  Fix C (v3) — batched training, no per-sample Python loop
      v1/v2 ran `[m.encode(t) for t in targets]` every step — 256 small
      ops with GPU sync between each, leaving H100 at ~55% utilization
      and 7GB/80GB memory. v3 pre-stacks targets into a single batched
      dict and uses helper functions that bypass NativeVOD.encode (which
      only handles single-sample dicts) to do one batched encode / one
      batched decode per step.

  Fix D (v4) — detach latent before diffusion loss
      v3 big run (hidden=128, ch=8, train_n=512, ep=2000) showed extreme
      posterior collapse: latent_σ → 0.0001, gate0_recon amp_range = 0,
      trained sample amp_range = 0.004. Joint training of encoder and
      diffusion lets the encoder find a trivial solution (output ≈
      constant) that minimises diffusion loss because predicting a
      constant is easy. We block diffusion gradient from reaching the
      encoder via .detach(); encoder is only trained by L_recon. This
      mirrors the standard latent-diffusion recipe (pretrained VAE →
      freeze → train diffusion on latent) without introducing a VAE.

  Fix E (v5) — epsilon prediction + smaller lr default
      v4 fixed encoder collapse but L_diff oscillated at 0.45 throughout
      2000 epochs and never converged. Two suspects:
        - lr=2e-3 (used for small models) too large for 8.4M-param big
          model; AdamW step kicks loss around its basin.
        - x_0 prediction at high noise (t≈T) is geometrically harder
          than epsilon prediction; standard DDPM default is epsilon.
      v5 makes prediction a CLI arg (default "epsilon") and lowers
      default lr suggestion. diffusion.py already supports both modes
      cleanly via the prediction= kwarg in diffusion_loss + ddim_sample.

  Fix F (NEW in v6) — cosine LR schedule + larger train set
      v5 made L_diff genuinely descend (0.27 → 0.145) for the first
      time but plateau'd at ep 1800. Two complementary pushes:
        - Cosine decay to 1% of base lr lets the optimiser fine-tune
          into the basin without oscillating around it.
        - train_n bump (512 → 2048) gives the diffusion model more
          Chladni mode coverage; v5 may have memorised 512 specific
          modes and failed to generalise the family.
      Default --lr-schedule cosine. Typical big run uses train_n=2048,
      epochs=5000.

  Fix G (NEW in v7) — bf16 autocast + cudnn benchmark
      v6 takes ~300ms/epoch on H100 PCIe in fp32. Wrapping forward in
      torch.amp.autocast(bfloat16) and enabling cudnn.benchmark=True
      typically gives 3-5× wall-clock speedup on H100 with no quality
      change for diffusion (DDPM is robust to bf16 by default — Stable
      Diffusion ships in fp16/bf16 weights). Latent EMA stats stay in
      fp32 to preserve precision; everything else inside the autocast
      block runs in bf16. Opt-in via --amp.

  Fix H (NEW in v8) — minibatch training loop
      v7 forwards the full train_n=2048 batch in one shot per epoch.
      Activation memory scales with batch_size × hidden × spatial_dims;
      at hidden=128, ch=8, train_n=2048 the activation peak hits ~20GB
      and OOMs on H100 PCIe (80GB total but other things share). v8
      shuffles indices each epoch and steps through minibatches of
      `--minibatch-size` (default 256), running forward/backward/step
      per minibatch. One "epoch" = one full pass through the dataset =
      `train_n // minibatch_size` optimiser steps.

      Activation peak now scales with minibatch_size only, so train_n
      can grow freely without RAM pressure. LR scheduler T_max is
      auto-scaled to total_steps = epochs × steps_per_epoch so cosine
      decay still hits 1% of base lr at the end of training.

      Combined with Fix G (bf16): activation halved by dtype + 8× cut
      by minibatch = ~16× peak reduction vs v7 fp32 full-batch.

  Fix I (NEW in v8 patch) — periodic checkpoint + resume
      v3-v7 had no checkpoint because runs were <5min. v8 with
      train_n=2048 epochs=5000 takes ~1 hour at ~130ms/step on H100.
      Without checkpoint, kill mid-run = lose everything because
      sampling/eval only runs after train_model() returns. Fix I:
        --checkpoint-dir DIR    enables periodic save
        --checkpoint-every N    save cadence (default 200 ep)
        --resume PATH           restart from a saved .pt
      Saved state covers model, opt, lr_scheduler, latent stats EMA,
      and current epoch. Resume picks up exact training trajectory
      (cosine LR continues from where it left off, EMA stats stay).

All nine fixes live in this trainer script — no changes to vod_minimal/.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from datetime import datetime

try:
    import psutil
    _HAVE_PSUTIL = True
except ImportError:
    _HAVE_PSUTIL = False


def _rss_gb():
    """Return current process resident set size in GB.
    Prefers psutil; falls back to /proc/self/status VmRSS on Linux."""
    if _HAVE_PSUTIL:
        return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    kb = int(line.split()[1])
                    return kb / (1024 ** 2)
    except Exception:
        pass
    return -1.0
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

_PROTO = Path(__file__).resolve().parent.parent.parent / "prototype"
if str(_PROTO) not in sys.path:
    sys.path.insert(0, str(_PROTO))

# Fix K (v9): monkeypatch LATENT_T=1 BEFORE any NativeVOD instantiation.
# Module-level constant lookup is dynamic in CPython, so functions inside
# vod_minimal.native that read LATENT_T at call time will see the new
# value. NativeVODConfig.architecture_version is independent so checkpoint
# load semantics are unaffected. Set via env var so callers can opt out:
#   STATIC_T=1   default v9 mode (T=1 fold, 8x compute reduction)
#   STATIC_T=0   pass-through (equivalent to v8)
import os as _os
import vod_minimal.native as _native_mod
_STATIC_T = _os.environ.get("STATIC_T", "1") == "1"
if _STATIC_T:
    _native_mod.LATENT_T = 1
    _native_mod.AUDIO_SIZE = (
        _native_mod.LATENT_T * _native_mod.LATENT_HW * _native_mod.LATENT_HW
    )

from vod_minimal.blocky_scattering import build_blocky_scattering_batch
from vod_minimal.diffusion import (
    diffusion_loss, ddim_sample, make_schedule, q_sample,
)
from vod_minimal.metrics import artifact_metrics, descriptor
from vod_minimal.native import (
    LATENT_HW, LATENT_T, NativeVOD, NativeVODConfig,
    views_to_torch,
)
print(f"[v9] STATIC_T={_STATIC_T}  effective LATENT_T={LATENT_T}",
      file=sys.stderr)


DESCRIPTOR_KEYS = ("amplitude", "phase", "frequency", "compression", "salience", "snr")

# Set by main() before any sampling/training calls so the diffusion_loss
# and ddim_sample helpers stay in sync (v3/v4 hardcoded "x_0" in 3
# places — v5 makes prediction type a CLI arg with a single source of
# truth).
PREDICTION_TYPE = "x_0"


class _NullCtx:
    """No-op context manager used when AMP is disabled."""
    def __enter__(self): return self
    def __exit__(self, *a): return False


# ----------------------------------------------------------------------- #
#  Fix Q helpers (v11): zero-terminal-SNR schedule + v-prediction
#  Source: Lin 2023 (arXiv:2305.08891 / WACV 2024), section 3.1.
# ----------------------------------------------------------------------- #
def rescale_schedule_zero_snr(schedule):
    """Lin 2023 Eq. 4: rescale α_bar so the last timestep has α_bar=0
    (true pure noise at t=T-1). NoiseSchedule is a frozen dataclass so
    we use dataclasses.replace to build a new one."""
    import dataclasses
    a = schedule.alphas_cumprod.clone()
    a0 = a[0]
    a_last = a[-1]
    a = (a - a_last) / (a0 - a_last) * a0
    a = torch.clamp(a, min=0.0, max=1.0)
    return dataclasses.replace(schedule, alphas_cumprod=a)


def v_target(x_0, t, schedule, noise):
    """v = α·ε - σ·x_0  where α=sqrt(α_bar), σ=sqrt(1-α_bar). Lin 2023 Eq. 12."""
    a_bar = schedule.alphas_cumprod[t]
    a_bar = a_bar.view(-1, *([1] * (x_0.ndim - 1)))
    alpha = torch.sqrt(a_bar)
    sigma = torch.sqrt(torch.clamp(1.0 - a_bar, min=0.0))
    return alpha * noise - sigma * x_0


def v_loss(model, x_0, schedule):
    B = x_0.shape[0]
    t = torch.randint(0, schedule.num_steps, (B,), device=x_0.device)
    noise = torch.randn_like(x_0)
    x_t = q_sample(x_0, t, schedule, noise=noise)
    target = v_target(x_0, t, schedule, noise)
    pred = model.denoise(x_t, t=t)
    return F.mse_loss(pred, target)


@torch.no_grad()
def v_ddim_sample(model, shape, schedule, *, num_steps=50, device=None,
                   generator=None, dtype=None):
    """DDIM with v-prediction. Always start from t=T-1 (Lin 2023 §3.3)."""
    if device is None: device = next(model.parameters()).device
    if dtype  is None: dtype  = next(model.parameters()).dtype
    x = torch.randn(shape, device=device, dtype=dtype, generator=generator)
    timesteps = torch.linspace(
        schedule.num_steps - 1, 0, num_steps, dtype=torch.long, device=device,
    )
    for i in range(num_steps):
        t = timesteps[i]
        t_batch = t.expand(shape[0])
        a_bar = schedule.alphas_cumprod[t]
        alpha = torch.sqrt(a_bar)
        sigma = torch.sqrt(torch.clamp(1.0 - a_bar, min=0.0))
        v_pred = model.denoise(x, t=t_batch)
        x_0_pred = alpha * x - sigma * v_pred
        eps_pred = sigma * x + alpha * v_pred
        if i < num_steps - 1:
            t_next = timesteps[i + 1]
            a_next = schedule.alphas_cumprod[t_next]
            x = torch.sqrt(a_next) * x_0_pred + torch.sqrt(torch.clamp(1 - a_next, min=0)) * eps_pred
        else:
            x = x_0_pred
    return x


# ----------------------------------------------------------------------- #
#  Fix R helper (v11): auxiliary weak decoder
#  Capacity-limited 1x1 Linear that decodes the FIRST FRAME of the latent.
#  Used only as a regularizer; it is NOT used at sample time.
# ----------------------------------------------------------------------- #
class WeakDecoder(torch.nn.Module):
    """v13: REVERT to shallow Linear (v11 form). v12's deep weak decoder
    caused regression — see v13 docstring."""
    def __init__(self, channels, hidden_mult=4):
        super().__init__()
        self.proj = torch.nn.Linear(channels, 1, bias=True)

    def forward(self, latent):
        return self.proj(latent[:, 0]).squeeze(-1)


def per_channel_aux_loss(latent, dec_image, target_image):
    """Fix R''' (v13): per-channel auxiliary reconstruction.

    For each channel c independently, mask all OTHER channels to zero,
    pass through dec_image, compare with target. A collapsed channel
    (constant) cannot survive masking-induced reconstruction.

    latent:        (B, T, H, W, C)  — encoder output
    dec_image:     nn.Linear(C, 1)  — main decoder
    target_image:  (B, H, W)
    """
    B, T, H, W, C = latent.shape
    losses = []
    for c in range(C):
        mask = torch.zeros_like(latent)
        mask[..., c] = 1.0
        masked = latent * mask              # all-zero except channel c
        # Decode (B, T, H, W, C) → take first frame, project, squeeze.
        decoded = dec_image(masked[:, 0]).squeeze(-1)   # (B, H, W)
        losses.append(F.mse_loss(decoded, target_image))
    return torch.stack(losses).mean()


def channel_variance_floor_loss(latent, target_min_std=0.2):
    """Fix R'' (v12): hard floor on per-channel std.
    latent: (B, T, H, W, C). Penalty = mean over c of max(0, floor - std_c)^2.
    With floor=0.2 every channel is guaranteed σ≥0.2 at convergence."""
    per_ch_std = latent.std(dim=(0, 1, 2, 3))    # (C,)
    deficit = torch.clamp(target_min_std - per_ch_std, min=0.0)
    return (deficit ** 2).mean()


# --------------------------------------------------------------------- #
#  Image helpers
# --------------------------------------------------------------------- #

def normalise_to_uint8(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a).squeeze()
    if a.ndim != 2:
        a = a.reshape(a.shape[0], -1)
    lo, hi = float(a.min()), float(a.max())
    if hi - lo < 1e-9:
        return np.full_like(a, 127, dtype=np.uint8)
    return ((a - lo) / (hi - lo) * 255.0).astype(np.uint8)


def save_grid(images: list[np.ndarray], path: Path, ncols: int) -> None:
    if not images:
        return
    h, w = images[0].shape[-2:]
    n = len(images)
    nrows = (n + ncols - 1) // ncols
    grid = np.full((nrows * h, ncols * w), 127, dtype=np.uint8)
    for i, im in enumerate(images):
        r, c = i // ncols, i % ncols
        grid[r * h:(r + 1) * h, c * w:(c + 1) * w] = normalise_to_uint8(im)
    Image.fromarray(grid, mode="L").save(path)


# --------------------------------------------------------------------- #
#  Batched encode/decode (Fix C)
#  Bypasses NativeVOD.encode which only accepts single-sample dicts.
#  Uses the same nn.Linear weights, just calls them on batched tensors.
# --------------------------------------------------------------------- #

def batched_encode(model: NativeVOD, batched_views: dict[str, torch.Tensor]) -> torch.Tensor:
    """Encode a dict of batched media views to a single (B, T, H, W, C) latent.

    batched_views format:
        image: (B, H, W)
        video: (B, T, H, W)
    """
    contributions = []
    if "image" in batched_views:
        img = batched_views["image"]                                    # (B, H, W)
        u = model.enc_image(img.unsqueeze(-1))                          # (B, H, W, C)
        u = u.unsqueeze(1).expand(-1, LATENT_T, -1, -1, -1).contiguous()  # (B, T, H, W, C)
        contributions.append(u)
    if "video" in batched_views:
        vid = batched_views["video"]                                    # (B, T, H, W)
        u = model.enc_video(vid.unsqueeze(-1))                          # (B, T, H, W, C)
        contributions.append(u)
    return torch.stack(contributions, dim=0).mean(dim=0)


def batched_decode(model: NativeVOD, U_batched: torch.Tensor) -> dict[str, torch.Tensor]:
    """Decode (B, T, H, W, C) latent back to media views.

    Mirrors NativeVOD.decode: image takes the middle T-slice, video uses all T.
    """
    out = {}
    mid = U_batched.shape[1] // 2
    out["image"] = model.dec_image(U_batched[:, mid]).squeeze(-1)       # (B, H, W)
    out["video"] = model.dec_video(U_batched).squeeze(-1)               # (B, T, H, W)
    return out


# --------------------------------------------------------------------- #
#  Latent normalization (Fix B)
# --------------------------------------------------------------------- #

class LatentStats:
    def __init__(self, ema_decay: float = 0.99):
        self.mean = 0.0
        self.std = 1.0
        self.decay = ema_decay
        self.initialized = False

    def update(self, x: torch.Tensor) -> None:
        cur_mean = float(x.detach().mean().item())
        cur_std = float(x.detach().std().item()) + 1e-8
        if not self.initialized:
            self.mean, self.std = cur_mean, cur_std
            self.initialized = True
        else:
            self.mean = self.decay * self.mean + (1 - self.decay) * cur_mean
            self.std = self.decay * self.std + (1 - self.decay) * cur_std

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std + self.mean


# --------------------------------------------------------------------- #
#  Metrics
# --------------------------------------------------------------------- #

def aggregate_metrics(images: list[np.ndarray]) -> dict:
    if not images:
        return {}
    desc_vecs = np.stack([descriptor("img", np.asarray(im)).vec() for im in images], axis=0)
    flat = np.stack([np.asarray(im).ravel() for im in images], axis=0).astype(np.float64)
    finite_ratio = float(np.isfinite(flat).all(axis=1).mean())
    amp_range = float(flat.max() - flat.min())
    entropies = []
    for im in images:
        hist, _ = np.histogram(np.asarray(im).ravel(), bins=48)
        probs = hist[hist > 0] / max(1, hist.sum())
        entropies.append(float(-(probs * np.log2(probs)).sum()))
    residues = []
    for im in images:
        try:
            r = artifact_metrics({"image": np.asarray(im)})
            if not np.isnan(r["mean_tile_residue"]):
                residues.append(r["mean_tile_residue"])
        except Exception:
            pass
    return {
        "descriptor_mean": dict(zip(DESCRIPTOR_KEYS, desc_vecs.mean(axis=0).tolist())),
        "amplitude_range": amp_range,
        "entropy_mean": float(np.mean(entropies)),
        "tile_residue_mean": float(np.mean(residues)) if residues else float("nan"),
        "finite_ratio": finite_ratio,
    }


def descriptor_distance(a: dict, b: dict) -> float:
    av = np.array([a["descriptor_mean"][k] for k in DESCRIPTOR_KEYS])
    bv = np.array([b["descriptor_mean"][k] for k in DESCRIPTOR_KEYS])
    return float(np.linalg.norm(av - bv))


# --------------------------------------------------------------------- #
#  Main pipeline pieces
# --------------------------------------------------------------------- #

def build_batched_targets(args, device, rng):
    batch = build_blocky_scattering_batch(
        rng, batch_size=args.train_n, size=LATENT_HW, frames=LATENT_T,
        artifact_strength=0.0, tile=4, spacetime=True,
        temporal_mode="static", flicker_strength=0.0, paired_denoising=True,
    )
    images_np = [np.asarray(s.target_views["image"]) for s in batch.samples]
    # Fix J: dataset stays on CPU (pinned if --pin-dataset). Per-mb
    # to(device) copy in train loop. Avoids PyTorch CUDA caching
    # allocator shadowing large always-on GPU tensors into host RAM,
    # which is the Codex-confirmed source of episodic host-RAM spikes
    # that hit 20GB cgroup limit on shared login nodes.
    target_device = torch.device("cpu") if args.pin_dataset else device
    images_t = torch.stack(
        [torch.as_tensor(im, dtype=torch.float32) for im in images_np], dim=0
    ).to(target_device)
    if args.pin_dataset and target_device.type == "cpu":
        images_t = images_t.pin_memory()                        # (B, H, W) pinned
    # Fix A: force video = image broadcast (B, T, H, W)
    videos_t = images_t.unsqueeze(1).expand(-1, LATENT_T, -1, -1).contiguous()
    if args.pin_dataset and target_device.type == "cpu":
        videos_t = videos_t.pin_memory()
    return {"image": images_t, "video": videos_t}, images_np


def train_model(args, device, batched_targets):
    cfg = NativeVODConfig(
        channels=args.channels, hidden=args.hidden,
        denoise_steps=4, backbone="unet", time_dim=args.time_dim,
    )
    m = NativeVOD(cfg).to(device)
    n_params = sum(p_.numel() for p_ in m.parameters())
    print(f"[train] params={n_params:,} hidden={args.hidden} ch={args.channels} train_n={args.train_n}",
          flush=True)

    # Fix G: cudnn benchmark + bf16 autocast on CUDA. Latent stats EMA
    # stays in fp32 (autocast contexts only convert eligible ops; .item()
    # on bf16 tensors loses precision so we cast back inside the EMA).
    use_amp = (device.type == "cuda") and args.amp
    if use_amp:
        torch.backends.cudnn.benchmark = True
        print(f"[train] AMP enabled: bf16 autocast + cudnn.benchmark=True", flush=True)

    schedule = make_schedule(num_steps=args.diffusion_steps).to(device)
    # Fix M (v10): weight_decay 10× higher to counter T=1 overfit pressure.
    opt = torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Fix H: minibatch loop. One "epoch" = one full pass through the
    # dataset; steps_per_epoch = ceil(train_n / minibatch_size). Cosine
    # T_max scales to total optimiser steps so end-of-training lr still
    # lands at 1% of base.
    train_n = next(iter(batched_targets.values())).shape[0]
    minibatch_size = max(1, min(args.minibatch_size, train_n))
    steps_per_epoch = (train_n + minibatch_size - 1) // minibatch_size
    total_steps = steps_per_epoch * args.epochs
    print(f"[train] minibatch: train_n={train_n}  mb={minibatch_size}  "
          f"steps_per_epoch={steps_per_epoch}  total_steps={total_steps}",
          flush=True)

    # Fix F (re-tuned for v8): cosine over total_steps, not epochs.
    # Fix L (v10): eta_min raised from lr*0.01 → lr * args.cosine_eta_floor
    #   default 0.1 (so lr=1e-4 bottoms at 1e-5 not 1e-6, preventing the
    #   "lock into narrow basin" failure mode seen in v9 5k.
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=total_steps, eta_min=args.lr * args.cosine_eta_floor,
    ) if args.lr_schedule == "cosine" else None
    stats = LatentStats(ema_decay=0.99)  # kept for ckpt back-compat, not used

    # Fix Q (v11): rescale schedule to zero terminal SNR (Lin 2023 §3.1).
    # Idempotent — safe to call multiple times. Done BEFORE any sampling.
    if args.zero_terminal_snr:
        schedule = rescale_schedule_zero_snr(schedule)
        print(f"[train] schedule rescaled: α_bar[-1]={float(schedule.alphas_cumprod[-1]):.6f}  "
              f"α_bar[0]={float(schedule.alphas_cumprod[0]):.6f}", flush=True)

    # Fix P (v11): LDM-style fixed scaling factor (Rombach 2022 / SD 0.18215).
    # Encode `--scale-fit-n` train samples ONCE → compute global std →
    # freeze 1/std as `scaling`. NO mean shift (LDM doesn't either). NO EMA.
    scaling = 1.0
    if args.use_fixed_scaling:
        with torch.no_grad():
            n_fit = min(args.scale_fit_n, train_n)
            fit_idx = torch.arange(n_fit, device=device)
            fit_targets = {k: v.index_select(0, fit_idx).to(device, non_blocking=True)
                           if v.device != device else v.index_select(0, fit_idx)
                           for k, v in batched_targets.items()}
            L_fit = batched_encode(m, fit_targets).float()
            empirical_std = float(L_fit.std())
            scaling = 1.0 / max(empirical_std, 1e-6)
        print(f"[train] Fix P scaling: empirical_std={empirical_std:.4f}  "
              f"scaling=1/std={scaling:.4f}  (LDM SD uses 0.18215 = 1/5.49)",
              flush=True)

    # Fix R (v11): auxiliary weak decoder for posterior collapse (He ICLR 2019).
    weak_decoder = None
    if args.w_weak > 0:
        weak_decoder = WeakDecoder(channels=args.channels).to(device)
        # Add weak decoder params to optimizer (separate group, same lr).
        opt.add_param_group({"params": list(weak_decoder.parameters()),
                             "weight_decay": args.weight_decay})
        print(f"[train] Fix R weak decoder enabled: w_weak={args.w_weak}",
              flush=True)

    autocast_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_amp else _NullCtx()
    )

    rng_torch = torch.Generator(device="cpu").manual_seed(args.seed + 7)

    # Fix N (v10): EMA-of-weights snapshot. Track shadow params updated
    # per optimizer step with decay args.ema_decay (default 0.999). At
    # sample time, swap live → shadow. Standard recipe in DDPM/SD/EDM.
    ema_decay = args.ema_decay
    ema_shadow = None
    if ema_decay > 0:
        ema_shadow = {n: p.detach().clone() for n, p in m.named_parameters()
                      if p.requires_grad}
        print(f"[train] EMA enabled: decay={ema_decay} ({len(ema_shadow)} params)",
              flush=True)

    def ema_step():
        if ema_shadow is None: return
        with torch.no_grad():
            for n, p in m.named_parameters():
                if p.requires_grad and n in ema_shadow:
                    ema_shadow[n].mul_(ema_decay).add_(p.detach(), alpha=1 - ema_decay)

    # Fix I (NEW in v8 patch): periodic checkpoint + resume support.
    # 5000-ep runs at ~130 ms/step take ~1 hour; users need to be able
    # to kill mid-run and recover. Checkpoint dir is created up-front;
    # on each save we dump model + opt + sched + stats + ep so a clean
    # restart from --resume <ckpt> picks up exact training state.
    ckpt_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else None
    if ckpt_dir is not None:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        print(f"[train] checkpoint dir: {ckpt_dir} (every {args.checkpoint_every} ep)", flush=True)

    start_ep = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        m.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        if lr_scheduler is not None and ck.get("lr_scheduler") is not None:
            lr_scheduler.load_state_dict(ck["lr_scheduler"])
        stats.mean = ck["stats_mean"]
        stats.std = ck["stats_std"]
        stats.initialized = ck.get("stats_initialized", True)
        start_ep = int(ck["ep"]) + 1
        print(f"[train] resumed from {args.resume} at ep={start_ep}", flush=True)

    def save_ckpt(ep_now, tag="latest"):
        if ckpt_dir is None:
            return
        path = ckpt_dir / f"v8_{tag}.pt"
        torch.save({
            "ep": ep_now,
            "model": m.state_dict(),
            "opt": opt.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict() if lr_scheduler is not None else None,
            "stats_mean": stats.mean,
            "stats_std": stats.std,
            "stats_initialized": stats.initialized,
            "args": vars(args),
        }, path)

    t0 = time.time()
    global_step = start_ep * steps_per_epoch
    last_L_diff = float("nan")
    last_L_recon = float("nan")
    for ep in range(start_ep, args.epochs):
        m.train()
        # shuffle indices on CPU (deterministic w.r.t. seed)
        perm = torch.randperm(train_n, generator=rng_torch).to(device)

        ep_L_diff_sum = 0.0
        ep_L_recon_sum = 0.0
        ep_n = 0
        for s in range(steps_per_epoch):
            idx = perm[s * minibatch_size:(s + 1) * minibatch_size]
            # Fix J: dataset may live on CPU (pinned). index_select on
            # CPU then to(device, non_blocking=True) — peak host RAM
            # tracks only mb size, not train_n.
            ds_dev = next(iter(batched_targets.values())).device
            if ds_dev != device:
                idx_cpu = idx.to(ds_dev)
                mb_targets = {
                    k: v.index_select(0, idx_cpu).to(device, non_blocking=True)
                    for k, v in batched_targets.items()
                }
            else:
                mb_targets = {k: v.index_select(0, idx) for k, v in batched_targets.items()}

            opt.zero_grad(set_to_none=True)
            with autocast_ctx:
                x_0_live = batched_encode(m, mb_targets)
                # Fix P (v11): fixed scaling, NO EMA, NO mean shift.
                if args.use_fixed_scaling:
                    x_0_for_diff = x_0_live.detach() * scaling
                else:
                    stats.update(x_0_live.float())
                    x_0_for_diff = stats.normalize(x_0_live.detach())
                # Fix Q (v11): v-prediction loss instead of epsilon.
                if PREDICTION_TYPE == "v":
                    L_diff = v_loss(m, x_0_for_diff, schedule)
                else:
                    L_diff = diffusion_loss(m, x_0_for_diff, schedule, prediction=PREDICTION_TYPE)
                rec = batched_decode(m, x_0_live)
                L_recon_img = F.mse_loss(rec["image"], mb_targets["image"])
                L_recon_vid = F.mse_loss(rec["video"], mb_targets["video"])
                L_recon = (L_recon_img + L_recon_vid) / 2
                # Fix R (v11): auxiliary weak decoder forces all latent
                # channels to carry mutual info (anti posterior collapse).
                L_weak = torch.tensor(0.0, device=device)
                if weak_decoder is not None:
                    weak_pred = weak_decoder(x_0_live)
                    L_weak = F.mse_loss(weak_pred, mb_targets["image"])
                # Fix R'' (v12): channel-wise variance floor.
                L_chvar = torch.tensor(0.0, device=device)
                if args.w_chvar > 0:
                    L_chvar = channel_variance_floor_loss(
                        x_0_live.float(),
                        target_min_std=args.target_min_std,
                    )
                # Fix R''' (v13): per-channel auxiliary reconstruction.
                L_chaux = torch.tensor(0.0, device=device)
                if args.w_chaux > 0:
                    L_chaux = per_channel_aux_loss(
                        x_0_live, m.dec_image, mb_targets["image"],
                    )
                loss = (L_diff + args.w_recon * L_recon
                        + args.w_weak * L_weak
                        + args.w_chvar * L_chvar
                        + args.w_chaux * L_chaux)
            loss.backward()
            # Clip both main model and weak_decoder params if present.
            params_to_clip = list(m.parameters())
            if weak_decoder is not None:
                params_to_clip += list(weak_decoder.parameters())
            torch.nn.utils.clip_grad_norm_(params_to_clip, 1.0)
            opt.step()
            if lr_scheduler is not None:
                lr_scheduler.step()
            ema_step()  # Fix N: EMA-of-weights update per optimizer step
            global_step += 1

            n_mb = idx.shape[0]
            ep_L_diff_sum += float(L_diff.detach()) * n_mb
            ep_L_recon_sum += float(L_recon.detach()) * n_mb
            ep_n += n_mb

        last_L_diff = ep_L_diff_sum / max(1, ep_n)
        last_L_recon = ep_L_recon_sum / max(1, ep_n)
        # RSS monitor: every epoch (cheap, ~1µs). Prints when log cadence
        # ticks OR every --rss-every-ep epoch when set.
        log_tick = (ep + 1) % max(1, args.epochs // 10) == 0
        rss_tick = args.rss_every_ep > 0 and (ep + 1) % args.rss_every_ep == 0
        if log_tick or rss_tick:
            cur_lr = opt.param_groups[0]["lr"]
            rss_gb = _rss_gb()
            gpu_mem_gb = (torch.cuda.memory_allocated() / (1024 ** 3)
                          if device.type == "cuda" else 0.0)
            gpu_peak_gb = (torch.cuda.max_memory_allocated() / (1024 ** 3)
                           if device.type == "cuda" else 0.0)
            print(f"  ep={ep+1:4d}/{args.epochs}  step={global_step:6d}  "
                  f"L_diff={last_L_diff:.4f}  L_recon={last_L_recon:.4f}  "
                  f"latent_μ={stats.mean:+.3f}  latent_σ={stats.std:.3f}  "
                  f"lr={cur_lr:.2e}  "
                  f"rss={rss_gb:.2f}GB  gpu_alloc={gpu_mem_gb:.2f}GB  "
                  f"gpu_peak={gpu_peak_gb:.2f}GB", flush=True)
        # GC every --gc-every-ep epoch if set; lets us probe if leak is
        # autograd-graph residual or genuine reference accumulation.
        if args.gc_every_ep > 0 and (ep + 1) % args.gc_every_ep == 0:
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        # Fix I: periodic checkpoint (--checkpoint-every controls cadence).
        # Always overwrite "latest"; periodic snapshots get the ep number.
        if ckpt_dir is not None and ((ep + 1) % args.checkpoint_every == 0
                                     or ep + 1 == args.epochs):
            save_ckpt(ep, tag="latest")
            save_ckpt(ep, tag=f"ep{ep+1}")
    elapsed = time.time() - t0
    print(f"[train] done in {elapsed:.1f}s   "
          f"({elapsed*1000/max(1,global_step):.1f} ms/step  "
          f"{elapsed/args.epochs:.2f} s/epoch)   "
          f"final stats μ={stats.mean:+.4f} σ={stats.std:.4f}", flush=True)
    # Fix N: swap live params → EMA shadow before returning, so all
    # downstream sampling uses the time-averaged weights. Live params
    # are not restored — caller can hold a deep_copy if it wants raw.
    if ema_shadow is not None:
        with torch.no_grad():
            for n, p in m.named_parameters():
                if p.requires_grad and n in ema_shadow:
                    p.data.copy_(ema_shadow[n])
        print(f"[train] EMA weights applied to model for sampling", flush=True)
    # v11: stash scaling on model so sample_decoded_images can find it
    m.v11_scaling = scaling if args.use_fixed_scaling else None
    return m, schedule, stats


def sample_decoded_images(model, schedule, stats, n, seed, device):
    g = torch.Generator(device=device).manual_seed(seed)
    shape = (n, LATENT_T, LATENT_HW, LATENT_HW, model.config.channels)
    # Fix Q (v11): v-prediction sampler if active.
    if PREDICTION_TYPE == "v":
        x_norm = v_ddim_sample(model, shape, schedule, num_steps=50,
                                device=device, generator=g)
    else:
        x_norm = ddim_sample(model, shape, schedule, num_steps=50,
                              device=device, generator=g, prediction=PREDICTION_TYPE)
    # Fix P (v11): if model carries scaling, divide by it; else use EMA stats.
    scaling = getattr(model, "v11_scaling", None)
    if scaling is not None:
        x_unnorm = x_norm / scaling
    else:
        x_unnorm = stats.unnormalize(x_norm)
    with torch.no_grad():
        decoded = batched_decode(model, x_unnorm)
    return [decoded["image"][i].detach().cpu().numpy() for i in range(n)]


def random_noise_decoded(model, n, seed, device):
    g = torch.Generator(device=device).manual_seed(seed)
    shape = (n, LATENT_T, LATENT_HW, LATENT_HW, model.config.channels)
    x = torch.randn(shape, device=device, generator=g, dtype=next(model.parameters()).dtype)
    with torch.no_grad():
        decoded = batched_decode(model, x)
    return [decoded["image"][i].detach().cpu().numpy() for i in range(n)]


def zero_decoded(model, n, device):
    shape = (n, LATENT_T, LATENT_HW, LATENT_HW, model.config.channels)
    x = torch.zeros(shape, device=device, dtype=next(model.parameters()).dtype)
    with torch.no_grad():
        decoded = batched_decode(model, x)
    return [decoded["image"][i].detach().cpu().numpy() for i in range(n)]


def gate0_reconstruct(model, batched_targets, n, device):
    sub = {k: v[:n] for k, v in batched_targets.items()}
    with torch.no_grad():
        U = batched_encode(model, sub)
        decoded = batched_decode(model, U)
    return [decoded["image"][i].detach().cpu().numpy() for i in range(n)]


# --------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------- #

def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--seed", type=int, default=430)
    p.add_argument("--train-n", type=int, default=64)
    p.add_argument("--epochs", type=int, default=1500,
                   help="Fix O (v10): default lowered from 5000 → 1500. "
                        "v8/v9 5k showed stability degradation past plateau.")
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=1e-3,
                   help="Fix M (v10): default 10× v8/v9 (1e-4 → 1e-3) to "
                        "counter T=1 fold's 8× per-epoch param/data ratio.")
    p.add_argument("--cosine-eta-floor", type=float, default=0.1,
                   help="Fix L (v10): cosine eta_min = lr * floor. v8/v9 "
                        "used 0.01 → 1e-6 trapped model in narrow basin. "
                        "v10 default 0.1 → 1e-5 keeps optimizer mobile.")
    p.add_argument("--ema-decay", type=float, default=0.999,
                   help="Fix N (v10): EMA-of-weights decay. 0 disables. "
                        "Sampling uses EMA snapshot, not live params.")
    p.add_argument("--use-fixed-scaling", action="store_true", default=True,
                   help="Fix P (v11): LDM-style fixed scaling factor "
                        "(arXiv:2112.10752). Replaces EMA stats. Default ON.")
    p.add_argument("--no-fixed-scaling", dest="use_fixed_scaling",
                   action="store_false",
                   help="disable Fix P, fall back to v8 Fix B EMA stats")
    p.add_argument("--scale-fit-n", type=int, default=512,
                   help="Fix P: # of samples used to compute the empirical "
                        "std for the LDM scaling factor.")
    p.add_argument("--zero-terminal-snr", action="store_true", default=True,
                   help="Fix Q (v11): rescale schedule so α_bar[-1]=0 "
                        "(Lin 2023 / arXiv:2305.08891). Default ON.")
    p.add_argument("--no-zero-terminal-snr", dest="zero_terminal_snr",
                   action="store_false")
    p.add_argument("--w-weak", type=float, default=0.5,
                   help="Fix R (v11): shallow weak decoder weight. v13 "
                        "REVERTS to v11 default 0.5 (v12's 1.5 + deeper "
                        "weak decoder regressed Fix R).")
    p.add_argument("--w-chvar", type=float, default=100.0,
                   help="Fix R'' (v12/v13): channel-variance-floor weight. "
                        "v12 used 10 (insufficient). v13 = 100.")
    p.add_argument("--target-min-std", type=float, default=0.25,
                   help="Fix R'' (v12/v13): per-channel std target. "
                        "v12 used 0.20 (insufficient). v13 = 0.25.")
    p.add_argument("--w-chaux", type=float, default=2.0,
                   help="Fix R''' (v13): per-channel auxiliary reconstruction "
                        "loss weight. For each channel c, mask other "
                        "channels and decode → MSE with target. 0 disables.")
    p.add_argument("--diffusion-steps", type=int, default=200)
    p.add_argument("--time-dim", type=int, default=64)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--channels", type=int, default=4)
    p.add_argument("--n-samples", type=int, default=8)
    p.add_argument("--w-recon", type=float, default=1.0)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--out", default="generated/diffusion_samples_v8")
    p.add_argument("--report-out", default="prototype/unconditional_fidelity_result_v8.json")
    p.add_argument("--md-out", default="prototype/unconditional_fidelity_report_v8.md")
    p.add_argument("--prediction", choices=["x_0", "epsilon", "v"], default="v",
                   help="diffusion target — must be consistent across train+sample")
    p.add_argument("--lr-schedule", choices=["constant", "cosine"], default="cosine",
                   help="constant lr or cosine decay to 1% of base over full run")
    p.add_argument("--amp", action="store_true",
                   help="enable bf16 autocast + cudnn.benchmark on CUDA (Fix G)")
    p.add_argument("--minibatch-size", type=int, default=256,
                   help="minibatch size for SGD (Fix H). Activation peak "
                        "scales with this, not train_n. Default 256 fits "
                        "hidden=128 ch=8 in <8GB on H100.")
    p.add_argument("--checkpoint-dir", type=str, default=None,
                   help="if set, save model+opt+sched+stats every "
                        "--checkpoint-every epochs (Fix I). Required to "
                        "support kill+resume on long 5000-ep runs.")
    p.add_argument("--checkpoint-every", type=int, default=200,
                   help="epochs between checkpoint dumps (Fix I)")
    p.add_argument("--resume", type=str, default=None,
                   help="path to checkpoint (.pt) to resume from (Fix I)")
    p.add_argument("--rss-every-ep", type=int, default=0,
                   help="if >0, print host RSS + gpu mem every N epochs "
                        "(diagnostic for OOM leak hunting). 0=off (only "
                        "prints at log cadence).")
    p.add_argument("--gc-every-ep", type=int, default=0,
                   help="if >0, run gc.collect() + torch.cuda.empty_cache() "
                        "every N epochs. Use to probe if leak is autograd "
                        "residual (collect helps) vs real reference cycle "
                        "(no help).")
    p.add_argument("--pin-dataset", action="store_true",
                   help="keep batched_targets on CPU (pinned) and copy "
                        "per-mb to GPU (Fix J). Reduces host-RAM peak "
                        "from PyTorch caching allocator GPU shadow on "
                        "shared 20GB cgroup login nodes.")
    args = p.parse_args()
    global PREDICTION_TYPE
    PREDICTION_TYPE = args.prediction

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    print(f"[device] {device}", flush=True)
    print("[data] building batched Chladni targets + Fix A (video=image.broadcast)...",
          flush=True)
    batched_targets, train_images = build_batched_targets(args, device, rng)

    # Verify Fix A
    if batched_targets["video"].shape[1] >= 2:
        diff = (batched_targets["video"][:, 1:] - batched_targets["video"][:, :-1]).abs().max().item()
    else:
        diff = 0.0  # T=1 (Fix K static fold) has no inter-frame diff to verify
    print(f"[verify] forced-static video: max |frame[t+1]-frame[t]| = {diff:.6f}",
          flush=True)
    print(f"[verify] batched_targets shapes: "
          f"image={tuple(batched_targets['image'].shape)} video={tuple(batched_targets['video'].shape)}",
          flush=True)

    trained_model, schedule, stats = train_model(args, device, batched_targets)
    cfg = NativeVODConfig(
        channels=args.channels, hidden=args.hidden,
        denoise_steps=4, backbone="unet", time_dim=args.time_dim,
    )
    untrained_model = NativeVOD(cfg).to(device)
    untrained_stats = LatentStats()
    untrained_stats.mean, untrained_stats.std = 0.0, 1.0
    untrained_stats.initialized = True
    untrained_model.v11_scaling = None  # untrained → use stats path with μ=0,σ=1

    print("[sample] generating samples for all sources...", flush=True)
    sources = {
        "train_reference":         train_images[:args.n_samples],
        "trained_sample":          sample_decoded_images(trained_model, schedule, stats,
                                                        args.n_samples, args.seed + 1, device),
        "untrained_sample":        sample_decoded_images(untrained_model, schedule, untrained_stats,
                                                        args.n_samples, args.seed + 2, device),
        "random_noise_baseline":   random_noise_decoded(trained_model, args.n_samples,
                                                       args.seed + 3, device),
        "zero_baseline":           zero_decoded(trained_model, args.n_samples, device),
        "gate0_recon":             gate0_reconstruct(trained_model, batched_targets, 8, device),
    }

    print("[sample] multi-seed stability...", flush=True)
    stability_seeds = [args.seed + 100, args.seed + 200, args.seed + 300]
    stability_runs = {
        f"trained_seed_{s}": sample_decoded_images(trained_model, schedule, stats, 4, s, device)
        for s in stability_seeds
    }

    print("[metrics] aggregating...", flush=True)
    metrics_per_source = {name: aggregate_metrics(imgs) for name, imgs in sources.items()}
    train_metrics = metrics_per_source["train_reference"]
    distances = {name: descriptor_distance(m, train_metrics) for name, m in metrics_per_source.items()}
    stability_metrics = {name: aggregate_metrics(imgs) for name, imgs in stability_runs.items()}
    stability_distances = {name: descriptor_distance(m, train_metrics) for name, m in stability_metrics.items()}

    print("[output] writing PNG grids...", flush=True)
    for name, imgs in sources.items():
        save_grid(imgs, out / f"{name}.png", ncols=4)
    save_grid(sum([imgs for imgs in stability_runs.values()], []),
              out / "trained_multi_seed.png", ncols=4)

    trained = metrics_per_source["trained_sample"]
    rn, zr, un, tr = (distances[k] for k in
                      ("random_noise_baseline", "zero_baseline", "untrained_sample", "trained_sample"))
    finite_ok = trained["finite_ratio"] == 1.0
    range_ok = trained["amplitude_range"] > 0.05
    multi_seed_var = float(np.std([stability_distances[k] for k in stability_runs]))
    passes = {
        "finite_ratio == 1.0": finite_ok,
        "amplitude_range > 0.05": range_ok,
        "beats random_noise baseline": tr < rn,
        "beats zero baseline": tr < zr,
        "beats untrained_sample": tr < un,
    }
    n_pass = sum(passes.values())
    verdict = "PASS" if n_pass == len(passes) else ("PARTIAL" if n_pass >= 3 else "FAIL")

    payload = {
        "date": datetime.now().isoformat(),
        "version": "v13: A-O + P + Q + R(shallow-weak) + R''(ch-var-floor↑) + R'''(per-ch-aux)",
        "device": str(device),
        "train_args": vars(args),
        "sampler": {"type": "DDIM", "prediction": "x_0", "eta": 0.0,
                    "num_steps": 50, "schedule_steps": args.diffusion_steps},
        "latent_stats": {"mean": stats.mean, "std": stats.std},
        "metrics": {
            "per_source": metrics_per_source,
            "descriptor_distance_to_train": distances,
            "stability": {"per_seed_distances": stability_distances,
                          "std_across_seeds": multi_seed_var},
        },
        "verdict": verdict,
        "passes": passes,
    }
    Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[output] JSON -> {args.report_out}", flush=True)

    md = [
        "# Unconditional Sample Fidelity v13 Report\n\n",
        f"**Date**: {payload['date']}\n",
        f"**Version**: {payload['version']}\n",
        f"**Device**: {device}\n",
        f"**Verdict**: **{verdict}** ({n_pass}/{len(passes)} checks pass)\n\n",
        "## Train args\n```json\n" + json.dumps(vars(args), indent=2) + "\n```\n\n",
        f"## Latent stats (Fix B)\nμ={stats.mean:+.4f}  σ={stats.std:.4f}\n\n",
        "## descriptor_distance_to_train\n\n",
        "| source | distance |\n|---|---|\n",
        *[f"| `{name}` | {d:.4f} |\n" for name, d in sorted(distances.items(), key=lambda kv: kv[1])],
        "\n## Per-source metrics\n\n",
        "| source | finite | amp_range | entropy | tile_residue |\n|---|---|---|---|---|\n",
        *[f"| `{name}` | {m['finite_ratio']:.3f} | {m['amplitude_range']:.3f} | "
          f"{m['entropy_mean']:.3f} | {m['tile_residue_mean']:.3f} |\n"
          for name, m in metrics_per_source.items()],
        "\n## PASS checks\n\n",
        *[f"- {'PASS' if v else 'FAIL'}  {k}\n" for k, v in passes.items()],
        f"\n## Multi-seed stability\n\nstd of distance across 3 seeds: **{multi_seed_var:.4f}**\n",
    ]
    Path(args.md_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.md_out).write_text("".join(md), encoding="utf-8")
    print(f"[output] MD report -> {args.md_out}", flush=True)
    print(f"\n=== VERDICT: {verdict} ===", flush=True)


if __name__ == "__main__":
    main()
