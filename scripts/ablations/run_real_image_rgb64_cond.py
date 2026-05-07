"""VOD-real-image-rgb64-cond — Stage 2 class conditioning smoke (Codex 协议).

Builds on Stage 1 (rgb64 conv head) by adding minimal class-id conditioning:
  • CIFAR-10 class id (0-9) → nn.Embedding(11, time_dim)  [+1 null token]
  • Class embedding is ADDED to the existing time embedding before
    concatenation with denoiser features (additive AdaLN-zero pattern).
  • Condition dropout p=0.1: train-time replace cond with null token.
  • cond=None path preserved (uses null token) for CFG/unconditional.
  • Sample: per-class grid (4 images each class) + random-cond sanity.

Type-B substrate intact: shared U(t,y,x,c) + 1×1 projection heads.
NO external CLIP / VAE / SD encoder.

Goal: confirm v16 substrate handles RGB 64×64 without external VAE/CLIP.

Design (preserves type-B):
  enc_image:  (H, W, 3) RGB → DCT lift per RGB channel → (H, W, 24) →
              Linear(24, C=8) → 8-channel substrate
  dec_image:  (H, W, 8) substrate → Linear(8, 3) → (H, W, 3) RGB
  enc_video:  same as enc_image (T=1 fold)
  dec_video:  same as dec_image (T=1 fold)

Single shared field U(t,y,x,c) c=8. RGB enters/exits through 1×1 projection
heads only. NO external pretrained encoder, NO SD/VAE/CLIP. Type-B intact.

Inherits v16 fix bundle: A-O + P (LDM scaling) + Q (v-pred + zsnr) +
R (weak decoder) + S' (DCT lift, RGB-aware here).

This is a **smoke test**, not a research conclusion. Goal: confirm
that the v16 fix bundle (P+Q+R+S' on top of A-O) trained on Chladni
toy data also trains/reconstructs/samples something image-like when
the data distribution is replaced with real natural images.

Inherits from `run_unconditional_fidelity_v16.py`:
  Fix A-O (substrate plumbing, mb SGD, ckpt, pinned dataset, ...)
  Fix P  (LDM-style fixed scaling factor)
  Fix Q  (zero-terminal-SNR + v-prediction)
  Fix R  (auxiliary weak decoder)
  Fix S' (DCT-II 2D orthogonal field lift)

Key changes:
  • Dataset: CIFAR-10 (uoft-cs/cifar10, HF parquet) → grayscale 32×32
  • LATENT_HW monkeypatched to args.image_size (default 32)
  • Image pixel range normalized to [-1, 1]
  • Per-task verdict: visual + descriptor + Codex-style 5 criteria

Hard rule (per the project instruction): if Gate-0 reconstruction
fails, do NOT continue to diffusion training. Report the minimal
fix instead.
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

# Real-image smoke uses LATENT_HW=32 by default (CIFAR-10 native size),
# overridable from CLI via --image-size before NativeVOD instantiation.
# We do this in main() right after parsing args; the default below
# matches the most common --image-size value so direct module imports
# still see a sensible value.
_IMAGE_SIZE_DEFAULT = int(_os.environ.get("VOD_IMAGE_SIZE", "32"))
_native_mod.LATENT_HW = _IMAGE_SIZE_DEFAULT
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
    def __init__(self, channels):
        super().__init__()
        self.proj = torch.nn.Linear(channels, 1, bias=True)

    def forward(self, latent):
        return self.proj(latent[:, 0]).squeeze(-1)


# ----------------------------------------------------------------------- #
#  Fix S (v15): multi-scale field-feature lift
# ----------------------------------------------------------------------- #
def _make_dct2_kernels(K=8, kH=3, kW=3):
    """Build K 2D DCT-II basis kernels of size (kH, kW).
    DCT-II 1D: φ_n[k] = cos(π/N (k+0.5) n).  2D: outer product.
    We pick the K lowest-(u+v) frequency pairs, sorted by (u+v, u, v).
    Return tensor (K, 1, kH, kW) ready for conv2d."""
    import math
    # 1D bases of length kH and kW
    basis_h = []
    for u in range(kH):
        b = torch.tensor([math.cos(math.pi * u * (i + 0.5) / kH) for i in range(kH)])
        if u > 0:
            b = b * math.sqrt(2.0 / kH)
        else:
            b = b * math.sqrt(1.0 / kH)
        basis_h.append(b)
    basis_w = []
    for v in range(kW):
        b = torch.tensor([math.cos(math.pi * v * (i + 0.5) / kW) for i in range(kW)])
        if v > 0:
            b = b * math.sqrt(2.0 / kW)
        else:
            b = b * math.sqrt(1.0 / kW)
        basis_w.append(b)
    # Pick K lowest-frequency (u, v) pairs
    pairs = sorted(((u, v) for u in range(kH) for v in range(kW)),
                   key=lambda uv: (uv[0] + uv[1], uv[0], uv[1]))[:K]
    kernels = []
    for u, v in pairs:
        kern = torch.outer(basis_h[u], basis_w[v])    # (kH, kW)
        kernels.append(kern)
    return torch.stack(kernels, dim=0).unsqueeze(1)    # (K, 1, kH, kW)


class FieldLift(torch.nn.Module):
    """v16: DCT-II 2D orthogonal lift. K=8 lowest-frequency basis
    functions on a 3×3 neighborhood. Orthogonal by construction."""
    def __init__(self, K=8, kH=3, kW=3):
        super().__init__()
        kernels = _make_dct2_kernels(K=K, kH=kH, kW=kW)   # (K, 1, kH, kW)
        self.register_buffer("kernels", kernels)
        self.K = K
        self.kH = kH
        self.kW = kW

    def forward(self, x_last1):
        """Input shape (..., 1). Output shape (..., K)."""
        x = x_last1.squeeze(-1)
        original_shape = x.shape
        H, W = original_shape[-2], original_shape[-1]
        x_flat = x.reshape(-1, 1, H, W).contiguous()
        out = F.conv2d(x_flat, self.kernels,
                       padding=(self.kH // 2, self.kW // 2))
        out = out.permute(0, 2, 3, 1).contiguous()
        out_shape = original_shape + (self.K,)
        return out.reshape(out_shape)


class FieldLiftedEncoder(torch.nn.Module):
    """Grayscale-input encoder (v16 form). Kept for back-compat."""
    def __init__(self, channels):
        super().__init__()
        self.lift = FieldLift(K=channels)
        self.scale = torch.nn.Parameter(torch.ones(channels))
        self.bias = torch.nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        feats = self.lift(x)
        return feats * self.scale + self.bias


# ----------------------------------------------------------------------- #
#  RGB-aware encoder/decoder for Stage 1 (rgb64)
# ----------------------------------------------------------------------- #
def _channels_first_apply(x_last_axis_C, fn):
    """Helper: apply a Conv2d-stack `fn` to a tensor whose channel
    axis is the LAST. Input shape (..., H, W, C_in). Output shape
    (..., H, W, C_out) where C_out is determined by fn."""
    orig_shape = x_last_axis_C.shape
    H, W = orig_shape[-3], orig_shape[-2]
    x = x_last_axis_C.movedim(-1, -3).contiguous()       # (..., C, H, W)
    x_flat = x.reshape(-1, x.shape[-3], H, W)
    out_flat = fn(x_flat)                                # (-1, C_out, H, W)
    out = out_flat.movedim(-3, -1).contiguous()          # (-1, H, W, C_out)
    return out.reshape(*orig_shape[:-3], H, W, out_flat.shape[-3])


class RGBConvEncoder(torch.nn.Module):
    """Learned RGB→C-channel projection head (Codex Stage-1 design).

    Architecture (per Codex spec):
      Conv2d(3, width, k=3, p=1) → SiLU → Conv2d(width, C, k=1)

    No spatial downsample (substrate stays at H×W). The 3×3 conv lets
    the encoder learn local RGB combinations; the 1×1 conv projects to
    substrate channels. Type-B substrate U(t,y,x,c) is preserved —
    only the modality projection head is learnable now.
    """
    def __init__(self, channels: int, width: int = 64):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, width, kernel_size=3, padding=1)
        self.act = torch.nn.SiLU()
        self.conv2 = torch.nn.Conv2d(width, channels, kernel_size=1)

    def forward(self, x):
        """Input (..., H, W, 3). Output (..., H, W, C)."""
        def fn(t):
            return self.conv2(self.act(self.conv1(t)))
        return _channels_first_apply(x, fn)


class RGBConvDecoder(torch.nn.Module):
    """Mirror of RGBConvEncoder: C-channel substrate → RGB image.

    Conv2d(C, width, k=3, p=1) → SiLU → Conv2d(width, 3, k=1)
    """
    def __init__(self, channels: int, width: int = 64):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(channels, width, kernel_size=3, padding=1)
        self.act = torch.nn.SiLU()
        self.conv2 = torch.nn.Conv2d(width, 3, kernel_size=1)

    def forward(self, latent):
        """Input (..., H, W, C). Output (..., H, W, 3)."""
        def fn(t):
            return self.conv2(self.act(self.conv1(t)))
        return _channels_first_apply(latent, fn)


# ----- Aliases for back-compat with code that imports these names ------- #
RGBFieldLiftedEncoder = RGBConvEncoder    # legacy name → new conv head
RGBDecoder = RGBConvDecoder               # legacy name → new conv head


# ----------------------------------------------------------------------- #
#  Stage 2: class conditioning — additive embedding into time path
# ----------------------------------------------------------------------- #
NUM_CLASSES_DEFAULT = 10            # CIFAR-10
NULL_CLASS_TOKEN = 10               # +1 token for unconditional / CFG


class ClassConditioner(torch.nn.Module):
    """nn.Embedding(num_classes + 1, time_dim).
    The +1 token is the 'null' / unconditional condition (CFG infrastructure).
    Output is added to the timestep embedding before being broadcast into
    the denoiser feature stack (additive AdaLN-zero pattern)."""
    def __init__(self, num_classes: int, embed_dim: int):
        super().__init__()
        self.num_classes = num_classes
        self.null_token = num_classes        # last index reserved for null
        self.embed = torch.nn.Embedding(num_classes + 1, embed_dim)
        # Init the null token to zero so unconditional == "no shift"
        # at training start (CFG-style).
        with torch.no_grad():
            self.embed.weight[self.null_token].zero_()

    def forward(self, cond):
        """cond: (B,) int64 class ids in [0, num_classes-1], or None / -1
        for unconditional (uses null token)."""
        if cond is None:
            return None
        # Replace -1 sentinel with null_token
        c = torch.where(cond < 0, torch.full_like(cond, self.null_token), cond)
        c = torch.clamp(c, min=0, max=self.num_classes)
        return self.embed(c)                  # (B, embed_dim)


def _patch_denoise_with_cond(m, conditioner):
    """Override m.denoise to accept an optional cond= kwarg. The class
    embedding is added to the timestep embedding before NativeVOD's
    standard concat-into-feature-stack path. Preserves cond=None
    behaviour for unconditional sampling."""
    import types
    orig_denoise = m.denoise
    time_dim = m.config.time_dim

    def denoise_with_cond(self, u_noisy, *, t=None, cond=None):
        # Fast path: no conditioner (just delegate to original).
        if conditioner is None or time_dim <= 0:
            return orig_denoise(u_noisy, t=t)
        # We replicate the original feature-build path so we can mix
        # the class embedding INTO t_emb before concat.
        squeeze_batch = False
        if u_noisy.ndim == 4:
            u = u_noisy.unsqueeze(0)
            squeeze_batch = True
        else:
            u = u_noisy
        feats = self._build_features(u)
        from vod_minimal.diffusion import sinusoidal_time_embedding
        B, T, H, W, _ = feats.shape
        if t is None:
            t_batch = torch.zeros(B, device=feats.device, dtype=torch.long)
        elif t.ndim == 0:
            t_batch = t.expand(B)
        else:
            t_batch = t
        t_emb = sinusoidal_time_embedding(t_batch, time_dim, dtype=feats.dtype)
        # Class embedding (additive). cond=None → null token.
        if cond is None:
            null = torch.full((B,), conditioner.null_token,
                              device=feats.device, dtype=torch.long)
            c_emb = conditioner(null)
        else:
            c_emb = conditioner(cond)
        c_emb = c_emb.to(dtype=feats.dtype)
        combined = t_emb + c_emb               # (B, time_dim) AdaLN-zero
        combined_grid = combined.view(B, 1, 1, 1, time_dim).expand(B, T, H, W, time_dim)
        feats = torch.cat([feats, combined_grid], dim=-1)
        delta = self.denoiser(feats)
        # Original denoise applies a step_logit-scaled delta on top of u_noisy.
        out = u + torch.sigmoid(self.step_logit) * delta
        if squeeze_batch:
            out = out.squeeze(0)
        return out

    m.denoise = types.MethodType(denoise_with_cond, m)


def v_loss_cond(model, x_0, schedule, cond, *, p_drop_cond: float = 0.1):
    """v-prediction loss with class conditioning + condition dropout."""
    B = x_0.shape[0]
    t = torch.randint(0, schedule.num_steps, (B,), device=x_0.device)
    noise = torch.randn_like(x_0)
    x_t = q_sample(x_0, t, schedule, noise=noise)
    target = v_target(x_0, t, schedule, noise)
    # CFG: with prob p_drop_cond, replace condition with null token.
    if cond is not None and p_drop_cond > 0:
        drop_mask = torch.rand(B, device=cond.device) < p_drop_cond
        cond_in = torch.where(drop_mask, torch.full_like(cond, -1), cond)
    else:
        cond_in = cond
    pred = model.denoise(x_t, t=t, cond=cond_in)
    return F.mse_loss(pred, target)


@torch.no_grad()
def v_ddim_sample_cond(model, shape, schedule, *, num_steps=50, device=None,
                        generator=None, dtype=None, cond=None):
    """v-prediction DDIM with optional class conditioning.
    cond: (n,) int64 class ids OR None for unconditional."""
    if device is None: device = next(model.parameters()).device
    if dtype is None: dtype = next(model.parameters()).dtype
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
        v_pred = model.denoise(x, t=t_batch, cond=cond)
        x_0_pred = alpha * x - sigma * v_pred
        eps_pred = sigma * x + alpha * v_pred
        if i < num_steps - 1:
            t_next = timesteps[i + 1]
            a_next = schedule.alphas_cumprod[t_next]
            x = torch.sqrt(a_next) * x_0_pred + torch.sqrt(torch.clamp(1 - a_next, min=0)) * eps_pred
        else:
            x = x_0_pred
    return x


# --------------------------------------------------------------------- #
#  Image helpers
# --------------------------------------------------------------------- #

def normalise_to_uint8(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    # Preserve trailing-3 RGB axis; squeeze only leading singletons.
    while a.ndim > 0 and a.shape[0] == 1 and a.ndim > 2:
        a = a[0]
    is_rgb = a.ndim == 3 and a.shape[-1] == 3
    if not is_rgb:
        a = a.squeeze()
        if a.ndim != 2:
            a = a.reshape(a.shape[0], -1)
    lo, hi = float(a.min()), float(a.max())
    if hi - lo < 1e-9:
        return np.full(a.shape, 127, dtype=np.uint8)
    return ((a - lo) / (hi - lo) * 255.0).astype(np.uint8)


def save_grid(images: list[np.ndarray], path: Path, ncols: int) -> None:
    if not images:
        return
    arr0 = np.asarray(images[0])
    is_rgb = arr0.ndim >= 3 and arr0.shape[-1] == 3
    if is_rgb:
        h, w = arr0.shape[-3], arr0.shape[-2]
    else:
        h, w = arr0.shape[-2], arr0.shape[-1]
    n = len(images)
    nrows = (n + ncols - 1) // ncols
    if is_rgb:
        grid = np.full((nrows * h, ncols * w, 3), 127, dtype=np.uint8)
    else:
        grid = np.full((nrows * h, ncols * w), 127, dtype=np.uint8)
    for i, im in enumerate(images):
        r, c = i // ncols, i % ncols
        grid[r * h:(r + 1) * h, c * w:(c + 1) * w] = normalise_to_uint8(im)
    Image.fromarray(grid, mode=("RGB" if is_rgb else "L")).save(path)


# --------------------------------------------------------------------- #
#  Batched encode/decode (Fix C)
#  Bypasses NativeVOD.encode which only accepts single-sample dicts.
#  Uses the same nn.Linear weights, just calls them on batched tensors.
# --------------------------------------------------------------------- #

def batched_encode(model: NativeVOD, batched_views: dict[str, torch.Tensor]) -> torch.Tensor:
    """RGB-aware batched encode.

    batched_views format (Stage-1 RGB64):
        image: (B, H, W, 3)
        video: (B, T, H, W, 3)
    """
    contributions = []
    if "image" in batched_views:
        img = batched_views["image"]                                    # (B, H, W, 3)
        u = model.enc_image(img)                                        # (B, H, W, C)
        u = u.unsqueeze(1).expand(-1, LATENT_T, -1, -1, -1).contiguous()  # (B, T, H, W, C)
        contributions.append(u)
    if "video" in batched_views:
        vid = batched_views["video"]                                    # (B, T, H, W, 3)
        u = model.enc_video(vid)                                        # (B, T, H, W, C)
        contributions.append(u)
    return torch.stack(contributions, dim=0).mean(dim=0)


def batched_decode(model: NativeVOD, U_batched: torch.Tensor) -> dict[str, torch.Tensor]:
    """RGB-aware batched decode. Output image (B, H, W, 3), video (B, T, H, W, 3)."""
    out = {}
    mid = U_batched.shape[1] // 2
    out["image"] = model.dec_image(U_batched[:, mid])                   # (B, H, W, 3)
    out["video"] = model.dec_video(U_batched)                           # (B, T, H, W, 3)
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

def _load_cifar10_rgb(args, rng) -> tuple:
    """Return (images (B,H,W,3) float32 in [-1, 1], labels (B,) int64).
    Uses HF datasets parquet (uoft-cs/cifar10)."""
    from datasets import load_dataset
    from PIL import Image
    print(f"[data] loading CIFAR-10 RGB from uoft-cs/cifar10 (HF parquet) ...",
          flush=True)
    ds = load_dataset("uoft-cs/cifar10", split="train",
                      cache_dir=args.data_cache_dir)
    n = min(args.train_n, len(ds))
    idx = rng.choice(len(ds), size=n, replace=False)
    out = np.empty((n, args.image_size, args.image_size, 3), dtype=np.float32)
    labels = np.empty(n, dtype=np.int64)
    for i, di in enumerate(idx):
        ex = ds[int(di)]
        img = ex["img"]    # PIL RGB 32×32
        labels[i] = int(ex["label"])
        if img.size != (args.image_size, args.image_size):
            # bilinear upsample 32→64 (NEAREST loses too much fidelity)
            img = img.resize((args.image_size, args.image_size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0   # → [-1, 1]
        out[i] = arr
    print(f"[data] CIFAR-10 RGB loaded: shape={out.shape} "
          f"range=[{out.min():.3f}, {out.max():.3f}] "
          f"mean={out.mean():.4f} std={out.std():.4f}",
          flush=True)
    return out, labels


def _load_local_imagefolder(args, rng) -> np.ndarray:
    """Fallback: load from a local directory of image files.
    Glob *.{png,jpg,jpeg} under --data-dir."""
    from PIL import Image
    paths = []
    for ext in ("png", "jpg", "jpeg", "bmp", "PNG", "JPG"):
        paths.extend(Path(args.data_dir).glob(f"**/*.{ext}"))
    paths = sorted(paths)
    if not paths:
        raise SystemExit(f"[data] no images under {args.data_dir!r}")
    n = min(args.train_n, len(paths))
    sel = rng.choice(len(paths), size=n, replace=False)
    out = np.empty((n, args.image_size, args.image_size), dtype=np.float32)
    for i, si in enumerate(sel):
        img = Image.open(paths[int(si)]).convert("L")
        img = img.resize((args.image_size, args.image_size))
        out[i] = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
    print(f"[data] local imagefolder loaded: shape={out.shape} "
          f"range=[{out.min():.3f}, {out.max():.3f}] "
          f"mean={out.mean():.4f} std={out.std():.4f}",
          flush=True)
    return out


def build_batched_targets(args, device, rng):
    """RGB 64×64 real-image smoke + class labels."""
    if args.dataset == "cifar10":
        images_np_arr, labels_np = _load_cifar10_rgb(args, rng)
    else:
        raise SystemExit(f"--dataset {args.dataset!r} not supported in rgb64 yet")

    images_np = [images_np_arr[i] for i in range(images_np_arr.shape[0])]

    target_device = torch.device("cpu") if args.pin_dataset else device
    images_t = torch.from_numpy(images_np_arr).contiguous().to(target_device)
    labels_t = torch.from_numpy(labels_np).contiguous().to(target_device)
    if args.pin_dataset and target_device.type == "cpu":
        images_t = images_t.pin_memory()
        labels_t = labels_t.pin_memory()
    videos_t = images_t.unsqueeze(1).contiguous()
    if args.pin_dataset and target_device.type == "cpu":
        videos_t = videos_t.pin_memory()
    return {"image": images_t, "video": videos_t, "labels": labels_t}, images_np


def train_model(args, device, batched_targets):
    cfg = NativeVODConfig(
        channels=args.channels, hidden=args.hidden,
        denoise_steps=4, backbone="unet", time_dim=args.time_dim,
    )
    m = NativeVOD(cfg).to(device)
    # Stage-1 RGB64: replace enc/dec heads to handle RGB (H, W, 3) input/output.
    # Type-B preserved: shared substrate U(t,y,x,c), only projections change.
    if args.use_field_lift:
        m.enc_image = RGBFieldLiftedEncoder(channels=args.channels).to(device)
        m.enc_video = RGBFieldLiftedEncoder(channels=args.channels).to(device)
        m.dec_image = RGBDecoder(channels=args.channels).to(device)
        m.dec_video = RGBDecoder(channels=args.channels).to(device)
        # Override _encode_image / _encode_video to skip the original
        # unsqueeze(-1) trick (the original adds a singleton channel for
        # Linear(1, C); for RGB input the channel dim already exists).
        import types
        def _encode_image_rgb(self, image):
            # image: (H, W, 3) — no unsqueeze
            u_hw = self.enc_image(image)                  # (H, W, C)
            return u_hw.unsqueeze(0).expand(LATENT_T, *u_hw.shape)
        def _encode_video_rgb(self, video):
            # video: (T, H, W, 3) — no unsqueeze
            return self.enc_video(video)                  # (T, H, W, C)
        m._encode_image = types.MethodType(_encode_image_rgb, m)
        m._encode_video = types.MethodType(_encode_video_rgb, m)
        print(f"[train] Stage-1 RGB64: enc/dec heads replaced "
              f"(Conv2d 3→{64}→{args.channels})", flush=True)

    # Stage-2: class conditioner.
    conditioner = None
    if args.num_classes > 0 and args.time_dim > 0:
        conditioner = ClassConditioner(
            num_classes=args.num_classes, embed_dim=args.time_dim,
        ).to(device)
        # Add conditioner params as a sub-module so optimiser sees them
        m.conditioner = conditioner
        _patch_denoise_with_cond(m, conditioner)
        # Re-instantiate optimizer to include conditioner params (if it
        # already exists below this block, we'll handle there). We
        # build opt after, so just record.
        print(f"[train] Stage-2: class conditioner enabled "
              f"({args.num_classes} classes + 1 null, embed_dim={args.time_dim}, "
              f"p_drop_cond={args.p_drop_cond})", flush=True)

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
                # Fix Q (v11) + Stage-2 (cond): v-prediction loss with class conditioning.
                if PREDICTION_TYPE == "v":
                    if conditioner is not None and "labels" in mb_targets:
                        L_diff = v_loss_cond(m, x_0_for_diff, schedule,
                                              cond=mb_targets["labels"],
                                              p_drop_cond=args.p_drop_cond)
                    else:
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
                loss = L_diff + args.w_recon * L_recon + args.w_weak * L_weak
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


def sample_decoded_images(model, schedule, stats, n, seed, device, cond=None):
    """If cond is provided (and model has conditioner), sample with class cond."""
    g = torch.Generator(device=device).manual_seed(seed)
    shape = (n, LATENT_T, LATENT_HW, LATENT_HW, model.config.channels)
    has_cond = (cond is not None) and getattr(model, "conditioner", None) is not None
    if PREDICTION_TYPE == "v":
        if has_cond:
            cond_t = cond if torch.is_tensor(cond) else torch.tensor(
                cond, device=device, dtype=torch.long
            )
            cond_t = cond_t.to(device=device, dtype=torch.long)
            x_norm = v_ddim_sample_cond(model, shape, schedule, num_steps=50,
                                          device=device, generator=g, cond=cond_t)
        else:
            x_norm = v_ddim_sample(model, shape, schedule, num_steps=50,
                                    device=device, generator=g)
    else:
        x_norm = ddim_sample(model, shape, schedule, num_steps=50,
                              device=device, generator=g, prediction=PREDICTION_TYPE)
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
    p.add_argument("--dataset", choices=["cifar10", "local"], default="cifar10",
                   help="real-image smoke dataset. cifar10 uses HF "
                        "uoft-cs/cifar10 (no trust_remote_code needed). "
                        "local reads images from --data-dir.")
    p.add_argument("--data-dir", type=str, default="D:/VOD/data/real_images",
                   help="local imagefolder fallback path")
    p.add_argument("--data-cache-dir", type=str, default=None,
                   help="HF datasets cache dir (default: HF default)")
    p.add_argument("--image-size", type=int, default=32,
                   help="image resize target H=W. monkeypatches "
                        "vod_minimal.native.LATENT_HW. CIFAR-10 native "
                        "is 32, so 32 is no-op resize.")
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
                   help="Fix R (v11): auxiliary weak decoder loss weight.")
    p.add_argument("--use-field-lift", action="store_true", default=True,
                   help="Fix S (v15): replace enc_image/enc_video Linear(1,C) "
                        "with FieldLift+Linear(8,C). Default ON.")
    p.add_argument("--no-field-lift", dest="use_field_lift",
                   action="store_false")
    p.add_argument("--diffusion-steps", type=int, default=200)
    p.add_argument("--time-dim", type=int, default=64)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--channels", type=int, default=4)
    p.add_argument("--n-samples", type=int, default=8)
    p.add_argument("--num-classes", type=int, default=10,
                   help="Stage-2: number of classes for conditioning. "
                        "0 disables conditioning (pure unconditional).")
    p.add_argument("--p-drop-cond", type=float, default=0.1,
                   help="Stage-2: probability of replacing class id with "
                        "null token at training time (CFG dropout).")
    p.add_argument("--samples-per-class", type=int, default=4,
                   help="Stage-2: per-class grid size (samples per class).")
    p.add_argument("--w-recon", type=float, default=1.0)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--out", default="generated/rgb64_cond")
    p.add_argument("--report-out", default="prototype/rgb64_cond_result.json")
    p.add_argument("--md-out", default="prototype/rgb64_cond_report.md")
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

    # Real-image smoke: monkeypatch LATENT_HW to the requested image size
    # BEFORE NativeVOD instantiation. We also re-bind the module-level
    # LATENT_HW alias used by this script so all downstream shape
    # references match.
    import vod_minimal.native as _nm
    _nm.LATENT_HW = int(args.image_size)
    _nm.AUDIO_SIZE = _nm.LATENT_T * _nm.LATENT_HW * _nm.LATENT_HW
    global LATENT_HW
    LATENT_HW = int(args.image_size)
    print(f"[real-image] LATENT_HW set to {LATENT_HW} (was {_IMAGE_SIZE_DEFAULT} default)",
          flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    print(f"[device] {device}", flush=True)
    print(f"[data] building real-image batched targets (dataset={args.dataset}, "
          f"image_size={args.image_size}, train_n={args.train_n}) ...",
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
    if args.use_field_lift:
        untrained_model.enc_image = RGBFieldLiftedEncoder(channels=args.channels).to(device)
        untrained_model.enc_video = RGBFieldLiftedEncoder(channels=args.channels).to(device)
        untrained_model.dec_image = RGBDecoder(channels=args.channels).to(device)
        untrained_model.dec_video = RGBDecoder(channels=args.channels).to(device)
        import types
        def _encode_image_rgb_u(self, image):
            u_hw = self.enc_image(image)
            return u_hw.unsqueeze(0).expand(LATENT_T, *u_hw.shape)
        def _encode_video_rgb_u(self, video):
            return self.enc_video(video)
        untrained_model._encode_image = types.MethodType(_encode_image_rgb_u, untrained_model)
        untrained_model._encode_video = types.MethodType(_encode_video_rgb_u, untrained_model)
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

    # ----- Stage 2: per-class grid + condition vs random comparison ----- #
    cond_metrics = {}
    if getattr(trained_model, "conditioner", None) is not None:
        print(f"[stage2] generating per-class grid "
              f"({args.num_classes} classes × {args.samples_per_class} samples)...",
              flush=True)
        per_class_imgs = []
        per_class_metrics = []
        for cls in range(args.num_classes):
            cond_t = torch.full((args.samples_per_class,), cls,
                                device=device, dtype=torch.long)
            samples = sample_decoded_images(
                trained_model, schedule, stats,
                args.samples_per_class, args.seed + 1000 + cls, device,
                cond=cond_t,
            )
            per_class_imgs.extend(samples)
            per_class_metrics.append(aggregate_metrics(samples))
        save_grid(per_class_imgs, out / "per_class_grid.png",
                  ncols=args.samples_per_class)
        # Condition vs random condition: same noise seed, different cond.
        # If condition is honored, distances should be different.
        N = args.n_samples
        same_cls_imgs = sample_decoded_images(
            trained_model, schedule, stats, N, args.seed + 9999, device,
            cond=torch.zeros(N, device=device, dtype=torch.long),  # all class 0
        )
        rand_cls = torch.randint(0, args.num_classes, (N,),
                                  device=device, dtype=torch.long)
        rand_cls_imgs = sample_decoded_images(
            trained_model, schedule, stats, N, args.seed + 9999, device,
            cond=rand_cls,
        )
        save_grid(same_cls_imgs, out / "cond_class0.png", ncols=4)
        save_grid(rand_cls_imgs, out / "cond_random.png", ncols=4)
        same_metrics = aggregate_metrics(same_cls_imgs)
        rand_metrics = aggregate_metrics(rand_cls_imgs)
        # Effect size: pixel-wise MSE between same-cond and random-cond
        # samples (same noise, different cond) — should be > 0 if cond
        # actually changes output.
        a = np.stack(same_cls_imgs)
        b = np.stack(rand_cls_imgs)
        cond_effect_mse = float(np.mean((a - b) ** 2))
        cond_metrics = {
            "per_class_metrics": per_class_metrics,
            "same_class_descriptor_distance": descriptor_distance(same_metrics, train_metrics),
            "random_class_descriptor_distance": descriptor_distance(rand_metrics, train_metrics),
            "cond_vs_random_pixel_mse": cond_effect_mse,
            "stage2_class_list": list(range(args.num_classes)),
        }
        print(f"[stage2] cond_effect MSE (same-cond vs random-cond, same noise)"
              f" = {cond_effect_mse:.4f}", flush=True)
        print(f"[stage2]   {'PASS' if cond_effect_mse > 0.01 else 'FAIL'}: condition has measurable effect", flush=True)

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
        "version": "rgb64-cond: Stage-2 class conditioning on CIFAR-10 RGB 64×64",
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
        "stage2_cond": cond_metrics,
    }
    Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[output] JSON -> {args.report_out}", flush=True)

    md = [
        "# VOD RGB64 + Class Conditioning Smoke (Stage 2)\n\n",
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
