"""DDPM noise schedule + DDIM sampler for VOD's UNet denoiser.

Minimal viable diffusion training mode. Operates on the same Chladni
latent U(T, H, W, C) that NativeVOD encodes/decodes — only the
training objective and the inference loop change.

Training objective (x_0 prediction):
    sample t ∈ [0, T_diff)
    x_t = sqrt(α_bar_t) · x_0 + sqrt(1 - α_bar_t) · ε
    L_diffusion = MSE(model.denoise(x_t, t=t), x_0)

Sampling (DDIM, deterministic, η=0):
    x_T ~ N(0, I)
    for t = T-1 ... 0:
        x_0_pred = model.denoise(x_t, t=t)
        ε_pred   = (x_t - sqrt(α_bar_t) · x_0_pred) / sqrt(1 - α_bar_t)
        x_(t-1)  = sqrt(α_bar_(t-1)) · x_0_pred + sqrt(1 - α_bar_(t-1)) · ε_pred
"""

from __future__ import annotations
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class NoiseSchedule:
    """Linear β schedule with precomputed α and α_bar."""
    num_steps: int
    betas: torch.Tensor          # (T_diff,)
    alphas: torch.Tensor         # (T_diff,)
    alphas_cumprod: torch.Tensor # (T_diff,)

    def to(self, device) -> "NoiseSchedule":
        return NoiseSchedule(
            num_steps=self.num_steps,
            betas=self.betas.to(device),
            alphas=self.alphas.to(device),
            alphas_cumprod=self.alphas_cumprod.to(device),
        )


def make_schedule(num_steps: int = 200, beta_start: float = 1e-4,
                  beta_end: float = 2e-2) -> NoiseSchedule:
    betas = torch.linspace(beta_start, beta_end, num_steps, dtype=torch.float64)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return NoiseSchedule(
        num_steps=num_steps,
        betas=betas.to(torch.float32),
        alphas=alphas.to(torch.float32),
        alphas_cumprod=alphas_cumprod.to(torch.float32),
    )


def sinusoidal_time_embedding(
    t: torch.Tensor, dim: int, *, dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Standard sinusoidal positional encoding for timesteps.

    t: (B,) integer tensor
    dtype: target compute dtype. If None, defaults to torch.float32 to
        preserve legacy behaviour. Pass `feats.dtype` from the caller
        when running under bf16/fp16/fp8 to avoid downstream cast.
    returns: (B, dim) in the requested dtype
    """
    if dim % 2 != 0:
        raise ValueError(f"dim must be even, got {dim}")
    if dtype is None:
        dtype = torch.float32
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device, dtype=dtype) / max(half - 1, 1)
    )
    args = t.to(dtype)[:, None] * freqs[None, :]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


def _broadcast_alpha(alpha: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    """Broadcast alpha (B,) or scalar to shape compatible with x_shape."""
    if alpha.ndim == 0:
        return alpha.view(1).expand(x_shape[0]).view(-1, *([1] * (len(x_shape) - 1)))
    return alpha.view(-1, *([1] * (len(x_shape) - 1)))


def q_sample(x_0: torch.Tensor, t: torch.Tensor, schedule: NoiseSchedule,
             noise: torch.Tensor | None = None) -> torch.Tensor:
    """Forward process: x_t = sqrt(α_bar_t) x_0 + sqrt(1-α_bar_t) ε."""
    if noise is None:
        noise = torch.randn_like(x_0)
    a = schedule.alphas_cumprod[t]
    a = _broadcast_alpha(a, x_0.shape)
    return torch.sqrt(a) * x_0 + torch.sqrt(1.0 - a) * noise


def diffusion_loss(model, x_0: torch.Tensor, schedule: NoiseSchedule,
                   *, prediction: str = "x_0") -> torch.Tensor:
    """Random-t MSE on x_0 prediction (default) or ε prediction."""
    B = x_0.shape[0]
    t = torch.randint(0, schedule.num_steps, (B,), device=x_0.device)
    noise = torch.randn_like(x_0)
    x_t = q_sample(x_0, t, schedule, noise=noise)
    pred = model.denoise(x_t, t=t)
    if prediction == "x_0":
        return F.mse_loss(pred, x_0)
    elif prediction == "epsilon":
        return F.mse_loss(pred, noise)
    raise ValueError(f"unknown prediction={prediction!r}")


@torch.no_grad()
def ddim_sample(model, shape: tuple, schedule: NoiseSchedule, *,
                num_steps: int = 50, device=None,
                generator: torch.Generator | None = None,
                prediction: str = "x_0",
                dtype: torch.dtype | None = None) -> torch.Tensor:
    """Deterministic DDIM (η=0) reverse process, returns final x_0.

    `prediction` must match the training objective.
    `dtype` controls the latent compute dtype. If None, inferred from
    the model's first parameter (so bf16/fp16 models sample in their
    native dtype without explicit casts).
    """
    if device is None:
        device = next(model.parameters()).device
    if dtype is None:
        dtype = next(model.parameters()).dtype
    x = torch.randn(shape, device=device, dtype=dtype, generator=generator)
    timesteps = torch.linspace(
        schedule.num_steps - 1, 0, num_steps, dtype=torch.long, device=device,
    )
    for i in range(num_steps):
        t = timesteps[i]
        t_batch = t.expand(shape[0])
        pred = model.denoise(x, t=t_batch)
        if prediction == "x_0":
            x_0_pred = pred
            a_t = schedule.alphas_cumprod[t]
            eps_pred = (x - torch.sqrt(a_t) * x_0_pred) / torch.sqrt(torch.clamp(1 - a_t, min=1e-9))
        else:  # epsilon
            eps_pred = pred
            a_t = schedule.alphas_cumprod[t]
            x_0_pred = (x - torch.sqrt(1 - a_t) * eps_pred) / torch.sqrt(torch.clamp(a_t, min=1e-9))
        if i < num_steps - 1:
            t_next = timesteps[i + 1]
            a_next = schedule.alphas_cumprod[t_next]
            x = torch.sqrt(a_next) * x_0_pred + torch.sqrt(torch.clamp(1 - a_next, min=0)) * eps_pred
        else:
            x = x_0_pred
    return x


__all__ = [
    "NoiseSchedule",
    "make_schedule",
    "sinusoidal_time_embedding",
    "q_sample",
    "diffusion_loss",
    "ddim_sample",
]
