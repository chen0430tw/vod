"""Binary-Twin minimal implementation for VOD text/logo channels.

This module is intentionally small, but it is no longer a placeholder.
It implements the minimal continuous/discrete coupling needed by the
prototype:

    continuous text channel x ∈ [0, 1]^N
    discrete symbol field B ∈ {0, ..., levels-1}^N
    Φ(x) = round((levels-1) x)
    Ψ(B) = B / (levels-1)

The key point is not that this solves OCR. It gives VOD a real
Binary-Twin object and a real consistency loss/metric so text/logos are
not treated as just another smooth image signal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


DEFAULT_LEVELS: int = 16
EPS: float = 1e-9


@dataclass(frozen=True)
class BinaryTwinState:
    """Coupled continuous/discrete representation."""

    continuous: np.ndarray
    symbols: np.ndarray
    reconstructed: np.ndarray


def encode_symbols(values: np.ndarray, *, levels: int = DEFAULT_LEVELS) -> np.ndarray:
    """Φ: continuous [0, 1] values -> integer symbols."""

    if levels < 2:
        raise ValueError("levels must be >= 2")
    arr = np.asarray(values, dtype=np.float64)
    clipped = np.clip(arr, 0.0, 1.0)
    return np.rint(clipped * (levels - 1)).astype(np.int64)


def decode_symbols(symbols: np.ndarray, *, levels: int = DEFAULT_LEVELS) -> np.ndarray:
    """Ψ: integer symbols -> continuous [0, 1] reconstruction."""

    if levels < 2:
        raise ValueError("levels must be >= 2")
    sym = np.asarray(symbols, dtype=np.int64)
    clipped = np.clip(sym, 0, levels - 1)
    return clipped.astype(np.float64) / float(levels - 1)


def binary_twin_state(values: np.ndarray, *, levels: int = DEFAULT_LEVELS) -> BinaryTwinState:
    """Build the coupled state (x, B, Ψ(B))."""

    continuous = np.asarray(values, dtype=np.float64)
    symbols = encode_symbols(continuous, levels=levels)
    reconstructed = decode_symbols(symbols, levels=levels)
    return BinaryTwinState(continuous=continuous, symbols=symbols, reconstructed=reconstructed)


def symbol_accuracy(pred: np.ndarray, target: np.ndarray, *, levels: int = DEFAULT_LEVELS) -> float:
    """Exact symbol accuracy after Φ quantization."""

    p = encode_symbols(pred, levels=levels)
    t = encode_symbols(target, levels=levels)
    if p.shape != t.shape:
        raise ValueError(f"shape mismatch: pred={p.shape}, target={t.shape}")
    if p.size == 0:
        return float("nan")
    return float(np.mean(p == t))


def symbol_hamming(pred: np.ndarray, target: np.ndarray, *, levels: int = DEFAULT_LEVELS) -> float:
    """Normalized symbol mismatch rate."""

    acc = symbol_accuracy(pred, target, levels=levels)
    return float("nan") if not (acc == acc) else float(1.0 - acc)


def reconstruction_error(values: np.ndarray, *, levels: int = DEFAULT_LEVELS) -> float:
    """Mean squared error between x and Ψ(Φ(x))."""

    st = binary_twin_state(values, levels=levels)
    if st.continuous.size == 0:
        return float("nan")
    return float(np.mean((np.clip(st.continuous, 0.0, 1.0) - st.reconstructed) ** 2))


def binary_twin_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    *,
    levels: int = DEFAULT_LEVELS,
) -> dict[str, float]:
    """Numpy diagnostics for Binary-Twin text/logo consistency."""

    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    if p.shape != t.shape:
        raise ValueError(f"shape mismatch: pred={p.shape}, target={t.shape}")
    if p.size == 0:
        nan = float("nan")
        return {
            "symbol_accuracy": nan,
            "symbol_hamming": nan,
            "continuous_mse": nan,
            "pred_reconstruction_error": nan,
            "target_reconstruction_error": nan,
        }
    return {
        "symbol_accuracy": symbol_accuracy(p, t, levels=levels),
        "symbol_hamming": symbol_hamming(p, t, levels=levels),
        "continuous_mse": float(np.mean((p - t) ** 2)),
        "pred_reconstruction_error": reconstruction_error(p, levels=levels),
        "target_reconstruction_error": reconstruction_error(t, levels=levels),
    }


def ordinal_logits_from_values(
    values: torch.Tensor,
    *,
    levels: int = DEFAULT_LEVELS,
    temperature: float = 0.05,
) -> torch.Tensor:
    """Differentiable logits over symbols from a continuous prediction.

    The closer `values` is to a quantization level, the higher that
    level's logit. This lets the prototype use a real discrete CE term
    without changing the decoder to emit class logits.
    """

    if levels < 2:
        raise ValueError("levels must be >= 2")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    vals = values.clamp(0.0, 1.0).unsqueeze(-1)
    grid = torch.linspace(0.0, 1.0, levels, device=values.device, dtype=values.dtype)
    return -((vals - grid) ** 2) / temperature


def binary_twin_torch_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    levels: int = DEFAULT_LEVELS,
    ce_weight: float = 1.0,
    recon_weight: float = 0.25,
    temperature: float = 0.05,
) -> torch.Tensor:
    """Differentiable Binary-Twin loss.

    Components:
      * CE between Φ(target) and ordinal logits derived from pred
      * MSE between pred and Ψ(Φ(target))

    This is the minimal discrete/continuous coupling. It is not a full
    OCR system, but it stops the prototype from pretending text is only
    a smooth float channel.
    """

    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: pred={tuple(pred.shape)}, target={tuple(target.shape)}")
    if pred.numel() == 0:
        return pred.new_zeros(())

    target_symbols = torch.round(target.detach().clamp(0.0, 1.0) * (levels - 1)).long()
    logits = ordinal_logits_from_values(pred, levels=levels, temperature=temperature)
    ce = F.cross_entropy(logits.reshape(-1, levels), target_symbols.reshape(-1))
    target_recon = target_symbols.to(dtype=pred.dtype) / float(levels - 1)
    recon = F.mse_loss(pred, target_recon)
    return ce_weight * ce + recon_weight * recon


def binary_twin_pixel_torch_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    levels: int = DEFAULT_LEVELS,
    recon_weight: float = 0.25,
    ce_weight: float = 1.0,
    temperature: float = 0.05,
    value_range: tuple[float, float] = (-1.0, 1.0),
) -> torch.Tensor:
    """Per-pixel Binary-Twin loss for image / video tensors.

    The text-channel `binary_twin_torch_loss` assumes inputs already lie
    in [0, 1]. For pixel media the natural range is `value_range`
    (default [-1, 1]). This wrapper linearly remaps to [0, 1], flattens
    every pixel into one symbol, and applies the same CE + reconstruction
    loss over `levels` quantization buckets.

    Design choice (documented per Task A spec): for image inputs of shape
    `(C, H, W)` or `(H, W)` and video inputs of shape `(F, H, W)` or
    `(C, F, H, W)`, we flatten ALL spatial / temporal / channel axes
    into one 1-D symbol stream before applying ordinal CE. We do not
    treat each channel separately; the goal here is to give every pixel a
    discrete symbol commitment, not to learn a per-channel codebook.

    Empty inputs return a 0-d zero tensor. Target is detached internally,
    matching the convention in `binary_twin_torch_loss`.
    """

    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: pred={tuple(pred.shape)}, target={tuple(target.shape)}")
    if pred.numel() == 0:
        return pred.new_zeros(())

    lo, hi = float(value_range[0]), float(value_range[1])
    if not (hi > lo):
        raise ValueError(f"value_range must satisfy hi > lo, got ({lo}, {hi})")
    span = hi - lo

    # Flatten everything to 1-D and remap to [0, 1].
    pred_flat = (pred.reshape(-1) - lo) / span
    target_flat = (target.detach().reshape(-1) - lo) / span

    return binary_twin_torch_loss(
        pred_flat,
        target_flat,
        levels=levels,
        ce_weight=ce_weight,
        recon_weight=recon_weight,
        temperature=temperature,
    )


def binary_twin_torch_accuracy(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    levels: int = DEFAULT_LEVELS,
) -> torch.Tensor:
    """Torch exact symbol accuracy for tests/training logs."""

    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: pred={tuple(pred.shape)}, target={tuple(target.shape)}")
    if pred.numel() == 0:
        return pred.new_tensor(float("nan"))
    p = torch.round(pred.detach().clamp(0.0, 1.0) * (levels - 1)).long()
    t = torch.round(target.detach().clamp(0.0, 1.0) * (levels - 1)).long()
    return (p == t).to(dtype=pred.dtype).mean()


__all__ = [
    "DEFAULT_LEVELS",
    "BinaryTwinState",
    "binary_twin_metrics",
    "binary_twin_pixel_torch_loss",
    "binary_twin_state",
    "binary_twin_torch_accuracy",
    "binary_twin_torch_loss",
    "decode_symbols",
    "encode_symbols",
    "ordinal_logits_from_values",
    "reconstruction_error",
    "symbol_accuracy",
    "symbol_hamming",
]
