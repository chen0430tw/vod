"""AIMP / TPSR minimal physical-consistency tools.

This is the first executable slice of AI Manga Physics in the prototype.
It implements the TPSR eye-highlight invariant and a small AIMP card
layer that can be used by tests and future generators.

It does not attempt to solve full scene understanding. It gives the
project concrete objects and metrics:

    FieldCard       scene-level field assumptions
    PerspectiveCard camera scale / distance assumptions
    LightingCard    light geometry assumptions
    TPSRMeasurement measured triangular highlight statistics

and the TPSR invariants:

    K    = H / (L_l^2 A^(γ/2))
    Uij  = H_i A_j^(γ/2) L_j^2 / (H_j A_i^(γ/2) L_i^2)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


EPS: float = 1e-12


@dataclass(frozen=True)
class FieldCard:
    main_light_angle: float = 0.0
    ambient_contrast: float = 1.0
    tone_density: float = 0.5


@dataclass(frozen=True)
class PerspectiveCard:
    camera_distance: float = 1.0
    focal_scale: float = 1.0


@dataclass(frozen=True)
class LightingCard:
    light_distance: float = 1.0
    gamma: float = 4.0
    coaxial: bool = True

    @property
    def light_diopter(self) -> float:
        return 1.0 / max(float(self.light_distance), EPS)


@dataclass(frozen=True)
class TPSRMeasurement:
    highlight_energy: float
    highlight_area: float
    light_diopter: float = 1.0
    gamma: float = 4.0

    def k(self) -> float:
        return tpsr_k(
            self.highlight_energy,
            self.highlight_area,
            light_diopter=self.light_diopter,
            gamma=self.gamma,
        )


def tpsr_k(
    highlight_energy: float,
    highlight_area: float,
    *,
    light_diopter: float = 1.0,
    gamma: float = 4.0,
) -> float:
    """Single-frame TPSR invariant."""

    H = max(float(highlight_energy), EPS)
    A = max(float(highlight_area), EPS)
    L = max(float(light_diopter), EPS)
    return float(H / ((L ** 2) * (A ** (gamma / 2.0))))


def tpsr_pair_ratio(a: TPSRMeasurement, b: TPSRMeasurement) -> float:
    """U_ij paired TPSR consistency ratio; ideal value is 1."""

    Hi = max(float(a.highlight_energy), EPS)
    Hj = max(float(b.highlight_energy), EPS)
    Ai = max(float(a.highlight_area), EPS)
    Aj = max(float(b.highlight_area), EPS)
    Li = max(float(a.light_diopter), EPS)
    Lj = max(float(b.light_diopter), EPS)
    gamma = 0.5 * (float(a.gamma) + float(b.gamma))
    return float((Hi * (Aj ** (gamma / 2.0)) * (Lj ** 2)) / (Hj * (Ai ** (gamma / 2.0)) * (Li ** 2)))


def tpsr_pairwise_log_deviation(measurements: list[TPSRMeasurement]) -> float:
    """Median |ln U_ij| across all pairs."""

    if len(measurements) < 2:
        return float("nan")
    vals = []
    for i in range(len(measurements)):
        for j in range(i + 1, len(measurements)):
            vals.append(abs(np.log(max(tpsr_pair_ratio(measurements[i], measurements[j]), EPS))))
    return float(np.median(vals))


def tpsr_consistency_score(
    measurements: list[TPSRMeasurement],
    *,
    sigma: float = 0.2,
) -> float:
    """Score in (0, 1]; 1 means pairwise TPSR consistency."""

    if sigma <= 0:
        raise ValueError("sigma must be positive")
    dev = tpsr_pairwise_log_deviation(measurements)
    if not (dev == dev):
        return float("nan")
    return float(np.exp(-dev / sigma))


def synthesize_tpsr_measurements(
    distances: np.ndarray,
    *,
    base_area: float = 64.0,
    base_energy: float = 1.0,
    light_distance: float = 1.0,
    gamma: float = 4.0,
    brightness_error: float = 1.0,
) -> list[TPSRMeasurement]:
    """Generate a TPSR-consistent camera-distance sequence.

    `brightness_error` multiplies the final frame energy. Values other
    than 1.0 simulate an inconsistent AI highlight.
    """

    d = np.asarray(distances, dtype=np.float64)
    if d.ndim != 1 or d.size == 0:
        raise ValueError("distances must be a non-empty 1-D array")
    if np.any(d <= 0):
        raise ValueError("distances must be positive")
    light_diopter = 1.0 / max(float(light_distance), EPS)
    out: list[TPSRMeasurement] = []
    for idx, dist in enumerate(d):
        area = base_area / (dist ** 2)
        energy = base_energy / (dist ** gamma)
        if idx == d.size - 1:
            energy *= brightness_error
        out.append(
            TPSRMeasurement(
                highlight_energy=float(energy),
                highlight_area=float(area),
                light_diopter=light_diopter,
                gamma=gamma,
            )
        )
    return out


def aimp_tpsr_metrics(
    measurements: list[TPSRMeasurement],
    *,
    sigma: float = 0.2,
) -> dict[str, float]:
    """Aggregate TPSR/AIMP consistency metrics."""

    if not measurements:
        nan = float("nan")
        return {
            "tpsr_k_mean": nan,
            "tpsr_k_cv": nan,
            "tpsr_pair_logdev": nan,
            "tpsr_consistency_score": nan,
        }
    ks = np.array([m.k() for m in measurements], dtype=np.float64)
    k_mean = float(np.mean(ks))
    k_cv = float(np.std(ks, ddof=0) / (abs(k_mean) + EPS))
    logdev = tpsr_pairwise_log_deviation(measurements)
    score = tpsr_consistency_score(measurements, sigma=sigma)
    return {
        "tpsr_k_mean": k_mean,
        "tpsr_k_cv": k_cv,
        "tpsr_pair_logdev": logdev,
        "tpsr_consistency_score": score,
    }


def tpsr_video_consistency_loss(
    video: torch.Tensor,
    *,
    gamma: float = 4.0,
    sigma: float = 0.2,
    top_k_fraction: float = 0.1,
    eps: float = 1e-9,
) -> torch.Tensor:
    """Differentiable TPSR-style consistency loss across video frames.

    This is a synthetic-highlight surrogate: instead of detecting real
    triangular eye-highlights, we treat the brightest `top_k_fraction`
    pixels of each frame as a synthetic highlight region. Per-frame:

        top_k = max(8, int(top_k_fraction * H * W))
        energy_t = mean(topk(frame.flatten(), top_k)) * top_k
        area_t   = top_k                                  (constant)
        K_t      = energy_t / (area_t ** (gamma / 2.0))    (light_diopter = 1)

    The loss is the squared coefficient of variation of K, i.e.
    `var(K) / (mean(K) ** 2 + eps)`. This is zero for TPSR-consistent
    sequences (constant K) and positive for anything else, and it is
    differentiable through `topk`.

    Accepts:
      * `(F, H, W)`     monochrome video
      * `(C, F, H, W)`  multi-channel video — channels are reduced by
                        mean before highlight extraction so we get one
                        K per frame.

    Returns a 0-d tensor. If the shape doesn't fit (`F < 2` or fewer
    than 2 spatial dims), returns a zero tensor with same dtype/device.

    Design choice (documented per Task B spec): we use the squared CV of
    K as the consistency proxy rather than the more general
    log-pair-deviation form because it is symmetric, scale-invariant,
    differentiable everywhere `mean(K) > 0`, and zero exactly when K is
    constant — which is the TPSR consistency condition. `sigma` is kept
    as an unused-by-default parameter for API parity with
    `tpsr_consistency_score`; it is not consumed by the squared-CV form.
    """

    if not isinstance(video, torch.Tensor):
        raise TypeError(f"video must be a torch.Tensor, got {type(video).__name__}")
    if video.ndim == 4:
        # (C, F, H, W) -> (F, H, W) by channel mean.
        frames = video.mean(dim=0)
    elif video.ndim == 3:
        frames = video
    else:
        return video.new_zeros(())

    F_, H, W = frames.shape
    if F_ < 2 or H < 1 or W < 1:
        return video.new_zeros(())

    spatial = H * W
    # Same top_k for every frame keeps the area constant.
    top_k = max(8, int(top_k_fraction * spatial))
    if top_k > spatial:
        top_k = spatial

    flat = frames.reshape(F_, spatial)
    top_vals, _ = torch.topk(flat, top_k, dim=-1, sorted=False)
    energy = top_vals.mean(dim=-1) * top_k                          # (F,)
    area = float(top_k)
    # light_diopter = 1.0 (synthetic) -> K = energy / area ** (gamma/2)
    K = energy / (area ** (gamma / 2.0))                            # (F,)

    K_mean = K.mean()
    K_var = K.var(unbiased=False)
    return K_var / (K_mean * K_mean + eps)


__all__ = [
    "FieldCard",
    "LightingCard",
    "PerspectiveCard",
    "TPSRMeasurement",
    "aimp_tpsr_metrics",
    "synthesize_tpsr_measurements",
    "tpsr_consistency_score",
    "tpsr_k",
    "tpsr_pair_ratio",
    "tpsr_pairwise_log_deviation",
    "tpsr_video_consistency_loss",
]
