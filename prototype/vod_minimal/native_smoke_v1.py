"""FROZEN SNAPSHOT: native_vod_smoke_v1 — DO NOT MODIFY.

This file is a frozen copy of `vod_minimal/native.py` taken on
2026-04-27, immediately after the target-leak removal. It is kept as
a reference snapshot of the smallest model that can still produce a
visible (if blurry, baseline-losing) image round-trip on real photos.

Use cases:
  * Reproducing the cat / ant / dice round-trip outputs in
    `C:\\Users\\asus\\vod_real_imgs\\inspect_*.png`.
  * Comparing a future architecture against this exact baseline so the
    "did the new design help" question has an unambiguous reference.

If `vod_minimal/native.py` evolves and breaks compatibility with the
frozen `inspect_native_smoke_v1.py`, that's fine — the frozen pair is
self-contained and imports from this module name only.

Original docstring follows:
---------------------------
Native unified VOD — SMOKE PROTOTYPE (not v0.3).

STATUS: native_vod_smoke
This module is a code-shape smoke prototype. Shape contracts, gradient
flow and module wiring work, but the model has NOT been shown to learn
anything that beats a trivial baseline. Do not call this v0.3 and do
not present its numbers as evidence that the VOD mechanism works.

CHANGES vs the previous broken iteration
----------------------------------------
The previous iteration leaked the answer: `forward(noisy, target)`
encoded the target views and fed that encoding to the denoiser as a
"condition". That made the denoiser able to trivially copy the target
through the condition pathway. Every loss number, every stress
comparison, every "improvement" reported on top of that was invalid.

This iteration removes the leak hard:

    forward(noisy_views) → predicted_views, U_pred
    (no target argument anywhere on the model side)

Targets only enter `native_total_loss(model, noisy, target, ...)`,
where they are compared against the model output and never re-encoded.
The denoiser sees [u_noisy, pos_t, pos_y, pos_x] only.

Audio and text are now experimental and OFF by default. They remain in
the codebase as 1×1 linear reshape adapters, which is not a real
multi-medium codec — keeping them on by default would dilute every
image/video number with reshape garbage. Re-enable explicitly via
`NativeVODConfig(enable_audio=True, enable_text=True)` if you need to
exercise the wiring; do not interpret their numbers as media quality.

Latent geometry (toy)
---------------------
    U.shape = (T, H, W, C)
        T = 8     time slices
        H = 16    spatial height
        W = 16    spatial width
        C = 4     representation channels

External media views must have shapes that map into the latent grid:

    image  (H, W)
    video  (T, H, W)
    audio  (T*H*W,)   = 2048 for the default toy
    text   (TEXT_LEN,) = 32

`build_projection_batch(..., spacetime=True, size=16, frames=8)` already
produces those shapes, so the data path lines up with no extra glue.

Forward pipeline
----------------
    encode(noisy_views)  → U_noisy
    encode(target_views) → U_target  (used as supervised condition)
    denoise_path(U_noisy, U_target, steps=K) → U_pred
    decode(U_pred)       → predicted_views

Five losses are aggregated by `native_total_loss`:

    L_field      MSE between U_pred and U_target      (latent reconstruction)
    L_media      MSE on image / video / audio decoded views
    L_temporal   one-sided ReLU on temporal smoothness inflation
                 (relu(smooth(pred_video) − smooth(target_video)))
    L_artifact   one-sided ReLU on tile-residue inflation, image+video
    L_text       MSE on the toy text channel (placeholder for the future
                 binary-twin CE-style consistency loss; see docs)

Each loss has its own weight; default weights produce a balanced toy
training run, but every component can be set to 0 to ablate it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from .torch_artifacts import (
    RESIDUE_FLOOR,
    artifact_regularization_loss,
)


LATENT_T: int = 8
LATENT_HW: int = 16
TEXT_LEN: int = 32
AUDIO_SIZE: int = LATENT_T * LATENT_HW * LATENT_HW  # 2048


# --------------------------------------------------------------------------- #
#  Shapes / reshape helpers
# --------------------------------------------------------------------------- #

def audio_to_grid(audio: torch.Tensor) -> torch.Tensor:
    """(..., AUDIO_SIZE) → (..., T, H, W)."""
    return audio.reshape(*audio.shape[:-1], LATENT_T, LATENT_HW, LATENT_HW)


def text_to_grid(text: torch.Tensor) -> torch.Tensor:
    """(..., TEXT_LEN) → (..., H, W). Repeats the channel string by 8 to
    cover the H*W grid (TEXT_LEN * 8 = LATENT_HW * LATENT_HW)."""
    repeats = (LATENT_HW * LATENT_HW) // TEXT_LEN  # 256/32 = 8
    expanded = text.unsqueeze(-1).expand(*text.shape, repeats)
    return expanded.reshape(*text.shape[:-1], LATENT_HW, LATENT_HW)


def grid_to_audio(grid: torch.Tensor) -> torch.Tensor:
    """(..., T, H, W) → (..., AUDIO_SIZE)."""
    return grid.reshape(*grid.shape[:-3], AUDIO_SIZE)


def grid_to_text(grid: torch.Tensor) -> torch.Tensor:
    """(..., H, W) → (..., TEXT_LEN). Average-pools each block of 8."""
    pool = (LATENT_HW * LATENT_HW) // TEXT_LEN  # 8
    flat = grid.reshape(*grid.shape[:-2], LATENT_HW * LATENT_HW)
    return flat.reshape(*flat.shape[:-1], TEXT_LEN, pool).mean(dim=-1)


# --------------------------------------------------------------------------- #
#  Native model
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class NativeVODConfig:
    channels: int = 4
    hidden: int = 32
    denoise_steps: int = 4
    enable_audio: bool = False  # experimental: 1×1 reshape, not a real codec
    enable_text: bool = False   # experimental: float-MSE proxy, not discrete CE


class NativeVOD(nn.Module):
    """Native unified VOD generator.

    A single nn.Module that owns:
      * four thin per-medium encoders (linear pixel-wise channel project)
      * one shared field denoiser (pointwise MLP on (T,H,W,C))
      * four thin per-medium decoders (inverse linear projection)

    Each forward pass touches every medium present in the input dict.
    There is no per-medium control flow above the encoder line — once
    the views are in the field they are governed by the same dynamics.
    """

    def __init__(self, config: NativeVODConfig = NativeVODConfig()):
        super().__init__()
        self.config = config
        c = config.channels

        # Per-pixel channel projection ≡ 1×1 conv via Linear over the
        # last axis. The encoders accept any view shape that ends in 1
        # after `unsqueeze(-1)`; the broadcast to the (T,H,W) grid is
        # handled in `encode`.
        self.enc_image = nn.Linear(1, c)
        self.enc_video = nn.Linear(1, c)
        self.enc_audio = nn.Linear(1, c)
        self.enc_text = nn.Linear(1, c)

        # Pointwise field denoiser. Features per latent voxel:
        #   [u_noisy (C), pos_t, pos_y, pos_x] = C + 3
        # NO target / condition / answer-side input. The previous
        # iteration concatenated `condition` here; that was the leak.
        feat_in = c + 3
        self.denoiser = nn.Sequential(
            nn.Linear(feat_in, config.hidden),
            nn.SiLU(),
            nn.Linear(config.hidden, config.hidden),
            nn.SiLU(),
            nn.Linear(config.hidden, c),
        )
        self.step_logit = nn.Parameter(torch.tensor(0.0))

        # Per-pixel inverse projection. Output gets squeezed back to the
        # native medium shape inside `decode`.
        self.dec_image = nn.Linear(c, 1)
        self.dec_video = nn.Linear(c, 1)
        self.dec_audio = nn.Linear(c, 1)
        self.dec_text = nn.Linear(c, 1)

    # ---- encode ---------------------------------------------------- #

    def _encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """(H, W) → (T, H, W, C) via channel project + time broadcast."""
        u_hw = self.enc_image(image.unsqueeze(-1))            # (H, W, C)
        return u_hw.unsqueeze(0).expand(LATENT_T, *u_hw.shape)

    def _encode_video(self, video: torch.Tensor) -> torch.Tensor:
        return self.enc_video(video.unsqueeze(-1))            # (T, H, W, C)

    def _encode_audio(self, audio: torch.Tensor) -> torch.Tensor:
        grid = audio_to_grid(audio).unsqueeze(-1)             # (T, H, W, 1)
        return self.enc_audio(grid)                           # (T, H, W, C)

    def _encode_text(self, text: torch.Tensor) -> torch.Tensor:
        grid_hw = text_to_grid(text).unsqueeze(-1)            # (H, W, 1)
        u_hw = self.enc_text(grid_hw)                         # (H, W, C)
        return u_hw.unsqueeze(0).expand(LATENT_T, *u_hw.shape)

    def active_media(self) -> tuple[str, ...]:
        """Media this configuration actually consumes / emits."""
        out = ["image", "video"]
        if self.config.enable_audio:
            out.append("audio")
        if self.config.enable_text:
            out.append("text")
        return tuple(out)

    def encode(self, views: dict[str, torch.Tensor]) -> torch.Tensor:
        """Fuse media views into a single latent U(T, H, W, C).

        The fusion is the mean of the per-medium encodings. Missing
        media are skipped silently. Audio / text are silently ignored
        unless their config flag is on, regardless of whether they are
        present in `views` — this keeps callers from accidentally
        re-introducing the experimental media into evaluation paths.
        """
        active = set(self.active_media())
        contributions: list[torch.Tensor] = []
        if "image" in views and "image" in active:
            contributions.append(self._encode_image(views["image"]))
        if "video" in views and "video" in active:
            contributions.append(self._encode_video(views["video"]))
        if "audio" in views and "audio" in active:
            contributions.append(self._encode_audio(views["audio"]))
        if "text" in views and "text" in active:
            contributions.append(self._encode_text(views["text"]))
        if not contributions:
            raise ValueError(
                "encode requires at least one active media view; "
                f"active={sorted(active)}, present={sorted(views)}"
            )
        return torch.stack(contributions, dim=0).mean(dim=0)

    # ---- denoise --------------------------------------------------- #

    def denoise(self, u_noisy: torch.Tensor) -> torch.Tensor:
        """Single shared update step on U.

        Features per voxel: [u_noisy, pos_t, pos_y, pos_x] (no target,
        no condition). Applies a per-voxel delta scaled by a learnable
        sigmoid step size.
        """
        T, H, W, _ = u_noisy.shape
        device = u_noisy.device
        dtype = u_noisy.dtype

        pos_t = torch.linspace(0.0, 1.0, T, device=device, dtype=dtype).view(T, 1, 1, 1).expand(T, H, W, 1)
        pos_y = torch.linspace(0.0, 1.0, H, device=device, dtype=dtype).view(1, H, 1, 1).expand(T, H, W, 1)
        pos_x = torch.linspace(0.0, 1.0, W, device=device, dtype=dtype).view(1, 1, W, 1).expand(T, H, W, 1)

        feats = torch.cat([u_noisy, pos_t, pos_y, pos_x], dim=-1)
        delta = self.denoiser(feats)
        step = torch.sigmoid(self.step_logit)
        return u_noisy + step * delta

    def denoise_path(self, u_noisy: torch.Tensor, *, steps: int | None = None) -> torch.Tensor:
        if steps is None:
            steps = self.config.denoise_steps
        u = u_noisy
        for _ in range(steps):
            u = self.denoise(u)
        return u

    # ---- decode ---------------------------------------------------- #

    def decode(self, U: torch.Tensor) -> dict[str, torch.Tensor]:
        """Project U back into the active media views."""
        active = set(self.active_media())
        out: dict[str, torch.Tensor] = {}

        if "image" in active:
            out["image"] = self.dec_image(U.mean(dim=0)).squeeze(-1)
        if "video" in active:
            out["video"] = self.dec_video(U).squeeze(-1)            # (T, H, W)
        if "audio" in active:
            out["audio"] = grid_to_audio(self.dec_audio(U).squeeze(-1))
        if "text" in active:
            text_hw = self.dec_text(U.mean(dim=0)).squeeze(-1)
            out["text"] = grid_to_text(text_hw)
        return out

    # ---- one-shot forward ----------------------------------------- #

    def forward(
        self,
        noisy_views: dict[str, torch.Tensor],
        *,
        steps: int | None = None,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Return predicted_views, U_pred.

        The model takes ONLY the noisy views. Targets are not visible
        anywhere on the model side; they only appear inside the loss
        function for comparison.
        """
        u_noisy = self.encode(noisy_views)
        u_pred = self.denoise_path(u_noisy, steps=steps)
        predicted = self.decode(u_pred)
        return predicted, u_pred


# --------------------------------------------------------------------------- #
#  Loss components
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class NativeLossWeights:
    field: float = 0.5
    media: float = 1.0
    temporal: float = 0.1
    artifact: float = 0.1
    text: float = 0.3

    def asdict(self) -> dict[str, float]:
        return {
            "field": self.field,
            "media": self.media,
            "temporal": self.temporal,
            "artifact": self.artifact,
            "text": self.text,
        }


def _temporal_smoothness_torch(video: torch.Tensor) -> torch.Tensor:
    """Differentiable mean abs frame-to-frame difference. 0-d tensor."""
    if video.ndim < 3 or video.shape[-3] < 2:
        return video.new_zeros(())
    return (video[..., 1:, :, :] - video[..., :-1, :, :]).abs().mean()


def native_total_loss(
    model: NativeVOD,
    noisy_views: dict[str, torch.Tensor],
    target_views: dict[str, torch.Tensor],
    *,
    weights: NativeLossWeights = NativeLossWeights(),
    steps: int | None = None,
    artifact_tile: int = 4,
) -> tuple[torch.Tensor, dict[str, float]]:
    """One-shot loss for a single sample.

    Targets are only used HERE — never passed to the model. The model
    sees `noisy_views` only and produces `pred` / `u_pred`; this
    function compares them against `target_views` to build the loss.

    Targets are detached at the loss-function boundary: every
    `target_views[k]` and the re-encoded `u_target_ref` are detached
    before they enter any comparison, so no gradient can flow back
    through the data side. The encoders are still trained through
    `pred = decode(u_pred)` and `L_media` (which only requires grad on
    the prediction side).
    """
    pred, u_pred = model.forward(noisy_views, steps=steps)

    # Detach every target view at the loss boundary so the data side
    # cannot receive gradient through any comparison term.
    target_views = {k: v.detach() for k, v in target_views.items()}

    with torch.no_grad():
        u_target_ref = model.encode(target_views)
    L_field = F.mse_loss(u_pred, u_target_ref)

    active = set(model.active_media())
    media_keys = [k for k in ("image", "video", "audio") if k in target_views and k in active]
    if media_keys:
        L_media = torch.stack([F.mse_loss(pred[k], target_views[k]) for k in media_keys]).mean()
    else:
        L_media = u_pred.new_zeros(())

    if "video" in target_views and "video" in active:
        smooth_pred = _temporal_smoothness_torch(pred["video"])
        smooth_target = _temporal_smoothness_torch(target_views["video"]).detach()
        L_temporal = torch.relu(smooth_pred - smooth_target)
        L_artifact = artifact_regularization_loss(pred["video"], target_views["video"], tile=artifact_tile)
    else:
        L_temporal = u_pred.new_zeros(())
        L_artifact = u_pred.new_zeros(())

    if "text" in target_views and "text" in active:
        L_text = F.mse_loss(pred["text"], target_views["text"])
    else:
        L_text = u_pred.new_zeros(())

    total = (
        weights.field * L_field
        + weights.media * L_media
        + weights.temporal * L_temporal
        + weights.artifact * L_artifact
        + weights.text * L_text
    )
    components = {
        "L_field": float(L_field.detach()),
        "L_media": float(L_media.detach()),
        "L_temporal": float(L_temporal.detach()),
        "L_artifact": float(L_artifact.detach()),
        "L_text": float(L_text.detach()),
        "L_total": float(total.detach()),
    }
    return total, components


# --------------------------------------------------------------------------- #
#  Numpy I/O helpers
# --------------------------------------------------------------------------- #

def views_to_torch(
    views: dict[str, "object"],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    import numpy as np

    return {
        k: torch.from_numpy(np.asarray(v, dtype=np.float32)).to(device)
        for k, v in views.items()
    }


def views_to_numpy(views: dict[str, torch.Tensor]) -> dict[str, "object"]:
    return {k: v.detach().cpu().numpy() for k, v in views.items()}


__all__ = [
    "AUDIO_SIZE",
    "LATENT_HW",
    "LATENT_T",
    "NativeLossWeights",
    "NativeVOD",
    "NativeVODConfig",
    "TEXT_LEN",
    "audio_to_grid",
    "grid_to_audio",
    "grid_to_text",
    "native_total_loss",
    "text_to_grid",
    "views_to_numpy",
    "views_to_torch",
]
