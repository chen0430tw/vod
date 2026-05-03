"""Native unified VOD — SMOKE PROTOTYPE (not v0.3).

STATUS: native_vod_smoke
This module is a code-shape smoke prototype. Shape contracts, gradient
flow and module wiring work, but the model has NOT been shown to learn
anything that beats a trivial baseline. Do not call this v0.3 and do
not present its numbers as evidence that the VOD mechanism works.

Backbone (post-§12.19 upgrade)
------------------------------
The field denoiser used to be a per-voxel pointwise MLP over
`[u_noisy, smooth3, smooth5, pos_t, pos_y, pos_x]`. That gave the
substrate composite (5-distinctive `native_total_loss`) a working
proof-of-concept but no real spatial / spatiotemporal mixing.

The default backbone is now a small 3-level spatial UNet over (H, W)
with a single 1-D conv along T at the bottleneck — see
`vod_minimal.denoisers.UNetDenoiser`. The substrate (encoders /
decoders / smoothing taps / position grid / loss plumbing) is
unchanged; only the denoiser interior differs. The legacy pointwise
MLP backbone is preserved as `--backbone mlp` for ablation; pick it
via `NativeVODConfig(backbone="mlp")`.

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
    denoise_path(U_noisy, steps=K) → U_pred
    decode(U_pred)       → predicted_views

Five losses are aggregated by `native_total_loss`:

    L_field      MSE between U_pred and U_target      (latent reconstruction)
    L_media      MSE on image / video / audio decoded views
    L_temporal   one-sided ReLU on temporal smoothness inflation
                 (relu(smooth(pred_video) − smooth(target_video)))
    L_artifact   one-sided ReLU on tile-residue inflation, image+video
    L_text       Binary-Twin discrete/continuous consistency on the toy
                 text channel: CE over Φ(target) plus reconstruction to
                 Ψ(Φ(target)).

Each loss has its own weight; default weights produce a balanced toy
training run, but every component can be set to 0 to ablate it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from torch import nn

from .torch_artifacts import (
    RESIDUE_FLOOR,
    artifact_regularization_loss,
)
from .binary_twin import (
    binary_twin_pixel_torch_loss,
    binary_twin_torch_accuracy,
    binary_twin_torch_loss,
)
from .aimp import tpsr_video_consistency_loss
from .denoisers import PointwiseMLPDenoiser, UNetDenoiser


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

ARCHITECTURE_VERSION = "1.0"


@dataclass(frozen=True)
class NativeVODConfig:
    channels: int = 4
    hidden: int = 32
    denoise_steps: int = 4
    enable_audio: bool = False  # experimental: 1×1 reshape, not a real codec
    enable_text: bool = False   # experimental: float-MSE proxy, not discrete CE
    backbone: str = "unet"      # "unet" (default) or "mlp" (legacy ablation)
    time_dim: int = 0           # >0 enables time-conditioned denoise for diffusion training
    architecture_version: str = ARCHITECTURE_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NativeVODConfig":
        version = data.get("architecture_version", "0.0")
        if version.split(".")[0] != ARCHITECTURE_VERSION.split(".")[0]:
            raise ValueError(
                f"NativeVODConfig: incompatible architecture_version "
                f"{version!r} (this build expects {ARCHITECTURE_VERSION!r}). "
                f"Major version mismatch — refuse to load."
            )
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"NativeVODConfig.from_dict: unknown keys {sorted(unknown)}")
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class NativeVODOutput:
    """Forward output of NativeVOD.

    Field naming aligns with diffusers convention (`sample` for the
    primary decoded media). `latent` exposes the denoised substrate
    field U for downstream consumers (loss, sampler, debug).

    Implements `__iter__` so legacy tuple unpacking
    `predicted, u_pred = model(noisy)` continues to work. Do NOT add
    new positional fields without updating the iter contract.
    """
    sample: dict[str, torch.Tensor]
    latent: torch.Tensor

    def __iter__(self):
        yield self.sample
        yield self.latent


class NativeVOD(nn.Module):
    """Native unified VOD generator.

    A single nn.Module that owns:
      * four thin per-medium encoders (linear pixel-wise channel project)
      * one shared field denoiser (UNet by default, legacy pointwise
        MLP behind `NativeVODConfig(backbone="mlp")`)
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
        # Rolled back: weight tying enc_image=enc_video hurt video
        # (forced video's 8-frame temporal signal to share single-frame
        # weights). Independent encoders kept; test for grad flow now
        # uses attribute-based parameter iteration which is robust
        # either way.

        # Field denoiser backbone. Features per latent voxel are the
        # same regardless of backbone choice:
        #   [u_noisy (C), smooth3(u_noisy) (C), smooth5(u_noisy) (C),
        #    pos_t, pos_y, pos_x]   total: 3C + 3
        # Two smoothing scales (3-tap, 5-tap) give the denoiser a
        # multi-scale local-mean reference. NO target / condition /
        # answer-side input.
        if config.backbone == "mlp":
            self.denoiser = PointwiseMLPDenoiser(
                channels=c, hidden=config.hidden, extra_dims=config.time_dim,
            )
        elif config.backbone == "unet":
            self.denoiser = UNetDenoiser(
                channels=c, hidden=config.hidden, extra_dims=config.time_dim,
            )
        else:
            raise ValueError(
                f"NativeVODConfig.backbone must be 'unet' or 'mlp', got {config.backbone!r}"
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
        # Rolled back fusion=sum: it slightly hurt image (0.0375 → 0.0400)
        # and didn't help video. The image bottleneck wasn't in fusion.

    # ---- shared feature helpers ----------------------------------- #

    @staticmethod
    def _smooth_taps(u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """3-tap and 5-tap H/W mean of a (T, H, W, C) latent.

        Uses circular roll along the spatial axes (consistent with
        `projections.smooth`). No trainable parameters.
        """
        s3 = (u + torch.roll(u, 1, dims=-3) + torch.roll(u, -1, dims=-3)) / 3.0
        s3 = (s3 + torch.roll(s3, 1, dims=-2) + torch.roll(s3, -1, dims=-2)) / 3.0
        s5 = (
            u
            + torch.roll(u, 1, dims=-3) + torch.roll(u, -1, dims=-3)
            + torch.roll(u, 2, dims=-3) + torch.roll(u, -2, dims=-3)
        ) / 5.0
        s5 = (
            s5
            + torch.roll(s5, 1, dims=-2) + torch.roll(s5, -1, dims=-2)
            + torch.roll(s5, 2, dims=-2) + torch.roll(s5, -2, dims=-2)
        ) / 5.0
        return s3, s5

    @staticmethod
    def _position_grid(T: int, H: int, W: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Return a `(T, H, W, 3)` normalized (t, y, x) position grid."""
        pos_t = torch.linspace(0.0, 1.0, T, device=device, dtype=dtype).view(T, 1, 1, 1).expand(T, H, W, 1)
        pos_y = torch.linspace(0.0, 1.0, H, device=device, dtype=dtype).view(1, H, 1, 1).expand(T, H, W, 1)
        pos_x = torch.linspace(0.0, 1.0, W, device=device, dtype=dtype).view(1, 1, W, 1).expand(T, H, W, 1)
        return torch.cat([pos_t, pos_y, pos_x], dim=-1)

    def _build_features(self, u_noisy: torch.Tensor) -> torch.Tensor:
        """Build the (B, T, H, W, 3C+3) feature tensor that both backbones consume.

        `u_noisy` may be 4-D (T, H, W, C) — the legacy single-sample
        path — in which case a leading batch dim is inserted. The
        feature tensor is always returned with an explicit batch dim
        (whether or not it was present on input) so the UNet backbone
        can be called uniformly.
        """
        if u_noisy.ndim == 4:
            u = u_noisy.unsqueeze(0)
        elif u_noisy.ndim == 5:
            u = u_noisy
        else:
            raise ValueError(
                f"u_noisy must be (T,H,W,C) or (B,T,H,W,C), got shape {tuple(u_noisy.shape)}"
            )
        B, T, H, W, _ = u.shape
        s3, s5 = self._smooth_taps(u)
        pos = self._position_grid(T, H, W, u.device, u.dtype).unsqueeze(0).expand(B, T, H, W, 3)
        return torch.cat([u, s3, s5, pos], dim=-1)

    # ---- denoise --------------------------------------------------- #

    def denoise(self, u_noisy: torch.Tensor, *, t: torch.Tensor | None = None) -> torch.Tensor:
        """Single shared update step on U.

        Features per voxel:
            [u_noisy, smooth3(u_noisy), smooth5(u_noisy),
             pos_t, pos_y, pos_x]
        No target, no condition. Applies a per-voxel delta scaled by a
        learnable sigmoid step size. Backbone is whatever
        `config.backbone` selected (`unet` default, `mlp` legacy).

        Accepts either a 4-D `(T, H, W, C)` single-sample latent (the
        existing public contract) or a 5-D `(B, T, H, W, C)` batched
        latent. Output rank matches input rank.
        """
        squeeze_batch = False
        if u_noisy.ndim == 4:
            u = u_noisy.unsqueeze(0)
            squeeze_batch = True
        else:
            u = u_noisy

        feats = self._build_features(u)
        if self.config.time_dim > 0:
            from .diffusion import sinusoidal_time_embedding
            B, T, H, W, _ = feats.shape
            if t is None:
                t_batch = torch.zeros(B, device=feats.device, dtype=torch.long)
            elif t.ndim == 0:
                t_batch = t.expand(B)
            else:
                t_batch = t
            t_emb = sinusoidal_time_embedding(t_batch, self.config.time_dim, dtype=feats.dtype)
            t_emb_grid = t_emb.view(B, 1, 1, 1, self.config.time_dim).expand(B, T, H, W, self.config.time_dim)
            feats = torch.cat([feats, t_emb_grid], dim=-1)
        delta = self.denoiser(feats)
        step = torch.sigmoid(self.step_logit)
        out = u + step * delta
        return out.squeeze(0) if squeeze_batch else out

    def denoise_path(self, u_noisy: torch.Tensor, *, steps: int | None = None) -> torch.Tensor:
        if steps is None:
            steps = self.config.denoise_steps
        u = u_noisy
        for _ in range(steps):
            u = self.denoise(u)
        return u

    # ---- decode ---------------------------------------------------- #

    def decode(
        self,
        U: torch.Tensor,
        *,
        requested: tuple[str, ...] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Project U back into the active media views.

        `requested` (optional) restricts decoding to a subset of the
        active media — useful when the caller only needs e.g. image and
        wants to skip the video / audio / text head dispatch. Note this
        only saves the per-pixel inverse projection (1×1 Linear); the
        substrate sampling cost is media-agnostic.

        Default `requested=None` preserves the original behaviour
        (decode every active medium).
        """
        active = set(self.active_media())
        if requested is not None:
            req_set = set(requested)
            unknown = req_set - active
            if unknown:
                raise ValueError(
                    f"decode: requested media {sorted(unknown)} not in "
                    f"active set {sorted(active)}"
                )
            active = req_set
        out: dict[str, torch.Tensor] = {}

        # Image projection (in projections.project_all) takes the middle
        # frame of the spacetime field — `U[T//2]` — not the temporal mean.
        # Decode must mirror that: pull the middle slice of the latent
        # before per-pixel projection. Using mean here while projection
        # uses middle frame caused image MSE to plateau at ~0.04 because
        # the model could never represent the time-asymmetric image signal
        # via a temporal mean of an oscillating latent.
        if "image" in active:
            mid = U.shape[0] // 2
            out["image"] = self.dec_image(U[mid]).squeeze(-1)
        if "video" in active:
            out["video"] = self.dec_video(U).squeeze(-1)            # (T, H, W)
        if "audio" in active:
            out["audio"] = grid_to_audio(self.dec_audio(U).squeeze(-1))
        if "text" in active:
            # Text projection still uses temporal mean (audio/text are
            # time-aggregates, not snapshots).
            text_hw = self.dec_text(U.mean(dim=0)).squeeze(-1)
            out["text"] = grid_to_text(text_hw)
        return out

    # ---- one-shot forward ----------------------------------------- #

    def forward(
        self,
        noisy_views: dict[str, torch.Tensor],
        *,
        steps: int | None = None,
    ) -> "NativeVODOutput":
        """Return NativeVODOutput(sample=predicted_views, latent=U_pred).

        The model takes ONLY the noisy views. Targets are not visible
        anywhere on the model side; they only appear inside the loss
        function for comparison.

        Legacy `predicted, u_pred = model(noisy)` tuple unpacking is
        preserved via NativeVODOutput.__iter__.
        """
        u_noisy = self.encode(noisy_views)
        u_pred = self.denoise_path(u_noisy, steps=steps)
        predicted = self.decode(u_pred)
        return NativeVODOutput(sample=predicted, latent=u_pred)

    # ---- HF-compatible serialization (shadow contract, no HF deps) -- #

    CONFIG_FILE = "config.json"
    SAFETENSORS_FILE = "model.safetensors"
    PYTORCH_FILE = "pytorch_model.bin"

    def save_pretrained(self, save_directory: str | Path) -> None:
        """Write `config.json` + `model.safetensors` (or `pytorch_model.bin`
        fallback) to `save_directory`.

        Layout matches HF convention so future PyTorchModelHubMixin /
        ModelMixin migration is a parent-class swap, not a rewrite.
        Does NOT depend on huggingface_hub or safetensors libraries —
        safetensors is used when available, else falls back to torch.save.
        """
        d = Path(save_directory)
        d.mkdir(parents=True, exist_ok=True)
        (d / self.CONFIG_FILE).write_text(
            json.dumps(self.config.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        state = self.state_dict()
        try:
            from safetensors.torch import save_file  # type: ignore
            save_file(state, str(d / self.SAFETENSORS_FILE))
        except ImportError:
            torch.save(state, d / self.PYTORCH_FILE)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *,
        map_location: str | torch.device | None = None,
        strict: bool = True,
    ) -> "NativeVOD":
        """Load a NativeVOD saved by `save_pretrained`.

        Only supports local directories. Hub download is delegated to a
        future PyTorchModelHubMixin wrapper (Phase B / C).
        """
        d = Path(pretrained_model_name_or_path)
        if not d.is_dir():
            raise FileNotFoundError(
                f"from_pretrained expects a local directory containing "
                f"{cls.CONFIG_FILE}, got {d!r}"
            )
        config_path = d / cls.CONFIG_FILE
        if not config_path.exists():
            raise FileNotFoundError(f"missing {config_path}")
        config = NativeVODConfig.from_dict(
            json.loads(config_path.read_text(encoding="utf-8"))
        )
        model = cls(config)

        st_path = d / cls.SAFETENSORS_FILE
        pt_path = d / cls.PYTORCH_FILE
        if st_path.exists():
            from safetensors.torch import load_file  # type: ignore
            state = load_file(str(st_path), device=str(map_location) if map_location else "cpu")
        elif pt_path.exists():
            state = torch.load(pt_path, map_location=map_location)
        else:
            raise FileNotFoundError(
                f"missing weight file: neither {cls.SAFETENSORS_FILE} "
                f"nor {cls.PYTORCH_FILE} found in {d!r}"
            )
        model.load_state_dict(state, strict=strict)
        return model


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
    binary_twin_pixel: float = 0.0
    aimp: float = 0.0
    recon: float = 0.0          # Gate 0: encode/decode identity (opt-in)
    clean_noop: float = 0.0     # Gate 0: denoise_path no-op on clean latent (opt-in)

    def asdict(self) -> dict[str, float]:
        return {
            "field": self.field,
            "media": self.media,
            "temporal": self.temporal,
            "artifact": self.artifact,
            "text": self.text,
            "recon": self.recon,
            "clean_noop": self.clean_noop,
            "binary_twin_pixel": self.binary_twin_pixel,
            "aimp": self.aimp,
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
    artifact_residue_floor: float = RESIDUE_FLOOR,
    artifact_strength: float = 1.0,
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
        L_artifact = artifact_regularization_loss(
            pred["video"], target_views["video"],
            tile=artifact_tile,
            residue_floor=artifact_residue_floor,
            strength=artifact_strength,
        )
    else:
        L_temporal = u_pred.new_zeros(())
        L_artifact = u_pred.new_zeros(())

    if "text" in target_views and "text" in active:
        L_text = binary_twin_torch_loss(pred["text"], target_views["text"])
        text_acc = binary_twin_torch_accuracy(pred["text"], target_views["text"])
    else:
        L_text = u_pred.new_zeros(())
        text_acc = u_pred.new_tensor(float("nan"))

    # Per-pixel Binary-Twin on image / video. Average across whichever
    # spatial media are present. Default weight is 0 so this is a no-op
    # unless callers opt in (NativeLossWeights.binary_twin_pixel > 0).
    pixel_terms: list[torch.Tensor] = []
    for medium in ("image", "video"):
        if medium in target_views and medium in active and medium in pred:
            pixel_terms.append(
                binary_twin_pixel_torch_loss(pred[medium], target_views[medium])
            )
    if pixel_terms:
        L_binary_twin_pixel = torch.stack(pixel_terms).mean()
    else:
        L_binary_twin_pixel = u_pred.new_zeros(())

    # TPSR/AIMP video consistency loss. Default weight is 0 so this is
    # a no-op unless callers opt in (NativeLossWeights.aimp > 0).
    if "video" in target_views and "video" in active and "video" in pred:
        L_aimp = tpsr_video_consistency_loss(pred["video"])
    else:
        L_aimp = u_pred.new_zeros(())

    # Gate 0a: encoder/decoder identity. Encode → decode (no denoise) on
    # the clean target should reconstruct the target. Without this loss
    # the encode/decode pair degenerates to a near-constant-output map
    # (decode(encode(x)) ≈ const) — verified on toy stress training.
    u_target_grad = model.encode(target_views)
    pred_recon = model.decode(u_target_grad)
    recon_terms: list[torch.Tensor] = []
    for k in media_keys:
        if k in pred_recon:
            recon_terms.append(F.mse_loss(pred_recon[k], target_views[k]))
    if recon_terms:
        L_recon = torch.stack(recon_terms).mean()
    else:
        L_recon = u_pred.new_zeros(())

    # Gate 0b: denoise_path no-op stability on clean latent. denoise(clean)
    # must stay near clean — the architecture has a residual `u + step*delta`
    # but delta isn't bounded on inputs the model didn't see in training.
    # Iterating an unconstrained delta on clean input produces exponential
    # blowup (verified ρ ≈ 20 per step via tensorearch temporal). Loss
    # forces delta ≈ 0 on clean latents.
    u_clean_pred = model.denoise_path(u_target_grad, steps=steps)
    L_clean_noop = F.mse_loss(u_clean_pred, u_target_grad)

    total = (
        weights.field * L_field
        + weights.media * L_media
        + weights.temporal * L_temporal
        + weights.artifact * L_artifact
        + weights.text * L_text
        + weights.binary_twin_pixel * L_binary_twin_pixel
        + weights.aimp * L_aimp
        + weights.recon * L_recon
        + weights.clean_noop * L_clean_noop
    )
    components = {
        "L_field": float(L_field.detach()),
        "L_media": float(L_media.detach()),
        "L_temporal": float(L_temporal.detach()),
        "L_artifact": float(L_artifact.detach()),
        "L_text": float(L_text.detach()),
        "L_binary_twin_pixel": float(L_binary_twin_pixel.detach()),
        "L_aimp": float(L_aimp.detach()),
        "L_recon": float(L_recon.detach()),
        "L_clean_noop": float(L_clean_noop.detach()),
        "binary_twin_symbol_accuracy": float(text_acc.detach()),
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
