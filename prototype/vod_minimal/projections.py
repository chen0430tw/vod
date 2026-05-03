"""Toy media projections from one Chladni-like field."""

from __future__ import annotations

import math

import numpy as np

from .artifacts import oc_four_over_e


EPS = 1e-9


def smooth(values: np.ndarray) -> np.ndarray:
    arr = values.astype(np.float64)
    out = arr.copy()
    for axis in range(arr.ndim):
        out = (out + np.roll(out, 1, axis=axis) + np.roll(out, -1, axis=axis)) / 3.0
    return out


def project_image(field: np.ndarray) -> np.ndarray:
    return field.astype(np.float64).copy()


def project_video(field: np.ndarray, frames: int = 10) -> np.ndarray:
    """Legacy 2-D-derived video projection.

    Treats `field` as a single image and produces a (frames, H, W) clip
    via roll + sin-phase mixing. Kept verbatim so that runs predating
    the spacetime upgrade reproduce bit-for-bit. New code should prefer
    `project_video_3d` and a true U(t, y, x) field.
    """
    out = []
    for t in range(frames):
        shifted = np.roll(field, shift=t, axis=1)
        phase = math.sin(2 * math.pi * t / frames)
        out.append(0.82 * shifted + 0.18 * phase * smooth(field))
    return np.stack(out, axis=0)


def project_video_3d(field: np.ndarray, frames: int | None = None) -> np.ndarray:
    """3-D projection: directly slice U(t, y, x) into a video clip.

    When `field.ndim == 3` we treat axis 0 as the time axis and return
    the volume as-is (or temporally resampled if `frames` differs).

    When `field.ndim == 2` we have a 2-D Chladni image, so this falls
    back to the legacy `project_video`. Callers may pass `frames=None`
    in either case to keep the field's native frame count.
    """
    arr = field.astype(np.float64, copy=False)
    if arr.ndim == 2:
        return project_video(arr, frames=10 if frames is None else frames)
    if arr.ndim != 3:
        raise ValueError(f"project_video_3d expects a 2-D or 3-D array, got ndim={arr.ndim}")

    f_native = arr.shape[0]
    if frames is None or frames == f_native:
        return arr.copy()
    # Resample along the time axis with linear interpolation.
    src = np.linspace(0.0, 1.0, f_native, endpoint=False)
    dst = np.linspace(0.0, 1.0, frames, endpoint=False)
    out = np.empty((frames,) + arr.shape[1:], dtype=np.float64)
    for h in range(arr.shape[1]):
        for w in range(arr.shape[2]):
            out[:, h, w] = np.interp(dst, src, arr[:, h, w])
    return out


def project_audio(field: np.ndarray, samples: int = 2048) -> np.ndarray:
    profile = field.mean(axis=0)
    profile = np.interp(np.linspace(0, profile.size - 1, samples), np.arange(profile.size), profile)
    t = np.linspace(0, 1, samples, endpoint=False)
    carrier = np.sin(2 * math.pi * 110 * t) + 0.45 * np.sin(2 * math.pi * 220 * t + 0.2)
    return carrier * (0.6 + 0.4 * profile)


def project_text(field: np.ndarray, chars: int = 32) -> np.ndarray:
    profile = np.abs(field).mean(axis=0)
    profile = np.interp(np.linspace(0, profile.size - 1, chars), np.arange(profile.size), profile)
    quantized = np.floor(15 * (profile - profile.min()) / (np.ptp(profile) + EPS))
    return quantized / 15.0


def project_all(
    field: np.ndarray,
    *,
    video_mode: str = "auto",
    frames: int | None = None,
) -> dict[str, np.ndarray]:
    """Project a field into the four media views.

    `video_mode`:
        "auto"   pick "3d" when field.ndim == 3 else "2d"
        "2d"     legacy: pretend field is an image even if it is 3-D
                 (uses temporal mean for image/audio/text and the
                 sin-phase rolled video projection)
        "3d"    spacetime: slice U(t, y, x) directly. For 2-D inputs
                 falls back to the legacy roll behaviour.

    For 3-D fields the image / audio / text projections are computed
    against the temporal mean of the volume — they remain a single 2-D
    image, a single audio clip and a single text channel respectively
    (they're not videos themselves), but they reflect the volume
    rather than an arbitrary first frame.
    """

    arr = field.astype(np.float64, copy=False)
    if video_mode == "auto":
        video_mode = "3d" if arr.ndim == 3 else "2d"
    if video_mode not in {"2d", "3d"}:
        raise ValueError(f"video_mode must be '2d' / '3d' / 'auto', got {video_mode!r}")

    if arr.ndim == 3 and video_mode == "3d":
        # Image takes the *middle frame* — a snapshot of U at one moment —
        # rather than the temporal mean. Reason: Chladni temporal modes
        # carry cos(2π·m_t·t + φ); averaging 8 uniformly-spaced frames
        # collapses the oscillation toward zero, which made image targets
        # numerically ≈ 0 and turned the zero baseline trivially perfect.
        # The middle frame preserves a representative spatial structure.
        # (Audio / text still use the temporal mean — they aren't a
        # snapshot but a temporal aggregate of the whole volume.)
        snapshot = arr[arr.shape[0] // 2]
        mean_image = arr.mean(axis=0)
        return {
            "image": project_image(snapshot),
            "video": project_video_3d(arr, frames=frames),
            "audio": project_audio(mean_image),
            "text": project_text(mean_image),
        }

    if arr.ndim == 3 and video_mode == "2d":
        # Legacy semantics on a 3-D field: collapse to its first frame.
        legacy_image = arr[0]
        return {
            "image": project_image(legacy_image),
            "video": project_video(legacy_image, frames=arr.shape[0]),
            "audio": project_audio(legacy_image),
            "text": project_text(legacy_image),
        }

    # 2-D input
    return {
        "image": project_image(arr),
        "video": project_video(arr) if video_mode == "2d" else project_video_3d(arr, frames=frames),
        "audio": project_audio(arr),
        "text": project_text(arr),
    }


def add_noise(
    view: np.ndarray,
    rng: np.random.Generator,
    scale: float = 0.24,
    *,
    artifact_suppression: bool = False,
    artifact_scale: float | None = None,
    artifact_tile: int = 8,
) -> np.ndarray:
    out = view + rng.normal(0.0, scale, size=view.shape)
    # OC_{4/e} is defined on (..., H, W) inputs only — 1-D media (audio
    # waveforms, text channels) have no tile geometry. Skip them rather
    # than feed them an undefined operator.
    if artifact_suppression and out.ndim >= 2:
        out = oc_four_over_e(
            out,
            rng,
            beta=scale if artifact_scale is None else artifact_scale,
            tile=artifact_tile,
        )
    return out


__all__ = [
    "add_noise",
    "project_all",
    "project_audio",
    "project_image",
    "project_text",
    "project_video",
    "project_video_3d",
    "smooth",
]
