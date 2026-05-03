"""Metrics for the minimal VOD prototype."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .artifacts import tile_residue
from .projections import smooth


EPS = 1e-9


# Tile residue is a geometric property of a 2-D spatial grid (the
# canonical AI-renderer "tile light spot / boundary contour" failure).
# 1-D media (audio waveforms, text channel strings) do not carry that
# grid, so mixing them into the main artifact score dilutes a real
# image/video signal with noise that has no causal relationship to the
# 4/e Orthogonal Compression failure mode. The main score therefore
# restricts itself to the spatial set; 1-D media are reported separately
# under non_spatial_* keys.
SPATIAL_MEDIA: tuple[str, ...] = ("image", "video")


@dataclass(frozen=True)
class Descriptor:
    name: str
    amplitude: float
    phase: float
    frequency: float
    compression: float
    salience: float
    snr: float

    def vec(self) -> np.ndarray:
        return np.array(
            [self.amplitude, self.phase, self.frequency, self.compression, self.salience, self.snr],
            dtype=np.float64,
        )


def _entropy(values: np.ndarray, bins: int = 48) -> float:
    hist, _ = np.histogram(values.ravel(), bins=bins)
    probs = hist[hist > 0] / max(1, hist.sum())
    return float(-(probs * np.log2(probs)).sum())


def _salience(values: np.ndarray) -> float:
    arr = values.astype(np.float64)
    total = 0.0
    count = 0
    for axis in range(arr.ndim):
        diff = np.diff(arr, axis=axis)
        if diff.size:
            total += float(np.mean(np.abs(diff)))
            count += 1
    return total / max(1, count)


def _dominant_frequency(values: np.ndarray) -> float:
    flat = values.ravel().astype(np.float64)
    flat -= flat.mean()
    if flat.size < 4:
        return 0.0
    spec = np.abs(np.fft.rfft(flat))
    if spec.size <= 1:
        return 0.0
    idx = int(np.argmax(spec[1:]) + 1)
    return idx / flat.size


def _dominant_phase(values: np.ndarray) -> float:
    flat = values.ravel().astype(np.float64)
    flat -= flat.mean()
    if flat.size < 4:
        return 0.0
    coeff = np.fft.rfft(flat)
    if coeff.size <= 1:
        return 0.0
    idx = int(np.argmax(np.abs(coeff[1:])) + 1)
    return float(np.angle(coeff[idx]))


def descriptor(name: str, values: np.ndarray) -> Descriptor:
    arr = values.astype(np.float64)
    amp = float(np.sqrt(np.mean(arr * arr)))
    max_entropy = math.log2(48)
    comp = 1.0 - min(_entropy(arr) / max_entropy, 1.0)
    signal = float(np.mean(arr * arr))
    noise = float(np.var(arr - smooth(arr)))
    snr = 10.0 * math.log10((signal + EPS) / (noise + EPS))
    return Descriptor(name, amp, _dominant_phase(arr), _dominant_frequency(arr), comp, _salience(arr), snr)


def mean_target_error(views: dict[str, np.ndarray], targets: dict[str, np.ndarray]) -> float:
    errors = []
    for name, value in views.items():
        current = descriptor(name, value).vec()
        target = descriptor(name + "_target", targets[name]).vec()
        errors.append(float(np.linalg.norm(current - target)))
    return float(np.mean(errors))


def mean_tile_residue(
    views: dict[str, np.ndarray],
    *,
    tile: int = 8,
    spatial_media: tuple[str, ...] = SPATIAL_MEDIA,
) -> float:
    """Mean tile-boundary residue score across the *spatial* media views.

    Restricts the average to media in `spatial_media` (default
    image / video). Returns NaN if no spatial medium is present.
    """

    spatial_residues = [
        tile_residue(views[m], tile=tile)
        for m in spatial_media
        if m in views
    ]
    if not spatial_residues:
        return float("nan")
    return float(np.mean(spatial_residues))


def artifact_metrics(
    views: dict[str, np.ndarray],
    *,
    tile: int = 8,
    spatial_media: tuple[str, ...] = SPATIAL_MEDIA,
) -> dict[str, float]:
    """Aggregate tile-residue artifact diagnostics.

    Main artifact metrics are computed over *spatial* media only
    (default image / video). Non-spatial media (audio waveforms, text
    channels) are reported separately under `non_spatial_*` keys: they
    have a tile_residue value but no causal relationship to the
    4/e Orthogonal Compression failure mode, so mixing them into the
    main score would silently dilute a real failure signal.

    Returns
    -------
    dict with keys:
        mean_tile_residue              spatial only — mean across image/video
        max_tile_residue               spatial only — worst across image/video
        artifact_score                 spatial only — 1.0 = no detectable tile
                                       contour preference; → 0 as residue grows
        non_spatial_mean_tile_residue  audio/text/etc. mean (informational)
        non_spatial_max_tile_residue   audio/text/etc. max (informational)

    NaN is returned per-block when that block has no media. An empty
    `views` returns NaN for all five keys.

    Score formula: `1 / (1 + max(mean_residue - 1, 0))`:
        residue 1.0  → score 1.00
        residue 2.0  → score 0.50
        residue 3.0  → score 0.33
        residue → ∞  → score → 0
    """

    nan = float("nan")
    spatial_residues = [
        tile_residue(views[m], tile=tile)
        for m in spatial_media
        if m in views
    ]
    non_spatial_residues = [
        tile_residue(value, tile=tile)
        for name, value in views.items()
        if name not in spatial_media
    ]

    if spatial_residues:
        mean_res = float(np.mean(spatial_residues))
        max_res = float(np.max(spatial_residues))
        excess = max(mean_res - 1.0, 0.0)
        score = 1.0 / (1.0 + excess)
    else:
        mean_res = nan
        max_res = nan
        score = nan

    if non_spatial_residues:
        non_spatial_mean = float(np.mean(non_spatial_residues))
        non_spatial_max = float(np.max(non_spatial_residues))
    else:
        non_spatial_mean = nan
        non_spatial_max = nan

    return {
        "mean_tile_residue": mean_res,
        "max_tile_residue": max_res,
        "artifact_score": float(score) if score == score else nan,  # NaN-safe
        "non_spatial_mean_tile_residue": non_spatial_mean,
        "non_spatial_max_tile_residue": non_spatial_max,
    }


def modular_shrinking_number(path: list[np.ndarray]) -> float:
    total = 0.0
    for k, (a, b) in enumerate(zip(path, path[1:]), start=1):
        total += (1.0 / k) * float(np.mean(np.abs(b - a)))
    return total


# --------------------------------------------------------------------------- #
#  Temporal / spatiotemporal metrics
# --------------------------------------------------------------------------- #

def _video_array(video: np.ndarray) -> np.ndarray | None:
    """Validate a video tensor; return float64 view or None if too small."""
    arr = np.asarray(video, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[0] < 2:
        return None
    return arr


def temporal_smoothness(video: np.ndarray) -> float:
    """Mean absolute frame-to-frame difference.

    Lower = smoother motion. Returns NaN when the input has fewer than
    two frames or is not a video.
    """
    arr = _video_array(video)
    if arr is None:
        return float("nan")
    return float(np.mean(np.abs(np.diff(arr, axis=0))))


def frame_descriptor_drift(video: np.ndarray) -> float:
    """Std-dev of the per-frame descriptor amplitude across time.

    Uses the same `descriptor` family as `mean_target_error` so this
    drift is directly comparable to projection-space error. NaN when
    the clip has fewer than two frames.
    """
    arr = _video_array(video)
    if arr is None:
        return float("nan")
    amps = np.array(
        [descriptor("frame", frame).amplitude for frame in arr],
        dtype=np.float64,
    )
    if amps.size < 2:
        return float("nan")
    return float(amps.std(ddof=0))


def temporal_artifact_drift(video: np.ndarray, *, tile: int = 8) -> float:
    """Std-dev of per-frame `tile_residue`.

    A real renderer should keep tile residue stable (or absent) across
    frames; rapid drift in the artifact statistic itself is a flicker /
    jitter signal beyond plain temporal_smoothness. NaN below two
    frames or below the tile period spatially.
    """
    arr = _video_array(video)
    if arr is None:
        return float("nan")
    if min(arr.shape[1:]) <= tile:
        return float("nan")
    residues = np.array(
        [tile_residue(frame, tile=tile) for frame in arr],
        dtype=np.float64,
    )
    return float(residues.std(ddof=0))


def cross_frame_consistency_score(video: np.ndarray, *, tile: int = 8) -> float:
    """Aggregate consistency score in [0, 1].

    Higher = more temporally consistent. Combines:
      * temporal_smoothness  (lower is better)
      * descriptor drift     (lower is better)
      * artifact drift       (lower is better)

    Each component is mapped through `1 / (1 + value)` and then
    averaged. The mapping is monotonically decreasing and bounded, so
    the final score lives in [0, 1] regardless of input scale.

    Returns NaN when the input is not a multi-frame video.
    """
    arr = _video_array(video)
    if arr is None:
        return float("nan")

    smooth_v = temporal_smoothness(arr)
    drift_v = frame_descriptor_drift(arr)
    artifact_v = temporal_artifact_drift(arr, tile=tile)

    components: list[float] = []
    for value in (smooth_v, drift_v, artifact_v):
        if not (value == value):  # NaN-safe skip
            continue
        components.append(1.0 / (1.0 + max(value, 0.0)))
    if not components:
        return float("nan")
    return float(np.mean(components))


def temporal_metrics(views: dict[str, np.ndarray], *, tile: int = 8) -> dict[str, float]:
    """Aggregate temporal diagnostics for a media-views dict.

    Looks at `views["video"]` only — the other media are not videos.
    Returns NaN-filled keys when no video is present.
    """
    nan = float("nan")
    video = views.get("video")
    if video is None:
        return {
            "temporal_smoothness": nan,
            "frame_descriptor_drift": nan,
            "temporal_artifact_drift": nan,
            "cross_frame_consistency_score": nan,
        }
    return {
        "temporal_smoothness": temporal_smoothness(video),
        "frame_descriptor_drift": frame_descriptor_drift(video),
        "temporal_artifact_drift": temporal_artifact_drift(video, tile=tile),
        "cross_frame_consistency_score": cross_frame_consistency_score(video, tile=tile),
    }
