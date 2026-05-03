"""Toy validation for the VOD Chladni-like entropy field idea.

This is not a real generator. It checks whether text, image, video, and audio
can be mapped into one shared entron descriptor and updated by the same
Chladni-style denoising/shrinking operator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class EntropyPattern:
    medium: str
    field: np.ndarray
    amplitude: float
    phase: float
    frequency: float
    compression: float
    salience: float
    snr: float

    def vector(self) -> np.ndarray:
        return np.array(
            [self.amplitude, self.phase, self.frequency, self.compression, self.salience, self.snr],
            dtype=np.float64,
        )


def _entropy(values: np.ndarray, bins: int = 32) -> float:
    flat = values.astype(np.float64).ravel()
    if flat.size == 0:
        return 0.0
    hist, _ = np.histogram(flat, bins=bins, density=False)
    probs = hist[hist > 0] / max(1, hist.sum())
    return float(-(probs * np.log2(probs)).sum())


def _gradient_energy(values: np.ndarray) -> float:
    arr = values.astype(np.float64)
    if arr.ndim == 1:
        return float(np.mean(np.abs(np.diff(arr)))) if arr.size > 1 else 0.0
    energy = 0.0
    count = 0
    for axis in range(arr.ndim):
        diff = np.diff(arr, axis=axis)
        if diff.size:
            energy += float(np.mean(np.abs(diff)))
            count += 1
    return energy / max(1, count)


def _dominant_frequency(values: np.ndarray) -> float:
    flat = values.astype(np.float64).ravel()
    flat = flat - flat.mean()
    if flat.size < 4 or np.allclose(flat, 0):
        return 0.0
    spec = np.abs(np.fft.rfft(flat))
    if spec.size <= 1:
        return 0.0
    idx = int(np.argmax(spec[1:]) + 1)
    return idx / max(1, flat.size)


def _phase(values: np.ndarray) -> float:
    flat = values.astype(np.float64).ravel()
    if flat.size < 4:
        return 0.0
    coeff = np.fft.rfft(flat - flat.mean())
    if coeff.size <= 1:
        return 0.0
    idx = int(np.argmax(np.abs(coeff[1:])) + 1)
    return float(np.angle(coeff[idx]))


def to_entropy_pattern(medium: str, values: np.ndarray) -> EntropyPattern:
    arr = values.astype(np.float64)
    amplitude = float(np.sqrt(np.mean(arr * arr)))
    entropy = _entropy(arr)
    max_entropy = math.log2(32)
    compression = 1.0 - min(entropy / max_entropy, 1.0)
    salience = _gradient_energy(arr)
    freq = _dominant_frequency(arr)
    phase = _phase(arr)
    noise = float(np.var(arr - arr.mean()))
    signal = float(np.mean(arr * arr))
    snr = 10.0 * math.log10((signal + 1e-9) / (noise + 1e-9))
    return EntropyPattern(medium, arr, amplitude, phase, freq, compression, salience, snr)


def chladni_basis(shape: tuple[int, ...]) -> np.ndarray:
    coords = np.meshgrid(*[np.linspace(0.0, 1.0, n) for n in shape], indexing="ij")
    basis = np.zeros(shape, dtype=np.float64)
    for i, grid in enumerate(coords, start=1):
        basis += np.sin(math.pi * (i + 1) * grid) * np.cos(math.pi * i * grid)
    norm = np.max(np.abs(basis)) + 1e-9
    return basis / norm


def denoise_chladni(pattern: EntropyPattern, steps: int = 8, shrink: float = 0.68) -> EntropyPattern:
    field = pattern.field.copy().astype(np.float64)
    target = chladni_basis(field.shape)
    if target.shape != field.shape:
        target = np.resize(target, field.shape)

    # Shared toy update: shrink noise while pulling every medium toward a stable
    # Chladni-like standing-wave basis. Real VOD would learn this operator.
    for _ in range(steps):
        centered = field - field.mean()
        field = shrink * centered + (1.0 - shrink) * pattern.amplitude * target

    return to_entropy_pattern(pattern.medium + "_vod", field)


def text_sample(text: str) -> np.ndarray:
    data = np.frombuffer(text.encode("utf-8"), dtype=np.uint8).astype(np.float64)
    return data / 255.0


def image_sample(size: int = 64) -> np.ndarray:
    y, x = np.mgrid[0:size, 0:size]
    img = np.sin(x / 4.0) + np.cos(y / 7.0) + 0.25 * np.sin((x + y) / 3.0)
    return img / np.max(np.abs(img))


def video_sample(frames: int = 12, size: int = 32) -> np.ndarray:
    t, y, x = np.mgrid[0:frames, 0:size, 0:size]
    vid = np.sin((x + t * 2) / 4.0) + np.cos((y - t) / 6.0)
    return vid / np.max(np.abs(vid))


def audio_sample(samples: int = 2048, rate: int = 2048) -> np.ndarray:
    t = np.arange(samples) / rate
    wav = 0.65 * np.sin(2 * math.pi * 110 * t) + 0.35 * np.sin(2 * math.pi * 220 * t + 0.4)
    return wav.astype(np.float64)


def summarize(patterns: Iterable[EntropyPattern]) -> str:
    lines = []
    header = f"{'medium':<14} {'amp':>9} {'phase':>9} {'freq':>9} {'compress':>10} {'salience':>10} {'snr':>9}"
    lines.append(header)
    lines.append("-" * len(header))
    for p in patterns:
        lines.append(
            f"{p.medium:<14} {p.amplitude:9.4f} {p.phase:9.4f} {p.frequency:9.4f} "
            f"{p.compression:10.4f} {p.salience:10.4f} {p.snr:9.4f}"
        )
    return "\n".join(lines)


def main() -> None:
    raw = [
        to_entropy_pattern("text", text_sample("VOD 熵纹 Chladni 文字")),
        to_entropy_pattern("image", image_sample()),
        to_entropy_pattern("video", video_sample()),
        to_entropy_pattern("audio", audio_sample()),
    ]
    updated = [denoise_chladni(p) for p in raw]

    print("Raw media mapped to shared entron descriptors:")
    print(summarize(raw))
    print()
    print("After shared Chladni-like denoising/shrinking update:")
    print(summarize(updated))
    print()
    print("Vector-space movement under one shared operator:")
    for before, after in zip(raw, updated):
        delta = np.linalg.norm(after.vector() - before.vector())
        print(f"{before.medium:<6} -> {after.medium:<10} delta={delta:.6f}")


if __name__ == "__main__":
    main()
