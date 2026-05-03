"""Deeper toy validation for the VOD Chladni model.

This script still is not a real generator. It validates the proposed interface:

1. One latent Chladni-like field can be projected into text/image/video/audio views.
2. Each view can be re-encoded into a comparable entropy descriptor.
3. One shared denoising/shrinking path improves cross-view agreement.
4. A linear regression head can estimate stability from interpretable signals.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


EPS = 1e-9


@dataclass
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


def chladni_field(size: int = 64, modes: tuple[tuple[int, int, float], ...] | None = None) -> np.ndarray:
    if modes is None:
        modes = ((2, 3, 0.70), (5, 4, 0.45), (7, 2, 0.25))
    y, x = np.mgrid[0:size, 0:size]
    x = x / (size - 1)
    y = y / (size - 1)
    field = np.zeros((size, size), dtype=np.float64)
    for mx, my, weight in modes:
        field += weight * (np.cos(math.pi * mx * x) * np.cos(math.pi * my * y))
        field -= weight * (np.cos(math.pi * my * x) * np.cos(math.pi * mx * y))
    field /= np.max(np.abs(field)) + EPS
    return field


def random_chladni_field(rng: np.random.Generator, size: int = 64, n_modes: int = 3) -> np.ndarray:
    modes = []
    for _ in range(n_modes):
        mx = int(rng.integers(2, 9))
        my = int(rng.integers(2, 9))
        weight = float(rng.uniform(0.15, 0.9))
        modes.append((mx, my, weight))
    return chladni_field(size=size, modes=tuple(modes))


def entropy(values: np.ndarray, bins: int = 48) -> float:
    hist, _ = np.histogram(values.ravel(), bins=bins)
    probs = hist[hist > 0] / max(1, hist.sum())
    return float(-(probs * np.log2(probs)).sum())


def salience(values: np.ndarray) -> float:
    arr = values.astype(np.float64)
    total = 0.0
    count = 0
    for axis in range(arr.ndim):
        diff = np.diff(arr, axis=axis)
        if diff.size:
            total += float(np.mean(np.abs(diff)))
            count += 1
    return total / max(1, count)


def dominant_frequency(values: np.ndarray) -> float:
    flat = values.ravel().astype(np.float64)
    flat = flat - flat.mean()
    if flat.size < 4:
        return 0.0
    spec = np.abs(np.fft.rfft(flat))
    if spec.size <= 1:
        return 0.0
    idx = int(np.argmax(spec[1:]) + 1)
    return idx / flat.size


def dominant_phase(values: np.ndarray) -> float:
    flat = values.ravel().astype(np.float64)
    flat = flat - flat.mean()
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
    comp = 1.0 - min(entropy(arr) / max_entropy, 1.0)
    signal = float(np.mean(arr * arr))
    noise = float(np.var(arr - smooth(arr)))
    snr = 10.0 * math.log10((signal + EPS) / (noise + EPS))
    return Descriptor(name, amp, dominant_phase(arr), dominant_frequency(arr), comp, salience(arr), snr)


def smooth(values: np.ndarray) -> np.ndarray:
    arr = values.astype(np.float64)
    out = arr.copy()
    for axis in range(arr.ndim):
        out = (out + np.roll(out, 1, axis=axis) + np.roll(out, -1, axis=axis)) / 3.0
    return out


def project_image(field: np.ndarray) -> np.ndarray:
    return field.copy()


def project_video(field: np.ndarray, frames: int = 10) -> np.ndarray:
    out = []
    for t in range(frames):
        shifted = np.roll(field, shift=t, axis=1)
        phase = math.sin(2 * math.pi * t / frames)
        out.append(0.82 * shifted + 0.18 * phase * smooth(field))
    return np.stack(out, axis=0)


def project_audio(field: np.ndarray, samples: int = 2048) -> np.ndarray:
    profile = field.mean(axis=0)
    profile = np.interp(np.linspace(0, profile.size - 1, samples), np.arange(profile.size), profile)
    t = np.linspace(0, 1, samples, endpoint=False)
    carrier = np.sin(2 * math.pi * 110 * t) + 0.45 * np.sin(2 * math.pi * 220 * t + 0.2)
    return carrier * (0.6 + 0.4 * profile)


def project_text(field: np.ndarray, chars: int = 32) -> np.ndarray:
    # Symbolic view: quantize stable vertical energy profile into pseudo character IDs.
    profile = np.abs(field).mean(axis=0)
    profile = np.interp(np.linspace(0, profile.size - 1, chars), np.arange(profile.size), profile)
    quantized = np.floor(15 * (profile - profile.min()) / (np.ptp(profile) + EPS))
    return quantized / 15.0


def add_noise(values: np.ndarray, scale: float, rng: np.random.Generator) -> np.ndarray:
    return values + rng.normal(0.0, scale, size=values.shape)


def shared_shrink_path(values: np.ndarray, target: np.ndarray, steps: int = 12) -> list[np.ndarray]:
    path = [values.astype(np.float64)]
    cur = path[0]
    target_resized = np.resize(target, cur.shape)
    for k in range(steps):
        rate = 0.78 - 0.025 * min(k, 8)
        cur = rate * smooth(cur) + (1.0 - rate) * target_resized
        path.append(cur)
    return path


def pairwise_mean_distance(descs: list[Descriptor]) -> float:
    vecs = [d.vec() for d in descs]
    distances = []
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            distances.append(float(np.linalg.norm(vecs[i] - vecs[j])))
    return float(np.mean(distances))


def modular_shrinking_number(path: list[np.ndarray]) -> float:
    total = 0.0
    for k, (a, b) in enumerate(zip(path, path[1:]), start=1):
        total += (1.0 / k) * float(np.mean(np.abs(b - a)))
    return total


def fit_linear_regression(features: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float]:
    x = np.column_stack([np.ones(features.shape[0]), features])
    beta, *_ = np.linalg.lstsq(x, target, rcond=None)
    pred = x @ beta
    mse = float(np.mean((pred - target) ** 2))
    return beta, mse


def mean_target_error(descs: list[Descriptor], target_descs: list[Descriptor]) -> float:
    errors = [
        float(np.linalg.norm(desc.vec() - target_desc.vec()))
        for desc, target_desc in zip(descs, target_descs)
    ]
    return float(np.mean(errors))


def validate_one(
    base: np.ndarray,
    target: np.ndarray,
    rng: np.random.Generator,
) -> tuple[float, float, float, float, list[float], list[Descriptor], list[list[np.ndarray]]]:
    raw_views = {
        "image": add_noise(project_image(base), 0.24, rng),
        "video": add_noise(project_video(base), 0.24, rng),
        "audio": add_noise(project_audio(base), 0.24, rng),
        "text": add_noise(project_text(base), 0.24, rng),
    }

    target_views = {
        "image": project_image(target),
        "video": project_video(target),
        "audio": project_audio(target),
        "text": project_text(target),
    }
    raw_desc = [descriptor(name, raw_views[name]) for name in raw_views]
    target_desc = [descriptor(name + "_target", target_views[name]) for name in raw_views]
    updated_paths_by_name = {
        name: shared_shrink_path(value, target_views[name], steps=12)
        for name, value in raw_views.items()
    }
    paths = list(updated_paths_by_name.values())
    updated_desc = [descriptor(name + "_vod", path[-1]) for name, path in updated_paths_by_name.items()]
    msn = [modular_shrinking_number(path) for path in paths]
    return (
        pairwise_mean_distance(raw_desc),
        pairwise_mean_distance(updated_desc),
        mean_target_error(raw_desc, target_desc),
        mean_target_error(updated_desc, target_desc),
        msn,
        updated_desc,
        paths,
    )


def stress_test(rng: np.random.Generator, n: int = 80) -> tuple[float, float, float, float]:
    before_values = []
    after_values = []
    features = []
    target_score = []

    for _ in range(n):
        base = random_chladni_field(rng)
        target = random_chladni_field(rng, n_modes=1)
        _pair_before, _pair_after, err_before, err_after, msns, descs, _paths = validate_one(base, target, rng)
        before_values.append(err_before)
        after_values.append(err_after)
        for item, msn in zip(descs, msns):
            features.append([item.amplitude, item.compression, item.salience, item.snr, msn])
            target_score.append(item.snr + 2.0 * item.compression - 4.0 * msn + 0.35 * item.salience * item.salience)

    features_arr = np.array(features)
    target_arr = np.array(target_score)
    order = rng.permutation(features_arr.shape[0])
    split = int(features_arr.shape[0] * 0.75)
    train_idx = order[:split]
    test_idx = order[split:]

    x_train = np.column_stack([np.ones(train_idx.size), features_arr[train_idx]])
    beta, *_ = np.linalg.lstsq(x_train, target_arr[train_idx], rcond=None)
    x_test = np.column_stack([np.ones(test_idx.size), features_arr[test_idx]])
    pred = x_test @ beta
    test_mse = float(np.mean((pred - target_arr[test_idx]) ** 2))
    return float(np.mean(before_values)), float(np.mean(after_values)), float(np.mean(np.array(before_values) - np.array(after_values))), test_mse


def main() -> None:
    rng = np.random.default_rng(430)
    base = chladni_field()
    target = chladni_field(modes=((2, 3, 1.0),))

    pair_before, pair_after, target_before, target_after, msns, updated_desc, paths = validate_one(base, target, rng)

    print("Descriptor geometry:")
    print(f"  raw pairwise view distance       : {pair_before:.6f}")
    print(f"  updated pairwise view distance   : {pair_after:.6f}")
    print("Target-projection agreement:")
    print(f"  mean target error before update  : {target_before:.6f}")
    print(f"  mean target error after update   : {target_after:.6f}")
    print()

    print("Modular shrinking path:")
    for name, msn in zip(("image", "video", "audio", "text"), msns):
        print(f"  {name:<5} MSN={msn:.6f}")
    print()

    print("Descriptor movement:")
    for name, path, after in zip(("image", "video", "audio", "text"), paths, updated_desc):
        before = descriptor(name, path[0])
        print(f"  {before.name:<5} -> {after.name:<10} delta={np.linalg.norm(after.vec() - before.vec()):.6f}")
    print()

    print("Final descriptors:")
    print("  name          amp      phase      freq      comp       sal      snr")
    for item in updated_desc:
        print(
            f"  {item.name:<10} {item.amplitude:8.4f} {item.phase:9.4f} "
            f"{item.frequency:9.4f} {item.compression:9.4f} {item.salience:9.4f} {item.snr:8.4f}"
        )
    print()

    # Regression target: synthetic stability score built from final descriptor quality.
    features = []
    target_score = []
    for item, msn in zip(updated_desc, msns):
        features.append([item.amplitude, item.compression, item.salience, item.snr, msn])
        target_score.append(item.snr + 2.0 * item.compression - 4.0 * msn + 0.35 * item.salience * item.salience)
    beta, mse = fit_linear_regression(np.array(features), np.array(target_score))

    print("Linear regression calibration:")
    print("  target = stability_score = snr + 2*compression - 4*MSN + 0.35*salience^2")
    print("  beta   =", np.array2string(beta, precision=5))
    print(f"  mse    = {mse:.8f}")
    print()

    before_mean, after_mean, improvement_mean, stress_mse = stress_test(rng, n=80)
    print("Stress test over 80 random Chladni fields:")
    print(f"  mean target error before update: {before_mean:.6f}")
    print(f"  mean target error after update : {after_mean:.6f}")
    print(f"  mean target-error improvement  : {improvement_mean:.6f}")
    print(f"  held-out regression mse         : {stress_mse:.8f}")


if __name__ == "__main__":
    main()
