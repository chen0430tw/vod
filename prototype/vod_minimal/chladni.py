"""Synthetic Chladni-like field generation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


EPS = 1e-9


@dataclass(frozen=True)
class Boundary:
    size: int = 64
    modes: tuple[tuple[int, int, float], ...] = ((2, 3, 0.7), (5, 4, 0.45), (7, 2, 0.25))


def chladni_field(boundary: Boundary) -> np.ndarray:
    size = boundary.size
    y, x = np.mgrid[0:size, 0:size]
    x = x / max(1, size - 1)
    y = y / max(1, size - 1)
    field = np.zeros((size, size), dtype=np.float64)

    for mx, my, weight in boundary.modes:
        a = np.cos(np.pi * mx * x) * np.cos(np.pi * my * y)
        b = np.cos(np.pi * my * x) * np.cos(np.pi * mx * y)
        field += weight * (a - b)

    return field / (np.max(np.abs(field)) + EPS)


def random_boundary(rng: np.random.Generator, size: int = 64, n_modes: int = 3) -> Boundary:
    modes: list[tuple[int, int, float]] = []
    for _ in range(n_modes):
        mx = int(rng.integers(2, 9))
        my = int(rng.integers(2, 9))
        weight = float(rng.uniform(0.15, 0.9))
        modes.append((mx, my, weight))
    return Boundary(size=size, modes=tuple(modes))


def random_chladni_field(rng: np.random.Generator, size: int = 64, n_modes: int = 3) -> np.ndarray:
    return chladni_field(random_boundary(rng, size=size, n_modes=n_modes))
