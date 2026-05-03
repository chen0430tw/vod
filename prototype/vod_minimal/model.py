"""Minimal shared VOD field updater."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .projections import smooth


@dataclass(frozen=True)
class MinimalVOD:
    """A tiny shared update rule for all toy media projections.

    This is not VDiT. It is a testable stand-in for Algorithm 3:

        du/dtau = div(D * grad(u)) + R

    The target projection is only used because this minimal prototype is a
    supervised denoising sanity check.
    """

    diffusion: float = 0.55
    reaction: float = 0.18
    step_size: float = 0.9
    steps: int = 12

    def update_path(self, noisy: np.ndarray, target_projection: np.ndarray) -> list[np.ndarray]:
        current = noisy.astype(np.float64)
        target = target_projection.astype(np.float64)
        path = [current]

        for _ in range(self.steps):
            diffusivity = 0.15 + 0.85 * np.abs(target) / (np.max(np.abs(target)) + 1e-9)
            diffusion_term = diffusivity * (smooth(current) - current)
            reaction_term = target - current
            delta = self.diffusion * diffusion_term + self.reaction * reaction_term
            current = current + self.step_size * delta
            path.append(current)

        return path

    def denoise_views(
        self,
        noisy_views: dict[str, np.ndarray],
        target_views: dict[str, np.ndarray],
    ) -> tuple[dict[str, np.ndarray], dict[str, list[np.ndarray]]]:
        paths = {
            name: self.update_path(noisy_views[name], target_views[name])
            for name in noisy_views
        }
        return {name: path[-1] for name, path in paths.items()}, paths
