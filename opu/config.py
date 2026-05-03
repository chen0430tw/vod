"""
OPU Config -- governance unit standalone configuration
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class OPUConfig:
    """OPU configuration (decoupled from inference engine config)"""

    enabled: bool = True
    ema_alpha: float = 0.15

    # Resource loop
    high_water: float = 0.85
    low_water: float = 0.6
    cooldown_steps: int = 6
    max_tighten_streak: int = 5
    max_relax_streak: int = 3

    # Friction loop
    mu_threshold: float = 0.1
    tau_threshold: float = 0.15
    prefetch_window: int = 2

    # Quality loop
    quality_alarm_threshold: float = 0.5
    quality_recover_threshold: float = 0.7

    # Resource ratios (OPU can dynamically adjust)
    hot_ratio: float = 0.6
    warm_ratio: float = 0.3

    # Health check
    health_interval: int = 16

    @classmethod
    def from_infer_config(cls, icfg) -> 'OPUConfig':
        """Extract OPU-related fields from InferConfig"""
        return cls(
            enabled=getattr(icfg, 'opu_enabled', True),
            high_water=getattr(icfg, 'opu_high_water', 0.85),
            low_water=getattr(icfg, 'opu_low_water', 0.6),
            cooldown_steps=getattr(icfg, 'opu_cooldown', 6),
            prefetch_window=getattr(icfg, 'prefetch_window', 2),
            hot_ratio=getattr(icfg, 'hot_ratio', 0.6),
            warm_ratio=getattr(icfg, 'warm_ratio', 0.3),
        )
