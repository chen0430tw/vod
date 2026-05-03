"""
Resource loop policy (Loop A) -- v2

v2 fixes:
  - evaluate() returns list[OPUAction] (no more tuple returns, conforms to ABC contract)
  - State changes exposed via self._state_changed
  - Evict/Tighten/Relax actions include reason + source

Threshold + hysteresis + cooldown + rate limiter, all four required.
"""

from __future__ import annotations
from typing import Any

from ..actions import OPUAction, Evict, Tighten, Relax
from ..config import OPUConfig
from .base import PolicyBase


class ResourcePolicy(PolicyBase):
    """
    Resource loop: pressure-based tighten/relax/evict.

    State:
      _pressure_level: 'normal' | 'high' (hysteresis)
      _tighten_count:  consecutive tighten count (rate limiter)
      _relax_count:    consecutive relax count
    """

    def __init__(self, cfg: OPUConfig):
        super().__init__(cfg)
        self._pressure_level = 'normal'
        self._tighten_count = 0
        self._relax_count = 0
        self._effective_hot_ratio = cfg.hot_ratio
        self._effective_warm_ratio = cfg.warm_ratio

    def evaluate(self, ledger: dict[str, Any]) -> list[OPUAction]:
        acts: list[OPUAction] = []
        self._state_changed = False

        if ledger.get('cooldown_left', 0) > 0:
            return acts

        pressure = ledger.get('hot_pressure', 0.0)
        faults = ledger.get('faults', 0.0)
        mu = ledger.get('mu', 0.0)
        tau = ledger.get('tau', 0.0)

        # Hysteresis: pressure state transition
        old = self._pressure_level
        if pressure >= self.cfg.high_water:
            self._pressure_level = 'high'
        elif pressure <= self.cfg.low_water:
            self._pressure_level = 'normal'

        if self._pressure_level != old:
            self._state_changed = True

        # High pressure: evict + tighten (rate limited)
        if self._pressure_level == 'high':
            reason = (f"pressure={pressure:.2f}>{self.cfg.high_water}")
            acts.append(Evict(
                target_free_ratio=pressure - self.cfg.low_water,
                pressure=pressure,
                reason=reason,
                source=self.name,
            ))
            if self._tighten_count < self.cfg.max_tighten_streak:
                new_hot = max(0.2, self._effective_hot_ratio * 0.85)
                new_warm = max(0.1, self._effective_warm_ratio * 0.90)
                acts.append(Tighten(
                    hot_ratio=new_hot, warm_ratio=new_warm,
                    reason=f"memory;streak={self._tighten_count+1}/{self.cfg.max_tighten_streak}",
                    source=self.name,
                ))
                self._effective_hot_ratio = new_hot
                self._effective_warm_ratio = new_warm
                self._tighten_count += 1
                self._relax_count = 0

        # Normal + all KPIs healthy: relax (rate limited)
        elif (self._pressure_level == 'normal'
              and faults < 0.5 and mu < 0.05 and tau < 0.05):
            if self._relax_count < self.cfg.max_relax_streak:
                new_hot = min(0.7, self._effective_hot_ratio * 1.05)
                acts.append(Relax(
                    hot_ratio=new_hot,
                    reason=f"healthy;p={pressure:.2f},f={faults:.1f},μ={mu:.3f},τ={tau:.3f}",
                    source=self.name,
                ))
                self._effective_hot_ratio = new_hot
                self._relax_count += 1
                self._tighten_count = 0

        return acts

    def reset(self):
        self._pressure_level = 'normal'
        self._tighten_count = 0
        self._relax_count = 0
        self._effective_hot_ratio = self.cfg.hot_ratio
        self._effective_warm_ratio = self.cfg.warm_ratio
        self._state_changed = False
