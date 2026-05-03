"""
Quality loop policy (Loop C) -- v2

v2 fixes:
  - QualityEscalation includes reason + source
  - suppress_evict=True on quality degradation (suppresses subsequent evictions)
  - state_changed used for sigma tracking

R2 S1C + S5:
  "Once quality signal degrades -> force promote critical tile precision/residency/lower gate strength"
"""

from __future__ import annotations
from typing import Any

from ..actions import OPUAction, QualityEscalation
from ..config import OPUConfig
from .base import PolicyBase


class QualityPolicy(PolicyBase):
    """
    Quality gatekeeper.

    - quality_score EMA below threshold -> alarm + escalation + suppress_evict
    - Automatically clears alarm on recovery
    - On degradation, lowers gate_level (via core.py policy interaction)
    """

    def __init__(self, cfg: OPUConfig):
        super().__init__(cfg)
        self._quality_ema: float = 1.0
        self._alarm: bool = False

    def evaluate(self, ledger: dict[str, Any]) -> list[OPUAction]:
        acts: list[OPUAction] = []
        self._state_changed = False

        # Not subject to cooldown: quality gatekeeper has highest priority
        quality = ledger.get('quality', self._quality_ema)

        if quality < self.cfg.quality_alarm_threshold:
            if not self._alarm:
                self._state_changed = True
            self._alarm = True
            acts.append(QualityEscalation(
                quality_score=quality,
                suppress_evict=True,
                reason=f"q_ema={quality:.3f}<{self.cfg.quality_alarm_threshold};promote+suppress_evict",
                source=self.name,
            ))

        elif quality > self.cfg.quality_recover_threshold and self._alarm:
            self._alarm = False
            self._state_changed = True

        return acts

    def update_ema(self, quality_score: float, alpha: float = 0.15):
        """Called by OPU core to update quality EMA"""
        self._quality_ema = self._quality_ema * (1 - alpha) + quality_score * alpha

    @property
    def alarm(self) -> bool:
        return self._alarm

    @property
    def quality_ema(self) -> float:
        return self._quality_ema

    def reset(self):
        self._quality_ema = 1.0
        self._alarm = False
        self._state_changed = False
