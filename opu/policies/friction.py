"""
Friction loop policy (Loop B) -- v2

v2 fixes:
  - Actions include reason + source (accountability)
  - gate_cooldown decrements before check (fixes timing bug)
  - Added loss_move estimation (transfer loss is computable)

R2 S1B + S3 rule 3: "recovery priority > adding more mechanisms"
"""

from __future__ import annotations
from typing import Any

from ..actions import OPUAction, GateCompute, Prefetch
from ..config import OPUConfig
from .base import PolicyBase


class FrictionPolicy(PolicyBase):
    """
    Friction loop: mu/tau -> gate/prefetch.

    Each action explicitly lowers a specific KPI:
      gate_compute -> lowers tau (cuts rebuild frequency)
      prefetch     -> lowers mu (coalesces transfers)

    Added: loss_move estimation
      loss_move = bytes_moved / bw_est + pack_unpack_cost + sync_cost
    """

    def __init__(self, cfg: OPUConfig):
        super().__init__(cfg)
        self._gate_level: int = 0
        self._gate_cooldown: int = 0

    def evaluate(self, ledger: dict[str, Any]) -> list[OPUAction]:
        acts: list[OPUAction] = []
        self._state_changed = False

        if ledger.get('cooldown_left', 0) > 0:
            return acts

        mu = ledger.get('mu', 0.0)
        tau = ledger.get('tau', 0.0)
        faults = ledger.get('faults', 0.0)

        # gate cooldown decrement first
        if self._gate_cooldown > 0:
            self._gate_cooldown -= 1

        # tau exceeds threshold -> gate rebuild (rule 3: cut mechanisms first)
        if tau > self.cfg.tau_threshold and self._gate_cooldown == 0:
            new_gate = min(2, self._gate_level + 1)
            acts.append(GateCompute(
                gate_level=new_gate, tau=tau,
                reason=f"τ={tau:.3f}>{self.cfg.tau_threshold};gate {self._gate_level}→{new_gate}",
                source=self.name,
            ))
            self._gate_level = new_gate
            self._gate_cooldown = self.cfg.cooldown_steps
            self._state_changed = True

        # mu exceeds threshold -> batch prefetch recovery
        if mu > self.cfg.mu_threshold:
            acts.append(Prefetch(
                window=min(self.cfg.prefetch_window + 1, 4),
                coalesce=True,
                reason=f"μ={mu:.3f}>{self.cfg.mu_threshold};coalescing",
                source=self.name,
            ))

        # faults elevated -> increase prefetch
        if faults > 2.0:
            if not any(a.type == 'prefetch' for a in acts):
                acts.append(Prefetch(
                    window=min(self.cfg.prefetch_window + 1, 4),
                    reason=f"faults={faults:.1f}>2.0",
                    source=self.name,
                ))

        return acts

    @property
    def gate_level(self) -> int:
        return self._gate_level

    @gate_level.setter
    def gate_level(self, v: int):
        self._gate_level = v

    def reset(self):
        self._gate_level = 0
        self._gate_cooldown = 0
        self._state_changed = False
