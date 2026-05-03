"""
OPU Core -- three-loop orchestration (v2)

v2 fixes:
  - sigma tracking uses policy.state_changed (no more tuple parsing)
  - suppress_evict on quality degradation: filters out low-priority evict actions
  - action accountability: each action's trace aggregated into summary

Observe -> Ledger -> [Policy A + B + C] -> Merge -> Act
"""

from __future__ import annotations
from collections import deque
from typing import Any

from .actions import OPUAction, Health, Noop
from .config import OPUConfig
from .stats import StepStats
from .policies.base import PolicyBase
from .policies.resource import ResourcePolicy
from .policies.friction import FrictionPolicy
from .policies.quality import QualityPolicy


class OPU:
    """
    Orchestration Processing Unit v3 (separated)

    Three loops:
      A. ResourcePolicy  -- resource governance (tighten/relax)
      B. FrictionPolicy  -- friction recovery (mu/tau -> gate/prefetch)
      C. QualityPolicy   -- quality gatekeeper (entropy/repeat -> escalation)
    """

    def __init__(self, cfg: OPUConfig,
                 policies: list[PolicyBase] | None = None):
        self.cfg = cfg
        self.step = 0

        # Default three policies
        if policies is None:
            self._resource = ResourcePolicy(cfg)
            self._friction = FrictionPolicy(cfg)
            self._quality = QualityPolicy(cfg)
            self._policies: list[PolicyBase] = [
                self._resource, self._friction, self._quality]
        else:
            self._policies = policies
            self._resource = next((p for p in policies
                                   if isinstance(p, ResourcePolicy)), None)
            self._friction = next((p for p in policies
                                   if isinstance(p, FrictionPolicy)), None)
            self._quality = next((p for p in policies
                                  if isinstance(p, QualityPolicy)), None)

        # EMA Ledger
        self._ema: dict[str, float] = {}
        self._cooldown_left: int = 0

        # sigma (policy jitter) tracking
        self._policy_changes: int = 0
        self._policy_change_window: deque = deque(maxlen=100)

        # History
        self._history: list[StepStats] = []
        self._last_actions: list[OPUAction] = []

    # Hook 1: Observe

    def observe(self, stats: StepStats) -> None:
        """Receive one step's observable signals, update EMA ledger."""
        self._history.append(stats)
        a = self.cfg.ema_alpha

        for key in ('step_time_s', 'hot_pressure', 'rebuild_time_s'):
            val = getattr(stats, key, 0.0)
            self._ema[key] = self._ema.get(key, val) * (1-a) + val * a

        faults_f = float(stats.faults)
        self._ema['faults'] = self._ema.get('faults', faults_f) * (1-a) + faults_f * a

        # mu/tau tracking
        self._ema['mu'] = self._ema.get('mu', 0.0) * (1-a) + stats.wait_time_s * a
        self._ema['tau'] = self._ema.get('tau', 0.0) * (1-a) + stats.rebuild_cost_s * a

        # aperture loss
        self._ema['aperture_loss'] = (self._ema.get('aperture_loss', 0.0) * (1-a)
                                      + stats.aperture_loss_s * a)

        # quality signal
        if self._quality is not None:
            self._quality.update_ema(stats.quality_score, a)
            self._ema['quality'] = self._quality.quality_ema

        if self._cooldown_left > 0:
            self._cooldown_left -= 1

        self.step += 1

    # Hook 2: Decide

    def decide(self) -> list[OPUAction]:
        """Merge three-loop policies, output action list."""
        if not self.cfg.enabled:
            return [Noop()]

        ledger = {
            **self._ema,
            'step': self.step,
            'cooldown_left': self._cooldown_left,
        }

        all_acts: list[OPUAction] = []
        any_state_changed = False
        suppress_evict = False

        # Quality loop (highest priority, not subject to cooldown)
        if self._quality is not None:
            q_acts = self._quality.evaluate(ledger)
            all_acts.extend(q_acts)
            if self._quality.state_changed:
                any_state_changed = True
            # On quality degradation: lower friction gate + suppress subsequent evict
            if q_acts:
                if self._friction is not None and self._friction.gate_level > 0:
                    self._friction.gate_level = max(0, self._friction.gate_level - 1)
                self._cooldown_left = self.cfg.cooldown_steps
                # Check whether to suppress evict
                for qa in q_acts:
                    if qa.payload.get('suppress_evict', False):
                        suppress_evict = True

        # Friction loop
        if self._friction is not None:
            f_acts = self._friction.evaluate(ledger)
            all_acts.extend(f_acts)
            if self._friction.state_changed:
                any_state_changed = True

        # Resource loop
        if self._resource is not None:
            r_acts = self._resource.evaluate(ledger)
            all_acts.extend(r_acts)
            if self._resource.state_changed:
                any_state_changed = True
            if r_acts and any(a.type in ('evict', 'tighten') for a in r_acts):
                self._cooldown_left = self.cfg.cooldown_steps

        # suppress_evict: filter out evict actions on quality degradation
        if suppress_evict:
            all_acts = [a for a in all_acts
                        if a.type != 'evict' or a.source == 'QualityPolicy']

        # sigma tracking
        if any_state_changed:
            self._policy_changes += 1
            self._policy_change_window.append(1)
        else:
            self._policy_change_window.append(0)

        # Periodic health
        if self.step % self.cfg.health_interval == 0:
            all_acts.append(Health(
                step=self.step,
                ema=dict(self._ema),
                sigma=self.policy_jitter,
            ))

        self._last_actions = all_acts
        return all_acts

    # Convenience: tick = observe + decide

    def tick(self, stats: StepStats) -> list[OPUAction]:
        """One full loop step: observe -> decide -> return actions"""
        self.observe(stats)
        return self.decide()

    # KPI

    @property
    def policy_jitter(self) -> float:
        """sigma = policy_changes / 100_steps"""
        if len(self._policy_change_window) == 0:
            return 0.0
        return sum(self._policy_change_window) / len(self._policy_change_window)

    @property
    def quality_ema(self) -> float:
        if self._quality is not None:
            return self._quality.quality_ema
        return self._ema.get('quality', 1.0)

    @property
    def quality_alarm(self) -> bool:
        if self._quality is not None:
            return self._quality.alarm
        return False

    @property
    def gate_level(self) -> int:
        if self._friction is not None:
            return self._friction.gate_level
        return 0

    def action_traces(self) -> list[str]:
        """Action accountability list from the most recent decide()"""
        return [a.trace for a in self._last_actions if a.type not in ('noop', 'health')]

    def summary(self) -> str:
        p = self._ema.get('hot_pressure', 0.0)
        f = self._ema.get('faults', 0.0)
        mu = self._ema.get('mu', 0.0)
        tau = self._ema.get('tau', 0.0)
        q = self.quality_ema
        g = self.gate_level
        return (f"[OPU] step={self.step}, p={p:.1%}, "
                f"f={f:.1f}, μ={mu:.3f}, τ={tau:.3f}, "
                f"σ={self.policy_jitter:.2f}, q={q:.2f}, "
                f"gate={g}, cd={self._cooldown_left}")
