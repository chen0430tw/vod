"""
OPU Actions -- stable ABI (v2)

v2 changes:
  - OPUAction adds reason (cause code) + source (originating policy)
  - Evict/Tighten/Relax support reason codes (memory/friction/quality)
  - Evict/Prefetch support tile_ids batch operations
  - All constructors unified source parameter for action accountability

ABI contract: new fields all have defaults, old engines won't break.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


@dataclass
class OPUAction:
    """OPU output atomic action."""
    type: str
    payload: dict[str, Any] = field(default_factory=dict)
    priority: int = 0
    # v2: accountability fields
    reason: str = ''       # trigger cause (e.g. "pressure=0.92>high_water=0.85")
    source: str = ''       # originating policy (e.g. "ResourcePolicy", "FrictionPolicy")

    def __repr__(self):
        src = f", src={self.source}" if self.source else ""
        rsn = f", reason={self.reason}" if self.reason else ""
        return f"OPUAction({self.type}, p={self.priority}{src}{rsn})"

    @property
    def trace(self) -> str:
        """Accountability string: source:type:reason"""
        return f"{self.source}:{self.type}:{self.reason}"


# Named action constructors

def Evict(target_free_ratio: float = 0.1, pressure: float = 0.0,
          tile_ids: list[str] | None = None,
          reason: str = '', source: str = '', **kw) -> OPUAction:
    """Evict hot tier tiles -> warm/cold. Lowers pressure, may raise mu."""
    return OPUAction(
        type='evict', priority=100,
        reason=reason, source=source,
        payload={'target_free_ratio': target_free_ratio,
                 'pressure': pressure,
                 'tile_ids': tile_ids or [], **kw})


def Prefetch(window: int = 2, coalesce: bool = False,
             tile_ids: list[str] | None = None,
             reason: str = '', source: str = '', **kw) -> OPUAction:
    """Prefetch warm/cold tiles -> hot. Lowers faults, may raise mu."""
    return OPUAction(
        type='prefetch', priority=80,
        reason=reason, source=source,
        payload={'window': window, 'coalesce': coalesce,
                 'tile_ids': tile_ids or [], **kw})


def Tighten(hot_ratio: float = 0.5, warm_ratio: float = 0.3,
            reason: str = '', source: str = '', **kw) -> OPUAction:
    """Shrink hot/warm ratios. reason: 'memory'|'friction'|'quality'"""
    return OPUAction(
        type='tighten', priority=90,
        reason=reason, source=source,
        payload={'hot_ratio': hot_ratio, 'warm_ratio': warm_ratio, **kw})


def Relax(hot_ratio: float = 0.6,
          reason: str = '', source: str = '', **kw) -> OPUAction:
    """Relax hot ratio. Only when all KPIs are healthy."""
    return OPUAction(
        type='relax', priority=30,
        reason=reason, source=source,
        payload={'hot_ratio': hot_ratio, **kw})


def GateCompute(gate_level: int = 1, tau: float = 0.0,
                reason: str = '', source: str = '', **kw) -> OPUAction:
    """Gate rebuild frequency. Lowers tau, may raise error.
    level 0=normal, 1=halved, 2=critical only."""
    return OPUAction(
        type='gate_compute', priority=85,
        reason=reason, source=source,
        payload={'gate_level': gate_level, 'tau': tau, **kw})


def QualityEscalation(quality_score: float = 0.0,
                      suppress_evict: bool = False,
                      reason: str = '', source: str = '', **kw) -> OPUAction:
    """On quality degradation, promote critical tile precision/residency.
    suppress_evict=True also suppresses subsequent evict actions."""
    return OPUAction(
        type='quality_escalation', priority=120,
        reason=reason, source=source,
        payload={'quality_score': quality_score,
                 'action': 'promote_critical',
                 'suppress_evict': suppress_evict, **kw})


def Health(**payload) -> OPUAction:
    """Periodic health check. Bookkeeping only."""
    return OPUAction(type='health', priority=5,
                     source='OPU', payload=payload)


def Noop() -> OPUAction:
    return OPUAction(type='noop', priority=0)
