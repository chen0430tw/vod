"""
OPU Actuators -- action executor interface

VA100 implements this interface; OPU does not touch implementation details.
OPU only outputs actions, it does not directly manipulate internal objects.
"""

from __future__ import annotations
from typing import Protocol, runtime_checkable

from .actions import OPUAction


@runtime_checkable
class ActionExecutor(Protocol):
    """
    VA100 engine implements this protocol.
    OPU calls execute(actions), engine translates to:
      - vram promote/evict/prefetch
      - kv_adapter materialize strategy adjustments
      - ghost move queue window adjustments

    Like "airport ground crew": OPU issues orders, crew executes, plane (inference) is unaware.
    """

    def execute_evict(self, target_free_ratio: float, **kw) -> None: ...
    def execute_prefetch(self, window: int, coalesce: bool = False, **kw) -> None: ...
    def execute_tighten(self, hot_ratio: float, warm_ratio: float, **kw) -> None: ...
    def execute_relax(self, hot_ratio: float, **kw) -> None: ...
    def execute_gate_compute(self, gate_level: int, **kw) -> None: ...
    def execute_quality_escalation(self, **kw) -> None: ...


def dispatch_actions(executor: ActionExecutor,
                     actions: list[OPUAction]) -> None:
    """
    Generic action dispatcher.
    Executes in descending priority order, translating OPUAction to executor method calls.
    """
    for a in sorted(actions, key=lambda x: -x.priority):
        p = a.payload
        if a.type == 'evict':
            executor.execute_evict(
                target_free_ratio=p.get('target_free_ratio', 0.1))
        elif a.type == 'prefetch':
            executor.execute_prefetch(
                window=p.get('window', 2),
                coalesce=p.get('coalesce', False))
        elif a.type == 'tighten':
            executor.execute_tighten(
                hot_ratio=p.get('hot_ratio', 0.5),
                warm_ratio=p.get('warm_ratio', 0.3))
        elif a.type == 'relax':
            executor.execute_relax(
                hot_ratio=p.get('hot_ratio', 0.6))
        elif a.type == 'gate_compute':
            executor.execute_gate_compute(
                gate_level=p.get('gate_level', 1))
        elif a.type == 'quality_escalation':
            executor.execute_quality_escalation()
        # health / noop: bookkeeping only, no execution
