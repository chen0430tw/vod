"""
OPU Policy Base -- policy base class (v2)

v2 fixes:
  - evaluate() strictly returns list[OPUAction], no tuple returns allowed
  - state_changed exposed via property, core.py reads after evaluate()
  - Added name attribute for action accountability
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..actions import OPUAction
    from ..config import OPUConfig


class PolicyBase(ABC):
    """
    Policy base class.

    Each policy handles one loop's decisions:
      ResourcePolicy  -> resource loop (tighten/relax)
      FrictionPolicy  -> friction loop (gate/prefetch)
      QualityPolicy   -> quality loop (escalation)

    Interface contract:
      - evaluate(ledger) -> list[OPUAction]  (strictly returns list, no tuples)
      - state_changed: bool  (read after evaluate, used for sigma tracking)
      - name: str  (used for action accountability)
    """

    def __init__(self, cfg: 'OPUConfig'):
        self.cfg = cfg
        self._state_changed: bool = False

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def state_changed(self) -> bool:
        """Whether the last evaluate() caused internal state change (for sigma tracking)"""
        return self._state_changed

    @abstractmethod
    def evaluate(self, ledger: dict[str, Any]) -> list['OPUAction']:
        """
        Output action list based on ledger (EMA state).

        Important: must return list[OPUAction], no tuples allowed.
        If policy state changed, set self._state_changed = True.
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset policy internal state"""
        ...
