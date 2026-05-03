"""Pytest config: make `vod_minimal` importable when running from anywhere.

We don't ship the prototype as an installable package, so the test runner
needs to know that `D:\\VOD\\prototype` is on sys.path before tests collect.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROTOTYPE_ROOT = Path(__file__).resolve().parent.parent
if str(_PROTOTYPE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROTOTYPE_ROOT))
