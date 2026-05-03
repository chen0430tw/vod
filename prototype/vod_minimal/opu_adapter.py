"""OPU adapter for VOD prototype control.

OPU is deliberately kept outside the minimal generation core. This adapter
translates VOD metrics into OPU StepStats and translates OPU actions into
sampling/training control suggestions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any


# OPU lives at D:\VOD\opu while this module lives at
# D:\VOD\prototype\vod_minimal. Add D:\VOD for this adapter only.
_VOD_ROOT = Path(__file__).resolve().parents[2]
if str(_VOD_ROOT) not in sys.path:
    sys.path.insert(0, str(_VOD_ROOT))

from opu.actions import OPUAction  # noqa: E402
from opu.config import OPUConfig  # noqa: E402
from opu.core import OPU  # noqa: E402
from opu.stats import StepStats  # noqa: E402


@dataclass(frozen=True)
class VODControlState:
    """Current controllable VOD runtime knobs."""

    steps: int = 1
    step_size: float = 1.0
    max_tokens: int = 512
    quality_strength: float = 1.0


@dataclass(frozen=True)
class VODControlSuggestion:
    """Result of applying OPU actions to VOD runtime knobs."""

    steps: int
    step_size: float
    max_tokens: int
    quality_strength: float
    actions: tuple[str, ...]
    summary: str


def quality_from_metrics(metrics: dict[str, float]) -> float:
    """Map projection metrics to OPU's 0..1 quality_score."""

    before = float(metrics.get("mean_before", 0.0))
    after = float(metrics.get("mean_after", before))
    success = float(metrics.get("success_rate", 0.0))
    if before <= 1e-9:
        improvement_quality = 1.0 if after <= before else 0.0
    else:
        improvement_quality = max(0.0, min(1.0, 1.0 - after / before))
    return max(0.0, min(1.0, 0.65 * improvement_quality + 0.35 * success))


def stats_from_metrics(
    metrics: dict[str, float],
    *,
    step: int = 0,
    step_time_s: float = 0.0,
    hot_pressure: float = 0.0,
    max_tokens: int = 512,
) -> StepStats:
    """Build OPU StepStats from VOD train/eval metrics."""

    before = float(metrics.get("mean_before", 0.0))
    after = float(metrics.get("mean_after", before))
    quality = quality_from_metrics(metrics)
    residual = max(0.0, after / max(before, 1e-9))

    return StepStats(
        step=step,
        step_time_s=step_time_s,
        hot_pressure=max(0.0, min(1.0, hot_pressure)),
        gpu_alloc_peak_mb=float(max_tokens),
        rebuild_time_s=residual * 0.02,
        wait_time_s=max(0.0, hot_pressure - 0.80) * 0.05,
        rebuild_cost_s=residual * 0.02,
        faults=0 if quality > 0.7 else 3,
        logits_entropy=after,
        repeat_rate=max(0.0, 1.0 - float(metrics.get("success_rate", 0.0))),
        quality_score=quality,
    )


def apply_opu_actions(
    state: VODControlState,
    actions: list[OPUAction],
    *,
    min_steps: int = 1,
    max_steps: int = 8,
    min_step_size: float = 0.25,
    min_tokens: int = 128,
) -> VODControlSuggestion:
    """Translate OPU actions into VOD runtime-control suggestions."""

    steps = state.steps
    step_size = state.step_size
    max_tokens = state.max_tokens
    quality_strength = state.quality_strength
    traces: list[str] = []

    for action in sorted(actions, key=lambda a: -a.priority):
        traces.append(action.trace)
        if action.type == "quality_escalation":
            steps = min(max_steps, steps + 1)
            quality_strength = min(2.0, quality_strength * 1.20)
        elif action.type == "gate_compute":
            gate_level = int(action.payload.get("gate_level", 1))
            steps = max(min_steps, steps - gate_level)
        elif action.type in {"tighten", "evict"}:
            max_tokens = max(min_tokens, int(max_tokens * 0.75))
        elif action.type == "relax":
            max_tokens = int(max_tokens * 1.10)
        elif action.type == "prefetch":
            step_size = max(min_step_size, step_size * 0.90)

    return VODControlSuggestion(
        steps=steps,
        step_size=step_size,
        max_tokens=max_tokens,
        quality_strength=quality_strength,
        actions=tuple(traces),
        summary=(
            f"steps={steps}, step_size={step_size:.3f}, "
            f"max_tokens={max_tokens}, quality_strength={quality_strength:.3f}"
        ),
    )


def suggest_from_metrics(
    metrics: dict[str, float],
    state: VODControlState,
    *,
    cfg: OPUConfig | None = None,
    step: int = 0,
    hot_pressure: float = 0.0,
) -> VODControlSuggestion:
    """One-shot VOD metrics -> OPU actions -> VOD control suggestion."""

    # For a one-shot checkpoint controller, use raw metric quality directly.
    # The default OPU EMA alpha is for online per-step loops and would dilute a
    # single checkpoint observation toward the initial quality=1.0.
    opu = OPU(cfg or OPUConfig(ema_alpha=1.0))
    stats = stats_from_metrics(metrics, step=step, hot_pressure=hot_pressure, max_tokens=state.max_tokens)
    actions = opu.tick(stats)
    return apply_opu_actions(state, actions)


def checkpoint_metrics(checkpoint: dict[str, Any], split: str = "test") -> dict[str, float]:
    key = f"{split}_metrics"
    metrics = checkpoint.get(key)
    if not isinstance(metrics, dict):
        raise KeyError(f"checkpoint does not contain dict {key!r}")
    return {k: float(v) for k, v in metrics.items()}
