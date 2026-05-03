from __future__ import annotations

from vod_minimal.opu_adapter import (
    VODControlState,
    apply_opu_actions,
    quality_from_metrics,
    suggest_from_metrics,
)
from opu.actions import Evict, Prefetch, QualityEscalation, Relax


def test_quality_from_metrics_rewards_improvement_and_success():
    good = {
        "mean_before": 10.0,
        "mean_after": 1.0,
        "success_rate": 1.0,
    }
    bad = {
        "mean_before": 10.0,
        "mean_after": 9.0,
        "success_rate": 0.0,
    }
    assert quality_from_metrics(good) > quality_from_metrics(bad)
    assert 0.0 <= quality_from_metrics(good) <= 1.0


def test_quality_escalation_increases_steps():
    state = VODControlState(steps=1, max_tokens=512)
    suggestion = apply_opu_actions(
        state,
        [QualityEscalation(quality_score=0.2, reason="test", source="test")],
    )
    assert suggestion.steps == 2
    assert suggestion.quality_strength > state.quality_strength


def test_resource_actions_reduce_max_tokens():
    state = VODControlState(steps=1, max_tokens=512)
    suggestion = apply_opu_actions(
        state,
        [Evict(pressure=0.95, reason="test", source="test")],
    )
    assert suggestion.max_tokens < state.max_tokens


def test_relax_increases_max_tokens():
    state = VODControlState(steps=1, max_tokens=512)
    suggestion = apply_opu_actions(state, [Relax(reason="test", source="test")])
    assert suggestion.max_tokens > state.max_tokens


def test_suggest_from_low_quality_metrics_increases_steps():
    metrics = {
        "mean_before": 10.0,
        "mean_after": 9.0,
        "success_rate": 0.0,
    }
    state = VODControlState(steps=1, max_tokens=512)
    suggestion = suggest_from_metrics(metrics, state)
    assert suggestion.steps > state.steps


def test_prefetch_reduces_step_size():
    state = VODControlState(step_size=1.0)
    suggestion = apply_opu_actions(
        state,
        [Prefetch(reason="test", source="test")],
    )
    assert suggestion.step_size < state.step_size
