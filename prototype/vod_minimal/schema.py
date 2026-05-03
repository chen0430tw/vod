"""Shared metadata and reporting schema for VOD prototype runs."""

from __future__ import annotations

from typing import Any


CORE_CONTRACT_VERSION = "vod-minimal-core-v1"
SCHEMA_VERSION = "vod-checkpoint-schema-v1"


def canonical_metrics(metrics: dict[str, float]) -> dict[str, float]:
    """Return metrics in stable key order, preserving optional extras."""

    ordered_keys = ("loss", "mean_before", "mean_after", "mean_improvement", "success_rate")
    result: dict[str, float] = {}
    for key in ordered_keys:
        if key in metrics:
            result[key] = float(metrics[key])
    for key in sorted(metrics):
        if key not in result:
            result[key] = float(metrics[key])
    return result


def checkpoint_payload(
    *,
    state_dict: Any,
    model_type: str,
    train_args: dict[str, Any],
    train_metrics: dict[str, float],
    test_metrics: dict[str, float],
    config: dict[str, Any] | None = None,
    best_epoch: int | None = None,
) -> dict[str, Any]:
    """Build the stable checkpoint payload shared by prototype trainers."""

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "core_contract_version": CORE_CONTRACT_VERSION,
        "model_type": model_type,
        "state_dict": state_dict,
        "train_args": dict(train_args),
        "train_metrics": canonical_metrics(train_metrics),
        "test_metrics": canonical_metrics(test_metrics),
    }
    if config is not None:
        payload["config"] = dict(config)
    if best_epoch is not None:
        payload["best_epoch"] = int(best_epoch)
    return payload


def print_metrics_block(title: str, metrics: dict[str, float]) -> None:
    print(title)
    for key, value in canonical_metrics(metrics).items():
        print(f"  {key:<18} {value:.6f}")
