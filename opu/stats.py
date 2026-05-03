"""
OPU StepStats -- per-step observable signals (v2)

v2 changes:
  - Added tile-level stats (tiles_hot/warm/cold, transfer counts)
  - Added ghost_move events (count/bytes/overhead)
  - Added stall_reason enum
  - Added phase marker (prefill/decode)
  - Added action accountability fields
  - All transfer/timing fields are explicitly "per-step delta", not cumulative

Aperture rule 1: "aperture must be measurable"
loss = wait_time + copy_bytes/bw + unpack_cost + rebuild_cost
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class StallReason(Enum):
    """Per-step main bottleneck reason (mutually exclusive, takes max time item)"""
    NONE = 'none'
    KV_TRANSFER = 'kv_transfer'
    WEIGHT_TRANSFER = 'weight_transfer'
    PACK_UNPACK = 'pack_unpack'
    REBUILD = 'rebuild'
    ALLOCATOR = 'allocator'
    KERNEL_WAIT = 'kernel_wait'


@dataclass
class GhostMoveEvent:
    """Single Ghost Move transfer event"""
    layer_id: int = 0
    tiles_moved: int = 0
    bytes_moved: int = 0
    reason: str = ''
    from_tier: str = ''
    to_tier: str = ''


@dataclass
class StepStats:
    """
    Per-step observable signals.
    VA100 collects, OPU interprets.

    Important: all time/byte fields must be **per-step deltas**, not cumulative.
    """
    step: int = 0
    step_time_s: float = 0.0
    phase: str = 'decode'

    # VRAM
    hot_usage_mb: float = 0.0
    hot_pressure: float = 0.0
    warm_usage_mb: float = 0.0
    cold_usage_mb: float = 0.0
    gpu_alloc_peak_mb: float = 0.0
    gpu_reserved_peak_mb: float = 0.0

    # KV tier bytes
    kv_bytes_hot: int = 0
    kv_bytes_warm: int = 0
    kv_bytes_cold: int = 0

    # Transfer (per-step delta)
    h2d_bytes: int = 0
    d2h_bytes: int = 0
    bw_est_gbs: float = 0.0

    # Tile-level stats
    tiles_hot_count: int = 0
    tiles_warm_count: int = 0
    tiles_cold_count: int = 0
    tiles_evicted: int = 0
    tiles_prefetched: int = 0
    tiles_promoted: int = 0
    tiles_demoted: int = 0

    # Tile operation timing (per-step, ms)
    tile_pack_ms: float = 0.0
    tile_unpack_ms: float = 0.0
    tile_gather_ms: float = 0.0

    # Prefetch hits
    prefetch_hits: int = 0
    prefetch_misses: int = 0

    # Rebuild
    rebuild_count: int = 0
    rebuild_time_s: float = 0.0

    # Faults
    faults: int = 0

    # Aperture loss breakdown (per-step delta)
    wait_time_s: float = 0.0
    copy_bytes: int = 0
    unpack_cost_s: float = 0.0
    rebuild_cost_s: float = 0.0

    @property
    def aperture_loss_s(self) -> float:
        """Total aperture loss = wait + unpack + rebuild"""
        return self.wait_time_s + self.unpack_cost_s + self.rebuild_cost_s

    # Bottleneck reason
    stall_reason: StallReason = StallReason.NONE

    # Ghost Move events
    ghost_move_events: list[GhostMoveEvent] = field(default_factory=list)
    ghost_move_overhead_ms: float = 0.0
    ghost_move_hide_rate: float = 1.0

    # Quality signals
    logits_entropy: float = 0.0
    repeat_rate: float = 0.0
    quality_score: float = 1.0

    # Action accountability
    actions_taken: list[str] = field(default_factory=list)

    def classify_stall(self) -> StallReason:
        """Auto-classify main bottleneck based on per-step time breakdown"""
        candidates = [
            (self.wait_time_s, StallReason.KV_TRANSFER),
            (self.unpack_cost_s, StallReason.PACK_UNPACK),
            (self.rebuild_cost_s, StallReason.REBUILD),
        ]
        worst_time, worst_reason = max(candidates, key=lambda x: x[0])
        if worst_time < 1e-6:
            return StallReason.NONE
        return worst_reason
