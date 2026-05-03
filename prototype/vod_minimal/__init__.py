"""Minimal VOD prototype package."""

from .artifacts import (
    D_AXIS,
    D_OC,
    boundary_visibility,
    excess_residue,
    fair_baseline_sigma,
    oc_four_over_e,
    perturbation_energy,
    tile_residue,
)
from .aimp import (
    FieldCard,
    LightingCard,
    PerspectiveCard,
    TPSRMeasurement,
    aimp_tpsr_metrics,
    synthesize_tpsr_measurements,
    tpsr_consistency_score,
    tpsr_k,
    tpsr_pair_ratio,
)
from .binary_twin import (
    binary_twin_metrics,
    binary_twin_state,
    binary_twin_torch_accuracy,
    binary_twin_torch_loss,
    decode_symbols,
    encode_symbols,
    symbol_accuracy,
)
from .chladni import Boundary, chladni_field, random_boundary, random_chladni_field
from .metrics import (
    SPATIAL_MEDIA,
    artifact_metrics,
    descriptor,
    mean_target_error,
    mean_tile_residue,
    modular_shrinking_number,
)
from .model import MinimalVOD
from .projections import project_all

__all__ = [
    "Boundary",
    "D_AXIS",
    "D_OC",
    "FieldCard",
    "LightingCard",
    "MinimalVOD",
    "PerspectiveCard",
    "SPATIAL_MEDIA",
    "TPSRMeasurement",
    "aimp_tpsr_metrics",
    "artifact_metrics",
    "binary_twin_metrics",
    "binary_twin_state",
    "binary_twin_torch_accuracy",
    "binary_twin_torch_loss",
    "boundary_visibility",
    "chladni_field",
    "decode_symbols",
    "descriptor",
    "encode_symbols",
    "excess_residue",
    "fair_baseline_sigma",
    "mean_target_error",
    "mean_tile_residue",
    "modular_shrinking_number",
    "oc_four_over_e",
    "perturbation_energy",
    "project_all",
    "random_boundary",
    "random_chladni_field",
    "symbol_accuracy",
    "synthesize_tpsr_measurements",
    "tile_residue",
    "tpsr_consistency_score",
    "tpsr_k",
    "tpsr_pair_ratio",
]
