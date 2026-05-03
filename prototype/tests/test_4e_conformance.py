"""Claim 1A conformance tests: OC_{4/e} matches the spec mathematical definition.

These tests pin the operator to the spec — not to any application metric.

A test failure here means the implementation has drifted from the math
definition in `docs/vod_full_mathematical_formulation.md` Section 7. It
does NOT mean the operator is "useless"; that question is Claim 1B and
lives elsewhere (run_claim1_four_over_e_ablation.py and the application
utility metric of the user's choice).

Anchors per spec section:

    Section 7.1   tile_residue gating: r=0 ⇒ identity
    Section 7.2   four 1-D processes, sigma_axis = beta * residue_gain * r
    Section 7.3   D_OC = 4/e, D_AXIS = 1/e (per-axis decay)
    Section 7.4   final operator OC_{4/e}(X) = X + w_q * N_4
    Section 7.5   covariance signature — non-zero on the four projection axes,
                  iid Gaussian has zero covariance for any non-zero shift
    Section 7.7   perturbation_energy + fair_baseline_sigma
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from vod_minimal.artifacts import (
    D_AXIS,
    D_OC,
    boundary_visibility,
    excess_residue,
    fair_baseline_sigma,
    oc_four_over_e,
    perturbation_energy,
    tile_residue,
)


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #

def _blocky(size: int = 64, tile: int = 8) -> np.ndarray:
    """Strong tile-aligned signal — guaranteed r > 0."""
    y = np.arange(size)[:, None] // tile
    x = np.arange(size)[None, :] // tile
    return ((x + y) % 2).astype(np.float64)


def _smooth(size: int = 64) -> np.ndarray:
    """Linear ramp — r = 0 (no boundary preference)."""
    return np.linspace(0.0, 1.0, size * size, dtype=np.float64).reshape(size, size)


def _perturbation(operator_out: np.ndarray, original: np.ndarray) -> np.ndarray:
    return operator_out - original


# --------------------------------------------------------------------------- #
#  Section 7.3 — decay constants
# --------------------------------------------------------------------------- #

class TestDecayConstants:
    def test_D_OC_equals_four_over_e(self):
        assert D_OC == pytest.approx(4.0 / math.e)

    def test_D_AXIS_equals_one_over_e(self):
        assert D_AXIS == pytest.approx(1.0 / math.e)

    def test_axis_sum_equals_orthogonal_compression_decay(self):
        # Spec 7.3: per-axis 1/e times four axes = 4/e
        assert 4.0 * D_AXIS == pytest.approx(D_OC)


# --------------------------------------------------------------------------- #
#  Section 7.1 — tile residue gating
# --------------------------------------------------------------------------- #

class TestGating:
    def test_smooth_input_yields_zero_excess_residue(self):
        # Linear ramp has J_tile ≈ J_all so R_tile ≈ 1 and r = 0.
        assert excess_residue(_smooth(), tile=8) == 0.0

    def test_blocky_input_yields_positive_excess_residue(self):
        assert excess_residue(_blocky(), tile=8) > 0.0

    def test_zero_residue_input_returns_unchanged(self):
        # Spec 7.4 trailing rule: if r(X,q) = 0 then OC_{4/e}(X) = X.
        smooth = _smooth()
        rng = np.random.default_rng(0)
        out = oc_four_over_e(smooth, rng, beta=0.5, tile=8)
        np.testing.assert_array_equal(out, smooth)

    def test_zero_beta_returns_unchanged(self):
        # sigma = beta * residue_gain * r; beta=0 ⇒ sigma=0 ⇒ identity.
        blocky = _blocky()
        rng = np.random.default_rng(0)
        out = oc_four_over_e(blocky, rng, beta=0.0, tile=8)
        np.testing.assert_array_equal(out, blocky)


# --------------------------------------------------------------------------- #
#  Section 7.4 — boundary visibility weight
# --------------------------------------------------------------------------- #

class TestBoundaryVisibility:
    def test_peaks_at_tile_boundary_columns(self):
        # On tile=8 the boundary columns are j ∈ {0, 7, 8, 15, ...}.
        w = boundary_visibility(8, 16, tile=8, lambda_q=1.0)
        # Top-left corner is on the boundary (d_q = 0) so w = 1.
        assert w[0, 0] == pytest.approx(1.0)
        # Tile interior — far from boundary — has w < 1.
        assert w[3, 3] < w[0, 0]

    def test_lambda_controls_decay_rate(self):
        w_fast = boundary_visibility(16, 16, tile=8, lambda_q=0.5)
        w_slow = boundary_visibility(16, 16, tile=8, lambda_q=2.0)
        # Slower decay ⇒ higher interior visibility.
        assert w_slow[3, 3] > w_fast[3, 3]

    def test_lambda_must_be_positive(self):
        with pytest.raises(ValueError, match="lambda_q"):
            boundary_visibility(16, 16, tile=8, lambda_q=0.0)


# --------------------------------------------------------------------------- #
#  Section 7.5 — covariance signature (the defining property)
# --------------------------------------------------------------------------- #

def _axis_covariance(perturbation: np.ndarray, *, max_shift: int = 8) -> dict[str, float]:
    """Mean |Cov| over k = 1..max_shift along the four projection axes.

    iid Gaussian noise has Cov(N(p), N(q)) = 0 for p ≠ q. The 4/e
    operator must be non-zero on each of the four axes by construction.
    """
    arr = perturbation.astype(np.float64)
    if arr.ndim == 3:
        arr = arr.mean(axis=0)
    H, W = arr.shape
    centered = arr - arr.mean()
    var = float(centered.var())
    if var < 1e-12:
        return {"vert": 0.0, "horiz": 0.0, "diag1": 0.0, "diag2": 0.0}

    def cov_at(dy: int, dx: int) -> float:
        if abs(dy) >= H or abs(dx) >= W:
            return 0.0
        if dx >= 0:
            a = centered[: H - dy, : W - dx]
            b = centered[dy:, dx:]
        else:
            a = centered[: H - dy, -dx:]
            b = centered[dy:, : W + dx]
        return float(np.mean(a * b))

    def avg_axis(get_cov):
        vals = [abs(get_cov(k)) for k in range(1, max_shift + 1)]
        return float(np.mean(vals)) / var

    return {
        "vert":  avg_axis(lambda k: cov_at(0, k)),
        "horiz": avg_axis(lambda k: cov_at(k, 0)),
        "diag1": avg_axis(lambda k: cov_at(k, k)),
        "diag2": avg_axis(lambda k: cov_at(k, -k)),
    }


class TestCovarianceSignature:
    """Spec 7.5 — the operator is non-iid by construction.

    These are the conformance tests that distinguish OC_{4/e} from a
    scalar multiplier on iid Gaussian noise. A scalar-iid placeholder
    would tie zero covariance on all axes (within MC noise).
    """

    @pytest.fixture
    def perturbation_4e(self):
        rng = np.random.default_rng(430)
        blocky = _blocky(size=128, tile=8)
        out = oc_four_over_e(blocky, rng, beta=0.5, tile=8)
        return _perturbation(out, blocky)

    @pytest.fixture
    def perturbation_iid(self):
        rng = np.random.default_rng(430)
        # Match perturbation energy: oc_four_over_e on the same blocky
        # field, then sigma_eq = sqrt(E_pert).
        blocky = _blocky(size=128, tile=8)
        oc_out = oc_four_over_e(blocky, np.random.default_rng(430), beta=0.5, tile=8)
        sigma_eq = fair_baseline_sigma(oc_out, blocky)
        return rng.normal(0.0, sigma_eq, size=blocky.shape)

    def test_oc_four_over_e_has_non_zero_axial_covariance(self, perturbation_4e):
        cov = _axis_covariance(perturbation_4e)
        # Each of the four axes must show measurable covariance.
        for axis in ("vert", "horiz", "diag1", "diag2"):
            assert cov[axis] > 0.01, f"axis={axis} cov={cov[axis]:.4g}"

    def test_iid_gaussian_baseline_has_near_zero_axial_covariance(self, perturbation_iid):
        cov = _axis_covariance(perturbation_iid)
        for axis in ("vert", "horiz", "diag1", "diag2"):
            assert cov[axis] < 0.02, f"axis={axis} cov={cov[axis]:.4g}"

    def test_oc_signature_dominates_iid_by_at_least_3x(self, perturbation_4e, perturbation_iid):
        # Spec 7.5 defining property: OC_{4/e} signature must be
        # qualitatively above iid floor on every axis.
        cov_4e = _axis_covariance(perturbation_4e)
        cov_iid = _axis_covariance(perturbation_iid)
        for axis in ("vert", "horiz", "diag1", "diag2"):
            ratio = cov_4e[axis] / max(cov_iid[axis], 1e-9)
            assert ratio >= 3.0, f"axis={axis} ratio={ratio:.2f}"


# --------------------------------------------------------------------------- #
#  Section 7.6 — video extension via leading-axis broadcast
# --------------------------------------------------------------------------- #

class TestVideoExtension:
    def test_3d_input_returns_3d_output_same_shape(self):
        rng = np.random.default_rng(0)
        v = np.tile(_blocky(size=32, tile=8)[None, :, :], (6, 1, 1))
        out = oc_four_over_e(v, rng, beta=0.3, tile=8)
        assert out.shape == v.shape

    def test_4d_input_returns_4d_output_same_shape(self):
        rng = np.random.default_rng(0)
        v = np.tile(_blocky(size=32, tile=8)[None, None, :, :], (2, 3, 1, 1))
        out = oc_four_over_e(v, rng, beta=0.3, tile=8)
        assert out.shape == v.shape

    def test_shared_spatial_pattern_across_frames(self):
        # Spec 7.6 default: same spatial OC_{4/e} pattern broadcast
        # across leading axes. Per-frame perturbation must be identical.
        rng = np.random.default_rng(0)
        v = np.tile(_blocky(size=32, tile=8)[None, :, :], (4, 1, 1))
        out = oc_four_over_e(v, rng, beta=0.3, tile=8)
        delta = out - v
        for t in range(1, delta.shape[0]):
            np.testing.assert_array_equal(delta[t], delta[0])


# --------------------------------------------------------------------------- #
#  Section 7.7 — fair ablation utilities
# --------------------------------------------------------------------------- #

class TestFairAblation:
    def test_perturbation_energy_is_mean_squared_difference(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal((32, 32))
        y = x + rng.normal(0.0, 0.1, size=x.shape)
        e = perturbation_energy(y, x)
        ref = float(np.mean((y - x) ** 2))
        assert e == pytest.approx(ref)

    def test_fair_baseline_sigma_is_sqrt_of_energy(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal((32, 32))
        y = x + rng.normal(0.0, 0.1, size=x.shape)
        s = fair_baseline_sigma(y, x)
        assert s == pytest.approx(math.sqrt(perturbation_energy(y, x)))

    def test_zero_perturbation_yields_zero_sigma(self):
        x = _smooth()
        rng = np.random.default_rng(0)
        # smooth input ⇒ OC_{4/e} returns identity ⇒ sigma_eq = 0
        out = oc_four_over_e(x, rng, beta=0.5, tile=8)
        assert fair_baseline_sigma(out, x) == 0.0


# --------------------------------------------------------------------------- #
#  Implementation contracts
# --------------------------------------------------------------------------- #

class TestImplementationContracts:
    def test_one_dim_input_raises(self):
        # Spec defines OC_{4/e} only on (..., H, W). Caller decides what
        # to do for 1-D media (audio, text).
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="ndim"):
            oc_four_over_e(np.zeros(64), rng, beta=0.1, tile=8)

    def test_negative_beta_raises(self):
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="beta"):
            oc_four_over_e(_blocky(), rng, beta=-0.1, tile=8)

    def test_negative_residue_gain_raises(self):
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="residue_gain"):
            oc_four_over_e(_blocky(), rng, beta=0.1, residue_gain=-0.1, tile=8)

    def test_tile_must_be_greater_than_one(self):
        with pytest.raises(ValueError, match="tile"):
            tile_residue(_blocky(), tile=1)
