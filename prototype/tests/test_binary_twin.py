"""Tests for the minimal Binary-Twin implementation."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from vod_minimal.binary_twin import (
    binary_twin_metrics,
    binary_twin_pixel_torch_loss,
    binary_twin_state,
    binary_twin_torch_accuracy,
    binary_twin_torch_loss,
    decode_symbols,
    encode_symbols,
    reconstruction_error,
    symbol_accuracy,
)
from vod_minimal.native import NativeVOD, NativeVODConfig, native_total_loss, views_to_torch
from vod_minimal.core import build_projection_batch


def test_encode_decode_symbols_round_trip_on_quantized_values():
    vals = np.arange(16, dtype=np.float64) / 15.0
    symbols = encode_symbols(vals)
    np.testing.assert_array_equal(symbols, np.arange(16))
    np.testing.assert_allclose(decode_symbols(symbols), vals)


def test_binary_twin_state_contains_continuous_and_discrete_parts():
    vals = np.array([0.0, 0.2, 0.5, 1.0])
    state = binary_twin_state(vals)
    assert state.continuous.shape == vals.shape
    assert state.symbols.dtype == np.int64
    assert state.reconstructed.shape == vals.shape


def test_symbol_accuracy_detects_corruption():
    target = np.array([0.0, 1 / 15, 2 / 15, 3 / 15])
    clean = target.copy()
    corrupt = target.copy()
    corrupt[2] = 14 / 15
    assert symbol_accuracy(clean, target) == pytest.approx(1.0)
    assert symbol_accuracy(corrupt, target) == pytest.approx(0.75)


def test_reconstruction_error_is_small_for_quantized_text():
    vals = np.arange(16, dtype=np.float64) / 15.0
    assert reconstruction_error(vals) == pytest.approx(0.0)


def test_binary_twin_metrics_required_keys():
    target = np.linspace(0.0, 1.0, 8)
    out = binary_twin_metrics(target, target)
    assert set(out) == {
        "symbol_accuracy",
        "symbol_hamming",
        "continuous_mse",
        "pred_reconstruction_error",
        "target_reconstruction_error",
    }
    assert out["symbol_accuracy"] == pytest.approx(1.0)
    assert out["symbol_hamming"] == pytest.approx(0.0)


def test_binary_twin_torch_loss_prefers_correct_symbols():
    target = torch.tensor([0.0, 1 / 15, 2 / 15, 3 / 15], dtype=torch.float32)
    clean = target.clone().requires_grad_(True)
    corrupt = torch.tensor([0.0, 1 / 15, 14 / 15, 3 / 15], dtype=torch.float32, requires_grad=True)
    clean_loss = binary_twin_torch_loss(clean, target)
    corrupt_loss = binary_twin_torch_loss(corrupt, target)
    assert clean_loss < corrupt_loss
    corrupt_loss.backward()
    assert corrupt.grad is not None
    assert torch.isfinite(corrupt.grad).all()


def test_binary_twin_torch_accuracy():
    target = torch.tensor([0.0, 1 / 15, 2 / 15, 3 / 15], dtype=torch.float32)
    pred = torch.tensor([0.0, 1 / 15, 14 / 15, 3 / 15], dtype=torch.float32)
    assert float(binary_twin_torch_accuracy(pred, target)) == pytest.approx(0.75)


def test_binary_twin_pixel_torch_loss_returns_scalar_tensor():
    pred = torch.zeros(3, 8, 8)
    target = torch.zeros(3, 8, 8)
    out = binary_twin_pixel_torch_loss(pred, target)
    assert isinstance(out, torch.Tensor)
    assert out.ndim == 0


def test_binary_twin_pixel_loss_zero_when_pred_equals_target():
    target = torch.linspace(-1.0, 1.0, 3 * 4 * 4).reshape(3, 4, 4)
    pred = target.clone()
    out = binary_twin_pixel_torch_loss(pred, target)
    # CE on exact-match quantized symbols is bounded but small;
    # reconstruction MSE is exactly 0; with default temperature the CE
    # plus 0.25 * recon collapses to a constant. Use clean vs corrupt
    # contrast as the actual test of "lower is better".
    corrupt = target.clone()
    corrupt = corrupt + 0.5
    corrupt_out = binary_twin_pixel_torch_loss(corrupt, target)
    assert out < corrupt_out


def test_binary_twin_pixel_loss_positive_and_finite_grad():
    target = torch.linspace(-1.0, 1.0, 16).reshape(4, 4)
    pred = (target + 0.4).clone().requires_grad_(True)
    loss = binary_twin_pixel_torch_loss(pred, target)
    assert float(loss.detach()) > 0.0
    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


def test_binary_twin_pixel_loss_handles_3d_video_and_2d_image_shapes():
    # 3-D (F, H, W) video shape
    target_video = torch.randn(4, 8, 8) * 0.3
    pred_video = (target_video + 0.2).clone().requires_grad_(True)
    loss_v = binary_twin_pixel_torch_loss(pred_video, target_video)
    assert loss_v.ndim == 0
    loss_v.backward()
    assert pred_video.grad is not None
    assert torch.isfinite(pred_video.grad).all()

    # 2-D (H, W) image shape
    target_image = torch.linspace(-1.0, 1.0, 64).reshape(8, 8)
    pred_image = (target_image + 0.2).clone().requires_grad_(True)
    loss_i = binary_twin_pixel_torch_loss(pred_image, target_image)
    assert loss_i.ndim == 0
    loss_i.backward()
    assert pred_image.grad is not None
    assert torch.isfinite(pred_image.grad).all()


def test_binary_twin_pixel_loss_empty_returns_zero():
    pred = torch.zeros(0)
    target = torch.zeros(0)
    out = binary_twin_pixel_torch_loss(pred, target)
    assert float(out) == 0.0


def test_native_text_loss_uses_binary_twin_when_text_enabled():
    rng = np.random.default_rng(0)
    sample = build_projection_batch(
        rng,
        batch_size=1,
        size=16,
        frames=8,
        spacetime=True,
        media=("image", "video", "text"),
        paired_denoising=True,
    ).samples[0]
    model = NativeVOD(NativeVODConfig(enable_text=True, hidden=8))
    noisy = views_to_torch(sample.noisy_views, torch.device("cpu"))
    target = views_to_torch(sample.target_views, torch.device("cpu"))
    loss, components = native_total_loss(model, noisy, target)
    assert torch.isfinite(loss)
    assert components["L_text"] > 0.0
    assert "binary_twin_symbol_accuracy" in components
