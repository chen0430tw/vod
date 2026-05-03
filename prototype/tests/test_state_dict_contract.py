"""state_dict key contract tests.

Locks down NativeVOD parameter naming so that future refactors cannot
silently rename keys and break published checkpoints. The expected
key sets here MUST match `STATE_DICT_KEYS.md` at the prototype root.

If a test in this file fails because you intentionally added /
removed / renamed a parameter:
  1. Update STATE_DICT_KEYS.md to reflect the new contract.
  2. If renaming, follow the migration procedure in §3 of that file
     (alias hook + version bump).
  3. Update the EXPECTED_* sets below.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import torch

from vod_minimal.native import (
    ARCHITECTURE_VERSION,
    NativeVOD,
    NativeVODConfig,
    NativeVODOutput,
)


# Top-level keys that must always be present, regardless of media flags
# or backbone choice. Each entry is a parameter / buffer key; nn.Linear
# contributes both `.weight` and `.bias`.
EXPECTED_TOP_LEVEL = {
    "enc_image.weight", "enc_image.bias",
    "enc_video.weight", "enc_video.bias",
    "enc_audio.weight", "enc_audio.bias",
    "enc_text.weight", "enc_text.bias",
    "dec_image.weight", "dec_image.bias",
    "dec_video.weight", "dec_video.bias",
    "dec_audio.weight", "dec_audio.bias",
    "dec_text.weight", "dec_text.bias",
    "step_logit",
}

EXPECTED_UNET_DENOISER = {
    "denoiser.down1.conv1.weight", "denoiser.down1.conv1.bias",
    "denoiser.down1.conv2.weight", "denoiser.down1.conv2.bias",
    "denoiser.down2.conv1.weight", "denoiser.down2.conv1.bias",
    "denoiser.down2.conv2.weight", "denoiser.down2.conv2.bias",
    "denoiser.bot.conv1.weight",   "denoiser.bot.conv1.bias",
    "denoiser.bot.conv2.weight",   "denoiser.bot.conv2.bias",
    "denoiser.temporal_conv.weight", "denoiser.temporal_conv.bias",
    "denoiser.up2.conv1.weight",   "denoiser.up2.conv1.bias",
    "denoiser.up2.conv2.weight",   "denoiser.up2.conv2.bias",
    "denoiser.up1.conv1.weight",   "denoiser.up1.conv1.bias",
    "denoiser.up1.conv2.weight",   "denoiser.up1.conv2.bias",
    "denoiser.out_conv.weight",    "denoiser.out_conv.bias",
}

EXPECTED_MLP_DENOISER = {
    "denoiser.net.0.weight", "denoiser.net.0.bias",
    "denoiser.net.2.weight", "denoiser.net.2.bias",
    "denoiser.net.4.weight", "denoiser.net.4.bias",
}


def test_state_dict_keys_unet_backbone():
    m = NativeVOD(NativeVODConfig(backbone="unet"))
    keys = set(m.state_dict().keys())
    expected = EXPECTED_TOP_LEVEL | EXPECTED_UNET_DENOISER
    extra = keys - expected
    missing = expected - keys
    assert not extra, (
        f"unexpected new state_dict keys (rename or new module?): {sorted(extra)}. "
        f"If intentional, update STATE_DICT_KEYS.md and EXPECTED_* in this test."
    )
    assert not missing, (
        f"missing expected state_dict keys: {sorted(missing)}"
    )


def test_state_dict_keys_mlp_backbone():
    m = NativeVOD(NativeVODConfig(backbone="mlp"))
    keys = set(m.state_dict().keys())
    expected = EXPECTED_TOP_LEVEL | EXPECTED_MLP_DENOISER
    extra = keys - expected
    missing = expected - keys
    assert not extra, f"unexpected keys: {sorted(extra)}"
    assert not missing, f"missing keys: {sorted(missing)}"


def test_state_dict_independent_of_media_flags():
    """enc/dec for audio/text are always present so checkpoint shape
    doesn't depend on enable_audio/enable_text. This is intentional —
    flipping the flag at inference shouldn't require a different
    checkpoint."""
    a = set(NativeVOD(NativeVODConfig(enable_audio=False, enable_text=False)).state_dict().keys())
    b = set(NativeVOD(NativeVODConfig(enable_audio=True, enable_text=True)).state_dict().keys())
    assert a == b, f"flag-dependent state_dict: diff={sorted(a ^ b)}"


def test_config_roundtrip():
    cfg = NativeVODConfig(channels=4, hidden=32, time_dim=64)
    d = cfg.to_dict()
    assert d["architecture_version"] == ARCHITECTURE_VERSION
    cfg2 = NativeVODConfig.from_dict(d)
    assert cfg2 == cfg


def test_config_rejects_incompatible_major_version():
    cfg = NativeVODConfig().to_dict()
    cfg["architecture_version"] = "99.0"
    with pytest.raises(ValueError, match="incompatible architecture_version"):
        NativeVODConfig.from_dict(cfg)


def test_config_rejects_unknown_keys():
    cfg = NativeVODConfig().to_dict()
    cfg["bogus_field"] = 42
    with pytest.raises(ValueError, match="unknown keys"):
        NativeVODConfig.from_dict(cfg)


def test_save_pretrained_roundtrip():
    m = NativeVOD(NativeVODConfig(channels=4, hidden=16, backbone="unet"))
    with tempfile.TemporaryDirectory() as td:
        m.save_pretrained(td)
        td_path = Path(td)
        assert (td_path / "config.json").exists()
        # one of safetensors or .bin must exist
        weight_exists = (td_path / "model.safetensors").exists() or (td_path / "pytorch_model.bin").exists()
        assert weight_exists, "no weight file written"
        # config.json shape
        cfg_dict = json.loads((td_path / "config.json").read_text(encoding="utf-8"))
        assert cfg_dict["channels"] == 4
        assert cfg_dict["hidden"] == 16
        assert cfg_dict["architecture_version"] == ARCHITECTURE_VERSION

        m2 = NativeVOD.from_pretrained(td)
        # state equality on every key
        s1 = m.state_dict()
        s2 = m2.state_dict()
        assert set(s1) == set(s2)
        for k in s1:
            assert torch.equal(s1[k], s2[k]), f"weight mismatch on {k}"


def test_native_vod_output_iter_unpacks_as_tuple():
    """Legacy callers like `predicted, u_pred = model(noisy)` must keep
    working after the dataclass migration."""
    m = NativeVOD(NativeVODConfig())
    noisy = {"image": torch.randn(16, 16), "video": torch.randn(8, 16, 16)}
    out = m(noisy)
    assert isinstance(out, NativeVODOutput)
    predicted, u_pred = out  # tuple unpacking via __iter__
    assert isinstance(predicted, dict)
    assert isinstance(u_pred, torch.Tensor)
    assert "image" in predicted and "video" in predicted


def test_decode_requested_subset():
    m = NativeVOD(NativeVODConfig())
    noisy = {"image": torch.randn(16, 16), "video": torch.randn(8, 16, 16)}
    out = m(noisy)
    U = out.latent
    only_image = m.decode(U, requested=("image",))
    assert set(only_image.keys()) == {"image"}
    only_video = m.decode(U, requested=("video",))
    assert set(only_video.keys()) == {"video"}
    # requested=None preserves legacy behaviour
    full = m.decode(U)
    assert "image" in full and "video" in full


def test_decode_rejects_inactive_media():
    m = NativeVOD(NativeVODConfig(enable_audio=False))
    noisy = {"image": torch.randn(16, 16)}
    out = m(noisy)
    with pytest.raises(ValueError, match="not in active set"):
        m.decode(out.latent, requested=("audio",))
