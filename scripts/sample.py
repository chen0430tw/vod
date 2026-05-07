"""VOD minimal sampling CLI (Phase 1).

Default behaviour: Gate 0 / Chladni round-trip demo. For each sample
we generate a clean Chladni field, encode it through the substrate,
and emit three PNGs into a per-sample subdirectory:

    orig.png             — the synthetic Chladni target
    recon_no_denoise.png — decode(encode(target))   (encoder/decoder identity)
    pipeline.png         — decode(denoise_path(encode(target)))   (full path)

This is intentionally a round-trip demo, not a text-to-image pipeline:
the unconditional sample fidelity required for a clean random-noise
generation is the active quality target and not yet at publication
quality. Round-trip lets a fresh user sanity-check the substrate
end-to-end without requiring trained checkpoints or external data.

Usage
-----
    py -3.13 scripts/sample.py --out generated/sample
    py -3.13 scripts/sample.py --checkpoint /path/to/saved_dir --samples 4

If --checkpoint is omitted the model is constructed from default config
with random initialisation; the run prints an UNTRAINED warning so the
user knows the resulting PNGs are not a quality demo.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Phase 1 portability: allow running this script directly without
# `pip install -e .`. The prototype/ tree is added to sys.path if
# vod_minimal is not already importable.
_PROTO = Path(__file__).resolve().parent.parent / "prototype"
if str(_PROTO) not in sys.path:
    sys.path.insert(0, str(_PROTO))

from vod_minimal.blocky_scattering import build_blocky_scattering_batch
from vod_minimal.native import (
    LATENT_HW, LATENT_T, NativeVOD, NativeVODConfig,
    views_to_torch, views_to_numpy,
)


def to_png(a, p):
    a = np.asarray(a).squeeze()
    if a.ndim != 2:
        a = a.reshape(a.shape[0], -1)
    a = (a - a.min()) / (a.max() - a.min() + 1e-9) * 255
    Image.fromarray(a.astype(np.uint8), 'L').save(p)


def build_or_load_model(checkpoint: str | None, device: torch.device) -> tuple[NativeVOD, bool]:
    """Return (model, is_trained). Falls back to random init if no checkpoint."""
    if checkpoint:
        ckpt_path = Path(checkpoint)
        if not ckpt_path.is_dir():
            raise FileNotFoundError(
                f"--checkpoint must be a directory containing config.json + "
                f"model.safetensors (or pytorch_model.bin), got {ckpt_path!r}"
            )
        model = NativeVOD.from_pretrained(ckpt_path).to(device)
        return model, True

    cfg = NativeVODConfig(channels=4, hidden=32, denoise_steps=4, backbone="unet")
    model = NativeVOD(cfg).to(device)
    return model, False


def main():
    p = argparse.ArgumentParser(
        description="VOD Phase 1 round-trip sampling CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Directory written by NativeVOD.save_pretrained. "
                        "If omitted, an untrained model is used.")
    p.add_argument("--out", type=str, required=True,
                   help="Output directory; one subdirectory per sample is created.")
    p.add_argument("--samples", type=int, default=4,
                   help="Number of round-trip samples to generate.")
    p.add_argument("--seed", type=int, default=430,
                   help="RNG seed for both data generation and torch init.")
    p.add_argument("--cpu", action="store_true",
                   help="Force CPU even if CUDA is available.")
    p.add_argument("--denoise-steps", type=int, default=8,
                   help="Iterative denoise_path step count for pipeline.png.")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model, is_trained = build_or_load_model(args.checkpoint, device)
    n_params = sum(p_.numel() for p_ in model.parameters())
    if not is_trained:
        print("=" * 72, flush=True)
        print("[UNTRAINED] No --checkpoint supplied. Model is random-initialised.", flush=True)
        print("[UNTRAINED] Output PNGs are a wiring sanity check, NOT a quality demo.", flush=True)
        print("[UNTRAINED] Train with scripts/ablations/run_gate0_verify.py first.", flush=True)
        print("=" * 72, flush=True)
    print(f"device={device}  params={n_params:,}  samples={args.samples}", flush=True)

    rng = np.random.default_rng(args.seed)
    batch = build_blocky_scattering_batch(
        rng, batch_size=args.samples, size=LATENT_HW, frames=LATENT_T,
        artifact_strength=0.0, tile=4, spacetime=True,
        temporal_mode='static', flicker_strength=0.0, paired_denoising=True,
    )

    model.eval()
    with torch.no_grad():
        for i, samp in enumerate(batch.samples):
            sd = out / f"sample_{i:02d}"
            sd.mkdir(exist_ok=True)
            target = views_to_torch(samp.target_views, device)

            # Encode → decode (identity, no denoise)
            U = model.encode(target)
            recon_views = model.decode(U)
            recon_np = views_to_numpy(recon_views)

            # Encode → denoise_path → decode (full pipeline)
            U_pred = model.denoise_path(U, steps=args.denoise_steps)
            pipeline_views = model.decode(U_pred)
            pipeline_np = views_to_numpy(pipeline_views)

            to_png(samp.target_views['image'], sd / 'orig.png')
            to_png(recon_np['image'], sd / 'recon_no_denoise.png')
            to_png(pipeline_np['image'], sd / 'pipeline.png')
            print(f"  sample {i}: {sd}/{{orig,recon_no_denoise,pipeline}}.png", flush=True)

    print(f"\ndone. inspect {out}/sample_*/")


if __name__ == "__main__":
    main()
