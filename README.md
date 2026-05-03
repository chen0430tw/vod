# VOD — Visual Output Diffusion (research prototype)

> **Status: research prototype, toy scale (~524K params).** Generation
> capability has just passed Gate 0 (encode/decode identity + denoiser
> stability) and produces unconditional samples via DDIM, but quality
> is not yet publishable. Architecture and interface contracts are
> documented to support future scaling and HF/Diffusers integration
> without breaking changes.

VOD explores a **type B substrate-shared** multimodal diffusion
architecture: a single shared entropy field `U(t, y, x, c)` is sampled
by one diffusion process, and per-medium decoder heads project it back
to image / video / audio / text views. This contrasts with type C
designs (e.g. Sora, LPM 1.0) that route each modality through
independent encoders into a shared backbone.

## Repository layout

```
VOD/
├── prototype/
│   ├── vod_minimal/              # Core package
│   │   ├── native.py             # NativeVOD substrate + encode/decode/denoise
│   │   ├── denoisers.py          # UNet (default) + pointwise MLP (legacy)
│   │   ├── diffusion.py          # DDPM schedule + DDIM sampler
│   │   └── ...
│   ├── tests/                    # pytest suite
│   ├── run_*.py                  # Training / evaluation entry points
│   ├── STATE_DICT_KEYS.md        # Frozen parameter naming contract
│   └── model_index.json          # HF Diffusers pipeline draft (Phase C)
├── docs/
│   ├── vod_full_mathematical_formulation.md
│   ├── vod_hf_compatibility_plan.md
│   ├── hf_modular_mapping.md
│   └── ...
└── LICENSE                       # Apache 2.0
```

## Quick start

```bash
cd prototype
py -3.13 -m pytest tests/                  # full test suite
py -3.13 run_gate0_verify.py               # encode/decode identity + denoiser stability
py -3.13 run_diffusion_train.py --epochs 1000 --diffusion-steps 200
```

Outputs land in `prototype/generated/sample_*/`.

## Architecture summary

- **Latent substrate**: `U(T=8, H=16, W=16, C=4)` — single shared field.
- **Encoders**: 4 thin per-medium 1×1 Linear layers (image / video / audio / text).
- **Denoiser**: 3-level spatial UNet over `(H, W)` with a 1-D
  conv along `T` at the bottleneck. Time conditioning via sinusoidal
  embedding when `time_dim > 0`.
- **Decoders**: 4 thin per-medium 1×1 Linear inverses.
- **Sampling**: DDPM-trained denoiser sampled with DDIM (η=0).

See `docs/vod_full_mathematical_formulation.md` for the full
formulation and `docs/vod_hf_compatibility_plan.md` for the HF /
Diffusers integration roadmap.

## HF / Diffusers compatibility

This prototype follows a **shadow-contract** approach: HF-compatible
serialization (`config.json` + `model.safetensors`),
`save_pretrained` / `from_pretrained`, and a locked `state_dict` key
contract are implemented **without** depending on `huggingface_hub`
or `diffusers`. Future Phase C migration to `ModularPipelineBlocks`
becomes a wrapper-only change. See `docs/hf_modular_mapping.md`.

## License

Apache License 2.0 — see [LICENSE](LICENSE).

## Acknowledgements

Architectural exploration informed by:
- Stable Diffusion / SDXL (latent diffusion baseline)
- Mochi 1, CogVideoX, Wan 2.2 (open video diffusion priors)
- LPM 1.0 (Anuttacon) — type C reference for full-duplex avatar performance
- Diffusers / Modular Diffusers (HF, 2026)
