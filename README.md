<div align="center">

# VOD

**Visual Output Diffusion — substrate-shared multimodal generation**

[![License](https://img.shields.io/badge/license-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/status-research_prototype-orange.svg)]()
[![HF compat](https://img.shields.io/badge/HF/Diffusers-shadow_contract-yellow.svg)](docs/hf_modular_mapping.md)

[Architecture](docs/vod_full_mathematical_formulation.md) ·
[Roadmap](docs/vod_hf_compatibility_plan.md) ·
[HF mapping](docs/hf_modular_mapping.md) ·
[Tests](prototype/tests/)

</div>

---

## TL;DR

VOD is a research prototype exploring a **type B substrate-shared**
diffusion architecture: a single shared entropy field
`U(t, y, x, c)` is sampled by **one** diffusion process, and per-medium
1×1 decoder heads project it back to image / video / audio / text views.
This contrasts with type C designs (Sora, LPM 1.0) that route each
modality through independent encoders into a shared backbone.

**Status: toy scale (~524K parameters, 16×16 latent).** Gate 0
(encode/decode identity + denoiser stability) passed; unconditional
DDIM sampling produces structured outputs but is not yet
publication-quality. Architecture and interface contracts are
documented to support future scaling and HF/Diffusers integration
without breaking changes.

---

## News

- **2026-05-05** — Phase 1 complete. Repo layout aligned to standard
  open-source ML format: `pyproject.toml` + `requirements.txt`,
  `scripts/sample.py` minimal CLI, 19 ablation entry points moved to
  `scripts/ablations/`. 235/235 tests passing.
- **2026-05-04** — Repository public. Phase 1 / 1A (HF/Diffusers shadow
  contract: serialization, `state_dict` key contract, output dataclass,
  dtype audit) merged.
- **2026-05-03** — Gate 0 passed (`L_recon=0.0285`, `L_clean_noop=0.0000`).
  DDPM training + DDIM sampling pipeline operational.

---

## Architecture

```
   image  ─┐
   video  ─┤   ┌────────────┐    ┌──────────────┐    ┌────────────┐   image  ─┐
   audio  ─┼─► │  encoder   │ ─► │  shared U    │ ─► │  decoder   ├── video   │
   text   ─┘   │ (1×1 Linear│    │ (T,H,W,C)    │    │ (1×1 Linear│   audio   │
               │  per medium)    │  + UNet      │    │  per medium)   text    │
               └────────────┘    │  denoiser    │    └────────────┘           │
                                 └──────────────┘                             │
                                        │                                     │
                                  DDIM sampling                               │
                                  (unconditional)                             │
                                                                              ▼
                                                                       requested
                                                                       subset only
```

- **Substrate**: `U(T=8, H=16, W=16, C=4)` — single shared field.
- **Encoder**: 4 thin per-medium 1×1 Linear projections.
- **Denoiser**: 3-level spatial UNet over `(H, W)` with 1-D conv along `T`
  at the bottleneck. Sinusoidal time conditioning when `time_dim > 0`.
- **Decoder**: 4 thin per-medium 1×1 Linear inverses, with optional
  `requested=("image",)` selective dispatch.
- **Sampler**: DDPM-trained denoiser sampled with DDIM (η=0).

Full formulation: [`docs/vod_full_mathematical_formulation.md`](docs/vod_full_mathematical_formulation.md).

---

## Installation

```bash
git clone https://github.com/chen0430tw/vod.git
cd vod
py -3.13 -m pip install -e .
```

This installs the `vod_minimal` package and pulls in `torch`, `numpy`,
`pillow`, `safetensors` from `pyproject.toml`. No `huggingface_hub` or
`diffusers` dependency required — VOD implements HF-compatible
serialization (`save_pretrained`, `from_pretrained`, `config.json`,
`model.safetensors`) standalone. See [Roadmap](#roadmap) for the
migration plan.

> **Status reminder.** VOD is currently a research prototype.
> Gate 0 visible output is passed. **Unconditional sample fidelity is
> still the active quality target** — `scripts/sample.py` defaults to
> a Gate 0 / Chladni round-trip demo, not text-to-image.

---

## Quickstart

```bash
# 1. Run the test suite (235 tests, ~25s on CPU)
py -3.13 -m pytest prototype/tests -q

# 2. Round-trip sampling demo (3 PNGs per sample: orig / recon / pipeline)
py -3.13 scripts/sample.py --out generated/sample --samples 4 --seed 430
# If no checkpoint is supplied, the model is random-initialised and
# the run prints an UNTRAINED warning — outputs are a wiring sanity
# check, not a quality demo.

# 3. (Optional) train a tiny model first, then sample with checkpoint
py -3.13 scripts/ablations/run_gate0_verify.py --epochs 400
py -3.13 scripts/sample.py --checkpoint <saved_dir> --out generated/sample
```

### Save / load (HF-compatible layout)

```python
from vod_minimal.native import NativeVOD, NativeVODConfig

model = NativeVOD(NativeVODConfig(channels=4, hidden=32))
model.save_pretrained("./my_vod")
# writes ./my_vod/config.json + ./my_vod/model.safetensors

reloaded = NativeVOD.from_pretrained("./my_vod")
```

### Selective decoding

```python
out = model(noisy_views)           # NativeVODOutput(sample=dict, latent=Tensor)
predicted, u_pred = out            # tuple unpacking still works (legacy)

# Decode only what you need:
image_only = model.decode(out.latent, requested=("image",))
```

### Ablations and experimental scripts

The full set of training / ablation / diagnostic scripts lives in
`scripts/ablations/`. Each is self-contained and runs from any cwd
once `pip install -e .` has been done:

```bash
py -3.13 scripts/ablations/run_diffusion_train.py --help
py -3.13 scripts/ablations/run_gate0_verify.py --help
py -3.13 scripts/ablations/run_msn_diagnostic.py --help
# ...19 entry points total
```

### Save / load (HF-compatible layout)

```python
from vod_minimal.native import NativeVOD, NativeVODConfig

model = NativeVOD(NativeVODConfig(channels=4, hidden=32))
model.save_pretrained("./my_vod")
# writes ./my_vod/config.json + ./my_vod/model.safetensors

reloaded = NativeVOD.from_pretrained("./my_vod")
```

### Selective decoding

```python
out = model(noisy_views)           # NativeVODOutput(sample=dict, latent=Tensor)
predicted, u_pred = out            # tuple unpacking still works (legacy)

# Decode only what you need:
image_only = model.decode(out.latent, requested=("image",))
```

---

## Repository layout

```
vod/
├── pyproject.toml                # Phase 1: pip install -e . + pytest config
├── requirements.txt              # Loose pin (numpy / torch / safetensors / Pillow)
├── prototype/
│   ├── vod_minimal/              # Core package (current import name; renamed in Phase 2)
│   │   ├── native.py             # NativeVOD substrate (encode/decode/denoise)
│   │   ├── denoisers.py          # UNet (default) + pointwise MLP (legacy)
│   │   ├── diffusion.py          # DDPM schedule + DDIM sampler
│   │   ├── chladni.py            # Chladni mode basis
│   │   ├── binary_twin.py        # Binary-Twin symbol coupling loss
│   │   ├── aimp.py               # TPSR / AIMP physical consistency
│   │   └── ...
│   ├── tests/                    # pytest suite (235 tests)
│   ├── STATE_DICT_KEYS.md        # Frozen parameter naming contract
│   ├── model_index.json          # HF Diffusers manifest draft
│   └── modular_model_index.json  # Modular Diffusers manifest draft
├── scripts/
│   ├── sample.py                 # Phase 1: minimal round-trip CLI
│   └── ablations/                # 19 training / ablation / diagnostic scripts
├── docs/
│   ├── vod_full_mathematical_formulation.md
│   ├── vod_hf_compatibility_plan.md   # Phase 1/2/3 plan
│   ├── hf_modular_mapping.md
│   ├── vod_chladni_model.md
│   └── ...
├── opu/                          # Operator policies (resource control)
├── LICENSE                       # Apache 2.0
└── README.md
```

---

## Roadmap

Engineering Phases (interface + repo layout, planned together):

| Phase | Scope | Status |
|---|---|---|
| **Phase 1** — Now | 1A: HF shadow contract (config.json, save/from_pretrained, state_dict key contract, output dataclass, dtype audit). 1B: pyproject.toml, requirements.txt, scripts/sample.py, scripts/ablations/ collection. | done |
| **Phase 2** — After sample fidelity | 2A: HF Spaces demo + model card + optional `PyTorchModelHubMixin`. 2B: introduce `vod/` public package with `vod_minimal` compatibility shim. | gated by sample fidelity |
| **Phase 3** — HF Hub / Diffusers integration | 3A: `ModularPipelineBlocks` wrapper + `ModelMixin`/`SchedulerMixin` + layerwise casting / offload hooks. 3B: deep modularisation (`vod/{models,fields,projections,constraints,sampling,io}/`). | gated by Phase 2 |

Research milestones (quality gates, not engineering Phases):

| Milestone | Status |
|---|---|
| Gate 0 — encode/decode identity + denoiser no-op | done |
| DDPM/DDIM — unconditional generation pipeline | done |
| **Unconditional sample fidelity** — crisp Chladni samples | **active focus** |
| Scaling — hidden ≥ 128, train set ≥ 1k, 32×32 / 64×64 latent | pending |
| Conditional generation — text / class label / image conditioning hook | pending |

See [`docs/vod_hf_compatibility_plan.md`](docs/vod_hf_compatibility_plan.md)
for the full Phase 1/2/3 plan and design decisions.

---

## Why type B?

Most current open multimodal video models (Mochi, CogVideoX, Wan,
HunyuanVideo, LPM 1.0) follow **type C**: independent per-modality
encoders feeding a shared backbone (DiT / UNet). VOD explores the
alternative: a substrate-shared field where modalities are projections
of a single underlying state, not distinct streams that meet in
attention.

Trade-offs (working hypothesis, not yet validated at scale):

| | Type B (VOD) | Type C (Sora / LPM) |
|---|---|---|
| Cross-modality consistency | substrate-enforced | learned via cross-attention |
| Per-modality capacity | shared field bottleneck | per-encoder allocation |
| Inference cost (single modality) | substrate sampling cost is media-agnostic | DiT cost is media-agnostic |
| Conditioning flexibility | requires substrate adaptation | independent per-encoder |
| Architectural simplicity | one denoiser, thin heads | many components |

VOD does **not** claim type B is universally superior — it claims
type B is a coherent design point that has been under-explored, and
this repo is the minimal apparatus to study it.

---

## Documentation

- [`docs/vod_full_mathematical_formulation.md`](docs/vod_full_mathematical_formulation.md) — substrate equations, loss components, ablation results
- [`docs/vod_hf_compatibility_plan.md`](docs/vod_hf_compatibility_plan.md) — Phase A/B/C roadmap, design decisions
- [`docs/hf_modular_mapping.md`](docs/hf_modular_mapping.md) — future Modular Diffusers wrapper contract
- [`docs/vod_chladni_model.md`](docs/vod_chladni_model.md) — Chladni mode basis derivation
- [`docs/vod_hypothesis_validation_plan.md`](docs/vod_hypothesis_validation_plan.md) — research claims and evaluation protocol
- [`prototype/STATE_DICT_KEYS.md`](prototype/STATE_DICT_KEYS.md) — parameter naming contract

---

## Citation

If you use VOD in your research, please cite:

```bibtex
@misc{chen2026vod,
  title  = {VOD: Substrate-Shared Multimodal Diffusion},
  author = {Chen, [author list]},
  year   = {2026},
  url    = {https://github.com/chen0430tw/vod},
  note   = {Research prototype; technical report in preparation}
}
```

A formal technical report is in preparation. This citation block will
be updated when arXiv preprint becomes available.

---

## Acknowledgements

Architectural exploration informed by:

- **Stable Diffusion / SDXL** — latent diffusion baseline
- **Mochi 1**, **CogVideoX**, **Wan 2.2**, **HunyuanVideo** — open
  video diffusion priors and Apache 2.0 ecosystem
- **LPM 1.0** (Anuttacon) — type C reference for full-duplex avatar
  performance and identity-aware multi-reference conditioning
- **Diffusers** and **Modular Diffusers** (Hugging Face, 2026) —
  composable pipeline reference and `ModularPipelineBlocks` design

---

## Disclaimer

This is a **research prototype**, not a production system. The model
is intentionally toy-scale to make architecture iteration fast and
cheap. Do not deploy generated outputs in user-facing products
without significant additional training, evaluation, and safety
review. The substrate-shared (type B) hypothesis is **under
investigation** and has not been validated at scale.

---

## License

Released under the [Apache License 2.0](LICENSE). You are free to use,
modify, and distribute this code commercially, subject to the terms
of the license, including patent grant and attribution requirements.
