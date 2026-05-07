# Unconditional Sample Fidelity Report
**Date**: 2026-05-05T02:22:15.133498
**Verdict**: **PASS** (5/5 checks pass)

## Train args
```json
{
  "seed": 430,
  "train_n": 64,
  "epochs": 500,
  "lr": 0.002,
  "diffusion_steps": 200,
  "time_dim": 64,
  "hidden": 32,
  "channels": 4,
  "n_samples": 8,
  "w_recon": 1.0,
  "cpu": true,
  "out": "generated/diffusion_samples",
  "report_out": "prototype/unconditional_fidelity_result.json",
  "md_out": "prototype/unconditional_fidelity_report.md"
}
```
## Sampler
DDIM, η=0, prediction=x_0, schedule=linear β [1e-4, 2e-2], num_steps=200, sample_steps=50

## Visible output
- `generated\diffusion_samples/train_reference.png` — Chladni training samples
- `generated\diffusion_samples/trained_sample.png` — DDIM samples from N(0,I) (TRAINED model)
- `generated\diffusion_samples/untrained_sample.png` — DDIM samples from N(0,I) (UNTRAINED control)
- `generated\diffusion_samples/random_noise_baseline.png` — pure N(0,I) decoded
- `generated\diffusion_samples/zero_baseline.png` — zeros decoded
- `generated\diffusion_samples/gate0_recon.png` — encode→decode of training samples
- `generated\diffusion_samples/trained_multi_seed.png` — same model, 3 different seeds

## descriptor_distance_to_train (L2 over [amp, phase, freq, comp, sal, snr])

| source | distance |
|---|---|
| `train_reference` | 0.0000 |
| `gate0_recon` | 0.0238 |
| `trained_sample` | 0.6891 |
| `untrained_sample` | 2.5953 |
| `random_noise_baseline` | 4.3907 |
| `zero_baseline` | 71.6071 |

## Per-source key metrics

| source | finite | amp_range | entropy | tile_residue |
|---|---|---|---|---|
| `train_reference` | 1.000 | 2.000 | 4.818 | 1.018 |
| `trained_sample` | 1.000 | 1.593 | 4.623 | 0.787 |
| `untrained_sample` | 1.000 | 4.905 | 4.926 | 0.978 |
| `random_noise_baseline` | 1.000 | 7.205 | 4.981 | 0.995 |
| `zero_baseline` | 1.000 | 0.000 | 0.000 | 0.000 |
| `gate0_recon` | 1.000 | 1.819 | 4.770 | 1.018 |

## PASS checks

- ✅ finite_ratio == 1.0
- ✅ amplitude_range > 0.05
- ✅ beats random_noise baseline
- ✅ beats zero baseline
- ✅ beats untrained_sample

## Multi-seed stability (3 seeds, n=4 each)

std of descriptor_distance across seeds: **0.4468**
(lower = more stable; if std > 0.5 the sampler is highly seed-dependent)

