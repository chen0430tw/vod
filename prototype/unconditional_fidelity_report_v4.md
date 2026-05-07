# Unconditional Sample Fidelity v4 Report

**Date**: 2026-05-05T04:31:36.059456
**Version**: v4: Fix A (video static) + B (latent norm) + C (batched) + D (detach latent)
**Device**: cuda
**Verdict**: **PARTIAL** (3/5 checks pass)

## Train args
```json
{
  "seed": 430,
  "train_n": 512,
  "epochs": 2000,
  "lr": 0.002,
  "diffusion_steps": 200,
  "time_dim": 64,
  "hidden": 128,
  "channels": 8,
  "n_samples": 8,
  "w_recon": 1.0,
  "cpu": false,
  "out": "generated/diffusion_samples_v4",
  "report_out": "prototype/unconditional_fidelity_result_v4.json",
  "md_out": "prototype/unconditional_fidelity_report_v4.md"
}
```

## Latent stats (Fix B)
μ=+0.0711  σ=0.4419

## descriptor_distance_to_train

| source | distance |
|---|---|
| `train_reference` | 0.0000 |
| `gate0_recon` | 0.0082 |
| `untrained_sample` | 3.9390 |
| `random_noise_baseline` | 4.0777 |
| `trained_sample` | 4.2448 |
| `zero_baseline` | 71.6466 |

## Per-source metrics

| source | finite | amp_range | entropy | tile_residue |
|---|---|---|---|---|
| `train_reference` | 1.000 | 2.000 | 4.818 | 1.018 |
| `trained_sample` | 1.000 | 3.705 | 4.983 | 0.963 |
| `untrained_sample` | 1.000 | 5.725 | 4.934 | 0.988 |
| `random_noise_baseline` | 1.000 | 5.431 | 4.967 | 1.029 |
| `zero_baseline` | 1.000 | 0.000 | 0.000 | 0.000 |
| `gate0_recon` | 1.000 | 2.000 | 4.781 | 1.018 |

## PASS checks

- PASS  finite_ratio == 1.0
- PASS  amplitude_range > 0.05
- FAIL  beats random_noise baseline
- PASS  beats zero baseline
- FAIL  beats untrained_sample

## Multi-seed stability

std of distance across 3 seeds: **0.0228**
