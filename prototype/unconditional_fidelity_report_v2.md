# Unconditional Sample Fidelity v2 Report

**Date**: 2026-05-05T04:15:26.905827
**Version**: v2: force video=image broadcast + latent normalization
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
  "out": "generated/diffusion_samples_v2",
  "report_out": "prototype/unconditional_fidelity_result_v2.json",
  "md_out": "prototype/unconditional_fidelity_report_v2.md"
}
```

## Latent stats (Fix B)
μ=-0.2498  σ=0.1877

## descriptor_distance_to_train

| source | distance |
|---|---|
| `train_reference` | 0.0000 |
| `gate0_recon` | 0.4093 |
| `trained_sample` | 1.4872 |
| `untrained_sample` | 2.5953 |
| `random_noise_baseline` | 4.3064 |
| `zero_baseline` | 74.0966 |

## Per-source key metrics

| source | finite | amp_range | entropy | tile_residue |
|---|---|---|---|---|
| `train_reference` | 1.000 | 2.000 | 4.818 | 1.018 |
| `trained_sample` | 1.000 | 0.441 | 4.308 | 0.789 |
| `untrained_sample` | 1.000 | 4.905 | 4.926 | 0.978 |
| `random_noise_baseline` | 1.000 | 7.064 | 4.969 | 1.036 |
| `zero_baseline` | 1.000 | 0.000 | 0.000 | 0.000 |
| `gate0_recon` | 1.000 | 0.432 | 4.768 | 1.018 |

## PASS checks

- PASS  finite_ratio == 1.0
- PASS  amplitude_range > 0.05
- PASS  beats random_noise baseline
- PASS  beats zero baseline
- PASS  beats untrained_sample

## Multi-seed stability

std of distance across 3 seeds: **0.3235**
