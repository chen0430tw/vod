# Unconditional Sample Fidelity v3 Report

**Date**: 2026-05-05T04:12:58.053940
**Version**: v3: Fix A (video static) + Fix B (latent norm) + Fix C (batched training)
**Device**: cuda
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
  "cpu": false,
  "out": "generated/diffusion_samples_v3",
  "report_out": "prototype/unconditional_fidelity_result_v3.json",
  "md_out": "prototype/unconditional_fidelity_report_v3.md"
}
```

## Latent stats (Fix B)
μ=-0.2541  σ=0.1876

## descriptor_distance_to_train

| source | distance |
|---|---|
| `train_reference` | 0.0000 |
| `gate0_recon` | 0.5078 |
| `trained_sample` | 2.1533 |
| `untrained_sample` | 2.8599 |
| `random_noise_baseline` | 4.3591 |
| `zero_baseline` | 73.6537 |

## Per-source metrics

| source | finite | amp_range | entropy | tile_residue |
|---|---|---|---|---|
| `train_reference` | 1.000 | 2.000 | 4.818 | 1.018 |
| `trained_sample` | 1.000 | 0.312 | 4.550 | 0.928 |
| `untrained_sample` | 1.000 | 5.956 | 4.995 | 0.886 |
| `random_noise_baseline` | 1.000 | 8.259 | 4.902 | 0.968 |
| `zero_baseline` | 1.000 | 0.000 | 0.000 | 0.000 |
| `gate0_recon` | 1.000 | 0.391 | 4.779 | 1.018 |

## PASS checks

- PASS  finite_ratio == 1.0
- PASS  amplitude_range > 0.05
- PASS  beats random_noise baseline
- PASS  beats zero baseline
- PASS  beats untrained_sample

## Multi-seed stability

std of distance across 3 seeds: **0.1462**
