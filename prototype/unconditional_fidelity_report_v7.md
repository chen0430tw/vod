# Unconditional Sample Fidelity v6 Report

**Date**: 2026-05-05T05:38:47.560317
**Version**: v6: A+B+C+D+E + F (cosine LR + larger train set)
**Device**: cpu
**Verdict**: **PASS** (5/5 checks pass)

## Train args
```json
{
  "seed": 430,
  "train_n": 16,
  "epochs": 50,
  "lr": 0.0001,
  "diffusion_steps": 200,
  "time_dim": 64,
  "hidden": 32,
  "channels": 4,
  "n_samples": 4,
  "w_recon": 1.0,
  "cpu": true,
  "out": "generated/diffusion_samples_v7",
  "report_out": "prototype/unconditional_fidelity_result_v7.json",
  "md_out": "prototype/unconditional_fidelity_report_v7.md",
  "prediction": "epsilon",
  "lr_schedule": "cosine",
  "amp": false
}
```

## Latent stats (Fix B)
μ=-0.2431  σ=0.2424

## descriptor_distance_to_train

| source | distance |
|---|---|
| `train_reference` | 0.0000 |
| `trained_sample` | 2.5250 |
| `untrained_sample` | 5.2651 |
| `random_noise_baseline` | 5.3936 |
| `gate0_recon` | 31.6135 |
| `zero_baseline` | 70.5531 |

## Per-source metrics

| source | finite | amp_range | entropy | tile_residue |
|---|---|---|---|---|
| `train_reference` | 1.000 | 2.000 | 4.880 | 1.170 |
| `trained_sample` | 1.000 | 0.410 | 4.974 | 1.029 |
| `untrained_sample` | 1.000 | 2.526 | 4.959 | 1.032 |
| `random_noise_baseline` | 1.000 | 2.987 | 4.917 | 0.917 |
| `zero_baseline` | 1.000 | 0.000 | 0.000 | 0.000 |
| `gate0_recon` | 1.000 | 0.067 | 4.765 | 1.018 |

## PASS checks

- PASS  finite_ratio == 1.0
- PASS  amplitude_range > 0.05
- PASS  beats random_noise baseline
- PASS  beats zero baseline
- PASS  beats untrained_sample

## Multi-seed stability

std of distance across 3 seeds: **0.1729**
