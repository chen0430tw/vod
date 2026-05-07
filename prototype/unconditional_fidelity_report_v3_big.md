# Unconditional Sample Fidelity v3 Report

**Date**: 2026-05-05T04:22:54.407826
**Version**: v3: Fix A (video static) + Fix B (latent norm) + Fix C (batched training)
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
  "out": "generated/diffusion_samples_v3_big",
  "report_out": "prototype/unconditional_fidelity_result_v3_big.json",
  "md_out": "prototype/unconditional_fidelity_report_v3_big.md"
}
```

## Latent stats (Fix B)
μ=+0.0440  σ=0.0001

## descriptor_distance_to_train

| source | distance |
|---|---|
| `train_reference` | 0.0000 |
| `untrained_sample` | 3.9390 |
| `trained_sample` | 4.2048 |
| `random_noise_baseline` | 5.8392 |
| `gate0_recon` | 19.6158 |
| `zero_baseline` | 48.9770 |

## Per-source metrics

| source | finite | amp_range | entropy | tile_residue |
|---|---|---|---|---|
| `train_reference` | 1.000 | 2.000 | 4.818 | 1.018 |
| `trained_sample` | 1.000 | 0.004 | 4.939 | 0.973 |
| `untrained_sample` | 1.000 | 5.725 | 4.934 | 0.988 |
| `random_noise_baseline` | 1.000 | 19.978 | 4.937 | 0.965 |
| `zero_baseline` | 1.000 | 0.000 | 0.000 | 0.000 |
| `gate0_recon` | 1.000 | 0.000 | 4.761 | 1.018 |

## PASS checks

- PASS  finite_ratio == 1.0
- FAIL  amplitude_range > 0.05
- PASS  beats random_noise baseline
- PASS  beats zero baseline
- FAIL  beats untrained_sample

## Multi-seed stability

std of distance across 3 seeds: **0.1359**
