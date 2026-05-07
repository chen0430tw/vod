# Unconditional Sample Fidelity v5 Report

**Date**: 2026-05-05T04:49:24.298029
**Version**: v5: A+B+C+D + E (epsilon prediction + tunable lr)
**Device**: cuda
**Verdict**: **PASS** (5/5 checks pass)

## Train args
```json
{
  "seed": 430,
  "train_n": 512,
  "epochs": 2000,
  "lr": 0.0001,
  "diffusion_steps": 200,
  "time_dim": 64,
  "hidden": 128,
  "channels": 8,
  "n_samples": 8,
  "w_recon": 1.0,
  "cpu": false,
  "out": "generated/diffusion_samples_v5",
  "report_out": "prototype/unconditional_fidelity_result_v5.json",
  "md_out": "prototype/unconditional_fidelity_report_v5.md",
  "prediction": "epsilon"
}
```

## Latent stats (Fix B)
μ=+0.0774  σ=0.4417

## descriptor_distance_to_train

| source | distance |
|---|---|
| `train_reference` | 0.0000 |
| `gate0_recon` | 0.0802 |
| `trained_sample` | 2.4541 |
| `untrained_sample` | 2.8238 |
| `random_noise_baseline` | 4.2696 |
| `zero_baseline` | 65.6613 |

## Per-source metrics

| source | finite | amp_range | entropy | tile_residue |
|---|---|---|---|---|
| `train_reference` | 1.000 | 2.000 | 4.818 | 1.018 |
| `trained_sample` | 1.000 | 1.125 | 4.950 | 0.926 |
| `untrained_sample` | 1.000 | 2.016 | 4.935 | 0.988 |
| `random_noise_baseline` | 1.000 | 6.506 | 5.012 | 1.064 |
| `zero_baseline` | 1.000 | 0.000 | 0.000 | 0.000 |
| `gate0_recon` | 1.000 | 1.367 | 4.773 | 1.018 |

## PASS checks

- PASS  finite_ratio == 1.0
- PASS  amplitude_range > 0.05
- PASS  beats random_noise baseline
- PASS  beats zero baseline
- PASS  beats untrained_sample

## Multi-seed stability

std of distance across 3 seeds: **0.1810**
