# Unconditional Sample Fidelity v16 Report

**Date**: 2026-05-07T04:35:59.067887
**Version**: v16: A-O + P + Q + R + S'(DCT-II orthogonal lift)
**Device**: cuda
**Verdict**: **PASS** (5/5 checks pass)

## Train args
```json
{
  "seed": 430,
  "train_n": 2048,
  "epochs": 1500,
  "lr": 0.0001,
  "weight_decay": 0.001,
  "cosine_eta_floor": 0.1,
  "ema_decay": 0.999,
  "use_fixed_scaling": true,
  "scale_fit_n": 512,
  "zero_terminal_snr": true,
  "w_weak": 0.5,
  "use_field_lift": true,
  "diffusion_steps": 200,
  "time_dim": 64,
  "hidden": 128,
  "channels": 8,
  "n_samples": 8,
  "w_recon": 1.0,
  "cpu": false,
  "out": "generated/v16_main",
  "report_out": "prototype/unconditional_fidelity_result_v16.json",
  "md_out": "prototype/unconditional_fidelity_report_v16.md",
  "prediction": "v",
  "lr_schedule": "cosine",
  "amp": true,
  "minibatch_size": 256,
  "checkpoint_dir": "runs/v16_ckpt",
  "checkpoint_every": 200,
  "resume": null,
  "rss_every_ep": 25,
  "gc_every_ep": 0,
  "pin_dataset": false
}
```

## Latent stats (Fix B)
μ=+0.0000  σ=1.0000

## descriptor_distance_to_train

| source | distance |
|---|---|
| `train_reference` | 0.0000 |
| `gate0_recon` | 0.1495 |
| `trained_sample` | 0.4306 |
| `random_noise_baseline` | 2.7754 |
| `untrained_sample` | 6.9328 |
| `zero_baseline` | 50.8049 |

## Per-source metrics

| source | finite | amp_range | entropy | tile_residue |
|---|---|---|---|---|
| `train_reference` | 1.000 | 2.000 | 4.693 | 0.668 |
| `trained_sample` | 1.000 | 1.381 | 4.606 | 0.884 |
| `untrained_sample` | 1.000 | 0.790 | 4.983 | 0.960 |
| `random_noise_baseline` | 1.000 | 4.300 | 5.011 | 1.024 |
| `zero_baseline` | 1.000 | 0.000 | 0.000 | 0.000 |
| `gate0_recon` | 1.000 | 2.011 | 4.816 | 0.659 |

## PASS checks

- PASS  finite_ratio == 1.0
- PASS  amplitude_range > 0.05
- PASS  beats random_noise baseline
- PASS  beats zero baseline
- PASS  beats untrained_sample

## Multi-seed stability

std of distance across 3 seeds: **0.2792**
