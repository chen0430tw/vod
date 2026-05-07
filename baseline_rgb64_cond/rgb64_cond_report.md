# VOD RGB64 + Class Conditioning Smoke (Stage 2)

**Date**: 2026-05-07T08:39:39.052228
**Version**: rgb64-cond: Stage-2 class conditioning on CIFAR-10 RGB 64×64
**Device**: cuda
**Verdict**: **PASS** (5/5 checks pass)

## Train args
```json
{
  "seed": 430,
  "dataset": "cifar10",
  "data_dir": "D:/VOD/data/real_images",
  "data_cache_dir": null,
  "image_size": 64,
  "train_n": 1024,
  "epochs": 1500,
  "lr": 0.0001,
  "weight_decay": 0.001,
  "cosine_eta_floor": 0.1,
  "ema_decay": 0.999,
  "use_fixed_scaling": true,
  "scale_fit_n": 512,
  "zero_terminal_snr": true,
  "w_weak": 0.0,
  "use_field_lift": true,
  "diffusion_steps": 200,
  "time_dim": 64,
  "hidden": 128,
  "channels": 8,
  "n_samples": 8,
  "num_classes": 10,
  "p_drop_cond": 0.1,
  "samples_per_class": 4,
  "w_recon": 1.0,
  "cpu": false,
  "out": "generated/rgb64_cond",
  "report_out": "prototype/rgb64_cond_result.json",
  "md_out": "prototype/rgb64_cond_report.md",
  "prediction": "v",
  "lr_schedule": "cosine",
  "amp": true,
  "minibatch_size": 32,
  "checkpoint_dir": "runs/rgb64_cond_ckpt",
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
| `gate0_recon` | 0.0042 |
| `trained_sample` | 3.0511 |
| `zero_baseline` | 9.0734 |
| `random_noise_baseline` | 9.6075 |
| `untrained_sample` | 9.9978 |

## Per-source metrics

| source | finite | amp_range | entropy | tile_residue |
|---|---|---|---|---|
| `train_reference` | 1.000 | 1.992 | 5.071 | 0.000 |
| `trained_sample` | 1.000 | 0.693 | 4.797 | 0.000 |
| `untrained_sample` | 1.000 | 0.430 | 5.142 | 0.000 |
| `random_noise_baseline` | 1.000 | 7.519 | 4.577 | 0.000 |
| `zero_baseline` | 1.000 | 0.088 | 1.585 | 0.000 |
| `gate0_recon` | 1.000 | 2.003 | 5.084 | 0.000 |

## PASS checks

- PASS  finite_ratio == 1.0
- PASS  amplitude_range > 0.05
- PASS  beats random_noise baseline
- PASS  beats zero baseline
- PASS  beats untrained_sample

## Multi-seed stability

std of distance across 3 seeds: **0.9823**
