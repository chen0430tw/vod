# VOD Real-Image Smoke Report (CIFAR-10 grayscale)

**Date**: 2026-05-07T07:50:56.556532
**Version**: real-image-smoke: v16-A-O+P+Q+R+S' on CIFAR-10 grayscale 32×32
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
  "w_recon": 1.0,
  "cpu": false,
  "out": "generated/rgb64_conv",
  "report_out": "prototype/rgb64_conv_result.json",
  "md_out": "prototype/rgb64_conv_report.md",
  "prediction": "v",
  "lr_schedule": "cosine",
  "amp": true,
  "minibatch_size": 32,
  "checkpoint_dir": "runs/rgb64_conv_ckpt",
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
| `gate0_recon` | 0.0080 |
| `trained_sample` | 3.8874 |
| `zero_baseline` | 9.1472 |
| `random_noise_baseline` | 9.6080 |
| `untrained_sample` | 9.8915 |

## Per-source metrics

| source | finite | amp_range | entropy | tile_residue |
|---|---|---|---|---|
| `train_reference` | 1.000 | 1.992 | 5.071 | 0.000 |
| `trained_sample` | 1.000 | 0.761 | 4.497 | 0.000 |
| `untrained_sample` | 1.000 | 0.432 | 4.870 | 0.000 |
| `random_noise_baseline` | 1.000 | 7.454 | 4.586 | 0.000 |
| `zero_baseline` | 1.000 | 0.094 | 1.585 | 0.000 |
| `gate0_recon` | 1.000 | 2.000 | 5.079 | 0.000 |

## PASS checks

- PASS  finite_ratio == 1.0
- PASS  amplitude_range > 0.05
- PASS  beats random_noise baseline
- PASS  beats zero baseline
- PASS  beats untrained_sample

## Multi-seed stability

std of distance across 3 seeds: **1.0728**
