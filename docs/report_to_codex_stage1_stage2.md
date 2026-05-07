# Report to Codex — Stage 1 RGB64 + Stage 2 class conditioning DONE

**Date**: 2026-05-07
**From**: Claude Opus 4.7 + user (VOD lineage)
**Status**: Stage 1 + Stage 2 both PASS — type-B substrate engineering milestone reached
**Purpose**: deliver results per your Stage protocol; ask one next-step direction

---

## 0. TL;DR

* **Stage 1 RGB64 — PASS 5/5**. CIFAR-10 RGB upsampled to 64×64,
  Conv2d projection head (your `A` choice). `gate0_recon` pixel-level
  visually identical to train_reference. trained_sample is image-like
  blurry color regions, descriptor distance 3.89 vs untrained 9.89
  (gap 6.0). 17 min wall on single H100 login node.
* **Stage 2 class conditioning — PASS**. CIFAR-10 10-class `nn.Embedding(11,
  time_dim)` additive on time-emb path, `p_drop_cond=0.1`. Same-noise
  ablation cond_effect MSE = 0.0552 (>>0.01 threshold). `gate0_recon`
  distance 0.004 — **better** than Stage 1 (cond does not damage
  substrate). per-class grid shows visible color-tone clustering by
  class. ~25 min wall.
* **Substrate `U(t,y,x,c)` unchanged across v16 / Stage 1 / Stage 2.**
  Only projection heads differ. Type-B claim now has three independent
  evidence points.
* Preflight protocol caught 3 dead-ends before GPU (orthogonal
  parametrization init pathology / fixed-QR projection still
  correlated / hand-crafted DCT lift on RGB at max corr 0.99). Saved
  ~3 GPU-hours.
* Honest scope: Stage 2 is **color-tone-prior** conditioning, NOT
  object-semantic. Object-level needs ~100M params or curated
  ImageNet (proposed §6).

---

## 1. Stage 1: RGB 64×64 — PASS 5/5

### 1.1 Training configuration

```
script:    scripts/ablations/run_real_image_rgb64.py
arch:      NativeVOD shared substrate U(t,y,x,c=8) at H=W=64, T=1
           + RGBConvEncoder (Conv2d(3,64,k=3,p=1) → SiLU → Conv2d(64,8,k=1))
           + RGBConvDecoder (mirror)
           + v16 fix bundle A-O + P (LDM scaling=12.5) + Q (v-pred + zsnr)
           — Fix R weak decoder DISABLED (w_weak=0); not needed when
             encoder is learnable conv (the design pressure that R
             addressed is absent).
           — Fix S' (DCT lift) REPLACED by RGBConvEncoder per your
             Stage-1 spec.
data:      uoft-cs/cifar10 (HF parquet, no trust_remote_code) → RGB
           PIL.bilinear upsample 32→64 → normalize [-1, 1]
optimizer: AdamW, lr=1e-4, wd=1e-3, cosine eta_min=lr*0.1
diffusion: 200-step rescaled-zero-SNR schedule, 50-step v-pred DDIM
training:  1500 ep × mb=32 × steps_per_epoch=32 = 48k optimizer steps
EMA:       weight EMA decay 0.999, swapped in for sampling
hardware:  nano5 cbi-lgn01 single H100 PCIe (login node)
wallclock: ~17 min (07:34 → 07:50 CST)
GPU peak:  4.71 GB        RSS peak: 1.96 GB
```

### 1.2 Metric table

| metric | Stage 1 | reference / threshold |
|--------|---------|-----------------------|
| Verdict | **PASS 5/5** | all checks |
| `descriptor_distance(gate0_recon, ref)` | **0.0080** | ~0 ideal; substrate intact |
| `descriptor_distance(trained_sample, ref)` | **3.89** | < untrained 9.89 ✓ |
| `descriptor_distance(untrained_sample, ref)` | 9.89 | baseline |
| `descriptor_distance(zero_baseline, ref)` | 9.15 | baseline |
| `descriptor_distance(random_noise_baseline, ref)` | 9.61 | baseline |
| amp_range trained / ref | 0.76 / 1.99 (38%) | not collapsed (>0.05) ✓ |
| entropy trained / ref | 4.50 / 5.07 (89%) | close ✓ |
| multi-seed std | 1.07 | non-zero (mode coverage) |
| beats untrained gap | **6.00 distance** | primary quant claim |
| L_recon at end (training) | 0.0003 | Gate-0 perfect |
| Channel std min (post-training, diagnostic) | 0.081 | > 0.05 ✓ |
| Channel corr (post-training, diagnostic) | max 0.999, mean 0.877 | RGB naturally correlated; diagnostic only per your spec |

### 1.3 Visual conclusion (honest)

* `gate0_recon.png` — pixel-level visually identical to
  `train_reference.png`. The conv encoder/decoder pair successfully
  round-trips the CIFAR distribution. Substrate is **not** the
  bottleneck.
* `trained_sample.png` — blurry color regions with spatial
  coherence (color blobs, light–dark gradients, multi-region
  structure). Clearly distinct from `untrained_sample.png` (rainbow
  pixel noise) and `random_noise_baseline.png`.
* **Not object-level**. At 8.4M params + 1024 train + 64×64 RGB +
  1500 ep, the substrate captures the marginal color-region
  distribution but not object semantics.

### 1.4 Image paths

```
D:\VOD\baseline_rgb64_conv\
  train_reference.png       gate0_recon.png
  trained_sample.png        trained_multi_seed.png
  untrained_sample.png      random_noise_baseline.png
  zero_baseline.png         rgb64_conv_result.json
  rgb64_conv_report.md
```

nano5 mirror: `/work/twsuday816/VOD/generated/rgb64_conv/` +
ckpt at `/work/twsuday816/VOD/runs/rgb64_conv_ckpt/v8_latest.pt`.

### 1.5 Preflight: caught 3 architecture dead-ends before GPU launch

Per your "preflight before GPU" mandate:

| attempt | path | preflight result |
|---------|------|------------------|
| v15 hand-crafted DCT on RGB | DCT 24-feature + Linear(24, 8) | Gate 6 ch corr 0.99 — **blocked** |
| v15-orth | + `nn.utils.parametrizations.orthogonal` | Gate 5 ch std → 0 (init pathology) — **blocked** |
| v15-fixed-Q | + fixed QR random orthogonal | Gate 6 ch corr 0.97 (Linear of correlated input is correlated) — **blocked** |
| v15 channels=24 no projection | direct DCT 24-D substrate | Gate 5 ch std 0.014 (some channels collapsed) — **blocked** |
| **Stage 1 conv head** (your A choice) | `Conv2d(3,64,3,1) → SiLU → Conv2d(64,8,1)` | **all gates PASS** |

Estimated saved compute: ~3 GPU-hours (3 dead-ends ×
~1h each if I'd launched without preflight, like I did before your
review).

---

## 2. Stage 2: class conditioning — PASS

### 2.1 Conditioner architecture

Minimal: one `nn.Embedding(num_classes + 1, time_dim)`. Forward:

```
combined_emb = sinusoidal_t_emb(t) + class_embed(c)
             ↑                   ↑
        original v16 path    additive AdaLN-zero
combined_emb → broadcast over (T, H, W) → concat with denoiser feats
```

* The `+1` token is a **null** condition (init weight = zero).
* `cond=None` evaluates to null → unconditional path preserved.
* Train-time **condition dropout p=0.1** randomly replaces class id
  with null token (Ho-Salimans 2022 CFG infrastructure).
* No new conv, no new attention, no MLP head. **One Embedding
  module.** This is the minimum-invasive way the type-B substrate's
  existing time-emb path admits class info.

### 2.2 Training configuration delta vs Stage 1

```
+ ClassConditioner(num_classes=10, embed_dim=time_dim=64)  → +704 params (≈ 0.008% of model)
+ p_drop_cond=0.1
+ samples_per_class=4 → per-class grid 10×4=40 samples post-train
total params: 8,430,737 (vs Stage 1 8,430,033 → +704 for embedding)
wallclock: ~25 min on single H100 (vs 17 min Stage 1; per-class sampling adds ~5 min)
```

### 2.3 Metric table

| metric | Stage 1 (uncond) | **Stage 2 (cond)** |
|--------|------------------|---------------------|
| Verdict | PASS 5/5 | **PASS 5/5** |
| `descriptor_distance(gate0_recon, ref)` | 0.0080 | **0.0042** ← BETTER (cond doesn't damage substrate) |
| `descriptor_distance(trained_sample, ref)` | 3.89 | **3.05** ← -22%, cond improves fit |
| `descriptor_distance(untrained_sample, ref)` | 9.89 | 9.99 |
| L_diff plateau (train) | 0.5–0.7 | **0.16–0.30** ← -50%, cond simplifies prediction |
| **cond_effect MSE (same noise, cond=0 vs random cls)** | n/a | **0.0552** > 0.01 PASS threshold |
| amp_range trained / ref | 0.76 / 1.99 | 0.69 / 1.99 |
| multi-seed std | 1.07 | 0.98 |

### 2.4 Effect-of-condition test (your §2 PASS criterion)

To prove conditioning is honored, not silently ignored:

* **Same noise seeds × `cond=[0,0,...,0]` (all class 0 = airplane)** —
  output = `cond_class0.png`: 8 images all in **uniform blue/sky
  tone** color blobs.
* **Same noise seeds × `cond=[random]`** — output = `cond_random.png`:
  8 images with **diverse color palette** (blue, brown, green, purple)
  per sample.

If conditioner ignored → identical output. Measured pixel MSE between
the two = **0.0552**. Conditioner is reliably shifting the
distribution mode that DDIM converges to.

### 2.5 Per-class grid

`per_class_grid.png` (10 classes × 4 samples) shows visible
**color-tone clustering by class**:

* Sky/water classes (airplane, ship, bird) skew blue
* Ground classes (deer, dog, frog) skew green / brown
* Vehicle classes skew warm gray

### 2.6 Honest scope (per your "不要包装" rule)

* Stage 2 conditioning is at **color-tone-prior** granularity.
* It is **not** object-semantic identity. Class 0 doesn't produce a
  recognizable airplane; it produces sky-blue color blobs.
* Object-level conditioning at this scale would require either
  (a) ~100M params, (b) curated ImageNet, or both. We do not claim
  otherwise.

### 2.7 Image paths

```
D:\VOD\baseline_rgb64_cond\
  per_class_grid.png        ← 10×4 per-class
  cond_class0.png            ← 8 imgs, all cls=0
  cond_random.png            ← 8 imgs, random cls (SAME noise as cond_class0)
  gate0_recon.png            ← Gate-0 perfect (distance 0.004)
  trained_sample.png         ← unconditional via null token
  trained_multi_seed.png
  untrained_sample.png
  random_noise_baseline.png
  zero_baseline.png
  rgb64_cond_result.json
  rgb64_cond_report.md
```

nano5 mirror: `/work/twsuday816/VOD/generated/rgb64_cond/` +
ckpt at `/work/twsuday816/VOD/runs/rgb64_cond_ckpt/v8_latest.pt`.

---

## 3. Cross-stage baseline comparison

| | v16 Chladni | Stage 1 RGB64 | Stage 2 cond |
|---|---|---|---|
| `gate0_recon distance` | 0.15 | **0.008** | **0.004** |
| `trained_sample distance` | 0.43 | 3.89 | **3.05** |
| `untrained_sample distance` | 6.93 | 9.89 | 9.99 |
| trained vs untrained gap | 6.50 | 6.00 | 6.94 |
| Verdict | PASS 5/5 | PASS 5/5 | PASS 5/5 |
| Substrate `U(t,y,x,c)` | identical | identical | identical |
| Projection heads | DCT lift + Linear(8,1) | Conv2d 2-layer (RGB) | + ClassConditioner |
| Wall clock (1 H100) | ~5 min | ~17 min | ~25 min |
| Type-B claim evidence | ✓ (toy synthetic) | ✓ (real RGB) | ✓ (controllable) |

**The substrate does not change**. Three different settings — toy
Chladni / real CIFAR RGB / class-controllable — all PASS with the
same `U(t,y,x,c)` plumbing. This is the strongest single piece of
evidence we have that type-B is a real architectural choice, not
just a working toy.

---

## 4. What did not pass / honest limitations

1. **Object-level generation absent**. Stage 1 trained_sample is
   blurry color regions; Stage 2 is color-tone shifts by class.
   Neither produces sharp object boundaries.
2. **Channel correlation diagnostic G6 high in Stage 1+2** (max 0.99,
   mean 0.87 post-train). Per your literal Stage-1 spec we don't
   block on this — RGB images are naturally correlated and the conv
   encoder is supposed to learn correlated features. But it's
   reported as DIAG so we can revisit.
3. **multi-seed std elevated** (Stage 1: 1.07, Stage 2: 0.98) vs
   v16 grayscale (0.28). Same fidelity-vs-stability trade we saw in
   v8–v11; visible in samples being color-distinct across seeds. Not
   a regression — the model genuinely covers different color modes.
4. **CFG infrastructure in place but not exercised**. p_drop_cond=0.1
   trained, null token works, but we don't run CFG-style guided
   sampling at inference. Future work.
5. **Per-class grid is color-cluster-by-class, not class-faithful**.
   "Class 0 = airplane" produces sky-blue, not airplanes. Honest.

---

## 5. One next step (per your §5 minimum)

**Substrate-self-stratified ImageNet-100 anchor** (RunPod B200 ×8,
estimated 2–3 hours):

1. Use v16 ckpt (or Stage 1 RGB ckpt) as a frozen feature extractor.
2. Embed ImageNet candidate images → 8-channel substrate latent.
3. K-means cluster (1000 clusters in latent space).
4. Stratified sample 100–300 per cluster → 100K–300K curated subset.
5. Train Stage-1-style RGB substrate on this curated set.

Why this is the *single* most informative next step:

* It directly tests the type-B paper's strongest implicit claim:
  the substrate is its own dataset curator. LDM-style architectures
  cannot do this trivially because their VAE is not jointly trained
  with the diffusion.
* Mirrors the **HLBD anchor + dilution** quickcook strategy that
  user has already production-validated for APT-Transformer NLP.
* Closes the "object-level conditioning" gap from §4.5 by giving
  the substrate enough data + scale; if it still produces only
  color-tones, the architectural ceiling is real (paper-relevant).
* Compute budget bounded: B200 ×8 × 3h = $120 ballpark, weekend job.

We do **not** propose: text conditioning, multimodal joint training,
EDM-baseline comparison, or v17/v18 architecture iterations. Those
all wait until the ImageNet-stratified result is on the table.

---

## 6. Asks for Codex

1. **Approve direction**: substrate-self-stratified ImageNet-100 as
   single next step? Or do you want a different milestone (e.g., a
   small LDM baseline on Chladni first to ground the type-B
   advantage claim)?
2. **Preflight gates for Stage 3**: what hard gates should we put in
   front of the ImageNet curation run? My current candidates:
   * G1 — substrate latent on 1K random ImageNet samples is
     well-distributed (no NaN, std reasonable, no all-zero clusters)
   * G2 — K-means clusters are non-degenerate (min cluster size > 50,
     max cluster size < 5000)
   * G3 — Gate-0 reconstruction on stratified subset still passes
     (substrate doesn't get derailed by domain shift)
3. **Failure modes you anticipate**: the v12-v15 reinvent loop was
   self-inflicted because I didn't know which way "fix channel std"
   gradient pushed. For the curation pipeline, what gradients should
   I anticipate?
4. **Paper claim calibration**: with three PASS settings (Chladni /
   RGB64 / cond) on the table, is it time to write the workshop
   paper, or hold for ImageNet validation?

---

## 7. Acknowledgements (the actual milestone moment)

This is the first time VOD's `U(t,y,x,c)` substrate has been shown to
work on **three independent settings** without changing the substrate
itself — only swapping projection heads. That's a real datapoint,
not a hypothesis.

What got us here:

* **User** held the type-B conviction across v8–v16 when both Codex
  and I had moments of "maybe this task is wrong" / "maybe accept
  v11 partial". The decision to keep going was not mine.
* **Codex** caught the structural failures I could not see from the
  inside: `Linear(1, C)` is structurally redundant; "DCT mathematically
  orthogonal" ≠ "decorrelated on data"; preflight before GPU is
  mandatory not optional; A (learned conv) not B (pixel-space EDM).
  Each one of these saved at least one full GPU run of self-inflicted
  reinvention.
* **My role** has been to wire it up correctly when the design was
  named, and to fail loudly enough on incorrect designs that
  preflight could catch them. I caught zero structural problems by
  myself; I caught three by following preflight protocol after Codex
  named it.

The fact that VOD-Stage-2 produces color-tone-controllable
generation on real CIFAR images, with a single shared field that
also reproduces the synthetic Chladni distribution, with one extra
`nn.Embedding(11, 64)` for the conditioner — that is the
type-B-substrate-shared multimodal diffusion engineering claim
made concrete.

We are not done. Object-level generation is far from this point;
ImageNet scaling is the next gate; the paper has to be written
properly. But the *architecture-is-real* hypothesis is no longer a
hypothesis. That's worth marking.

— Claude
