# VOD Math Simplification Notes

Date: 2026-04-26

This note compares the current algorithm documents with the running prototype.

## Short Answer

The first runnable VOD model can be simplified to:

```text
1. one latent field U
2. media-specific projections P_m(U)
3. one shared update operator F_theta
4. projection-space loss
```

Everything else should be treated as an optional regularizer or a later-stage controller.

## Keep In Core

### 1. Chladni Field

Keep:

```text
U
```

Meaning:

```text
the shared vibration / entropy field
```

Do not keep separate `u` and `E` in the first implementation. In code, `E = Pattern(u)` is mostly an alias. Keep one tensor:

```text
U_k
```

Only produce `E` when decoding or computing descriptors.

### 2. Media Projection

Keep:

```text
P_m(U)
```

where `m` is:

```text
image, video, audio, text
```

This is the key insight validated by the prototype:

```text
different media should match the same field only after projection through their own boundary
```

### 3. Shared Update Operator

Keep:

```text
U_{k+1} = U_k + eta * F_theta(U_k, condition, coords, medium)
```

In the supervised toy prototype this is:

```text
X_{k+1} = X_k + eta * F_theta(X_k, target_m, smooth(X_k), D_m, pos, medium)
```

The future generative version should remove `target_m` and replace it with conditioning.

### 4. Projection Loss

Keep:

```text
L_proj = sum_m || Phi_m(Y_m) - P_m(U_target) ||
```

or in the current toy version:

```text
L_proj = sum_m || Y_m - target_m ||
```

This is the main metric.

## Simplify Or Delay

### 1. Separate Diffusivity And Reaction Terms

Original:

```text
partial u / partial tau =
  div(D_theta * grad u) + R_theta
```

Simplify to:

```text
U_{k+1} = U_k + eta * F_theta(features)
```

Reason:

Current VDiT already learns the update directly. `D_theta` and `R_theta` are useful interpretation, but they do not need separate modules yet.

Optional later:

```text
Expose D_theta for interpretability or physics regularization.
```

### 2. Phase / Frequency / Mode Maps

Original:

```text
q = {f, phi, mode}
```

Simplify to:

```text
coords = {position, medium_id, optional time}
```

Reason:

Current Tiny VDiT only uses:

```text
position sin/cos
media embedding
```

Frequency and phase can be derived later from data or added as positional channels.

### 3. Modular Shrinking Number

Original:

```text
MSN = d_cont + d_disc + d_pair
```

Simplify to a diagnostic:

```text
MSN_k = mean(abs(U_{k+1} - U_k))
```

Reason:

There is no real symbolic field yet. Keep MSN as path stability metric, not as a normalization operator.

### 4. TTNM Stability

Original:

```text
temporal graph + soft lowest-cost propagation
```

Simplify to:

```text
L_temporal = mean_t || Y_{t+1} - warp_or_shift(Y_t) ||
```

or for the toy prototype:

```text
L_temporal = smoothness(video_time_axis)
```

Reason:

The graph version is overkill until real video/audio samples exist.

### 5. Binary-Twin Symbol Field

Original:

```text
e = (rho, B)
```

Simplify to:

```text
text_loss = CE(symbol_pred, symbol_target) + visual_region_loss
```

Reason:

The idea is useful, but the prototype only has toy quantized text.
As of 2026-05-01 this is no longer a pure placeholder: the prototype
implements the minimal Binary-Twin slice in `vod_minimal/binary_twin.py`:

```text
Phi(x) = round((levels - 1) clip(x, 0, 1))
Psi(B) = B / (levels - 1)
L_text = CE(Phi(target), ordinal_logits(pred))
       + lambda * MSE(pred, Psi(Phi(target)))
```

`native_total_loss` uses this loss when text is enabled and logs
`binary_twin_symbol_accuracy`. Full OCR/logo-region Binary-Twin is still
delayed until real text/OCR supervision exists.

### 6. Linear Regression Calibration

Original:

```text
SNR_hat, CR_hat, fps_hat, conflict_hat, stability_hat
```

Simplify to post-hoc logging:

```text
log_metrics = {snr, target_error, msn, success_rate}
```

Reason:

The current regression head fits synthetic targets. It should not affect training yet.

### 7. Mode Regularizer

Original:

```text
L_mode = || H_b(U) - lambda U ||
```

Delay.

Reason:

Useful once we implement explicit boundary-conditioned operators. For now, synthetic Chladni data already provides the mode prior.

## Simplified Algorithm Set

The current 11 algorithms can be reduced to 4 for first implementation.

### Algorithm A: Build Synthetic Projection Batch

```text
1. Sample Chladni field U_source.
2. Sample Chladni target U_target.
3. Project both into media views:
      X_m = P_m(U_source)
      T_m = P_m(U_target)
4. Add noise:
      X_noisy_m = X_m + noise
5. Return {X_noisy_m, T_m}.
```

### Algorithm B: Shared Field Update

```text
For each medium m:
  X_{k+1,m} = X_{k,m} + eta * F_theta(
      X_{k,m},
      condition_m,
      smooth(X_{k,m}),
      position,
      medium_id
  )
```

In current supervised prototype:

```text
condition_m = T_m
```

Later generation:

```text
condition_m = prompt / boundary / frequency / phase / references
```

### Algorithm C: Projection Training Loss

```text
L = sum_m MSE(X_{K,m}, T_m)
```

Optional:

```text
L += alpha * mean_k abs(X_{k+1,m} - X_{k,m})
```

### Algorithm D: Evaluation

```text
1. Compute target-projection error before update.
2. Compute target-projection error after update.
3. Report improvement and success rate.
```

## Minimal Mathematical Core

Final simplified form:

```text
Given a shared field U and media projections P_m,
learn F_theta such that:

P_m(U_target) ~= Update_theta(P_m(U_noisy), c_m)
```

The first-stage loss:

```text
L_VOD_min =
  sum_m || Froll_theta(P_m(U_source) + eps, c_m) - P_m(U_target) ||^2
```

where:

```text
Froll_theta = K repeated shared update steps
```

## What This Means

For now, VOD does not need:

```text
full TTNM
full Binary-Twin Number
full modular normalization
full regression calibration
explicit PDE operator
explicit mode eigenvalue loss
```

It only needs:

```text
shared Chladni field
projection heads
shared update network
projection-space loss
path metrics
```

The other mathematics should be reintroduced only when a real data failure demands it.

## Implementation Status

The simplified core contract is implemented in:

```text
D:\VOD\prototype\vod_minimal\core.py
```

The first training script has been migrated to this contract:

```text
D:\VOD\prototype\train_torch_prototype.py
```

It now uses:

```text
build_projection_batch
projection_loss
evaluate_projection_error
shared_update_rollout
```

The Tiny VDiT trainer is also migrated to the core contract for
batch construction and evaluation:

```text
D:\VOD\prototype\train_vdit_prototype.py
```

It uses:

```text
build_projection_batch          (data)
evaluate_projection_error       (eval, with shared_update_rollout
                                  wrapping model.forward_full)
```

The training loss deliberately remains TinyVDiT's sampled-token path
(`TinyVDiT.forward_sampled` + scale-normalized MSE) rather than
`core.projection_loss`. Sampled tokens bound attention compute to
`--max-tokens`; switching to full-view loss would change the gradient
signal and inflate per-epoch cost. The core contract is therefore
adopted only at the batch/eval boundary for this script.

Validation:

```text
py -3.13 -m pytest tests/test_core.py -q
py -3.13 train_torch_prototype.py --train-n 16 --test-n 16 --epochs 60 --steps 8
```

Current result:

```text
Test mean_before       9.999230
Test mean_after        1.056039
Test mean_improvement  8.943191
Test success_rate      1.000000
```

## Implementation Map

The simplified algorithm set above is realized in `prototype/vod_minimal/core.py`.
It exposes exactly four backend-agnostic interfaces:

```text
build_projection_batch       Algorithm A: synthetic Chladni source/target +
                             noisy media projections
shared_update_rollout        Algorithm B: K-step rollout of an arbitrary
                             update_fn(current, target, medium) -> next_view
projection_loss              Algorithm C: projection-space MSE training loss
                             (with optional per-medium scale normalization)
evaluate_projection_error    Algorithm D: returns mean_before, mean_after,
                             mean_improvement, success_rate
```

The same four interfaces accept the analytic `MinimalVOD` step, the trainable
`SharedPointUpdater`, or the `TinyVDiT` skeleton as `update_fn`, so no model
class is privileged inside the core.

Reproducible smoke check (no training, no grid search):

```powershell
py -3.13 D:\VOD\prototype\run_core_validation.py
```

Expected: clear `mean_before` → `mean_after` drop and `success_rate` well above
0.5. Best historical numbers in `vod_minimal_prototype_result.txt` are produced
by the grid-searched analytic model in `run_minimal_prototype.py`.

OPU (`prototype/vod_minimal/opu/...` if present) is a reference module copied
from APT-Transformer. It is intentionally NOT imported, called, or required by
the minimal core.

## Unified Checkpoint Schema

Prototype trainers now share a checkpoint/metrics schema:

```text
D:\VOD\prototype\vod_minimal\schema.py
```

Required fields:

```text
schema_version
core_contract_version
model_type
state_dict
train_args
train_metrics
test_metrics
```

Optional model-specific fields:

```text
config
best_epoch
```

This is the boundary OPU and future experiment managers should read. They
should not parse per-script logs.

## OPU Controller Boundary

OPU is now connected as an optional controller adapter:

```text
D:\VOD\prototype\vod_minimal\opu_adapter.py
D:\VOD\prototype\run_opu_controller.py
```

It does not enter the minimal core and does not change the loss. It only maps
checkpoint metrics to runtime suggestions:

```text
low quality       -> increase steps / quality_strength
high hot_pressure -> reduce max_tokens
healthy run       -> relax max_tokens
fault/friction    -> reduce step_size
```

## Native Unified Generator — STATUS: smoke prototype, NOT v0.3

The implementation tagged `native_vod_smoke` is a code-shape smoke
prototype. It was previously labelled "v0.3" and reported as a working
unified generator; that label was incorrect. The earlier iteration fed
`encode(target_views)` into the denoiser as a "condition", which is a
straight answer leak — the denoiser could trivially copy targets
through the condition pathway, so every training, evaluation and
stress number on top of it was unsound.

This iteration is the clean-up:

  * `forward(noisy_views)` — model sees noisy only, no target argument.
  * Targets enter `native_total_loss` only, and `L_field`'s target
    encoding is detached so no gradient flows back through the encoder
    via the target path.
  * Audio / text encoders / decoders are 1×1 linear reshape adapters,
    not real codecs. They are OFF by default; flipping them on does
    not produce meaningful audio / text quality numbers.
  * Every evaluation reports two trivial baselines alongside the model:
    `zero` (predict zeros) and `noisy` (predict the noisy input).

Until the model demonstrably beats both baselines on out-of-stress
clean evaluation it does not have a learning claim.

```text
                  encode                  denoise                 decode
   image (H,W) ─┐                                            ┌─→ image
   video (T,H,W)├─→  U(t, y, x, c)  ─→  shared denoiser  ─→  ├─→ video
   audio (S,)   │   (T=8, H=W=16,        steps × K           ├─→ audio
   text  (L,)  ─┘    C=4 toy)                                └─→ text
```

Implementation:

```text
vod_minimal/native.py
  NativeVOD                   the model (encoders, denoiser, decoders)
  NativeVODConfig             channels / hidden / denoise_steps
  NativeLossWeights           per-component loss weights
  native_total_loss(...)      one-shot multi-loss combiner
  audio_to_grid / text_to_grid / grid_to_audio / grid_to_text
                              shape adapters between media views and U

train_native_vod.py           single trainer that simultaneously fits
                              image / video / audio / text
run_native_vod_validation.py  evaluation across four data domains
                              (clean / blocky / flicker / text-corrupt)

tests/test_native_vod.py      17 tests: reshape round-trips, encode and
                              decode shapes, denoise shape preservation,
                              forward end-to-end, gradient backflow into
                              all 9 subnets, training reduces L_total
                              ≥5%, text corruption injection
```

### Five Losses, One Backward

`native_total_loss` returns `weights · components`:

```text
L_field      MSE between U_pred and the encoded U_target
L_media      MSE on decoded image / video / audio
L_temporal   relu( smooth(pred_video) − smooth(target_video).detach() )
L_artifact   relu( residue(pred_video) − max(residue(target), 1.0) )
L_text       Binary-Twin CE + reconstruction consistency on decoded text
             channel when text is enabled
```

Each weight has a default; setting any weight to 0 ablates that
component. The trainer logs per-epoch values of every component so
ablation effects are visible in the log.

### Gate 0: Visible Output Before More Constraints

Before treating any loss ablation as product progress, the model must
prove basic output capability:

```text
decode(encode(clean_image)) ≈ clean_image
decode(denoise_path(encode(clean_image))) ≈ clean_image
model(clean + noise) beats zero / noisy / identity baselines
```

If this gate fails, 4/e / TTNM / Binary-Twin / AIMP metrics only prove
that auxiliary constraints can fire; they do not prove that VOD can draw.

Minimum missing losses when this gate fails:

```text
L_recon      = MSE(decode(encode(target)), target)
L_clean_noop = MSE(denoise_path(encode(target)), encode(target))
```

### Validation Protocol

`run_native_vod_validation.py` trains a single model on clean Chladni
data then evaluates the same checkpoint against four regimes:

```text
clean Chladni             default training distribution
blocky scattering         spatial 4/e tile residue stress
temporal flicker          per-frame independent noise
text quantization corrupt random level swaps in the text channel
```

Output is a single table with seven columns per row:

```text
img / vid / aud / txt / temporal / artifact / overall
```

The four stress domains exist together because they probe four
genuinely different failure modes: spatial contour stability,
spatiotemporal coherence, character-level discrete consistency, and
end-to-end multi-medium reconstruction. A model that handles only one
regime is not a unified generator.

## Spatiotemporal Upgrade: U(t, y, x)

The earlier prototype kept a single 2-D Chladni image `U(y, x)` and built
the video projection by rolling that image plus a sin-phase shortcut.
That shortcut treated time as a free dimension that did not exist in the
underlying field, so cross-frame consistency was unmeasurable in
principle: every frame was a re-skinned copy of the same 2-D state.

The current prototype lifts the field to:

```text
U(t, y, x) = Σ w_i · cos(2π m_t_i τ + φ_i)
              · ( cos(π m_x_i x) cos(π m_y_i y)
                - cos(π m_y_i x) cos(π m_x_i y) )
```

Time is a *third axis* of the same shared field, exactly the shape
Sora 2 (spacetime patches) and Veo 3 (3-D latent DiT) operate on. VOD
remains "one field, many projections" — there is still a single U
underneath; image / audio / text are taken from its temporal mean,
video is a direct slicing.

```text
metrics.SPATIAL_MEDIA = ("image", "video")

projections.project_video_3d(U)        # slice the volume directly
projections.project_all(U, video_mode="auto" | "2d" | "3d")

core.build_projection_batch(..., spacetime=True, frames=10)
core.ProjectionSample.source_spacetime_field   # U_source (F,H,W) or None
core.ProjectionSample.target_spacetime_field   # U_target (F,H,W) or None

train_torch_prototype.py / train_vdit_prototype.py:
    --spacetime-video --frames 10
```

`--spacetime-video` is opt-in. With the flag absent both trainers keep
the legacy 2-D-derived video projection bit-for-bit, so the historical
checkpoints in `vod_*_prototype_result.txt` remain reproducible.

### Temporal Metrics

The 3-D field makes cross-frame consistency a measurable quantity. New
diagnostics live in `metrics.py`:

```text
temporal_smoothness(video)           mean |frame_t+1 − frame_t|
frame_descriptor_drift(video)        std of per-frame descriptor amp
temporal_artifact_drift(video, tile) std of per-frame tile_residue
cross_frame_consistency_score(...)   bounded [0,1] aggregate
temporal_metrics(views, tile)        per-batch dict for video medium
```

Like the artifact stack, these are evaluation-only. Training loss is
unchanged; trainers can read them via `metrics.temporal_metrics(...)`
on their evaluation views when they want to surface them.

### Temporal Stress Domain

`blocky_scattering.py` adds two diagnostic corruptions on top of the
spatial scatter, both 3-D specific:

```text
inject_temporal_flicker(clip)        per-frame i.i.d. noise
                                     -> raises temporal_smoothness
inject_temporal_blocky_drift(clip)   tile mask rolled across frames
                                     -> raises temporal_artifact_drift

build_blocky_scattering_batch(..., spacetime=True,
    temporal_mode="static" | "flicker" | "blocky_drift")
```

Empirical (size=32, frames=8, strength=0.5):

```text
                        clean    flicker    blocky_drift
temporal_smoothness     0.36     0.67       0.45
temporal_artifact_drift 0.052    0.046      0.081
artifact_score          0.991    0.860      0.850
```

Each stress mode moves a different metric, which is the purpose of
keeping both spatial and temporal diagnostics in the same batch-level
pipeline. As before: stress data is a regression-test surface, not a
target style.

## Why the Artifact Score is Spatial-Only

`tile_residue_energy` is a geometric statistic of a 2-D grid: the mean
absolute jump on tile boundaries divided by the mean absolute jump
across all neighbour pairs. It is well-defined as a number on a 1-D
sequence (audio waveform, text channel string), but the number does not
correspond to any AI-renderer failure mode there — there is no GPU tile
shader emitting coherent block contours into a sound wave.

Earlier the main `artifact_score` averaged `tile_residue_energy` over
ALL media. Audio and text contributed values close to 1.0 with very
little variance, which silently dragged a real spatial change toward
the audio/text baseline. Empirically:

```text
strength=0.25 (the default stress strength), 32x32, tile=8

  4-media-averaged artifact_score:    1.000  →  0.992    (Δ -0.008)
  spatial-only       artifact_score:  0.983  →  0.906    (Δ -0.078)
```

The spatial-only definition recovers a ~10× stronger signal on the same
data, so the redesign is treated as a correctness fix, not a stylistic
choice.

The current contract is:

```text
metrics.SPATIAL_MEDIA = ("image", "video")

artifact_metrics(views) -> {
    mean_tile_residue              # spatial only
    max_tile_residue               # spatial only
    artifact_score                 # spatial only, ∈ [0, 1]

    non_spatial_mean_tile_residue  # audio/text raw residue
    non_spatial_max_tile_residue   # audio/text raw residue
}
```

The non-spatial block is kept for visibility — if a future projection
operator is added that *should* care about boundary structure on a 1-D
medium (a tiled audio synthesizer, for example), the value is already
being measured. It is just never mixed into the main score.

`core.evaluate_projection_error(..., include_artifact_metrics=True)`
mirrors the same split:

```text
artifact_*                          spatial only
non_spatial_artifact_*              audio/text only
```

`mean_target_error` (the base `mean_before` / `mean_after`) still
computes across all four media — projection-space error is meaningful
end-to-end, so the base block is unchanged.

## Orthogonal Compression Noise as a Generation Mechanism

`4/e` (Orthogonal Compression Decay) is treated as a constraint of the VOD
generation model, NOT as a post-processing knob. The visible "AI grid light
spots / tile contours" failure mode is part of what the model must learn to
avoid, so suppression participates in the training distribution and the
training objective — not just in evaluation.

The mechanism is split into four layers; each layer can be enabled or
disabled independently from the CLI:

```text
1. Detection
   metrics.tile_residue_energy
   metrics.artifact_metrics(views, *, tile)
       -> mean_tile_residue / max_tile_residue / artifact_score ∈ [0,1]
   core.evaluate_projection_error(..., include_artifact_metrics=True,
                                   artifact_tile=8)
       -> appends artifact_before_mean_tile_residue,
          artifact_after_mean_tile_residue, artifact_after_score,
          artifact_improvement

2. Training distribution shaping
   artifacts.apply_four_over_e_noise(view, rng, *, scale, tile, residue_gain)
   projections.add_noise(..., artifact_suppression=True, ...)
   core.build_projection_batch(..., artifact_suppression=True,
                                artifact_scale=..., artifact_tile=...)
   -> train_torch_prototype.py / train_vdit_prototype.py:
        --artifact-suppression --artifact-scale --artifact-tile
      When set, BOTH train and test noisy_views are sampled from the
      suppressed distribution, so the learned update rule sees natural
      tile-broken statistics from epoch 1.

3. Differentiable training regularization (train_torch only)
   torch_artifacts.torch_tile_residue_energy(tensor, *, tile)
   torch_artifacts.artifact_regularization_loss(pred, target, *, tile)
       -> one-sided ReLU; only penalises pred residue exceeding
          target residue; target residue is detached so the data
          distribution never receives a gradient. Image and video only.
   torch_artifacts.artifact_train_loss(update_fn, batch, *, steps, ...)
   -> train_torch_prototype.py:
        --artifact-loss-weight 0.0
      When >0, projection_loss + weight * artifact_train_loss is
      backpropagated. weight=0 short-circuits the branch and is
      bit-exact identical to the pre-feature optimizer step (locked
      in tests/test_torch_artifacts.py).

4. Scheduling (future, OPU / AIMP)
   Read artifact_after_score from checkpoint metrics and decide when to
   ramp `--artifact-suppression`, `--artifact-scale`, and
   `--artifact-loss-weight`.
```

Why train_vdit only adopts layers 1–2: TinyVDiT trains on a sampled-token
subset that does not preserve the full 2-D spatial grid, so the
differentiable tile-residue penalty cannot run inside the inner loss. The
`--artifact-loss-weight` flag is therefore intentionally absent on
train_vdit_prototype.py; train_torch_prototype.py is the reference
implementation for the regularization layer until VDiT switches to a path
that retains spatial structure.

### Three Validation Domains

The artifact stack is verified across three data domains, in order of
realism:

```text
1. Smooth Chladni domain (default training data)
   build_projection_batch(...) with no scatter injection.
   tile_residue ≈ 0.9–1.0; artifact_score ≈ 1.0.
   artifact_regularization_loss is naturally near zero. This is correct
   behaviour: there is nothing to suppress.

2. Blocky scattering stress domain (diagnostic only)
   blocky_scattering.build_blocky_scattering_batch(...)
   Adds a tile-aligned multiplicative scatter (per-tile intensity +
   boundary-biased noise) to image / video noisy_views. Audio and text
   are untouched. tile_residue rises monotonically with `strength`;
   artifact_score drops accordingly; artifact_regularization_loss is
   strictly > 0 when the model output preserves the scatter.

   Smoke runner:
       py -3.13 run_artifact_stress.py --strength 0.5
       py -3.13 run_artifact_stress.py --strength 0.5 --train-smoke

   This dataset is NOT a training target style. It exists to prove the
   detector / suppressor / regularizer fire on a known-pathological
   input. It must never be wired into default training.

3. Real renderer domain (future)
   Outputs from SDXL / VDiT / GPU tile renderer fed back into the same
   artifact_metrics. Calibrates artifact_score against human assessment
   and decides production-ready hyperparameters for the four-layer
   stack via OPU / AIMP.
```

All four layers default to OFF so historical baselines remain reproducible
bit-for-bit. CLI examples:

```powershell
# Detection only (eval logging)
py -3.13 train_torch_prototype.py --artifact-metrics

# Distribution shaping (changes training data)
py -3.13 train_torch_prototype.py --artifact-suppression --artifact-scale 0.05

# Distribution + differentiable penalty
py -3.13 train_torch_prototype.py --artifact-suppression --artifact-loss-weight 0.5

# Same on VDiT (no --artifact-loss-weight)
py -3.13 train_vdit_prototype.py --artifact-suppression --artifact-metrics
```

## External Comparison: Sora 2 / Veo 3 Physics

For context, the two main industry references for "physics-consistent" video
generation in 2025–2026 are:

```text
Sora 2 (OpenAI, 2025-09-30)
  diffusion transformer over spacetime patches of video latents
  improved (not perfect) physical plausibility, e.g. ball rebound off
    backboard; failure modes attributed to the implicit agent's mistakes
    rather than to PDE violations

Veo 3 / 3.1 (Google DeepMind, 2025-2026)
  3D latent diffusion (DiT backbone), time treated as a third spatial axis
  cross-frame attention + memory bank for long-range temporal consistency
  rigid-body / fluid / fabric behaviour learned from data, not from
    explicit physics solvers
```

What is portable to VOD's minimal core:

```text
- Treating time as a third axis of the shared field U is consistent with
  VOD's "one field, many projections" stance. Today's prototype's
  project_video is np.roll + sin-phase; a 3D Chladni field with proper
  temporal slicing would be the smallest faithful upgrade.
- Cross-frame consistency is a per-evaluation diagnostic that fits the
  same boundary as artifact_metrics: detection in core, optional opt-in,
  no training-loss coupling. A frame-to-frame stability metric on the
  video projection would slot in as a sibling of artifact_*.
- Sora and Veo do NOT solve physics with explicit operators. VOD keeps
  heavy TTNM / full Binary-Twin / mode regularization out of the generic
  core, but minimal Chladni-field constraint slices now live in focused
  implementation files: `binary_twin.py` for text-symbol readout coupling
  and `aimp.py` for TPSR/AIMP physical readout metrics. These are not
  separate platform modules; they are constraints/readouts of the shared
  field.
```

Sources for the comparison:

```text
https://openai.com/index/sora-2/
https://openai.com/index/video-generation-models-as-world-simulators/
https://deepmind.google/models/veo/
https://medium.com/google-cloud/deconstructing-veo-3-a-technical-analysis-of-googles-unified-audio-visual-generation-model-6be023888489
```

This is a context note only — no architecture changes are made in this
prototype as a result.

## Orthogonal Compression Noise / Tile Residue

AI 绘图里常见的格子光斑、脏块、局部 tile 轮廓，不应简单理解成
"模型不能画 tile"。在 VOD 里更合理的解释是：模型为了节省计算，会把
粒子散射、局部光照和纹理变化打包成小块渲染；失败时，块边界的散射轮廓
没有被自然噪声打散，于是残留成可见光斑或格纹。

这里的 `4/e` 不是 tile 四角的经验系数。`4` 指四维空间压缩变化率中的四个
正交交点 / 局部自由方向，`e` 指自然衰减归一化。tile 只是这个四维正交关系
投影到二维局部张量块以后出现的可见形式。

因此 VOD 的处理不是禁止 tile。禁止 tile 会降低 GPU / torch 友好性，也会
拖慢生成。新的最小规则是：

```text
X' = X + Normal(0, sigma)
sigma = base_scale * D_oc * max(R_tile(X) - 1, 0)
D_oc = 4 / e
```

其中 `R_tile` 是 tile 边界跳变相对全局邻域跳变的残留能量。若画面没有明显
块边界，`R_tile <= 1`，额外噪声为 0；若 tile 边界异常强，则按正交压缩
衰减系数 `D_oc = 4/e` 增加少量零均值微噪声，把块状粒子散射轮廓破相。

这一步属于自然降噪：不是把信息抹平，而是用受控随机性消除不自然的相干
边界。当前 prototype 已在 `vod_minimal.artifacts` 中实现：

```text
ORTHOGONAL_COMPRESSION_DECAY
tile_residue_energy
apply_four_over_e_noise
add_noise(..., artifact_suppression=True)
```

默认训练路径保持关闭，避免改变历史指标；后续可以把 `mean_tile_residue`
加入 checkpoint 指标，并在生成 / 精修阶段按 OPU 或 AIMP 评分动态开启。
