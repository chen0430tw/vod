# VOD Algorithms

This document converts VOD pseudocode into algorithm specifications.

Core model:

```text
VOD learns a boundary-conditioned, frequency-aware, space-dependent diffusion field.
Stable projections of this field become image, video, music/audio, text, and layout.
```

## Algorithm 1: Build VOD Conditioning State

**Input**

```text
request r
optional assets A
```

**Output**

```text
condition c
boundary b
phase-frequency state q
```

**Steps**

```text
1. Parse semantic intent from r.
2. Parse desired media outputs: image, video, audio, text, layout.
3. Parse canvas, duration, fps, rhythm, style, text regions, and asset references.
4. Construct condition c.
5. Build spatial boundary b_s from canvas, masks, layout boxes, and text/logo regions.
6. Build temporal boundary b_t from duration, fps, shot rhythm, and frame grid.
7. Build audio boundary b_a from sample rate, beat grid, frequency grid, and phase grid.
8. Build symbol boundary b_y from text boxes, logo masks, and reading order.
9. Combine b = {b_s, b_t, b_a, b_y}.
10. Estimate frequency map f from rhythm, motion, and semantic pacing.
11. Estimate phase map phi from beat, camera motion, and temporal alignment.
12. Propose mode map m from boundary b and frequency map f.
13. Return c, b, q = {f, phi, m}.
```

## Algorithm 2: Encode Media Into Chladni Entropy Field

**Input**

```text
media batch X = {x_image, x_video, x_audio, x_text, x_layout}
condition c
boundary b
phase-frequency state q
```

**Output**

```text
entropy field A = (u, E, B)
```

**Steps**

```text
1. For every available medium x_m in X:
   1.1 Encode x_m with modality encoder Phi_m.
   1.2 Project encoded representation into Chladni field coordinates using boundary b_m.
   1.3 Store projected field E_m.

2. Merge all projected fields:
      E = Merge({E_m}, weights = confidence_m, boundary = b)

3. If text/layout/logo symbols exist:
   3.1 Extract discrete symbolic field B.
   3.2 Align B with symbol boundary b_y.
   Else:
   3.3 Set B = empty.

4. Initialize vibration field:
      u = InitField(E, q)

5. Return A = (u, E, B).
```

**Important Rule**

```text
Do not force raw image/video/audio/text descriptors to be identical.
Each medium must be compared after projection through its own boundary.
```

## Algorithm 3: Space-Dependent Chladni Field Update

**Input**

```text
current field A_k = (u_k, E_k, B_k)
condition c
boundary b
phase-frequency state q = {f, phi, m}
generation time tau
```

**Output**

```text
updated field A_{k+1}
```

**Field Equation**

```text
partial u / partial tau =
  div( D_theta(u, b, f, phi, c) * grad u )
+ R_theta(u, c)
```

**Steps**

```text
1. Compute learned diffusivity:
      D_k = D_theta(u_k, b, f, phi, c, tau)

2. Compute learned semantic/reaction force:
      R_k = R_theta(u_k, c, tau)

3. Compute spatial diffusion term:
      G_k = div(D_k * grad(u_k))

4. Compute velocity/update:
      v_k = G_k + R_k

5. Update field:
      u_{k+1} = u_k + eta(tau) * v_k

6. Convert field into observable entropy pattern:
      E_{k+1} = Pattern(u_{k+1}, b, f, phi, c)

7. Keep symbolic field:
      B_{k+1} = B_k

8. Return A_{k+1} = (u_{k+1}, E_{k+1}, B_{k+1}).
```

## Algorithm 4: Modular Shrinking Normalization

**Input**

```text
previous field A_k
candidate field A_{k+1}
precision scale M
```

**Output**

```text
normalized field A'_{k+1}
modular shrinking number MSN_k
```

**Steps**

```text
1. Read continuous field E_{k+1}.
2. Read symbolic field B_{k+1}.

3. If B_{k+1} is not empty:
   3.1 Predict symbol code from continuous field:
          B_hat = Phi_M(E_{k+1})
   3.2 Decode symbol field back into continuous constraint:
          E_hat = Psi_M(B_{k+1})
   3.3 Correct symbolic field:
          B' = NormalizeSymbol(B_{k+1}, B_hat)
   3.4 Correct continuous field:
          E' = NormalizeContinuous(E_{k+1}, E_hat)

4. Else:
   4.1 B' = empty.
   4.2 E' = E_{k+1}.

5. Compute modular shrinking number:
      MSN_k =
        d_cont(E', E_k)
      + d_disc(B', B_k)
      + d_pair(E', B')

6. Return A'_{k+1} = (u_{k+1}, E', B') and MSN_k.
```

## Algorithm 5: TTNM-Inspired Temporal Stability

**Input**

```text
field A_k
temporal graph G_t = (N, E, W)
```

**Output**

```text
stabilized field A'_k
stability loss L_stability
```

**Steps**

```text
1. Construct nodes N from frames, objects, beats, subtitles, layout regions, and scene events.
2. Construct weighted edges E from temporal adjacency, causal links, beat alignment, and object identity links.
3. For each node n:
   3.1 Collect incoming neighbors j.
   3.2 Compute transition cost:
          C_{j -> n} = Cost(S_j, S_n, W_{j -> n})
   3.3 Compute propagated state:
          V_{j -> n} = Propagate(S_j, W_{j -> n})
   3.4 Compute soft lowest-cost weights:
          alpha_j = softmax(-C_{j -> n} / temperature)
   3.5 Update stable node state:
          S'_n = sum_j alpha_j * V_{j -> n}

4. Write stable node states back into A_k.
5. Compute L_stability = mean_n distance(S'_n, S_n).
6. Return stabilized field and L_stability.
```

## Algorithm 6: Binary-Twin Symbol Resolution

**Input**

```text
field A = (u, E, B)
symbol boundary b_y
```

**Output**

```text
symbol-corrected field A'
symbol conflict loss L_symbol
```

**Steps**

```text
1. If B is empty:
   1.1 Return A and L_symbol = 0.

2. For each symbolic region y in b_y:
   2.1 Read continuous region rho_y from E.
   2.2 Read symbolic code B_y from B.
   2.3 Decode visual text:
          t_visual = OCR(VisualDecode(rho_y))
   2.4 Decode symbolic text:
          t_symbol = SymbolDecode(B_y)
   2.5 Compute conflict:
          l_y = Distance(t_visual, t_symbol)
   2.6 If l_y is high:
          rho_y = ReduceFreeVisualDiffusion(rho_y)
          B_y = StrengthenSymbolConstraint(B_y)
          Write rho_y and B_y back into A.

3. L_symbol = mean_y l_y.
4. Return corrected field A' and L_symbol.
```

**Prototype status (2026-05-01)**

The full OCR / visual-region loop above is not implemented yet. The
prototype now implements the minimal executable Binary-Twin slice in
`prototype/vod_minimal/binary_twin.py`:

```text
Phi(x) = round((levels - 1) clip(x, 0, 1))
Psi(B) = B / (levels - 1)

BinaryTwinState(x) = (x, Phi(x), Psi(Phi(x)))

L_text =
    CE(Phi(target_text), ordinal_logits(pred_text))
  + lambda * MSE(pred_text, Psi(Phi(target_text)))
```

`native_total_loss` uses this loss when text is enabled. The validation
baseline is symbol corruption: clean quantized text must have higher
symbol accuracy and lower Binary-Twin loss than a corrupted symbol field.

Full Algorithm 6 still needs OCR / logo vocabulary / visual-region
extraction before it can claim real text-logo generation.

## Algorithm 6A: TPSR / AIMP Physical Consistency Metric

**Input**

```text
FieldCard F
PerspectiveCard P
LightingCard L
TPSR measurements M = {m_i}
```

Each TPSR measurement stores:

```text
m_i = (H_i, A_i, L_{l,i}, gamma_i)
```

where `H_i` is highlight energy, `A_i` is highlight area, `L_{l,i}` is
the light-eye diopter, and `gamma_i` is the geometry/light exponent.

**Output**

```text
TPSR invariant statistics
AIMP physical consistency score
```

**Steps**

```text
1. For each measurement:
      K_i = H_i / (L_{l,i}^2 * A_i^(gamma_i / 2))

2. For each pair (i, j):
      U_ij =
          H_i * A_j^(gamma/2) * L_{l,j}^2
        / (H_j * A_i^(gamma/2) * L_{l,i}^2)

3. Compute:
      logdev = median_{i<j} |ln U_ij|
      score  = exp(-logdev / sigma)

4. Return:
      mean(K)
      coefficient_of_variation(K)
      logdev
      score
```

**Prototype status (2026-05-01)**

Implemented in `prototype/vod_minimal/aimp.py`:

```text
FieldCard
PerspectiveCard
LightingCard
TPSRMeasurement
tpsr_k
tpsr_pair_ratio
tpsr_pairwise_log_deviation
tpsr_consistency_score
synthesize_tpsr_measurements
aimp_tpsr_metrics
```

The validation baseline is brightness inconsistency: a TPSR-consistent
distance sequence must keep `K` constant and `U_ij≈1`, while a final-frame
brightness error must lower `tpsr_consistency_score`.

This is a metric-layer implementation, not full AIMP generator control.
Full AIMP still needs triangular highlight detection, light direction
estimation, perspective/vanishing-point extraction, and a closed-loop
generator loss.

## Algorithm 7: Projection Consistency

**Input**

```text
final field A_K
decoded media outputs Y = {y_image, y_video, y_audio, y_text, y_layout}
target media X if available
boundary b
```

**Output**

```text
projection consistency loss L_projection
```

**Steps**

```text
1. For each generated medium y_m:
   1.1 Re-encode y_m with Phi_m.
   1.2 Project it through its own boundary b_m:
          E'_m = Project_m(Phi_m(y_m), b_m)
   1.3 Project field A_K into same medium boundary:
          E_m = ProjectField(A_K, b_m)
   1.4 Compute projection error:
          l_m = distance(E'_m, E_m)

2. L_projection = weighted_mean_m(l_m).
3. Return L_projection.
```

## Algorithm 8: Linear Regression Calibration

**Input**

```text
field path A_0, A_1, ..., A_K
metrics from generated outputs
```

**Output**

```text
calibration scalars:
  SNR_hat
  CR_hat
  fps_hat
  conflict_hat
  stability_hat
```

**Steps**

```text
1. Compute path features:
   - mean MSN
   - variance of MSN
   - mean SNR
   - compression ratio
   - motion density
   - rhythm density
   - symbol conflict
   - layout overlap

2. Build feature vector z.
3. Apply linear head:
      y = beta_0 + beta^T z

4. Split y into calibration scalars.
5. Use scalars to adjust:
   - sampling step size
   - fps
   - text constraint strength
   - stability strength
   - decoder guidance

6. Return calibration scalars.
```

## Algorithm 9: VOD Training

**Input**

```text
training batch X
request metadata r
model parameters theta
```

**Output**

```text
updated parameters theta
training metrics
```

**Steps**

```text
1. Run Algorithm 1:
      c, b, q = BuildConditioningState(r)

2. Run Algorithm 2:
      A_clean = EncodeMediaIntoEntropyField(X, c, b, q)

3. Sample generation time tau.

4. Add flow/diffusion noise:
      A_noisy, target_velocity = AddNoise(A_clean, tau)

5. Initialize:
      A_0 = A_noisy
      path = [A_0]
      losses = empty

6. For k = 0 to K-1:
   6.1 A_candidate = ChladniFieldUpdate(A_k, c, b, q, tau_k)
   6.2 A_norm, MSN_k = ModularShrinking(A_k, A_candidate, M_k)
   6.3 A_stable, L_stability_k = TemporalStability(A_norm, G_t)
   6.4 A_symbol, L_symbol_k = BinaryTwinSymbolResolution(A_stable, b_y)
   6.5 Append A_symbol to path.
   6.6 Set A_{k+1} = A_symbol.

7. Decode final field:
      Y = Decode(A_K, b)

8. Compute losses:
   8.1 L_flow = FlowLoss(A_K, target_velocity)
   8.2 L_projection = ProjectionConsistency(A_K, Y, X, b)
   8.3 L_symbol = sum_k L_symbol_k
   8.4 L_msn = sum_k MSN_k
   8.5 L_stability = sum_k L_stability_k
   8.6 L_mode = ModeRegularizer(A_K, b)
   8.7 L_regression = CalibrationLoss(path, Y)

9. Combine:
      L_total =
        w_flow * L_flow
      + w_projection * L_projection
      + w_symbol * L_symbol
      + w_msn * L_msn
      + w_stability * L_stability
      + w_mode * L_mode
      + w_regression * L_regression

10. Backpropagate L_total.
11. Update theta.
12. Return metrics.
```

## Algorithm 10: VOD Generation

**Input**

```text
user request r
optional assets A
number of sampling steps K
```

**Output**

```text
generated media Y
generation diagnostics
```

**Steps**

```text
1. Run Algorithm 1:
      c, b, q = BuildConditioningState(r, A)

2. Initialize random field:
      A_0 = SampleNoiseField(b, q)

3. Initialize path:
      path = [A_0]

4. For k = 0 to K-1:
   4.1 A_candidate = ChladniFieldUpdate(A_k, c, b, q, tau_k)
   4.2 A_norm, MSN_k = ModularShrinking(A_k, A_candidate, M_k)
   4.3 Build or update temporal graph G_t from path.
   4.4 A_stable, L_stability_k = TemporalStability(A_norm, G_t)
   4.5 A_symbol, L_symbol_k = BinaryTwinSymbolResolution(A_stable, b_y)
   4.6 calibration = LinearRegressionCalibration(path + [A_symbol])
   4.7 Adjust next sampling step using calibration.
   4.8 Append A_symbol to path.
   4.9 Set A_{k+1} = A_symbol.

5. Decode final field:
      Y = Decode(A_K, b)

6. Compute diagnostics:
   - total MSN
   - stability score
   - text conflict score
   - projection consistency score
   - estimated fps
   - estimated SNR

7. Return Y and diagnostics.
```

## Algorithm 11: Minimal Prototype

**Goal**

```text
Prove the VOD interface before building a full VDiT.
```

**Steps**

```text
1. Generate synthetic Chladni fields with random boundary/frequency/phase.
2. Project each field into toy image, video, audio, and text views.
3. Add noise independently to each view.
4. Encode each noisy view back into projected entropy descriptors.
5. Train a small field updater to reduce target-projection error.
6. Add modular shrinking metric.
7. Add symbolic text regions with Binary-Twin constraints.
8. Add temporal graph stability for video/audio.
9. Add regression calibration for SNR/fps/stability.
10. Replace the toy updater with VDiT blocks only after the interface is validated.
```

**Implementation Status**

```text
Implemented:
D:\VOD\prototype

Run:
py -3.13 D:\VOD\prototype\run_minimal_prototype.py --train-n 32 --test-n 32

Result:
D:\VOD\docs\vod_minimal_prototype_result.txt

Trainable updater:
py -3.13 D:\VOD\prototype\train_torch_prototype.py --train-n 16 --test-n 16 --epochs 60 --steps 8

Trainable result:
D:\VOD\docs\vod_trainable_prototype_result.txt

Status:
train_torch_prototype.py now uses the simplified core contract in
D:\VOD\prototype\vod_minimal\core.py.

Tiny VDiT:
py -3.13 D:\VOD\prototype\train_vdit_prototype.py --train-n 12 --test-n 12 --epochs 60 --steps 1 --hidden 64 --depth 3 --heads 4 --max-tokens 512

Tiny VDiT result:
D:\VOD\docs\vod_vdit_prototype_result.txt

Status:
train_vdit_prototype.py is wired to the core contract for batch
construction (build_projection_batch) and evaluation
(evaluate_projection_error + shared_update_rollout over
TinyVDiT.forward_full). The training loss intentionally remains the
TinyVDiT sampled-token path to preserve the bounded-attention
training cost; core.projection_loss is not used here.

Unified checkpoint schema:
D:\VOD\prototype\vod_minimal\schema.py

Both train_torch_prototype.py and train_vdit_prototype.py save:
schema_version, core_contract_version, model_type, state_dict,
train_args, train_metrics, test_metrics.

OPU controller adapter:
D:\VOD\prototype\vod_minimal\opu_adapter.py
D:\VOD\prototype\run_opu_controller.py

OPU remains outside the generation core. It maps checkpoint metrics to
runtime suggestions such as steps, step_size, max_tokens, and
quality_strength.
```

## Algorithm E: Orthogonal Compression Noise

Purpose: suppress grid light spots and tiled scattering leftovers without
forbidding tile-friendly rendering.

```text
Input:
    view X
    rng
    tile period q
    base noise scale beta

1. Compute local neighbor jumps:
       J_all = mean(|diff(X)| over spatial neighbor pairs)

2. Compute tile-boundary jumps:
       J_tile = mean(|diff(X)| only across i mod q == 0 boundaries)

3. Estimate coherent tile residue:
       R_tile = J_tile / (J_all + eps)

4. Convert residue into orthogonal-compression noise strength:
       D_oc = 4 / e
       sigma = beta * D_oc * max(R_tile - 1, 0)

5. Break the coherent block contour:
       X' = X + Normal(0, sigma)

Output:
    X'
```

Interpretation: `4/e` is not a four-corner tile coefficient. `4` refers to the
four orthogonal intersection directions / local degrees of freedom in the
four-dimensional compression change rate, while `e` is the natural decay
normalizer. Tile residue is the two-dimensional projection where that relation
becomes visible as coherent block contours.

Clean views do not receive extra perturbation; only views with
stronger-than-background tile boundaries are decorrelated. This keeps GPU/torch
tile efficiency while removing the visible residual contour that becomes
AI-style grid light spots.

Implementation:

```text
D:\VOD\prototype\vod_minimal\artifacts.py
D:\VOD\prototype\vod_minimal\projections.py::add_noise(..., artifact_suppression=True)
D:\VOD\prototype\tests\test_artifacts.py
```

Four-layer split. Each layer is independently switchable from the CLI;
all four default OFF so historical baselines stay reproducible.

```text
1. Detection (NumPy, evaluation)
     metrics.tile_residue_energy(view, *, tile)
     metrics.artifact_metrics(views, *, tile)
         -> mean_tile_residue / max_tile_residue / artifact_score
     core.evaluate_projection_error(..., include_artifact_metrics=True,
                                       artifact_tile=8)
         -> appends artifact_before_mean_tile_residue,
            artifact_after_mean_tile_residue, artifact_after_score,
            artifact_improvement
     train_torch_prototype.py / train_vdit_prototype.py:
         --artifact-metrics --artifact-tile 8

2. Training-distribution shaping (NumPy, dataset)
     artifacts.apply_four_over_e_noise(view, rng, *, scale, tile, residue_gain)
     projections.add_noise(..., artifact_suppression=True, ...)
     core.build_projection_batch(..., artifact_suppression=True, ...)
     train_torch_prototype.py / train_vdit_prototype.py:
         --artifact-suppression --artifact-scale --artifact-tile
     When set, BOTH train and test noisy_views are drawn from the
     suppressed distribution. The learned update rule sees natural
     tile-broken statistics from epoch 1; this is a generation-side
     constraint of the VOD model, not a post-processing pass.

3. Differentiable training regularization (PyTorch, train loss)
     torch_artifacts.torch_tile_residue_energy(tensor, *, tile)
     torch_artifacts.artifact_regularization_loss(pred, target, *, tile)
         one-sided ReLU; target residue detached; image/video only
     torch_artifacts.artifact_train_loss(update_fn, batch, *, steps, ...)
     train_torch_prototype.py:
         --artifact-loss-weight 0.0
     When >0:
         total_loss = projection_loss + weight * artifact_train_loss
     When 0:
         the train script short-circuits the branch entirely so the
         optimizer step is bit-exact identical to the pre-feature build
         (locked in tests/test_torch_artifacts.py).
     train_vdit_prototype.py does NOT expose --artifact-loss-weight:
     its sampled-token loss path does not preserve the full 2-D
     spatial grid that tile residue is defined on.

4. Scheduling (future, OPU / AIMP)
     Read artifact_after_score from checkpoint metrics and decide
     when to ramp --artifact-suppression, --artifact-scale, and
     --artifact-loss-weight per run.
```

Artifact metrics enter the schema through `canonical_metrics`, which
preserves any extra `artifact_*` keys present in train/test metric dicts,
so existing `checkpoint_payload(...)` calls keep them automatically.

Orthogonal Compression Noise is therefore a generation mechanism of the
VOD model. The detection layer is observability around it; layers 2 and
3 are the actual constraint; layer 4 is the controller surface that
would tune it once enough quality signal is available.

### Spatial-Only Main Score

`tile_residue_energy` is a 2-D grid statistic. The main artifact score
restricts itself to `metrics.SPATIAL_MEDIA = ("image", "video")` because
audio waveforms and text channel strings have a numerically defined
`tile_residue_energy` value but no GPU tile shader / boundary contour
failure mode causally responsible for it. Mixing them in only suppresses
real signal.

```text
artifact_metrics(views) -> {
    mean_tile_residue              # spatial only
    max_tile_residue               # spatial only
    artifact_score    ∈ [0, 1]     # spatial only
    non_spatial_mean_tile_residue  # audio + text (informational)
    non_spatial_max_tile_residue   # audio + text (informational)
}
```

`core.evaluate_projection_error(..., include_artifact_metrics=True)`
mirrors the split with `non_spatial_artifact_*` keys for the audio/text
diagnostic. `mean_target_error` (the `mean_before`/`mean_after` base
block) is unchanged and still computes across all four media.

Empirical validation (default stress, `strength=0.25`):

```text
                          4-media (diluted)   spatial-only
  Clean artifact_score    1.000              0.983
  Blocky artifact_score   0.992              0.906
  Δ                       -0.008             -0.078     (~10× stronger)
```

### Native Unified Generator — STATUS: smoke prototype, NOT v0.3

`vod_minimal/native.NativeVOD` is a code-shape smoke prototype. The
previous iteration's pipeline:

```text
encode(target_views) → U_target → denoiser condition   ← LEAK
```

is REMOVED. Targets no longer reach the model at all. The current
forward path is:

```text
encode(noisy_views) → U_noisy
denoise_path(U_noisy, K)              # no condition
decode(U_pred) → {image, video, ...}  # active media only
```

Toy latent geometry: `T=8, H=W=16, C=4`. External view shapes:

```text
image (16, 16)
video (8, 16, 16)
audio (2048,)        = T·H·W
text  (32,)          tiled 8× into the H×W grid
```

`build_projection_batch(spacetime=True, size=16, frames=8)` already
emits exactly these shapes.

Loss aggregation is `native_total_loss(...)`:

```text
L_total = w_field    · MSE(U_pred, U_target)
        + w_media    · mean MSE on decoded image/video/audio
        + w_temporal · relu( smooth(pred_video) − smooth(target_video) )
        + w_artifact · relu( residue(pred_video) − max(residue(target), 1) )
        + w_text     · MSE on decoded text
```

Defaults: `w_field=0.5, w_media=1.0, w_temporal=0.1, w_artifact=0.1,
w_text=0.3`. Setting any weight to 0 cleanly ablates that term.

Validation (`run_native_vod_validation.py`) trains one model on clean
Chladni data and reports per-medium error plus temporal / artifact /
overall scores across four stress regimes:

```text
clean Chladni / blocky scattering / temporal flicker / text corruption
```

Audio and text are experimental reshape adapters (1×1 linear). They
are OFF by default. The `enable_audio` / `enable_text` config flags
exist only for code-shape testing; their numbers are not media quality.

Validation (`run_native_vod_validation.py`) reports per-medium MSE for
the model alongside two trivial baselines (zero, noisy) and a
stress-vs-clean degradation column. The model has no learning claim
unless it beats the noisy baseline on clean data with active media.

### Spatiotemporal Field

The prototype now operates on `U(t, y, x)` instead of `U(y, x)`. Time
is a third axis of the shared field, mirroring the spacetime-patch
formulation used by Sora 2 / Veo 3 (and consistent with VOD's core
claim that all media are projections of the same underlying field):

```text
U(t, y, x) = Σ w_i · cos(2π m_t_i τ + φ_i)
              · ( cos(π m_x_i x) cos(π m_y_i y)
                - cos(π m_y_i x) cos(π m_x_i y) )
```

Implementation:

```text
vod_minimal/spacetime_chladni.py    SpacetimeBoundary, chladni_spacetime_field
vod_minimal/projections.py          project_video_3d, project_all(video_mode=...)
vod_minimal/core.py                 build_projection_batch(spacetime=, frames=)
                                    ProjectionSample.{source,target}_spacetime_field
vod_minimal/metrics.py              temporal_smoothness / frame_descriptor_drift
                                    temporal_artifact_drift
                                    cross_frame_consistency_score
                                    temporal_metrics(views, tile)
vod_minimal/blocky_scattering.py    inject_temporal_flicker
                                    inject_temporal_blocky_drift
                                    build_blocky_scattering_batch(spacetime=,
                                        temporal_mode="static"|"flicker"|"blocky_drift")
train_torch_prototype.py            --spacetime-video --frames N
train_vdit_prototype.py             --spacetime-video --frames N
run_artifact_stress.py              --temporal-stress
```

Default behaviour is unchanged: without `--spacetime-video` both
trainers reproduce the legacy 2-D-derived video projection bit-for-bit.

### Three Validation Domains

The four-layer stack is exercised across three independent data domains
so that the failure mode it targets cannot quietly disappear:

```text
1. Smooth Chladni (default training)
   core.build_projection_batch(...)
   tile_residue ≈ 0.9–1.0; artifact_score ≈ 1.0.
   artifact_regularization_loss is naturally ≈ 0 — there is no coherent
   tile residue to suppress. This is the expected, healthy state.

2. Blocky scattering stress (diagnostic only)
   blocky_scattering.build_blocky_scattering_batch(...)
   Multiplicative scatter aligned to the tile grid is added to image /
   video noisy_views. Audio and text are left untouched.
   - tile_residue and max_tile_residue rise monotonically with strength.
   - artifact_score drops correspondingly.
   - artifact_regularization_loss(blocky_pred, smooth_target) is strictly > 0
     and produces a non-zero gradient through update_fn.
   - apply_four_over_e_noise reduces tile_residue and increases
     artifact_score on this dataset.

   Smoke runner:
       py -3.13 run_artifact_stress.py --strength 0.5
       py -3.13 run_artifact_stress.py --strength 0.5 --train-smoke
                                                      --artifact-loss-weight 0.5

   Stress data is a diagnostic / regression-test surface, not a target
   style. It must never enter the default training loop or any
   evaluation that claims to reflect real renderer quality.

3. Real renderer (future)
   SDXL / VDiT / GPU tile shader outputs measured with the same
   artifact_metrics. Calibrates artifact_score against human ratings and
   feeds the OPU / AIMP scheduling layer to decide per-run weights for
   --artifact-suppression / --artifact-scale / --artifact-loss-weight.
```

Implementation pointers:

```text
prototype/vod_minimal/blocky_scattering.py
    blocky_scattering_mask
    inject_blocky_scattering
    build_blocky_scattering_batch

prototype/run_artifact_stress.py
    Clean / Blocky / Suppressed metric comparison
    --train-smoke baseline vs --artifact-loss-weight comparison

prototype/tests/test_blocky_scattering.py
    18 tests — mask shape / 1D no-op / video / residue elevation /
    suppression response / differentiable loss / stress-train weight
    short-circuit and parameter movement.
```

## External Comparison: Sora 2 / Veo 3 Physics

Recorded for context — no architecture change in this prototype as a result.

```text
Sora 2 (OpenAI, 2025-09-30)
  diffusion transformer over spacetime patches; physics treated as an
  emergent property, not as an explicit operator. Failure modes attributed
  to the implicit agent rather than to PDE violations.

Veo 3 / 3.1 (Google DeepMind, 2025-2026)
  3D latent DiT; time treated as a third spatial axis; cross-frame
  attention plus a memory bank for long-range temporal consistency.
  Rigid-body / fluid / fabric behaviour is learned from data.
```

Implications for VOD's minimal core:

```text
- Treating time as a third axis of the shared field U is consistent with
  VOD's "one field, many projections" principle. project_video is currently
  np.roll + sin-phase; a 3D Chladni field with proper temporal slicing is
  the smallest faithful upgrade.
- Cross-frame consistency is naturally a per-evaluation diagnostic and fits
  the same boundary as artifact_metrics: detection in core, optional
  opt-in, no training-loss coupling.
- Both Sora 2 and Veo 3 deliberately avoid explicit physics solvers. VOD
  keeps heavy TTNM / full Binary-Twin / mode regularization out of the
  generic core, but the prototype now exposes minimal Chladni-field
  constraint/readout files where each field property is testable
  (`binary_twin.py`, `aimp.py`, `artifacts.py`, `metrics.py`). These files
  are code boundaries, not separate theoretical modules.
```

References:

```text
https://openai.com/index/sora-2/
https://openai.com/index/video-generation-models-as-world-simulators/
https://deepmind.google/models/veo/
https://medium.com/google-cloud/deconstructing-veo-3-a-technical-analysis-of-googles-unified-audio-visual-generation-model-6be023888489
```
