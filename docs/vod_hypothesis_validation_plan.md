# VOD Hypothesis Validation Plan

Date: 2026-05-01

This document indexes the VOD hypotheses (H0–H5) drawn from the early
AI-drawing research note (`docs/vod_full_mathematical_formulation.md`
§3A) and tracks each one against:

```text
Hypothesis
Current implementation status
Metric
Baseline
Falsification condition
Next minimal implementation
```

This is a *plan*, not a results report. For verified PASS/FAIL on the
implemented mechanisms, see `prototype/vod_product_validation_report.md`.

## Status legend

```text
DONE       PASS verified by run_vod_product_validation.py
PARTIAL    minimal implementation passes its narrow test, full claim NOT solved
OPEN       no implementation, hypothesis not yet tested
MONITORING diagnostic exists, no PASS/FAIL criterion yet
```

---

## H0  Shadow Sampling / 阴影采样

**Status**: PASS — operator conformance + test-time application + training-time
integration + continuous strength dial + composite stacking with TTNM,
all verified (2026-05-01 / 2026-05-02).

### Hypothesis

AI image generators behave like *shadow samplers*: prompt features induce
probability shadows over the image field rather than directly operating
on stable object structure. Visible failures (tile light spots, dirty
block contours) are residue of this shadow sampling, not "model can't
draw tiles".

### Current implementation status

`vod_minimal/artifacts.py` implements `OC_{4/e}` operator with the four
spec components: `R_tile`, `r(X,q)`, `N_4`, `w_q(i,j)`, and the final
`OC_{4/e}(X) = X + w_q · N_4`.

Verification cascade (full table in `vod_full_mathematical_formulation.md`
§13 + `VOD_agent_postmortem.md` §12.15-§12.18):

```text
Claim 1A operator conformance (AXCOV signature):       PASS ~13σ
Claim 1B test-time application (sign-agreement, s=0.3): PASS ~7.7σ
Claim 1B training-time integration (floor=0.0 fix):    PASS, monotonic
Claim 1B continuous strength dial:                      PASS, sweep 0-2.0
Composite stacking with TTNM:                           PASS, no interference
```

The original "PARTIAL — only phase-break verified" status was upgraded
when the training-time integration tests revealed (and fixed) the
default `RESIDUE_FLOOR=1.0` that was silently gating the loss to zero
on stress data with `target_tile_residue < 1.0`.

### Metric

```text
local luminance variance
boundary contour coherence
tile residue                       (R_tile)
boundary sign agreement            (operationalisation of "破相")
```

### Baseline

```text
no perturbation                    (X)
iid gaussian, matched energy
iid uniform,  matched energy
OC_{4/e}                           (under test)
```

### Falsification

```text
If OC_{4/e} cannot reduce coherent boundary sign agreement vs iid
under matched perturbation energy at low/medium strength, the
"shadow → constraint" claim is unsupported in its phase-break form.
```

### Next minimal implementation

```text
1. Cross-validate on real AI-renderer outputs (not synthetic halos).
   Acquire SDXL / SD3 / Flux samples with visible tile artifacts and
   re-run boundary_sign_agreement before/after OC_{4/e}.
2. Wire OC_{4/e} into a generation-time scheduler (currently it lives
   only in build_projection_batch's noise pipeline).
```

---

## H1  Mask-as-Boundary / 遮罩即边界

**Status**: OPEN

### Hypothesis

Masks are not just post-processing composites. They are *boundary
conditions* of the generation field. A field with a mask boundary has
a different valid mode set than the same field without it; the model
should learn this difference, not paint over the mask afterwards.

### Current implementation status

No implementation. `vod_minimal` has `Boundary` and `random_boundary`
but they parameterise Chladni mode shape, not spatial masks. There is
no "this region is forbidden / required" boundary in the prototype.

### Metric

```text
inside/outside leakage             (mass that lands on the wrong side)
edge consistency                   (boundary jump magnitude on mask edge)
boundary violation rate            (fraction of pixels violating mask)
```

### Baseline

```text
no mask                            (free generation)
hard mask compositor               (post-hoc paste-over)
boundary-conditioned VOD field     (mask as PDE boundary)
```

### Falsification

```text
If boundary-conditioned VOD does not reduce inside/outside leakage vs
the post-hoc compositor on hard-mask synthetic tasks, mask-as-boundary
gives no advantage and the hypothesis is unsupported.
```

### Next minimal implementation

```text
1. Add a mask channel to `chladni_field`: zero-pad outside mask region.
2. Define `boundary_violation_rate` metric.
3. Compare three baselines on synthetic 2-circle mask:
     - free Chladni
     - free Chladni + post-hoc mask multiply
     - mask-conditioned Chladni (zero outside the boundary in the field)
4. Single-script ablation, ~50 seeds, paired comparison.
```

Estimated: half-day of work, no new dependencies.

---

## H2  Dynamic Zoom / Micro-Imaging

**Status**: OPEN

### Hypothesis

Local detail at high zoom is not a *new object* hallucinated by the
model — it should be the same field projected at a finer scale.
"Zoom in" therefore should preserve descriptors at the parent scale
and reveal sub-scale structure consistent with parent-scale field
modes.

### Current implementation status

No implementation. `chladni_field` produces a single resolution; there
is no multi-scale projection pipeline. `descriptor()` runs at one
scale only.

### Metric

```text
multi-scale descriptor consistency (how stable amp/phase/freq are
                                    across scales)
zoom-in / zoom-out reconstruction error
frequency band preservation
```

### Baseline

```text
single-scale resize                (PIL/cv2 bilinear)
independent local regeneration     (model generates each scale fresh)
VOD multi-scale projection         (same field, different P_m)
```

### Falsification

```text
If VOD multi-scale projection does not show better cross-scale
descriptor consistency than independent local regeneration, multi-scale
is no better than the "generate-then-zoom" baseline.
```

### Next minimal implementation

```text
1. Build same Chladni field at two resolutions (e.g. 64x64 and 256x256).
2. Crop a 16x16 region from the 64x64 field; upsample to 64x64.
3. Compare descriptor(crop_upsampled) vs descriptor(corresponding 64x64
   region of the 256x256 field).
4. Pre-registered: VOD multi-scale RMSE on (amp, phase, freq) is lower
   than bilinear-upsample RMSE.
```

Estimated: half-day, no dependencies.

---

## H3  UV Text / Logo Plane

**Status**: PARTIAL upgraded — Binary-Twin minimal PASS verified at three levels:
text channel (one-step gradient direction), image+video pixel-level
discrete coupling (`L_binary_twin_pixel`), and composite stacking with
the other distinctives (no destructive interaction). Full OCR / 2-D
symbol grid / region-conflict resolver still NOT done.

### Hypothesis

Text and logos should not be free-painted by the image model. They
belong to a discrete symbol plane (UV plane / Binary-Twin object)
*coupled* to the continuous image field. Coupling = both sides must
agree under projection.

### Current implementation status

`vod_minimal/binary_twin.py`:
- Encode/decode maps `Φ`, `Ψ` between continuous `[0,1]` channel and
  integer symbols `{0..levels-1}`
- Differentiable loss `binary_twin_torch_loss = CE + MSE_to_recon`
- Paired diff test PASS in `run_vod_product_validation.py` §2.3:
  BT update improves symbol accuracy 13× more than MSE-only AND
  moves CLOSER to continuous target (0.000553 vs 0.000622).

This solves toy quantized-text channel only. NOT a full OCR / logo /
region-conflict solver.

### Metric

```text
symbol_accuracy                    (Φ exact match)
symbol_hamming                     (1 - accuracy)
continuous_mse                     (Ψ ∘ Φ recon distance)
OCR accuracy                       (full task, NOT done)
logo code accuracy                 (full task, NOT done)
region visual error                (full task, NOT done)
```

### Baseline

```text
MSE-only text channel
BinaryTwin minimal                 (verified)
post-hoc compositor                (NOT done)
full OCR-grounded loss             (NOT done)
```

### Falsification (for the FULL claim, not the minimal)

```text
If BinaryTwin loss + actual OCR-grounded sub-loss cannot reduce text
artifact rate vs free-painted baseline on real AI-render text outputs,
the UV-plane coupling hypothesis is not supported at scale.
```

### Next minimal implementation

```text
1. Extend BinaryTwin to 2-D symbol grids (current implementation is
   1-D channel only).
2. Add a real OCR sub-loss: render predicted text to image, run an OCR
   model, CE on character predictions.
3. Test on synthetic stroke-based text dataset (no external pretrained
   image generator needed).
```

Estimated: 1-2 days. OCR sub-loss requires pinning a small pretrained
OCR model (e.g. CRNN trained from scratch on synthetic chars) — NOT
fetching a large external model per project rules.

---

## H4  Layered Gestalt Text

**Status**: OPEN

### Hypothesis

Some image content (e.g. ASCII art of a face, stylised typography)
shows different valid readings at different scales — coarse view shows
a pattern, fine view shows characters. Single-level symbol loss cannot
capture this; multi-level symbol pyramid can.

### Current implementation status

No implementation. `binary_twin.py` is single-level (one quantization
to `levels` symbols). No coarse/fine pyramid.

### Metric

```text
coarse visual similarity           (downsampled image vs target gestalt)
fine symbol accuracy               (per-character accuracy)
scale-dependent consistency        (both readings must coexist)
```

### Baseline

```text
single-level text loss
BinaryTwin single-level
multi-level BinaryTwin             (NOT done — this hypothesis)
```

### Falsification

```text
If multi-level BinaryTwin cannot simultaneously satisfy coarse-pattern
similarity AND fine-symbol accuracy on a synthetic 2-level gestalt
dataset, the layered-symbol claim is unsupported.
```

### Next minimal implementation

```text
1. Build 2-level synthetic data: `coarse_group ∈ {0..3}`, `fine_symbol ∈
   {0..15}` per pixel cell. 16-level symbol = coarse * 4 + fine.
2. Train two losses:
     - flat 16-level BinaryTwin
     - hierarchical: separate coarse 4-class CE + fine 4-class CE
3. Measure: coarse_pattern_recall, fine_symbol_accuracy.
4. PASS iff hierarchical ≥ flat on both metrics simultaneously.
```

Estimated: 1 day. Pure synthetic, no external deps.

---

## H5  Manga Physics / AIMP

**Status**: PARTIAL upgraded — TPSR metric layer PASS verified, plus
`tpsr_video_consistency_loss` differentiable training-time integration
PASS with monotonic effect on `pred_video_tile_residue` (-5.1% on
inclusion in 5-layer composite). Full Field Card / Perspective Card
scene controller + vanishing-point / light-direction heuristics still
NOT done.

### Hypothesis

Manga / illustration quality issues (perspective inconsistency,
incoherent lighting, wrong eye-highlight geometry) can be constrained
by *physical cards* (Field, Perspective, Lighting) and the TPSR
invariant `K = H / (L_l^2 · A^(γ/2))`.

### Current implementation status

`vod_minimal/aimp.py`:
- `FieldCard`, `PerspectiveCard`, `LightingCard` dataclasses
- `TPSRMeasurement`, `tpsr_k`, `tpsr_pair_ratio`,
  `tpsr_consistency_score`, `aimp_tpsr_metrics`
- Synthetic `synthesize_tpsr_measurements` with `brightness_error`
  knob to construct consistent vs corrupted sequences
- Verified PASS in `run_vod_product_validation.py` §2.4:
  consistent_score > corrupted_score AND > random_score, K_cv lower
  for consistent.

This is the TPSR metric layer only. NOT a full AIMP scene controller.

### Metric

```text
TPSR K_cv                          (cross-frame K consistency)
median |ln U_ij|                   (pairwise ratio deviation)
tpsr_consistency_score             (exp(-deviation/sigma))
vanishing point error              (NOT done)
light direction consistency        (NOT done)
object placement violation         (NOT done)
```

### Baseline

```text
prompt-only generation
unconstrained generation
TPSR/AIMP metric layer             (verified diagnostic)
TPSR/AIMP loss feedback to model   (NOT done)
```

### Falsification (for the full claim)

```text
If a generator trained with TPSR/AIMP loss feedback does NOT produce
more consistent eye-highlight geometry on real character generation
tasks vs the same generator without it, the AIMP claim does not hold.
```

### Next minimal implementation

```text
1. Wire `tpsr_consistency_score` as a soft loss term in `native.py`'s
   composite loss, gated by spatial_media presence and a video-level
   highlight extractor.
2. Add `vanishing_point` and `light_direction` heuristics to
   PerspectiveCard / LightingCard (they're currently just dataclasses).
3. Compare `train_native_vod` runs with weights.aimp=0 vs >0 on a
   synthetic perspective-corrupted dataset, measure K_cv on outputs.
```

Estimated: 2-3 days for end-to-end. The TPSR metric layer is already
in place; the work is plumbing into training + adding two more cards
as actually-used controllers.

---

## Cross-cutting next-product step

Status rollup (post-2026-05-02 5-layer composite milestone):

```text
H0 Shadow Sampling     PASS    (4/e operator + training-time + composite stacking)
H1 Mask-as-Boundary    OPEN
H2 Dynamic Zoom        OPEN
H3 UV Text / Logo      PARTIAL (Binary-Twin text + pixel + composite, no OCR / 2-D)
H4 Layered Gestalt     OPEN
H5 Manga Physics/AIMP  PARTIAL (TPSR metric + L_aimp loss term + composite, no controller)
```

5-layer composite ablation (`stress_4layer.json`) confirms 4/e + TTNM +
Binary-Twin-pixel + AIMP + Chladni shared field can be wired into one
`native_total_loss` and trained jointly with monotonic per-distinctive
control and no destructive interaction. This is the architecture
verification milestone for VOD as a native multimodal diffusion
generator (see `VOD_agent_postmortem.md` §12.19).

If continuing within a 1-week budget, the highest-leverage path is:

```text
1. H1 (mask-as-boundary)  — half day, isolates a clear protocol gap
2. H2 (dynamic zoom)      — half day, validates field-vs-resize
3. H4 (layered gestalt)   — 1 day, extends Binary-Twin without OCR dep
```

These three new minimal implementations + the existing 4/e + Binary-Twin
+ TPSR + TTNM PASS would constitute "VOD distinctives 7 / 9 verified at
minimal scale", enough to motivate a real training-time integration test.

H3-full and H5-full both require either external OCR or full AIMP
controller loop and are 1-week+ items.
