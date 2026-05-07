# Omni-Diffusion lessons for VOD

**Status**: 2026-05-07 — strategy doc, no code changes. Inputs Stage 3
plan only.

---

## 1. Omni-Diffusion is what

[Omni-Diffusion](https://arxiv.org/abs/2603.06577) (VITA-MLLM, March
2026; based on [Dream-7B](https://arxiv.org/abs/2508.15487), HKU-NLP,
August 2025) is the first any-to-any multimodal language model built
entirely on **mask-based discrete diffusion**. Project page:
[omni-diffusion.github.io](https://omni-diffusion.github.io/).
Code: [VITA-MLLM/Omni-Diffusion](https://github.com/VITA-MLLM/Omni-Diffusion).

Key facts:

* **Discrete tokens**: text / speech / image are tokenized into a unified
  vocabulary (Dream-7B base + extended +8192 image tokens via MAGVIT-v2,
  +16384 speech tokens). All modalities then live in a single discrete
  token sequence.
* **Mask-based diffusion**: instead of next-token autoregression (or
  Gaussian-noise continuous diffusion), training perturbs the sequence
  by masking random tokens; the model predicts the masked tokens
  conditioned on the unmasked context. Inference iteratively unmasks,
  parallel, in arbitrary order.
* **Progressive training**: 3-stage pipeline — text-image alignment →
  text-speech-image alignment → speech-driven visual interaction (SDVI).
* **Variable-length generation**: *attenuated tail-pad masking* lets the
  model emit shorter or longer sequences without a hard length prior.
* **Inference tricks**: image generation uses a *position penalty* to
  constrain unmasking order (raster-friendly priority); speech uses a
  *special token pre-infilling* trick to inject text semantics.

This places Omni-Diffusion in the **discrete-token unification** family
(adjacent to Show-o, Chameleon, Transfusion). VOD is in the
**continuous-substrate unification** family (the third option in our
type-A / type-B / type-C taxonomy from `docs/paper_v16_baseline.md`).

The two architectures answer the same question — "how do you put many
modalities in one diffusion model" — with **structurally different
representations**. They are not in competition for the same niche;
they are evidence that "unified multimodal diffusion" works in at
least two distinct ways.

---

## 2. What VOD does NOT borrow

Hard rules. None of these enter VOD:

| ❌ NOT in VOD | Why this would break the type-B claim |
|---------------|---------------------------------------|
| **MAGVIT-v2 / external image tokenizer** | Tokenizing images means VOD becomes type-A / type-D (modality-private discrete encoder). Substrate-shared field claim collapses. |
| **Dream-7B / token-MLLM backbone** | Dream-7B is autoregressive-then-diffusion-LLM; substituting it for VOD's spatial UNet replaces continuous field with discrete token sequence. Different paper. |
| **Discrete vocabulary** for image/audio/text | The whole point of VOD is *continuous* `U(t,y,x,c)`. Adding discrete vocab buckets defeats the substrate-shared design. |
| **External speech tokenizer** | Same logic. Speech in VOD must enter through a 1×1 projection head, not a learned codebook. |
| **CFG-as-token-guidance**            | VOD CFG is on the additive class embedding (continuous). Token-classifier-free-guidance assumes discrete logit space. Not portable. |

Stage 3 specifically: if any "fix" requires importing a tokenizer or a
pretrained Dream-7B, **stop and reconsider**. That fix is solving the
wrong problem.

---

## 3. What VOD DOES borrow (concept-level only)

Six lessons, each ported as a continuous-substrate analogue.

### 3.1 Progressive curriculum

**Omni**: text-image → text-speech-image → speech-driven interaction.
Each stage is independently validated before the next is started.

**VOD analogue** (already partially executed):

1. ✅ **Chladni 16×16 grayscale** — synthetic substrate validation (v16,
   §5 of paper).
2. ✅ **Real RGB 64×64 unconditional** — Stage 1, learned conv head.
3. ✅ **Class conditioning** — Stage 2, additive embedding + CFG dropout.
4. ⏳ **ImageNet-100 RGB object-level** — Stage 3A (next).
5. ⏳ **Image + video shared field** — Stage 3 follow-up.
6. ⏳ **Audio / text projection heads** — later.

The principle: do not skip steps. If 4 fails, do not jump to 5; debug 4.

### 3.2 Requested field extent

**Omni**: variable-length token sequence; user/system requests how many
output tokens for which modalities.

**VOD analogue**: substrate `U(t,y,x,c)` shape is a *request*, not a
constant. Stage 3 must support:

* **Image request**: `T=1, H=W=64, C=8` (current Stage 1/2 default).
* **Short video request**: `T=8`, same H/W/C.
* **Longer video**: chunked temporal field — e.g. four `T=8` chunks
  conditioned on the previous chunk's last frame.
* **Per-modality decoder selection**: only invoke `dec_image` if image
  was requested; only invoke `dec_video` if video was requested. Do
  not always emit all heads.

This is API-level work, not architecture; the substrate already supports
arbitrary `T`. We just have not exposed it.

### 3.3 Partial field masking / inpainting

**Omni**: native — masked diffusion *is* inpainting by construction.
Mask any token positions, the model fills them.

**VOD analogue**: continuous-field analogue of "RePaint"-style masked
sampling (Lugmayr et al. 2022, CVPR; the standard inpainting recipe for
Gaussian-noise diffusion):

```
for each diffusion step t:
    x_t_known   = q_sample(x_0_target, t)        # forward-noise the known region
    x_t_unknown = denoise_step(x_t)              # model's reverse step
    x_t = mask * x_t_known + (1 - mask) * x_t_unknown
```

Use cases for VOD:

* **Image inpainting**: paint out a region, sample fill.
* **Video extension**: keyframes given, fill the rest of `T`.
* **Local repair**: region with artifact → noise it again, redenoise.

This is a **trainer-side sampling-loop change**, not an architecture
change. The same v16 substrate handles it.

### 3.4 CFG / condition dropout formalization

**Omni**: condition is text-prompt token. Guidance scale sweep is
standard.

**VOD analogue** (Stage 2 already has the infrastructure; Stage 3 must
exercise it):

* Train: `p_drop_cond = 0.1` already active (Stage 2).
* Inference: classifier-free guidance with scale `s`:
  `pred_guided = pred_uncond + s * (pred_cond - pred_uncond)`
* **Required Stage 3 sweep**: `s ∈ {0.0, 1.0, 2.0, 4.0, 7.5}`.
* Report: per-class grid + descriptor distance to train + per-class
  fidelity (if available).
* Failure mode to watch: `s` too high → over-saturated samples /
  collapse to per-class prototype. Standard CFG pathology.

### 3.5 Uncertainty-guided sampling

**Omni**: per-token entropy from the discrete-vocabulary distribution;
high-entropy positions get more refinement steps.

**VOD analogue** for continuous substrate (no token vocabulary, no
entropy in the discrete sense):

| signal | source | use |
|--------|--------|-----|
| **residual magnitude** `|delta|` | `denoise()` returns delta added to `u_noisy`; `delta` magnitude per-voxel | high-residual voxels need more steps |
| **denoise disagreement** | run 2-3 denoise() with different `t` perturbations on the same noisy state | high disagreement = high uncertainty |
| **predicted variance** | future: model could output `(mean, log_var)` instead of just delta — EDM-style | per-voxel uncertainty |
| **local refinement** | re-noise high-uncertainty regions to higher `t`, then redenoise | targeted compute |

**Stage 3 minimum**: just measure residual magnitude during sampling
and report; do not yet implement adaptive resampling. That is Stage 4.

### 3.6 Repetition / artifact control

**Omni**: position penalty stops images from "filling in raster order
with copies of the same token".

**VOD analogue**: VOD already has artifact metrics from the prototype:

| Omni concern | VOD existing mechanism |
|--------------|-------------------------|
| token-level repetition | `4/e` (claim 1) — entropy / phase-coverage check |
| temporal flicker | Fix A static-broadcast + temporal artifact metrics |
| boundary phase-break | TTNM (boundary topology metric) |
| spatial repetition | tile-residue + `mean_tile_residue` in `artifact_metrics` |

Stage 3 should add these to the verdict pipeline for ImageNet-100, not
just for Chladni.

---

## 4. Stage 3 redefinition

### 4.1 Old Stage 3 (per `report_to_codex_stage1_stage2.md` §5)

> Substrate-self-stratified ImageNet-100 anchor (RunPod B200×8, 2-3h).

This is correct in direction but **monolithic**. Breaking it down:

### 4.2 New Stage 3: ImageNet-100 + controlled field generation

Four sub-steps, each independently validated:

| step | scope | budget (B200 ×8) | passes if |
|------|-------|------------------|-----------|
| **3A** | ImageNet-100 object-level RGB unconditional | ~3h | trained_sample > untrained/random/zero, **object-level structure visible (not just color blobs)**, gate0_recon distance OK |
| **3B** | CFG sweep on Stage 2-style class conditioning + ImageNet-100 classes | ~1h | guidance scale sweep table + per-class grid; cond effect monotone in `s` until collapse threshold |
| **3C** | Partial field masking smoke (image inpainting) | ~30 min | known region preservation MSE low; unknown region plausible; mask boundary clean |
| **3D** | Requested-extent API smoke | ~30 min | `T=1`, `T=8` requests run end-to-end; only requested decoder called; shape contracts hold |

Total Stage 3 budget: ~5h B200×8 + ~6h human review = **one weekend**.

3A first. 3B/3C/3D only if 3A passes. If 3A fails, debug 3A; do **not**
jump to 3B as a workaround.

---

## 5. Stage 3 verdicts (acceptance criteria)

### 5.1 Stage 3A — ImageNet/RGB object-level

**PASS**:

* `trained_sample` descriptor distance < `untrained_sample` distance
  (gap ≥ 3.0).
* `gate0_recon` visually identical to `train_reference`.
* `trained_sample` shows **object-level structure** — not just color
  regions. Honest visual standard: a third party looking at
  `trained_sample.png` should be able to identify "this looks like an
  animal / vehicle / plant" *without being told the class*. This is
  the bar Stage 1/2 explicitly did not clear.
* Class conditioning (Stage 2 carryover) is now class-faithful, not
  just color-tone shift.

**FAIL** (do NOT escalate to 3B):

* `trained_sample` is still color blobs at 100M-param / ImageNet-100
  scale → architectural ceiling, not data scarcity. Diagnose: is the
  substrate lift basis still the bottleneck? Is the spatial UNet
  capacity adequate?

### 5.2 Stage 3B — CFG sweep

**PASS**:

* Guidance scale sweep `s ∈ {0, 1, 2, 4, 7.5}` produces a monotone
  trend: increasing `s` strengthens class effect (per-class fidelity
  goes up) up to a saturation point.
* No collapse (entropy of trained sample doesn't drop to zero) at
  reasonable `s` (≤ 4).
* Report includes the scale-sweep grid: 5 columns × 4 classes = 20
  images.

**FAIL**:

* Guidance has no measurable effect → conditioner was not actually
  trained (reproduction issue, not architecture).
* Mode collapse at `s = 2` → `p_drop_cond` too low, or condition
  injection point too dominant.

### 5.3 Stage 3C — Partial field masking

**PASS**:

* Known-region preservation: pixel-MSE between `mask * recon` and
  `mask * input` < 0.05 (i.e., the model does not damage what it was
  given).
* Unknown-region plausibility: visual inspection — fill is consistent
  with the known region, not random color blobs.
* Mask boundary: no obvious seam / phase-break artifact.

**FAIL**:

* Known region damaged (model overwrites what it was given) →
  RePaint-style mixing is not implemented correctly.
* Unknown fill ignores the known context (replaces with mean image).

### 5.4 Stage 3D — Requested extent API

**PASS**:

* `request={"image": True}` → returns image only, decoder graph for
  video / audio / text is *not invoked*.
* `request={"image": True, "video": True, "T": 8}` → returns both with
  shape `(B, 1, H, W, 3)` and `(B, 8, H, W, 3)` respectively, sampled
  from the same shared substrate.
* Per-modality decoder is conditioned only on the requested time slice
  of `U`.
* Smoke-only: video quality is not yet evaluated. We just confirm the
  pipeline doesn't error and shapes match.

**FAIL**:

* Always-on full decoder graph → wasted compute on irrelevant heads.
* Shape mismatch between requested `T` and substrate output `T`.

---

## 6. Differences summary table (VOD vs Omni-Diffusion)

| axis | Omni-Diffusion | VOD |
|------|----------------|-----|
| representation | discrete token vocabulary (image 8192, speech 16384, text 32K-ish) | continuous field `U(t,y,x,c) ∈ ℝ` |
| generation unit | masked token replacement (parallel iterative unmask) | continuous-noise denoise (DDIM v-prediction) |
| modality relation | shared *token* space; modality is a token-id range | shared *field*; modality is a 1×1 projection head |
| backbone | Dream-7B diffusion LLM (transformer over tokens) | spatial UNet + 1-D conv at bottleneck (NativeVOD) |
| training stages | text-image → text-speech-image → SDVI | Chladni → RGB64 → cond → ImageNet-100 → multi-modal |
| variable length | attenuated tail-pad masking | requested extent (`T` per request, conditional decoder activation) |
| inpainting | native (mask any tokens) | RePaint-style masked-noise mixing in sampling loop |
| conditioning | token-level (text prompt is just more tokens) | additive embedding on time-emb path (continuous) |
| CFG | on token-distribution logits | on continuous prediction (`pred_uncond + s*(pred_cond - pred_uncond)`) |
| uncertainty | per-token entropy from softmax | residual magnitude / denoise disagreement / future predicted-variance |
| **what VOD borrows** | curriculum, requested extent, masking, CFG sweep, uncertainty signals, position-penalty-style artifact control | — |
| **what VOD does NOT borrow** | tokenizer (MAGVIT/speech), Dream-7B backbone, discrete vocab, token-classifier-free-guidance | — |

---

## 7. Rule for Stage 3 design changes

**Hard rule**: any Stage 3 sub-step that requires changing the substrate
representation (introducing tokenization, replacing the spatial UNet
with a transformer-LLM-style backbone, adding an external pretrained
encoder/decoder) is **not Stage 3**. It is a different paper. Park
those ideas in a separate doc and do not let them sneak into the
Stage 3 plan.

Stage 3 is *strictly* about exercising the type-B substrate at object
scale + extending the inference / conditioning surface. Architecture
of the substrate itself is frozen at v16 + Stage 1 conv head.

---

## 8. References

1. Lijiang Li *et al.* 2026. *Omni-Diffusion: Unified Multimodal
   Understanding and Generation with Masked Discrete Diffusion.*
   arXiv:2603.06577. ICLR 2025.
   [paper](https://arxiv.org/abs/2603.06577) ·
   [project](https://omni-diffusion.github.io/) ·
   [code](https://github.com/VITA-MLLM/Omni-Diffusion).
2. Jiacheng Ye *et al.* 2025. *Dream 7B: Diffusion Large Language
   Models.* arXiv:2508.15487.
   [paper](https://arxiv.org/abs/2508.15487) ·
   [code](https://github.com/DreamLM/Dream).
3. Lugmayr *et al.* 2022. *RePaint: Inpainting using Denoising
   Diffusion Probabilistic Models.* CVPR. — masked-noise sampling
   recipe for Stage 3C.
4. Ho & Salimans 2022. *Classifier-Free Diffusion Guidance.*
   arXiv:2207.12598. — already adopted at Stage 2; Stage 3B exercises
   the scale sweep.

VOD's own context:

5. `docs/paper_v16_baseline.md` — paper draft covering v16 / Stage 1
   / Stage 2 results.
6. `docs/report_to_codex_stage1_stage2.md` — §5 single-next-step is
   superseded by this document's §4.2 four-sub-step Stage 3 plan.
