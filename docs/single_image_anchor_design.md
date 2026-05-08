# HVBD anchor design — 9-PNG static + 12-sheet HVBDT (per work guide)

**Status**: 2026-05-08 — aligned with `HVBD_VOD_Claude_Work_Guide.md`
v1 (the 430-authored plan). Supersedes the earlier "1 master 2048×2048
+ 16 sub-tiles" sketch (which was written before the work guide
existed).

---

## 1. Why 9 separate PNGs (not one master)

The earlier draft proposed cramming everything into a single 2048×2048
PNG with 16×16 cells. **Image2 实测 (430)** showed that any single
generated image with > 224 cells suffers from:

* bottom truncation
* cell-number jumps / skips
* row dropouts
* compressed cells in the last row
* model "summarizing" the bottom rows

Therefore: **224 cells per PNG is the hard ceiling**. Splitting into
9 specialized PNGs gives more total cells (1648) without exceeding
any single image's safe limit, AND keeps each PNG focused on one
hierarchical level (L1 primitives, L2 textures, L3 geometry, L4
sketches, ...).

This is a structural improvement over the 1-master approach AND it
maps directly onto the HLBD analogue (see `HVBD_introduction.md`),
where each level is its own conceptual frame.

---

## 2. The 9-PNG static plan

| # | filename | cells | layout | role |
|---|----------|-------|--------|------|
| 1 | `HVBD_anchor_core_v1.png` | 224 | 14×16 | main anchor — 16 domain × 14 variation |
| 2 | `HVBD_L1_primitives.png` | 160 | 10×16 | visual atoms |
| 3 | `HVBD_L2_textures_patterns.png` | 224 | 14×16 | texture / pattern |
| 4 | `HVBD_L3_geometry_spectrum.png` | 196 | 14×14 | formal geometry / spectrum |
| 5 | `HVBD_L4_sketch_edges.png` | 224 | 14×16 | line-art / edges |
| 6 | `HVBD_L5_grayscale_depth_channels.png` | 224 | 14×16 | gray / depth / channels |
| 7 | `HVBD_L6_rgb_natural_domains.png` | 224 | 14×16 | full RGB natural |
| 8 | `HVBD_L7_multiview_variants.png` | 72 | 8 groups of 3×3 | multi-view |
| 9 | `HVBD_L8_multistyle_multimedia.png` | 100 | 4 groups of 5×5 | multi-style / multi-media |
| **total** | | **1648 cells** | | |

`HVBD_anchor_core_v1.png` is the **primary** anchor — most VOD
training will random-sample cells from this one. L1-L8 are
specialized supplements: useful for ablations and for showing that
substrate trained on the full 1648-cell set generalizes better than
just the anchor_core 224.

---

## 3. Cut into cells, do NOT random-crop the master

**Key training decision** (per user 2026-05-08): split each PNG into
its constituent cells **at preprocessing time**, with each cell
saved as a separate file. **Do not random-crop across the whole
master at training time.**

Reason:

| approach | pro | con |
|----------|-----|-----|
| split into cells (chosen) | cell ≈ 1 sample, same shape as CIFAR/ImageNet samples; **substrate cannot learn the 14×16 grid geometry** as a prior | needs split script + metadata.jsonl |
| random-crop master | trivial, master PNG itself is the dataset | substrate **learns the grid lines** as part of the prior. This is a documented failure mode (guide §8); it produces samples that look like miniature mosaics. |

In-cell augmentation (random crop within a single cell, e.g., 56×56
from 64×64) is allowed and recommended; **never cross cell borders.**

Implementation:

* `scripts/split_hvbd_subanchors.py` — split uniform-grid PNGs into
  per-cell PNGs, write `metadata.jsonl` per the schema in
  `data/hvbd_static/metadata_schema.json`.
* `scripts/crop_anchor_dataset.py` — `HVBDAnchorDataset` PyTorch
  Dataset that loads cells, applies in-cell augmentation, returns
  `(B, H, W, 3)` tensors in `[-1, 1]`.

---

## 4. Anchor + dilution training schedule

Per `HVBD_VOD_Claude_Work_Guide.md` §8:

```
0     – 5k  step:  100% HVBD cells          (Phase A — planting prior)
5k    – 20k step:  50% HVBD / 50% real      (Phase B — co-training)
20k   – 0.85T:     10% HVBD anchor gravity  (Phase C — minor anchor)
0.85T – T:          0–5% HVBD anchor        (Phase D — late dilution)
```

`scripts/make_dilution_schedule.py` writes `configs/curriculum_default.json`
with these step thresholds. Trainer reads the file every step and
samples from `HVBDAnchorDataset` vs the real dataset (`CIFAR-10` /
`ImageNet-100` / etc.) according to the current `anchor_prob`.

If `eval_samples.py` reports `grid_artifact_score > 0.5` → reduce
anchor_prob earlier, increase random rotation/scale within cells, or
use multi-anchor rotation (Phase A draws from a *set* of HVBD
anchors instead of always `anchor_core_v1`).

---

## 5. HVBDT — 12 video sheets (motion / camera / anime / control / benchmark)

Parallel structure to the 9-PNG static plan, but each PNG is a
**frame-strip**: rows = motion atoms, columns = consecutive frames.

| # | sheet | rows × cols | role |
|---|-------|-------------|------|
| 1 | `HVBDT_core_motion_primitives.png` | 10 × 12 | basic motion atoms ✅ in repo |
| 2 | `HVBDT_core_dynamic_textures.png` | 12 × 12 | dynamic textures (water/fire/smoke/...) |
| 3 | `HVBDT_core_camera_motion.png` | 12 × 12 | camera moves (pan/tilt/zoom/orbit/...) |
| 4 | `HVBDT_anime_frame_strips.png` | 12 × 12 | anime base actions |
| 5 | `HVBDT_anime_timing_principles.png` | 12 × 12 | 一拍二 / smear / anticipation / ... |
| 6 | `HVBDT_anime_character_consistency.png` | 6 × 12 | 360 rotation / consistency |
| 7 | `HVBDT_anime_production_pipeline.png` | 8 × 12 | sketch → lineart → color → final |
| 8 | `HVBDT_control_keyframe_interpolation.png` | 6 × 3 | first/mid/last frame |
| 9 | `HVBDT_control_pose_depth_lineart.png` | 6 × 12 | pose / depth / lineart strip |
| 10 | `HVBDT_control_audio_mouth_face.png` | 8 × 12 | mouth / expression / audio align |
| 11 | `HVBDT_benchmark_motion_categories.png` | (irregular) | motion classification eval |
| 12 | `HVBDT_benchmark_consistency_quality.png` | (irregular) | consistency/quality eval |

Each row of an HVBDT sheet → one **T=12 frame sequence** training
sample. Substrate trained with `LATENT_T=12` (or T=8, taking only
first 8 frames as a fallback if memory budget is tight).

`scripts/extract_frame_strips.py` (TODO) splits each sheet into
per-row frame sequences.

---

## 6. LDM-baseline split (controlled type-A comparison)

For the type-B vs type-A paper claim (see `omni_diffusion_lessons_for_vod.md`
§4.5 framing), the **same** HVBD source PNGs are used to build a
type-A LDM baseline:

* `HVBD_anchor_core_v1.png` → split column-wise into **16 single-domain
  sub-anchor PNGs** (`HVBD_anchor_core_v1_col_F.png`, ..._col_G.png).
* Each per-domain sub-anchor → train its own VAE.
* 16 VAEs total = 16× compute inflation vs VOD's 1 substrate pass.
* Each VAE has only 14 cells (its column) of training data — at the
  floor where small-VAE collapse risk increases. This is the honest
  comparison: VOD shares info across domains, LDM cannot.

Compute economics table is the headline paper result.

---

## 7. Reference

* `docs/HVBD_introduction.md` — HVBD as the visual analogue of HLBD.
* `docs/B200_runbook.md` — 4-stage GPU training plan.
* `docs/experiment_matrix.md` — config × scale × dataset matrix.
* `HVBD_VOD_Claude_Work_Guide.md` (downloads) — 430's authored
  full plan; this document re-implements its design decisions.
* `data/hvbd_static/metadata_schema.json` — per-cell metadata JSON
  schema.
* `scripts/build_hvbd_registry.py` — verify all 9 + 12 PNGs present.
* `scripts/split_hvbd_subanchors.py` — uniform-grid splitter.
* `scripts/crop_anchor_dataset.py` — `HVBDAnchorDataset` for training.
* `scripts/make_dilution_schedule.py` — curriculum config generator.
* `scripts/eval_samples.py` — VOD/LDM verdict metrics + grid-artifact
  detection.
