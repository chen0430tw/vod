# Single-Image Anchor — VOD's image-domain HLBD analogue

**Status**: 2026-05-07 design proposal — paper-grade Stage 3 extension.
No code yet. Discusses the design of a single curated "surveillance
wall" image that, when shown to the type-B substrate, plants a
quickcook-style multi-domain prior.

---

## 1. Why the anchor exists

NLP quickcook (APT-Transformer `pretrain_quickcook.py`):

```
HLBD anchor (high-quality long-form corpus, multi-document)
        +
C4 / Wikipedia / arXiv / GitHub Code  (dilution streams)
        ↓ curriculum hot-swap weights
APT substrate sees coherent + diverse + structured language
```

The anchor's job is to **plant a high-density representative prior** of
the manifold the substrate must learn, before any dilution kicks in.
HLBD works for NLP because language information is multi-document
sequential — you need many documents to span the manifold.

**Image anchor proposed here**:

```
Single ANCHOR IMAGE — surveillance-wall composite (one PNG)
        +
ImageNet / CIFAR / personal image streams  (dilution)
        ↓ curriculum hot-swap weights
VOD substrate sees the entire image-mode manifold inside one frame
```

This works for VOD specifically because the type-B substrate
`U(t,y,x,c)` is **patch-locally addressable** — the spatial UNet
denoiser operates on local neighborhoods, so a `1024 × 1024` anchor
image is, from the substrate's point of view, **N×N independent
patches** (where N = image_size / patch_size). One physical image
yields hundreds of effective training views.

LDM can't do this. Its VAE compresses the whole image into a single
latent before diffusion sees it; patches are not first-class.

---

## 2. Anchor layout proposal

Working name: `vod_anchor_v1.png`

### 2.1 Resolution and grid

* Master canvas: **1024 × 1024 RGB**
* Primary grid: **8 × 8 of 128 × 128 cells** (64 cells)
* Each cell is **one curated visual concept** (see §3 for selection).

This maps cleanly onto:

* substrate spatial size 64×64 (Stage 1) → one substrate "view" = one
  cell at 1× downsample
* substrate spatial size 128×128 (future) → 4 cells per substrate view
* random crop training: each crop covers 1-4 cells worth of content,
  so every step sees a multi-mode mini-distribution

### 2.2 Multi-scale inset (optional v1.1)

Embed within the 8×8 grid: **4 corner cells** are themselves 4×4 sub-grids
of 32×32 inset images. So the anchor carries:

* 60 × `128²` cells (medium scale)
* 4 × 4×4 sub-grids = 64 × `32²` mini-tiles (fine scale)

This forces the substrate to see **scale-varied** patches without us
having to maintain a separate fine dataset.

### 2.3 Visual layout sketch (ASCII)

```
+---+---+---+---+---+---+---+---+
| F | F | A | A | V | V | T | T |     F = face-domain (closeup, profile, ...)
+---+---+---+---+---+---+---+---+     A = animal (mammal, bird, fish, insect)
| F | F | A | A | V | V | T | T |     V = vehicle (car, plane, ship, bike)
+---+---+---+---+---+---+---+---+     T = texture (cloth, wood, fur, metal)
| O | O | B | B | N | N | P | P |     O = object (tool, electronic, food)
+---+---+---+---+---+---+---+---+     B = building (urban, rural, ancient)
| O | O | B | B | N | N | P | P |     N = natural (mountain, sea, sky, plant)
+---+---+---+---+---+---+---+---+     P = pattern (geometric, fractal, tiles)
| C | M | M | I | I | S | S | E |     C = chart/diagram
+---+---+---+---+---+---+---+---+     M = micro-grid 4×4 of 32² insets (fine)
| H | H | L | L | D | D | G | G |     I = indoor scene
+---+---+---+---+---+---+---+---+     S = sketch/line-art
| H | H | L | L | D | D | G | G |     E = edge / silhouette
+---+---+---+---+---+---+---+---+     H = handwriting / text
| C | M | M | I | I | S | S | E |     L = lighting study (low/high key)
+---+---+---+---+---+---+---+---+     D = depth (near/far)
                                      G = gradient field
```

64 cells, 16 distinct domain categories, 4 cells each. Multi-scale via
the 4 `M`-cells which contain 4×4 of 32² tiles.

---

## 3. Content selection (the hard part)

### 3.1 Selection principle: maximize patch-distribution coverage, NOT class count

We are not building ImageNet-on-one-image. We are building a
**spectrum** of:

* spatial frequencies (low → high; cf. classical Chladni distribution)
* color statistics (warm / cool / muted / saturated / monochrome)
* edge density (smooth gradients → sharp boundaries → fractal edges)
* semantic class (object / animal / scene / texture / abstract)
* lighting (flat / directional / diffuse / specular)
* scale (close-up macro / mid-shot / wide-angle)
* view angle (frontal / 3⁄4 / profile / overhead / oblique)

Two cells can be the same class (both "face") but different lighting /
angle / age / sex / etc. — the goal is **coverage**, not taxonomic
balance.

### 3.2 Concrete content list (proposal v1)

64 cells, grouped by §2.3 categories. Mix CC0 / public-domain sources
where possible to keep the anchor publishable in the paper.

| code | category | cells | source candidates |
|------|----------|-------|-------------------|
| F | face | 4 | FFHQ samples (public, NVIDIA), real consented portrait set |
| A | animal | 4 | ImageNet animal subset / iNaturalist CC0 |
| V | vehicle | 4 | Stanford Cars / KITTI / COCO vehicle subset |
| T | texture | 4 | DTD (Describable Textures Dataset, Cimpoi 2014) |
| O | object | 4 | COCO objects / ImageNet artifacts |
| B | building | 4 | Places365 / LSUN-churches / OSM imagery |
| N | natural | 4 | Places365 outdoor / open-source landscape |
| P | pattern | 4 | Procedurally generated (Voronoi, Perlin, fractal, Chladni) |
| C | chart | 1 | Procedural matplotlib renders |
| M | micro-grid | 4 (= 64 inset tiles) | Random ImageNet 32×32 |
| I | indoor | 2 | Places365 indoor |
| S | sketch | 4 | Sketchy database / manga-line public sketches |
| E | edge/silhouette | 1 | Canny-on-clean-image renders |
| H | handwriting | 4 | IAM handwriting / printed-glyph mix |
| L | lighting study | 4 | Same scene 4 lighting conditions (procedural relight) |
| D | depth | 4 | NYU-Depth / KITTI depth visualization |
| G | gradient | 4 | Linear / radial / Perlin gradients (procedural) |

Total: 60 photographic + 4 micro-grid (=64 fine tiles) + 4-8 procedural.

The **procedural** entries (P, C, E, G, parts of M) are crucial: they
are mathematically clean and give the substrate gradient-direction
priors that natural photography hides under noise.

### 3.3 Build-time normalization

Before mosaic-ing:

* All source images resized to exactly 128 × 128 (or 32 × 32 for inset)
* Color: maintain native [0, 255] RGB (let substrate normalize)
* Aspect: center-crop to square if needed
* Sharpening: NO post-process; keep original spatial frequencies

---

## 4. Training strategy

### 4.1 Phase A — anchor-only quickcook (planting prior)

```
batch sampling:
    crop_size  = 64×64  (matches substrate Stage 1 spatial)
    crops_per_step  = 32 (one minibatch)
    crops are RANDOM offsets across the 1024² anchor
    => one minibatch covers ≈ 32×64²/1024² = 12.5% of the anchor
```

* Run for, say, **300 epochs** (~5 min on B200×8 if patches are
  small).
* Substrate develops a multi-mode prior: it has seen face-patches,
  texture-patches, edge-patches, etc. — all in one image, all aligned
  in coordinate space.

### 4.2 Phase B — anchor + dilution

After Phase A primes the substrate, mix in real ImageNet samples (or
CIFAR-10 RGB) at increasing weight. The anchor stays in the batch
distribution as a *gravity* term — at any step there is still a 10-30%
chance the crop comes from the anchor.

```
curriculum.json:
  step 0    – 5000:   100% anchor crops
  step 5000 – 20000:  50% anchor / 50% dataset
  step 20000+:        10% anchor / 90% dataset
```

The 10% anchor gravity in late training serves the same role as
HLBD-late-stage in NLP quickcook: prevent forgetting of the
multi-mode prior.

### 4.3 Why this should beat substrate-self-stratified

| | substrate-self-stratified | single-image anchor |
|---|---|---|
| Dataset prep | bootstrap chicken-and-egg (need v16 to stratify, but v16 was Chladni-only) | One PNG, prepared once, public |
| Reproducibility | depends on which substrate version stratifies | bit-exact: one PNG hash |
| Storage / sharing | thousands of files | one PNG (≈ 1-2 MB) |
| Paper artifact | dataset description | the anchor image *is* the artifact |
| Multi-scale | requires multi-res sampling | built-in via M cells |
| Domain coverage | depends on ImageNet noise | curated by hand for spectrum |
| Contradiction with type-B claim | none | none — substrate sees patches, anchor is just one input image |

---

## 5. Validation: did the anchor work?

After Phase A (anchor-only), measure on substrate:

### 5.1 Patch-distribution coverage

Encode the anchor → 8-channel substrate latent. K-means in latent
space with K=64 (one cluster per cell). Verify:

* Each cell occupies its own cluster (no two cells collapse).
* Cluster spread σ within each cell-cluster < cluster-to-cluster σ
  (cells distinguishable).

### 5.2 Untrained-vs-anchored sample diversity

* Sample 64 unconditional outputs from the anchored substrate.
* Compute pairwise descriptor distance.
* Threshold: pairwise std > pairwise std of substrate-self-stratified
  baseline. (Anchor should give *more* diversity, not less.)

### 5.3 Out-of-anchor recall

* Held-out test images that are NOT in the anchor (e.g. CIFAR test
  set).
* gate0_recon distance should still be reasonable (≤ 2× the
  in-anchor recon distance).
* If out-of-anchor recon explodes → anchor overfitted, dilute earlier.

### 5.4 Single-image-prior failure mode to watch

If anchored substrate produces samples that are **64-cell mosaic
themselves** (i.e. it learns "always emit a tile grid"), the anchor
geometry has leaked into the prior. Fix: Phase B dilution must start
earlier, or random-crop during anchor exposure must be less
geometrically aligned (use random rotation / scale).

---

## 6. Stage 3 integration

Insert into Stage 3 plan (per `docs/omni_diffusion_lessons_for_vod.md`
§4.2):

| sub-step | scope | budget (B200×8) | gate |
|----------|-------|-----------------|------|
| **3A-anchor** *(new)* | Build `vod_anchor_v1.png`; substrate Phase-A anchor-only training; validation per §5 above | ~30 min build + ~10 min train | per-cell cluster separable; out-of-anchor recall ≤ 2× in-anchor |
| **3A-imagenet** | Phase-B dilution: anchor + ImageNet-100 curated → object-level RGB | ~3h | trained_sample shows object-level structure |
| **3B** | CFG sweep on Stage-2-style class conditioner | ~1h | guidance scale monotone |
| **3C** | Partial field masking smoke (RePaint-style) | ~30 min | mask-boundary clean |
| **3D** | Requested extent API smoke | ~30 min | shape contracts |

3A-anchor is the *smallest* possible Stage 3 entry: < 1 hour total.
If it fails (substrate doesn't pick up multi-mode prior from a single
image), we learn that early — before committing to the full
ImageNet-100 budget.

---

## 7. Why this is paper-grade

The Stage-3-anchor result, if it works, is a publishable claim on its
own:

> *"A type-B substrate-shared diffusion model can be primed with a
> single curated anchor image, replacing thousand-document dataset
> curation. The anchor's patch-level mode coverage maps directly to
> the substrate's spatial-locality prior, a property absent in
> latent-diffusion (type-A) architectures."*

This puts VOD's substrate design on a footing where **dataset prep
becomes a one-image-design problem**, not a multi-million-sample
curation problem. That's a significant claim about type-B's
engineering economics relative to type-A LDM.

For the workshop paper:

* Section 6 (Stage 1) and 7 (Stage 2) are existing.
* Add a new Section 8.5 / 8.6 sub-section "Single-image anchor as
  quickcook initialiser" — small, paper-grade, replaces or augments
  the substrate-self-stratified anchor proposal.

---

## 8. Open questions for review

1. **Anchor source licensing**: which CC0 / public-domain image sets
   for the 60 photographic cells? Avoid copyright issues when the
   anchor itself ships with the paper. Candidate: FFHQ + DTD +
   Places365 + iNaturalist CC0 + Stanford Cars + procedural.
2. **Layout v2**: should the anchor be 1024×1024 (8×8 of 128²) or
   2048×2048 (16×16 of 128² = 256 cells)? More cells = more diversity,
   but training-step crops also become more sparse per cell.
3. **Multi-anchor vs single-anchor**: Phase A could use one anchor;
   later phases could rotate among 3-4 anchors of different curation
   bias. Cheaper than ImageNet, richer than one image.
4. **Procedural-only anchor as a baseline**: as an ablation, build an
   anchor entirely from procedurally generated content (Chladni,
   Voronoi, Perlin, fractal, gradient). If even a procedural anchor
   works, the claim is even stronger — substrate priors come from
   structure not semantics.
5. **Image-based vs caption-aligned anchor**: should the anchor have
   per-cell labels for Stage 3B class conditioning, or do we keep
   anchor and class-cond separate? Recommendation: keep separate.
   Anchor is unconditional prior; conditioning still uses CIFAR class
   ids.

---

## 9. References

1. Shaham *et al.* 2019. *SinGAN: Learning a Generative Model from a
   Single Natural Image.* ICCV. — single-image GAN training precedent.
2. Kulikov *et al.* 2023. *SinDDM: A Single Image Denoising Diffusion
   Model.* — single-image diffusion training precedent.
3. Cimpoi *et al.* 2014. *Describing Textures in the Wild (DTD).*
   CVPR. — texture dataset for the T cells.
4. Karras *et al.* 2019. *FFHQ.* StyleGAN release. — face cells.
5. Ye *et al.* 2025. *Dream 7B.* arXiv:2508.15487. — base for
   Omni-Diffusion (which planted the curriculum-quickcook idea VOD is
   borrowing).
6. APT-Transformer `apt/trainops/scripts/pretrain_quickcook.py` —
   internal HLBD anchor + dilution implementation that this proposal
   ports to the image domain.

VOD's own context:

7. `docs/omni_diffusion_lessons_for_vod.md` — Stage 3 four-sub-step
   plan; this anchor proposal extends 3A.
8. `docs/paper_v16_baseline.md` — paper draft (v16 / Stage 1 / Stage 2);
   anchor result would slot into a new Section 8.6 as a stand-alone
   contribution.
