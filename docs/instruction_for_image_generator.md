# Instructions for Image Generator (HVBD / HVBDT continuation)

> **Audience**: ChatGPT-4o / Image2 / similar. Read all 5 sections
> before generating any new HVBD/HVBDT sheet.
>
> **Goal**: continue generating the remaining 8 HVBD static PNGs and
> 11 HVBDT video PNGs to feed the VOD type-B substrate training
> pipeline. The two PNGs already generated (`HVBD_anchor_core_v1.png`
> and `HVBDT_core_motion_primitives.png`) are accepted as the
> reference style — please match their visual conventions.

---

## 1. How the images will be split

The trainer does NOT random-crop the whole sheet. Instead:

1. A Python script reads the sheet, splits by **uniform grid** rows×cols
2. Each cell is saved as a separate small PNG (e.g. `HVBD_anchor_core_v1__r03_c12_face.png`)
3. Cells are loaded one at a time during training as if they were
   ImageNet samples
4. In-cell augmentation only — random crop *within* one cell,
   never across grid lines
5. 14×16 = 224 cells from `HVBD_anchor_core_v1.png` already split
   successfully (verified)

**Implication**: as long as the sheet is a clean uniform grid, my
split script auto-handles it. No metadata files needed from you.

**What breaks split**:
- Irregular row heights / column widths
- Cells that span multiple grid positions
- Headers / legends embedded inside grid cells (they get split as
  if they were content cells)
- Stretched bottom row (sheet bottom-truncation)

---

## 2. Hard rules for every sheet you generate

### 2.1 Cell count ≤ 224

Image2 reliably truncates / drops cells when total cell count
exceeds 224. Use only these layouts:

| total cells | layout | notes |
|-------------|--------|-------|
| 224 | 14 × 16 | preferred for level sheets |
| 196 | 14 × 14 | preferred for `L3_geometry_spectrum` |
| 160 | 10 × 16 | preferred for `L1_primitives` |
| 144 | 12 × 12 | preferred for HVBDT video sheets |
| 100 | 10 × 10 / 4 groups of 5×5 | for `L8_multistyle` |
| 72 | 8 groups of 3×3 | for `L7_multiview` |

Do **not** exceed 224 cells, even if the layout looks "balanced".
Image2 will silently drop the bottom rows.

### 2.2 Uniform grid

* Every row has the same height; every column has the same width
* Cell borders are visible (1-2 px gray line) so the split script
  can detect them
* No nested grids inside one cell (except L7/L8 group-of-grids
  layouts which are explicitly listed)
* No row/column labels, no axis text, no decorative title bar —
  the split script treats those as content

### 2.3 Each cell is one self-contained visual concept

* Cell = one face, one car, one texture sample, one DCT basis, etc.
* Do not span a single object across two cells
* Do not put descriptive text on top of cell content (small floating
  labels in the corner are OK if they're small enough that
  random-crop will sometimes miss them)
* Maintain visual diversity within the sheet; do not repeat the
  same example in multiple cells

### 2.4 Cell size

* For 1024×1024 sheet @ 14×16: each cell ≈ 64-73 px
* For larger sheets: cell ≈ 64-128 px
* Cell pixel size doesn't have to be exact — the split script does
  `width // cols` and `height // rows`, any small remainder is
  trimmed off the right/bottom edges

---

## 3. The 8 missing HVBD static sheets

Each row shows: **filename**, **layout**, **per-row/column theme**.

### 3.1 `HVBD_L1_primitives.png` (160, 10×16)

Per-row theme (10 rows, 16 cells each):

1. Single point variations (size, position, color)
2. Single line (orientation, length, thickness)
3. Polyline / zigzag
4. Curved line / arc
5. Wave (sine, square, triangle, sawtooth)
6. Closed shape: circle, oval, ellipse
7. Closed shape: square, rectangle, rhombus
8. Closed shape: triangle, polygon (5/6/8-side)
9. Star / starburst / sun
10. Compound primitives (cross, target, arrow, spiral, icon, emoji glyph)

### 3.2 `HVBD_L2_textures_patterns.png` (224, 14×16)

Per-row theme:

1. Black-white basic line patterns (stripes, hatching)
2. Checkerboard / grid / dot patterns
3. Optical patterns (spirals, radial bursts)
4. Procedural noise (Perlin, Voronoi, fractal)
5. Color gradients and color fields
6. Paper / wall / plaster surfaces
7. Metal / rust / mechanical surfaces
8. Stone / brick / concrete
9. Wood grain / floorboards
10. Cloth / weave / knit
11. Animal patterns (leopard, zebra, scales, feathers)
12. Water / ice / snow / fire / cloud
13. Plant / soil / natural surface
14. Decorative pattern / UI pattern

### 3.3 `HVBD_L3_geometry_spectrum.png` (196, 14×14)

Per-row theme:

1. Orthographic projection (cube, cylinder, sphere variants)
2. Axonometric projection
3. Perspective view
4. Cube colored-face views (3D primitive)
5. Basic technical drawing (front/top/side)
6. DCT basis functions (low to high frequency)
7. FFT spectrum visualizations
8. Sobel / gradient fields
9. Laplacian filter outputs
10. Canny edge maps
11. Topographic contour map
12. Vector field visualization
13. Flow field / streamlines
14. Grid transform / topology graph

### 3.4 `HVBD_L4_sketch_edges.png` (224, 14×16)

Per-row theme: line art / outline only, **no color**, **no shading**:

1. Face line-art (head outline, basic features only)
2. Human body pose line-art (full body, action poses)
3. Hand / facial expression closeup line-art
4. Animal silhouette / contour
5. Vehicle line-art (car, plane, ship, bike)
6. Building line-art (urban, ancient)
7. Indoor space line-art (perspective interior)
8. Furniture / household object sketches
9. Plant line-art
10. Mechanical / tool sketches
11. Manga panel line-art
12. Blueprint / technical line-art
13. Pure silhouette (filled black against white)
14. Canny / contour extracts of complex scenes

### 3.5 `HVBD_L5_grayscale_depth_channels.png` (224, 14×16)

Per-row theme: grayscale, depth maps, channel-isolated views:

1. Grayscale faces (luminance only)
2. Grayscale animals
3. Grayscale vehicles
4. Grayscale buildings
5. Grayscale natural scenes
6. Grayscale indoor scenes
7. Depth maps (near=light, far=dark)
8. Surface normal maps (RGB encoding xyz)
9. Segmentation masks (one color per region)
10. Luminance-only versions of color images
11. Shadow-only renders
12. Specular highlight only
13. Alpha matte / silhouette mask
14. Thermal / infrared / x-ray / blueprint-like

### 3.6 `HVBD_L6_rgb_natural_domains.png` (224, 14×16)

Per-COLUMN theme (16 cols, 14 rows of variations each):

1. Face (full RGB, varied age/sex/lighting)
2. Animal (varied species)
3. Vehicle (car, plane, ship, ...)
4. Object (tools, electronics)
5. Building (urban, rural, ancient)
6. Natural landscape
7. Indoor scene
8. Food
9. Plant
10. City street
11. Sky / weather
12. Water / ocean
13. Material close-up
14. Human activity
15. Product / commodity
16. Abstract RGB scene

This sheet is the **closest to natural-image distribution**; 14
photographs per domain.

### 3.7 `HVBD_L7_multiview_variants.png` (72, 8 groups of 3×3)

Layout: 8 distinct themes, each shown as a 3×3 mini-grid (= 9
viewpoints per theme, 8 themes total = 72 cells).

Themes:

1. Cube / primitive 3D object
2. Room / indoor space
3. City street corner
4. Vehicle (single car or single plane)
5. Standing character (full body)
6. Animal (single individual)
7. Game scene (third-person typical view)
8. Stage / performance scene

Each theme's 3×3 grid shows 9 viewpoints in this consistent order:

```
front       side        rear
high-angle  low-angle   over-the-shoulder
close-up    top-down    distant-establishing
```

### 3.8 `HVBD_L8_multistyle_multimedia.png` (100, 4 groups of 5×5)

Layout: 4 distinct subjects, each shown as 5×5 grid of 25 styles
(= 100 cells total).

Subjects:

1. City
2. Character (single anime/realistic person)
3. Object (a single iconic object — a teapot or a chair)
4. Event / scene (e.g. a wedding, a battle, a market)

Each 5×5 shows 25 styles in this consistent order:

```
photograph  anime       cel-shaded  oil-painting    watercolor
pencil-sketch  manga-page  light-novel  magazine      newspaper
news-screenshot  website-UI  chatroom    BBS          terminal
game-screenshot  surveillance  infographic  product   advertisement
sticker     ticket      handwritten meme            poster
```

---

## 4. The 11 missing HVBDT video sheets

Format conventions match `HVBDT_core_motion_primitives.png`:

* Each row = one motion atom / scenario
* Each column = a consecutive frame in time
* Time flows left → right
* Frames must be a continuous animation, not abstract concept
  diagrams

### 4.1 `HVBDT_core_dynamic_textures.png` (12 × 12)

12 dynamic-texture types (water ripples, fire flicker, smoke flow,
cloud drift, snow fall, rain streaks, electric arc, gradient flow,
noise scrolling, loading spinner, UI shimmer, light spot motion).

### 4.2 `HVBDT_core_camera_motion.png` (12 × 12)

12 camera moves over a fixed scene (pan-left, pan-right, tilt-up,
tilt-down, zoom-in, zoom-out, dolly-in, dolly-out, orbit-around,
top-down-to-eye, low-to-normal, parallax background scroll).

### 4.3 `HVBDT_anime_frame_strips.png` (12 × 12)

12 anime base actions (walk loop, run loop, turn, jump, wave, draw
sword, hair sway, skirt sway, expression change, mouth shape change,
sit-down, camera-zoom-on-character).

### 4.4 `HVBDT_anime_timing_principles.png` (12 × 12)

12 timing concepts (1-on-1, 1-on-2, 1-on-3, hold frame,
anticipation, overshoot, squash-and-stretch, smear frame, impact
frame, speed line, limited animation, loop animation).

### 4.5 `HVBDT_anime_character_consistency.png` (6 × 12)

6 consistency challenges, 12 frames each (front-to-side rotation,
360 rotation, far-to-close zoom, expression set, pose set with
identical clothing, multi-shot identity).

### 4.6 `HVBDT_anime_production_pipeline.png` (8 × 12)

8 pipeline-stage rows of the same animation:

1. rough sketch frames
2. clean lineart frames
3. flat color frames
4. shaded color frames
5. composited final frames
6. background fixed + character moving
7. camera pan over background
8. final anime shot

### 4.7 `HVBDT_control_keyframe_interpolation.png` (6 × 3)

6 transitions, 3 frames each (first / mid / last):

1. pose A → pose B
2. expression A → expression B
3. camera angle A → camera angle B
4. object position A → object position B
5. lighting A → lighting B
6. background A → background B

### 4.8 `HVBDT_control_pose_depth_lineart.png` (6 × 12)

6 control-input strips, 12 frames each:

1. pose strip
2. depth strip
3. lineart strip
4. flat-color strip (output)
5. mask-guided localized motion
6. background-static / character-moving

### 4.9 `HVBDT_control_audio_mouth_face.png` (8 × 12)

8 lip-sync / face-control rows, 12 frames each:

1. mouth A
2. mouth I
3. mouth U
4. mouth E
5. mouth O
6. neutral → smile → angry → surprised expression
7. speaking face strip + waveform (optional small wave below)
8. blink timing + head nod

### 4.10 `HVBDT_benchmark_motion_categories.png` (irregular)

Evaluation set; you can use 12 × 12 = 144 cells where rows are
motion categories (walk, run, jump, turn, wave, attack, dodge, fall,
pan, zoom, object-move, expression-change) and columns are 12 example
clips per category.

### 4.11 `HVBDT_benchmark_consistency_quality.png` (irregular)

Evaluation table-like layout; rows = quality dimensions (character
consistency, motion consistency, background stability, line
stability, color stability, prompt alignment, keyframe adherence,
temporal flicker, anime timing quality, media-dynamics correctness),
columns = scoring exemplars (good / mid / bad). 10 × 6 = 60 cells
suffices.

---

## 5. How the substrate will train on these

You don't need this to generate the images, but if you want to
sanity-check that your generated layout matches my pipeline:

1. **Phase A** (steps 0 - 5k, 100% HVBD cells): substrate sees only
   HVBD cells, drawn at random across all 1648 cell PNGs. With
   in-cell crop + flip, each cell is seen many times. Substrate
   learns multi-domain visual primitives in one image's worth of
   compute.

2. **Phase B** (steps 5k - 20k, 50/50): start mixing CIFAR-10 /
   ImageNet-100 real samples. Anchor still strong but real images
   prevent over-fitting to anchor.

3. **Phase C** (steps 20k - 0.85T, 10% anchor): real images
   dominate; HVBD anchor is "gravity" — still seen 10% of the time
   to prevent forgetting the multi-domain prior.

4. **Phase D** (steps 0.85T - T, 0-5% anchor): finishing on real
   data; HVBD presence is residual.

The trainer reads `configs/curriculum_default.json` every step and
samples from `HVBDAnchorDataset` (all 1648 cells) vs the real
dataset (CIFAR/ImageNet) according to the current `anchor_prob`.

If `eval_samples.py` reports `grid_artifact_score > 0.5` after
training (= substrate is generating samples that look like grid
mosaics), I'll diagnose: increase random rotation/scale aug, or
swap in a different HVBD anchor every N steps. **You don't need to
worry about that** — you just need to generate clean uniform grids.

---

## 6. Generation order suggestion

Generate in this order. Stop after every 2-3 sheets so 430 / Claude
can verify split-script works on them:

1. `HVBD_L1_primitives.png` (simplest, hardest to fail)
2. `HVBD_L2_textures_patterns.png`
3. `HVBD_L3_geometry_spectrum.png`  ← stop, verify
4. `HVBD_L4_sketch_edges.png`
5. `HVBD_L5_grayscale_depth_channels.png`
6. `HVBD_L6_rgb_natural_domains.png`  ← stop, verify
7. `HVBD_L7_multiview_variants.png`
8. `HVBD_L8_multistyle_multimedia.png`  ← stop, all 9 static done
9. `HVBDT_core_camera_motion.png` (similar to motion_primitives,
   easiest HVBDT to bootstrap)
10. `HVBDT_anime_frame_strips.png`  ← stop, verify
11. ... rest of 11 HVBDT sheets

After each sheet generation, save as `HVBD_<level>.png` /
`HVBDT_<topic>.png` and 430 will drop into `data/hvbd_static/raw/`
or `data/hvbdt/sheets/`.

---

## 7. Style consistency

* Use the same overall color palette range as
  `HVBD_anchor_core_v1.png` (natural photographs in saturated but
  not over-blown color).
* For HVBDT: use the simple geometric style of
  `HVBDT_core_motion_primitives.png` — clean, uniform, no
  decorative borders, frame numbers in tiny gray text below each
  frame are OK.
* Don't add a header bar with the sheet name unless it's outside the
  grid area (small text outside the cell region is fine; embedded
  title inside cell area gets split as if it were content).

That's it. Generate clean grids, ≤ 224 cells, self-contained per-cell
content, and 430's pipeline does the rest.
