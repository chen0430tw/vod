# B200 Runbook (VOD + HVBD anchor + dilution)

**Audience**: 430 / Claude renting RunPod B200×8 instance for VOD
training. Per `HVBD_VOD_Claude_Work_Guide.md` §7.

---

## 0. Pre-rent checklist (do BEFORE renting B200)

Do **not** waste B200 minutes on file generation, downloads, or
directory wrangling. Before SSH-ing in, verify:

```
[host] D:\VOD\
  data/
    hvbd_static/
      raw/HVBD_anchor_core_v1.png            ✅ (already in repo)
      raw/HVBD_L1_primitives.png             [ ] needs generation
      raw/HVBD_L2_textures_patterns.png      [ ] needs generation
      raw/HVBD_L3_geometry_spectrum.png      [ ] needs generation
      raw/HVBD_L4_sketch_edges.png           [ ] needs generation
      raw/HVBD_L5_grayscale_depth_channels.png [ ] needs generation
      raw/HVBD_L6_rgb_natural_domains.png    [ ] needs generation
      raw/HVBD_L7_multiview_variants.png     [ ] needs generation
      raw/HVBD_L8_multistyle_multimedia.png  [ ] needs generation
    hvbdt/
      sheets/HVBDT_core_motion_primitives.png ✅
      sheets/HVBDT_core_camera_motion.png    [ ] needs generation
      sheets/HVBDT_anime_frame_strips.png    [ ] needs generation
      ... (12 sheets total, only 1 done)
    dilution_small/
      cifar10/                                [ ] need to download
      cifar100/                                [ ] need to download
      imagenet100/                             [ ] need to download
  configs/                                    ✅
  scripts/                                    ✅
```

Run `scripts/build_hvbd_registry.py` locally — it will tell you which
PNGs are missing. Do **not** rent B200 until all 9 + 12 PNGs are in
`data/hvbd_static/raw/` and `data/hvbdt/sheets/`.

---

## 1. SSH connect

```bash
# RunPod gives you something like:
wsl ssh root@<IP> -p <PORT>
# or via the on-cluster tooling.
```

First minute on box:

```bash
nvidia-smi -L                              # verify 8× B200
df -h /workspace                           # verify storage
which conda                                # verify env
```

---

## 2. Sync repo to box

```bash
# from local D:\VOD
wsl bash -c "rsync -av --exclude '.git' --exclude '*.pt' \
    /mnt/d/VOD/ root@<IP>:/workspace/VOD/"
```

Then on box:

```bash
cd /workspace/VOD
pip install -r requirements.txt
python scripts/build_hvbd_registry.py     # verify all PNGs present
python scripts/split_hvbd_subanchors.py   # cells/ ready
python scripts/make_dilution_schedule.py --total-steps 60000
```

CPU preflight (Codex protocol §):

```bash
py -3.13 scripts/preflight_rgb64.py --image-size 64 --train-n 8 --steps 50
```

If preflight fails, do **not** launch GPU. Diagnose first.

---

## 3. First B200 run — verify signal (guide §7.2)

Five experiments, each ~1-2h on B200×8. Total ~6-10h.

```bash
# 1. VOD-100M no-anchor (control)
python scripts/train_vod.py --config configs/vod_100m_no_anchor.yaml

# 2. VOD-100M HVBD-anchor (treatment)
python scripts/train_vod.py --config configs/vod_100m_hvbd.yaml

# 3-5: scale up
python scripts/train_vod.py --config configs/vod_300m_no_anchor.yaml
python scripts/train_vod.py --config configs/vod_300m_hvbd.yaml
python scripts/train_vod.py --config configs/vod_500m_hvbd.yaml
```

After each run:

```bash
python scripts/eval_samples.py \
    --generated-dir experiments/static_anchor_ablation/<run>/generated \
    --train-ref-dir data/hvbd_static/cells/anchor_core_v1 \
    --report-out experiments/static_anchor_ablation/<run>/eval.json
```

**Look for**:

* HVBD-anchor runs hit lower descriptor_distance_to_ref earlier than
  no-anchor? (Yes → HVBD accelerates structure formation, claim 1.)
* `grid_artifact_score < 0.3` on HVBD runs? (Yes → no mosaic
  collapse. No → reduce anchor_prob, increase random crop.)
* `diversity_generated > diversity_ref * 0.5`? (Yes → multi-domain
  diversity preserved.)

---

## 4. Second B200 — paper comparison (guide §7.3)

```bash
python scripts/train_vod.py --config configs/vod_500m_hvbd.yaml \
    --total-steps 80000   # full run
python scripts/train_ldm_baseline.py --config configs/ldm_baseline.yaml
```

Generate the paper comparison table:

```bash
python scripts/eval_samples.py --compare-runs \
    experiments/static_anchor_ablation/vod_500m_hvbd \
    experiments/ldm_vs_vod/ldm_baseline \
    --out reports/tables/type_b_vs_type_a.json
```

Decisive metrics:

| | VOD-500M-HVBD | LDM-baseline | Ratio |
|---|---|---|---|
| total wall-clock | T_vod | T_ldm | T_ldm/T_vod (target ~16×) |
| anchor PNGs needed | 1 | 16 | 16× saved |
| VAEs trained | 0 | 16 | type-B advantage |
| domain routing required | no | yes | type-B advantage |

---

## 5. Third B200 — 1B static (guide §7.4)

If second B200 verdict is favorable:

```bash
python scripts/train_vod.py --config configs/vod_1b_static.yaml
```

Targets:

* hidden=512, channels=8, ~1B params
* Full HVBD + multi-domain dilution
* prompt-conditioning enabled (Stage 2 class-id; Stage 3 future text)
* UI / anime / infographic / game-screenshot small-domain top-up

---

## 6. Fourth B200 — VOD-T video end (guide §7.5)

```bash
python scripts/train_vodt.py --config configs/vodt_300m_motion.yaml
python scripts/train_vodt.py --config configs/vodt_500m_frame_interp.yaml
python scripts/train_vodt.py --config configs/vodt_500m_anime_strips.yaml
python scripts/train_vodt.py --config configs/vodt_image_to_video_smoke.yaml
```

---

## 7. Cost-budget table

| stage | runs | wall-clock | rough cost (RunPod $5/h B200×8) |
|-------|------|-----------|---------------------------------|
| First B200 | 5 ablation | ~10h | ~$50 |
| Second B200 | 2 (VOD vs LDM full) | ~12h | ~$60 |
| Third B200 | 1 (1B static) | ~24h | ~$120 |
| Fourth B200 | 4 (VOD-T smokes) | ~16h | ~$80 |
| **Total** | | ~62h | **~$310** |

Plus storage / data prep / failed retries: budget **$400-500** for
the full Stage 3 plan.

---

## 8. Hard rules

1. **Preflight before every GPU launch** — Codex protocol. Do not
   skip.
2. **Checkpoint every 1000 steps** — B200 instances can be reclaimed;
   long unsaved runs = pure loss.
3. **rsync results back to local D:\VOD\experiments\ after each run**
   — do not leave artifacts on a rented box.
4. **Mosaic check after every HVBD run** — `grid_artifact_score >
   0.5` triggers mandatory diagnosis: reduce anchor_prob, add random
   rotation/scale, or use multi-anchor rotation.
5. **No 17B at first round** — guide §0.2.
