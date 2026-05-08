# Experiment matrix — VOD + HVBD + HVBDT (4 B200 stages)

Per `HVBD_VOD_Claude_Work_Guide.md` §7.

---

## Stage I — first B200, signal verification

| run | config | params | data | notes |
|-----|--------|--------|------|-------|
| 1 | `configs/vod_100m_no_anchor.yaml` | 100M | CIFAR-10 | control, no anchor |
| 2 | `configs/vod_100m_hvbd.yaml` | 100M | CIFAR-10 + HVBD anchor | treatment |
| 3 | `configs/vod_300m_no_anchor.yaml` | 300M | ImageNet-100 | control at scale |
| 4 | `configs/vod_300m_hvbd.yaml` | 300M | ImageNet-100 + HVBD | treatment at scale |
| 5 | `configs/vod_500m_hvbd.yaml` | 500M | ImageNet-100 + HVBD | scaling test |

**Verdict criteria**: HVBD runs vs no-anchor runs in trained_d / diversity / time-to-recognizable-structure.

---

## Stage II — paper comparison: type-B vs type-A

| run | config | role |
|-----|--------|------|
| 1 | `configs/vod_500m_hvbd.yaml` (full 80k steps) | type-B canonical |
| 2 | `configs/ldm_baseline.yaml` | type-A 16-VAE per-domain |
| 3 | ablation: HVBD vs no-anchor vs HVBD+dilution at 500M | claim isolation |
| 4 | ablation: HVBD 224 vs random mosaic vs class-sorted dataset | anchor quality |

**Decisive table**: `reports/tables/type_b_vs_type_a.json` —
wall-clock, params, multi-domain coverage, cross-domain composition.

---

## Stage III — 1B static end

`configs/vod_1b_static.yaml` (TBD) — full HVBD + multi-domain dilution
+ prompt conditioning + small-domain top-ups. ~24h B200×8.

---

## Stage IV — video (HVBDT)

| run | config | scope |
|-----|--------|-------|
| 1 | `configs/vodt_300m_motion.yaml` | motion primitives smoke |
| 2 | `configs/vodt_500m_frame_interp.yaml` (TBD) | frame interpolation |
| 3 | `configs/vodt_500m_anime_strips.yaml` (TBD) | anime frame-strip prior |
| 4 | `configs/vodt_image_to_video.yaml` (TBD) | I2V smoke test |
