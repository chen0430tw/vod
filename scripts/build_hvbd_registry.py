"""Register HVBD/HVBD-T source PNGs and metadata.

Per `HVBD_VOD_Claude_Work_Guide.md` §11 task 3: walk
`data/hvbd_static/raw/` and `data/hvbdt/sheets/`, validate that each
expected master PNG exists, record dimensions, write a registry
manifest at `data/hvbd_static/registry.jsonl` and `data/hvbdt/registry.jsonl`.

Does NOT split into cells — that is `split_hvbd_subanchors.py`.

Usage:
    py -3.13 scripts/build_hvbd_registry.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

from PIL import Image

# Per the guide §2.1 — 9 static masters with explicit cell counts/layout.
EXPECTED_STATIC = [
    # (filename, rows, cols, total_cells, level_id)
    ("HVBD_anchor_core_v1.png",            14, 16, 224, "anchor_core"),
    ("HVBD_L1_primitives.png",             10, 16, 160, "L1_primitives"),
    ("HVBD_L2_textures_patterns.png",      14, 16, 224, "L2_textures_patterns"),
    ("HVBD_L3_geometry_spectrum.png",      14, 14, 196, "L3_geometry_spectrum"),
    ("HVBD_L4_sketch_edges.png",           14, 16, 224, "L4_sketch_edges"),
    ("HVBD_L5_grayscale_depth_channels.png", 14, 16, 224, "L5_grayscale_depth_channels"),
    ("HVBD_L6_rgb_natural_domains.png",    14, 16, 224, "L6_rgb_natural_domains"),
    ("HVBD_L7_multiview_variants.png",     None, None, 72, "L7_multiview_variants"),  # 8 groups of 3×3
    ("HVBD_L8_multistyle_multimedia.png",  None, None, 100, "L8_multistyle_multimedia"),  # 4 groups of 5×5
]

EXPECTED_HVBDT = [
    # (filename, rows, cols, level_id)
    ("HVBDT_core_motion_primitives.png",   10, 12, "core_motion_primitives"),
    ("HVBDT_core_dynamic_textures.png",    12, 12, "core_dynamic_textures"),
    ("HVBDT_core_camera_motion.png",       12, 12, "core_camera_motion"),
    ("HVBDT_anime_frame_strips.png",       12, 12, "anime_frame_strips"),
    ("HVBDT_anime_timing_principles.png",  12, 12, "anime_timing_principles"),
    ("HVBDT_anime_character_consistency.png", 6, 12, "anime_character_consistency"),
    ("HVBDT_anime_production_pipeline.png", 8, 12, "anime_production_pipeline"),
    ("HVBDT_control_keyframe_interpolation.png", 6, 3, "control_keyframe_interpolation"),
    ("HVBDT_control_pose_depth_lineart.png", 6, 12, "control_pose_depth_lineart"),
    ("HVBDT_control_audio_mouth_face.png", 8, 12, "control_audio_mouth_face"),
    ("HVBDT_benchmark_motion_categories.png", None, None, "benchmark_motion_categories"),
    ("HVBDT_benchmark_consistency_quality.png", None, None, "benchmark_consistency_quality"),
]


def register_dir(raw_dir: Path, expected: list, kind: str) -> list:
    rows = []
    for entry in expected:
        if kind == "static":
            fn, r, c, total, level = entry
        else:
            fn, r, c, level = entry
            total = (r * c) if (r and c) else None
        path = raw_dir / fn
        if not path.exists():
            print(f"[register-{kind}] MISSING  {fn}")
            rows.append({
                "filename": fn, "exists": False,
                "expected_rows": r, "expected_cols": c,
                "expected_cells": total, "level_id": level,
            })
            continue
        try:
            with Image.open(path) as im:
                W, H = im.size
                mode = im.mode
        except Exception as e:
            print(f"[register-{kind}] CANNOT_OPEN {fn}: {type(e).__name__}: {e}")
            continue
        rec = {
            "filename": fn, "exists": True,
            "size_px": [W, H], "mode": mode,
            "expected_rows": r, "expected_cols": c,
            "expected_cells": total, "level_id": level,
        }
        rows.append(rec)
        print(f"[register-{kind}] OK  {fn}  {W}x{H} {mode}  cells={total}")
    return rows


def main():
    proj = Path(__file__).resolve().parent.parent
    static_dir = proj / "data/hvbd_static/raw"
    hvbdt_dir = proj / "data/hvbdt/sheets"

    print("=== HVBD static registry ===")
    static_rows = register_dir(static_dir, EXPECTED_STATIC, "static")
    print("\n=== HVBD-T registry ===")
    hvbdt_rows = register_dir(hvbdt_dir, EXPECTED_HVBDT, "hvbdt")

    out_static = proj / "data/hvbd_static/registry.jsonl"
    out_hvbdt = proj / "data/hvbdt/registry.jsonl"
    with out_static.open("w", encoding="utf-8") as f:
        for r in static_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with out_hvbdt.open("w", encoding="utf-8") as f:
        for r in hvbdt_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_static_present = sum(1 for r in static_rows if r["exists"])
    n_hvbdt_present = sum(1 for r in hvbdt_rows if r["exists"])
    print(f"\nWrote registries:")
    print(f"  {out_static}  ({n_static_present}/{len(static_rows)} present)")
    print(f"  {out_hvbdt}  ({n_hvbdt_present}/{len(hvbdt_rows)} present)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
