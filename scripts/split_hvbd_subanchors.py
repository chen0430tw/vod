"""Split HVBD master PNGs into per-cell PNGs + metadata.jsonl.

Uniform-grid splitter for the regular (rows × cols) HVBD masters.
Irregular masters (L7 groups-of-3×3, L8 groups-of-5×5, HVBDT benchmark
sheets) need per-master logic — TODO when those raw PNGs land.

Per the user instruction: split into single-cell PNGs (NOT random-crop
the whole sheet). This avoids the substrate learning the 14×16 grid
geometry as a prior.

Usage:
    py -3.13 scripts/split_hvbd_subanchors.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

from PIL import Image

# Domain labels per column for HVBD_anchor_core_v1 (14 rows × 16 cols).
# Inferred from the user-provided artifact (ChatGPT Image 2026-05-08 12_42_44):
ANCHOR_CORE_DOMAINS_BY_COL = [
    "Face", "Animal", "Vehicle", "Texture",
    "Object", "Building", "Natural", "Pattern",
    "Chart", "Indoor", "Sketch", "Edge",
    "Handwriting", "Lighting", "Depth", "Gradient",
]


def split_uniform_grid(
    src_path: Path, out_dir: Path,
    rows: int, cols: int, level_id: str,
    cell_id_prefix: str,
    domains_by_col: list | None = None,
) -> list:
    """Split a uniform rows×cols grid into per-cell PNGs.
    Returns list of metadata records."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(src_path) as im:
        W, H = im.size
        cell_w = W // cols
        cell_h = H // rows
        records = []
        for r in range(rows):
            for c in range(cols):
                x0, y0 = c * cell_w, r * cell_h
                x1, y1 = x0 + cell_w, y0 + cell_h
                cell = im.crop((x0, y0, x1, y1))
                domain = (domains_by_col[c] if domains_by_col and c < len(domains_by_col)
                          else f"col{c}")
                cell_id = f"{cell_id_prefix}__r{r:02d}_c{c:02d}_{domain.lower()}"
                cell_fn = f"{cell_id}.png"
                cell.save(out_dir / cell_fn)
                records.append({
                    "cell_id": cell_id,
                    "source_png": str(src_path.relative_to(src_path.parents[2])).replace("\\", "/"),
                    "level": level_id,
                    "domain": domain,
                    "row": r, "col": c,
                    "cell_pixel_bbox": [x0, y0, x1, y1],
                    "saved_to": str((out_dir / cell_fn).relative_to(src_path.parents[2])).replace("\\", "/"),
                    "variant_index": r,
                })
    return records


def main():
    proj = Path(__file__).resolve().parent.parent
    raw_dir = proj / "data/hvbd_static/raw"
    cells_dir = proj / "data/hvbd_static/cells"
    metadata_path = proj / "data/hvbd_static/metadata.jsonl"

    cells_dir.mkdir(parents=True, exist_ok=True)
    all_records = []

    # Anchor core: 14×16, columns are 16 visual domains.
    src = raw_dir / "HVBD_anchor_core_v1.png"
    if src.exists():
        recs = split_uniform_grid(
            src, cells_dir / "anchor_core_v1",
            rows=14, cols=16, level_id="anchor_core",
            cell_id_prefix="HVBD_anchor_core_v1",
            domains_by_col=ANCHOR_CORE_DOMAINS_BY_COL,
        )
        all_records.extend(recs)
        print(f"[split] HVBD_anchor_core_v1.png → {len(recs)} cells")
    else:
        print(f"[split] SKIP HVBD_anchor_core_v1.png (not present)")

    # L1-L6 will be added in similar fashion when their raw PNGs land.
    # L7/L8/HVBDT-benchmark are non-uniform — handled by separate scripts.

    with metadata_path.open("w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[split] wrote metadata: {metadata_path}  ({len(all_records)} cells)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
