"""Anchor-cell dataset + DataLoader for VOD training.

Loads cell PNGs from `data/hvbd_static/cells/`, applies in-cell
random crop / resize / horizontal flip / RGB-only conversion as
training-time augmentation. **Does NOT cross cell borders** — the
guide §8 warns: cross-cell crops would inject 14×16 grid geometry
into the substrate prior.

Usage as a module:

    from scripts.crop_anchor_dataset import HVBDAnchorDataset
    ds = HVBDAnchorDataset(
        cells_dir="data/hvbd_static/cells/anchor_core_v1",
        image_size=64,
        augment=True,
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True)

Returns tensors (B, H, W, 3) in [-1, 1].
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class HVBDAnchorDataset(Dataset):
    def __init__(
        self,
        cells_dir: str | Path,
        image_size: int = 64,
        augment: bool = True,
        in_cell_crop_min: float = 0.85,   # at least 85% of cell pixels
        rgb_only: bool = True,
        metadata_path: str | Path | None = None,
    ):
        self.cells_dir = Path(cells_dir)
        self.image_size = image_size
        self.augment = augment
        self.in_cell_crop_min = in_cell_crop_min
        self.rgb_only = rgb_only

        # Discover files
        self.paths = sorted(self.cells_dir.glob("*.png"))
        if not self.paths:
            raise FileNotFoundError(
                f"No cell PNGs in {self.cells_dir}. "
                f"Run scripts/split_hvbd_subanchors.py first."
            )

        # Optional metadata lookup (cell_id → record)
        self.metadata = None
        if metadata_path is not None and Path(metadata_path).exists():
            self.metadata = {}
            with Path(metadata_path).open("r", encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    self.metadata[rec["cell_id"]] = rec

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        p = self.paths[idx]
        img = Image.open(p)
        if self.rgb_only:
            img = img.convert("RGB")
        W, H = img.size

        # In-cell random crop (NEVER outside cell)
        if self.augment:
            crop_frac = np.random.uniform(self.in_cell_crop_min, 1.0)
            crop_w = max(1, int(W * crop_frac))
            crop_h = max(1, int(H * crop_frac))
            x0 = np.random.randint(0, max(1, W - crop_w + 1))
            y0 = np.random.randint(0, max(1, H - crop_h + 1))
            img = img.crop((x0, y0, x0 + crop_w, y0 + crop_h))

        if img.size != (self.image_size, self.image_size):
            img = img.resize((self.image_size, self.image_size), Image.BILINEAR)

        # Horizontal flip with 50% (skip for cells where direction matters
        # — TODO: domain-aware flag in metadata; for now flip all)
        if self.augment and np.random.rand() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0  # → [-1, 1]
        # Keep last-axis channels (B, H, W, 3) as in run_real_image_rgb64.py
        meta = {"cell_id": p.stem}
        if self.metadata and p.stem in self.metadata:
            meta.update(self.metadata[p.stem])
        return torch.from_numpy(arr), meta


def collate_with_meta(batch):
    """Return (images, list_of_meta) — images are stacked, metadata kept as list."""
    images = torch.stack([b[0] for b in batch], dim=0)
    metas = [b[1] for b in batch]
    return images, metas


if __name__ == "__main__":
    # Smoke test
    proj = Path(__file__).resolve().parent.parent
    cells = proj / "data/hvbd_static/cells/anchor_core_v1"
    if not cells.exists():
        print(f"Run scripts/split_hvbd_subanchors.py first ({cells} missing)")
        raise SystemExit(0)
    ds = HVBDAnchorDataset(cells_dir=cells, image_size=64, augment=True)
    print(f"[anchor-ds] {len(ds)} cells loaded")
    img, meta = ds[0]
    print(f"[anchor-ds] sample 0: shape={tuple(img.shape)} dtype={img.dtype} "
          f"range=[{img.min():.3f}, {img.max():.3f}]  cell_id={meta['cell_id']}")
