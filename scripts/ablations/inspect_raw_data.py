"""Print raw training image / video as ASCII so we see what VOD actually
gets fed, not just PNG renderings.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

_PROTO = Path(__file__).resolve().parent.parent.parent / "prototype"
if str(_PROTO) not in sys.path:
    sys.path.insert(0, str(_PROTO))

from vod_minimal.blocky_scattering import build_blocky_scattering_batch
from vod_minimal.native import LATENT_HW, LATENT_T

CHARS = " .:-=+*#%@"

def to_ascii(arr: np.ndarray) -> str:
    arr = np.asarray(arr, dtype=np.float64)
    lo, hi = float(arr.min()), float(arr.max())
    span = hi - lo if hi > lo else 1.0
    norm = (arr - lo) / span  # [0,1]
    rows = []
    for row in norm:
        rows.append("".join(CHARS[min(int(v * (len(CHARS) - 1) + 0.5), len(CHARS) - 1)] for v in row))
    return "\n".join(rows)


def main():
    rng = np.random.default_rng(430)
    batch = build_blocky_scattering_batch(
        rng, batch_size=4, size=LATENT_HW, frames=LATENT_T,
        artifact_strength=0.0, tile=4, spacetime=True,
        temporal_mode="static", flicker_strength=0.0, paired_denoising=True,
    )

    print(f"LATENT_HW = {LATENT_HW}, LATENT_T = {LATENT_T}")
    print()

    for i in range(2):
        s = batch.samples[i]
        img = np.asarray(s.target_views["image"])
        vid = np.asarray(s.target_views["video"])
        print("=" * 80)
        print(f"SAMPLE {i}")
        print(f"  image shape={img.shape}  std={img.std():.4f}  range=[{img.min():+.3f},{img.max():+.3f}]")
        print(f"  video shape={vid.shape}  std={vid.std():.4f}  range=[{vid.min():+.3f},{vid.max():+.3f}]")
        print()
        print(f"  IMAGE (single 2D frame, what diffuser learns to reproduce after broadcast to T={LATENT_T}):")
        print(to_ascii(img))
        print()
        print(f"  VIDEO frame 0:")
        print(to_ascii(vid[0]))
        print(f"  VIDEO frame 4:")
        print(to_ascii(vid[4]))
        print(f"  VIDEO frame 7:")
        print(to_ascii(vid[7]))
        # Per-frame std to verify "static" really means identical frames
        per_frame_std = [vid[t].std() for t in range(vid.shape[0])]
        per_frame_mean = [vid[t].mean() for t in range(vid.shape[0])]
        print(f"  per-frame std:  {[f'{x:.3f}' for x in per_frame_std]}")
        print(f"  per-frame mean: {[f'{x:+.3f}' for x in per_frame_mean]}")
        # Frame-to-frame difference to detect identical frames
        diff = np.abs(vid[1:] - vid[:-1]).mean(axis=(1, 2))
        print(f"  |frame[t+1] - frame[t]|.mean: {[f'{x:.4f}' for x in diff]}")
        print(f"  is video temporally static? {bool(np.all(diff < 1e-6))}")
        print(f"  is image == video[T//2]? {bool(np.allclose(img, vid[LATENT_T // 2]))}")
        print()


if __name__ == "__main__":
    main()
