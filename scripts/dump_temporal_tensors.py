"""Dump VOD spacetime field tensors for tensorearch temporal-* analysis.

Outputs three .npz files into tensorearch_reports/:
  - v8_chladni_T8.npz    (T,H,W) static-broadcast image stack (target)
  - v8_chladni_uv.npz    (T,H,W) u/v vector field decomposed via Sobel grads
  - v8_chladni_5d.npz    (T,H,W,C) full latent target (channels=1 placeholder)

These are produced from the same Chladni spacetime field that train uses
(build_blocky_scattering_batch + Fix A static broadcast). No GPU needed.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

_PROTO = Path(__file__).resolve().parent.parent / "prototype"
if str(_PROTO) not in sys.path:
    sys.path.insert(0, str(_PROTO))

from vod_minimal.blocky_scattering import build_blocky_scattering_batch
from vod_minimal.native import LATENT_HW, LATENT_T


def main():
    rng = np.random.default_rng(430)
    batch = build_blocky_scattering_batch(
        rng, batch_size=32, size=LATENT_HW, frames=LATENT_T,
        artifact_strength=0.0, tile=4, spacetime=True,
        temporal_mode="static", flicker_strength=0.0, paired_denoising=True,
    )
    # Stack first sample's image broadcast over T frames -> (T, H, W)
    img0 = np.asarray(batch.samples[0].target_views["image"], dtype=np.float32)
    # Fix A: static broadcast — frames are identical to image
    series_THW = np.broadcast_to(img0, (LATENT_T, LATENT_HW, LATENT_HW)).copy()

    # ----------------- 1. (T, H, W) for `temporal` --------------------- #
    out1 = Path("tensorearch_reports/v8_chladni_T8.npz")
    out1.parent.mkdir(parents=True, exist_ok=True)
    # Add tiny synthetic temporal variation so dispersion check has signal
    # (Fix A killed real T variation; for diagnostic we want the dispersion
    # detector to confirm "static" rather than fail on degenerate input)
    t = np.arange(LATENT_T, dtype=np.float32)[:, None, None]
    series = series_THW + 0.001 * np.sin(2 * np.pi * t / LATENT_T) * series_THW
    np.savez(out1, field=series.astype(np.float32), dt=1.0)
    print(f"wrote {out1}  shape={series.shape}")

    # ----------------- 2. (T, H, W) u/v for `temporal-radio` ----------- #
    # Sobel grads: u = ∂field/∂x, v = ∂field/∂y per timestep
    out2 = Path("tensorearch_reports/v8_chladni_uv.npz")
    u = np.zeros_like(series)
    v = np.zeros_like(series)
    for ti in range(LATENT_T):
        gy, gx = np.gradient(series[ti])
        u[ti], v[ti] = gx, gy
    np.savez(out2, u=u.astype(np.float32), v=v.astype(np.float32), dt=1.0)
    print(f"wrote {out2}  u.shape={u.shape}  v.shape={v.shape}")

    # ----------------- 3. h/uv triple for `temporal-couple` ------------ #
    # Treat field as height h, gradient as (u, v) — geostrophic-style
    out3 = Path("tensorearch_reports/v8_chladni_huv.npz")
    np.savez(out3, h=series.astype(np.float32), u=u.astype(np.float32),
             v=v.astype(np.float32), dt=1.0)
    print(f"wrote {out3}  h.shape={series.shape}")

    # ----------------- 4. balance (potential/response) ----------------- #
    # potential = field, response = ∂field/∂t (numerical), forcing = constant 0
    out4 = Path("tensorearch_reports/v8_chladni_balance.npz")
    dfdt = np.zeros_like(series)
    dfdt[:-1] = series[1:] - series[:-1]
    dfdt[-1] = dfdt[-2]
    forcing = np.zeros_like(series)
    np.savez(out4, potential=series.astype(np.float32),
             response=dfdt.astype(np.float32), forcing=forcing.astype(np.float32),
             dt=1.0)
    print(f"wrote {out4}  potential.shape={series.shape}")


if __name__ == "__main__":
    main()
