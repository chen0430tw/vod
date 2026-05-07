"""Convert v8_monitor_short.log into TrainingTrace JSON for forecast.

We have step + train_loss; grad_norm/curvature/direction_consistency are
estimated from the loss curve itself (numerical surrogates):
    grad_norm ≈ |dL/dstep| (forward diff, normalised)
    curvature ≈ |d²L/dstep²|
    direction_consistency ≈ 1 - sign-flip ratio

target_metric uses negative L_diff (lower-is-better → flip for higher-is-better).
"""
from __future__ import annotations
import json
import re
from pathlib import Path

import numpy as np

LOG_RE = re.compile(
    r"ep=\s*(\d+)/\d+\s+step=\s*(\d+)\s+L_diff=([\d.]+)\s+L_recon=([\d.]+)"
)


def main():
    log_path = Path("runs/v8_monitor_short.log")
    if not log_path.exists():
        # try mirror
        log_path = Path("runs_local_mirror/v8_monitor_short.log")
    text = log_path.read_text()

    rows = []
    for m in LOG_RE.finditer(text):
        ep, step, ld, lr = m.groups()
        rows.append((int(ep), int(step), float(ld), float(lr)))
    if not rows:
        raise SystemExit("no log rows matched")

    eps = np.array([r[0] for r in rows])
    steps = np.array([r[1] for r in rows])
    L = np.array([r[2] for r in rows])
    Lr = np.array([r[3] for r in rows])

    # Numerical surrogates
    dL = np.gradient(L, steps)
    d2L = np.gradient(dL, steps)
    grad_norm = np.abs(dL) * 1000.0   # scale ~1
    curvature = np.abs(d2L) * 1000.0
    sign_flips = np.cumsum(np.diff(L) > 0).astype(float)
    flip_ratio = np.concatenate([[0.0], sign_flips / np.maximum(1, np.arange(1, len(L)))])
    direction_consistency = 1.0 - flip_ratio

    out = {
        "run_id": "v8_monitor_short_lr1e-4",
        "checkpoint_path": "prototype/_monitor_short.json",
        "target_metric": "neg_L_diff",
        "steps": [
            {
                "step": int(steps[i]),
                "train_loss": float(L[i]),
                "val_metric": float(-L[i]),  # negate so higher=better
                "grad_norm": float(grad_norm[i]),
                "curvature": float(curvature[i]),
                "direction_consistency": float(direction_consistency[i]),
            }
            for i in range(len(rows))
        ],
    }
    out_path = Path("tensorearch_reports/v8_training_trace.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}  steps={len(rows)}  "
          f"L_diff range {L.min():.4f}..{L.max():.4f}  "
          f"final L_diff={L[-1]:.4f}")


if __name__ == "__main__":
    main()
