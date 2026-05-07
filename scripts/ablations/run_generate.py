"""Unconditional generation from a trained NativeVOD.

Pipeline:
    1. (optional) brief training with sweet-spot weights to get a real model
    2. Sample U_init ~ N(0, sigma_init) over (T, H, W, C)
    3. Run model.denoise_path(U_init, steps=K) — model has no target,
       just refines U toward "clean field" subspace it learned
    4. Decode each medium projection
    5. Render image as PNG, video as frame grid PNG + GIF if PIL/imageio,
       print text + audio summary

Output goes to D:\\VOD\\prototype\\generated\\<run_name>\\
"""

from __future__ import annotations
import argparse, time
from pathlib import Path
import numpy as np
import torch

import sys
sys.path.insert(0, "D:/VOD/prototype")

from vod_minimal.blocky_scattering import build_blocky_scattering_batch
from vod_minimal.native import (
    LATENT_HW, LATENT_T, NativeLossWeights, NativeVOD, NativeVODConfig,
    native_total_loss, views_to_numpy, views_to_torch,
)


def _normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float64)
    lo, hi = a.min(), a.max()
    if hi - lo < 1e-9:
        return np.zeros_like(a, dtype=np.uint8)
    return ((a - lo) / (hi - lo) * 255.0).clip(0, 255).astype(np.uint8)


def save_image_png(arr: np.ndarray, path: Path) -> None:
    from PIL import Image
    a = np.asarray(arr).squeeze()
    img = _normalize_to_uint8(a)
    if img.ndim == 2:
        Image.fromarray(img, mode="L").save(path)
    else:
        # Fallback: take first 2-D slice
        Image.fromarray(img.reshape(-1, img.shape[-1]) if img.ndim == 3 else img,
                        mode="L").save(path)


def save_video_grid_png(video: np.ndarray, path: Path, *, ncols: int = 4) -> None:
    """Render (T, H, W) video as a tiled grid PNG."""
    from PIL import Image
    v = np.asarray(video).squeeze()
    T, H, W = v.shape
    nrows = (T + ncols - 1) // ncols
    grid = np.zeros((nrows * H, ncols * W), dtype=np.uint8)
    for t in range(T):
        r, c = t // ncols, t % ncols
        grid[r*H:(r+1)*H, c*W:(c+1)*W] = _normalize_to_uint8(v[t])
    Image.fromarray(grid, mode="L").save(path)


def save_video_gif(video: np.ndarray, path: Path, *, fps: int = 4) -> None:
    """Save (T, H, W) video as animated GIF."""
    from PIL import Image
    v = np.asarray(video).squeeze()
    frames = [Image.fromarray(_normalize_to_uint8(f), mode="L") for f in v]
    duration_ms = int(1000 / fps)
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=duration_ms, loop=0)


def train_briefly(args, device) -> NativeVOD:
    rng = np.random.default_rng(args.seed)
    train_b = build_blocky_scattering_batch(
        rng, batch_size=args.train_n, size=LATENT_HW, frames=LATENT_T,
        artifact_strength=0.6, tile=4, spacetime=True, temporal_mode="flicker",
        flicker_strength=0.3, paired_denoising=True,
    )
    cfg = NativeVODConfig(channels=args.channels, hidden=args.hidden,
                           denoise_steps=args.steps, backbone="unet")
    m = NativeVOD(cfg).to(device)
    s = args.scale
    w = NativeLossWeights(
        field=0.5, media=1.0, text=0.0,
        temporal=0.1*s, artifact=0.1*s,
        binary_twin_pixel=0.1*s, aimp=0.1*s,
    )
    opt = torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=1e-4)
    print(f"training: epochs={args.epochs} backbone=unet weight_scale={s}", flush=True)
    t0 = time.time()
    for ep in range(args.epochs):
        m.train()
        opt.zero_grad(set_to_none=True)
        L = []
        for samp in train_b.samples:
            n_t = views_to_torch(samp.noisy_views, device)
            t_t = views_to_torch(samp.target_views, device)
            loss, _ = native_total_loss(m, n_t, t_t, weights=w)
            L.append(loss)
        loss_total = torch.stack(L).mean()
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if (ep + 1) % max(1, args.epochs // 5) == 0:
            print(f"  ep={ep+1:4d}  L={float(loss_total):.4f}", flush=True)
    print(f"trained in {time.time()-t0:.1f}s", flush=True)
    return m


def generate(model: NativeVOD, *, n_samples: int, sigma_init: float,
             denoise_steps: int, device, seed: int,
             init_mode: str = "perturbed_clean") -> list[dict]:
    """Sample U init, run denoise_path, decode each medium.

    init_mode:
        "random"           — U_init ~ N(0, sigma_init), pure noise. The
                              model is a paired denoiser, not a true
                              unconditional generator, so this usually
                              produces noise unless the model has been
                              trained for a very long time.
        "perturbed_clean"  — Build a fresh clean Chladni field, encode
                              to U_clean, add N(0, sigma_init) noise,
                              denoise. This is the model's actual
                              training domain (clean+noise→clean) and
                              should produce visible Chladni structure.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    out = []
    model.eval()
    with torch.no_grad():
        for i in range(n_samples):
            if init_mode == "random":
                U = sigma_init * torch.randn(
                    (1, LATENT_T, LATENT_HW, LATENT_HW, model.config.channels),
                    device=device, generator=g,
                )
            elif init_mode == "perturbed_clean":
                rng = np.random.default_rng(seed + 1000 * (i + 1))
                clean_batch = build_blocky_scattering_batch(
                    rng, batch_size=1, size=LATENT_HW, frames=LATENT_T,
                    artifact_strength=0.0, tile=4, spacetime=True,
                    temporal_mode="static", flicker_strength=0.0,
                    paired_denoising=True,
                )
                target = views_to_torch(clean_batch.samples[0].target_views, device)
                U_clean = model.encode(target)
                noise = sigma_init * torch.randn_like(U_clean)
                U = U_clean + noise
            else:
                raise ValueError(f"unknown init_mode={init_mode!r}")
            U_clean = model.denoise_path(U, steps=denoise_steps)
            views = model.decode(U_clean)
            out.append({k: v.squeeze(0).cpu().numpy() for k, v in views.items()})
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="default")
    p.add_argument("--seed", type=int, default=430)
    p.add_argument("--n-samples", type=int, default=4)
    p.add_argument("--sigma-init", type=float, default=0.5)
    p.add_argument("--denoise-steps", type=int, default=8,
                   help="iterations of model.denoise_path during generation")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--train-n", type=int, default=16)
    p.add_argument("--channels", type=int, default=4)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--steps", type=int, default=4,
                   help="model.config.denoise_steps (used during training only)")
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--scale", type=float, default=0.03,
                   help="distinctive weight scale (sweet spot 0.03 from sweetspot scan)")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--out-dir", default="D:/VOD/prototype/generated")
    p.add_argument("--init-mode", choices=("random", "perturbed_clean"),
                   default="perturbed_clean")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    out_root = Path(args.out_dir) / args.name
    out_root.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = train_briefly(args, device)
    samples = generate(
        model, n_samples=args.n_samples, sigma_init=args.sigma_init,
        denoise_steps=args.denoise_steps, device=device, seed=args.seed + 1,
        init_mode=args.init_mode,
    )

    print(f"\nsaving {len(samples)} samples to {out_root}", flush=True)
    for i, views in enumerate(samples):
        sample_dir = out_root / f"sample_{i:02d}"
        sample_dir.mkdir(exist_ok=True)
        if "image" in views:
            save_image_png(views["image"], sample_dir / "image.png")
        if "video" in views:
            v = np.asarray(views["video"]).squeeze()
            if v.ndim == 3:
                save_video_grid_png(v, sample_dir / "video_grid.png")
                try:
                    save_video_gif(v, sample_dir / "video.gif")
                except Exception as e:
                    print(f"  GIF write failed for sample {i}: {e}", flush=True)
        if "audio" in views:
            a = views["audio"].ravel()
            print(f"  sample {i}: audio  shape={a.shape}  range=[{a.min():.3f},{a.max():.3f}]", flush=True)
        if "text" in views:
            t = views["text"].ravel()
            quantized = np.round(np.clip(t, 0, 1) * 15).astype(int)
            print(f"  sample {i}: text   {quantized.tolist()[:16]}{'...' if len(quantized) > 16 else ''}", flush=True)
        print(f"  sample {i} → {sample_dir}", flush=True)

    print(f"\ndone. Open D:\\VOD\\prototype\\generated\\{args.name}\\sample_00\\ to inspect.", flush=True)


if __name__ == "__main__":
    main()
