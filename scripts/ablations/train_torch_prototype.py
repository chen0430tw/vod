"""Train the minimal VOD prototype with a shared PyTorch updater."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from vod_minimal.core import (
    MEDIA,
    ProjectionBatch,
    build_projection_batch,
    evaluate_projection_error,
    projection_loss,
    shared_update_rollout,
)
from vod_minimal.schema import checkpoint_payload, print_metrics_block
from vod_minimal.torch_artifacts import artifact_train_loss
from vod_minimal.torch_model import SharedPointUpdater, to_tensor


def build_dataset(
    seed: int,
    n: int,
    *,
    artifact_suppression: bool = False,
    artifact_scale: float | None = None,
    artifact_tile: int = 8,
    spacetime_video: bool = False,
    frames: int = 10,
) -> ProjectionBatch:
    rng = np.random.default_rng(seed)
    return build_projection_batch(
        rng,
        batch_size=n,
        artifact_suppression=artifact_suppression,
        artifact_scale=artifact_scale,
        artifact_tile=artifact_tile,
        spacetime=spacetime_video,
        frames=frames,
    )


def make_update_fn(model: SharedPointUpdater):
    def _update(current: torch.Tensor, target: torch.Tensor, medium: str) -> torch.Tensor:
        del medium
        return model.forward_step(current, target)

    return _update


@torch.no_grad()
def evaluate(
    model: SharedPointUpdater,
    batch: ProjectionBatch,
    device: torch.device,
    steps: int,
    *,
    include_artifact_metrics: bool = False,
    artifact_tile: int = 8,
) -> dict[str, float]:
    model.eval()
    update_fn = make_update_fn(model)

    def _rollout(noisy_views: dict[str, np.ndarray], target_views: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        denoised = {}
        for medium in MEDIA:
            noisy = to_tensor(noisy_views[medium], device)
            target = to_tensor(target_views[medium], device)
            pred = shared_update_rollout(update_fn, noisy, target, medium, steps=steps)
            denoised[medium] = pred.detach().cpu().numpy()
        return denoised

    metrics = evaluate_projection_error(
        _rollout,
        batch,
        include_artifact_metrics=include_artifact_metrics,
        artifact_tile=artifact_tile,
    )
    loss = projection_loss(update_fn, batch, steps=steps, device=device)
    metrics["loss"] = float(loss.detach().cpu())
    return metrics


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    torch.manual_seed(args.seed)
    train_samples = build_dataset(
        args.seed,
        args.train_n,
        artifact_suppression=args.artifact_suppression,
        artifact_scale=args.artifact_scale,
        artifact_tile=args.artifact_tile,
        spacetime_video=args.spacetime_video,
        frames=args.frames,
    )
    test_samples = build_dataset(
        args.seed + 1,
        args.test_n,
        artifact_suppression=args.artifact_suppression,
        artifact_scale=args.artifact_scale,
        artifact_tile=args.artifact_tile,
        spacetime_video=args.spacetime_video,
        frames=args.frames,
    )

    model = SharedPointUpdater(hidden=args.hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    last_artifact_loss = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        update_fn = make_update_fn(model)
        loss = projection_loss(update_fn, train_samples, device=device, steps=args.steps)
        # Short-circuit: when weight is zero we MUST NOT touch the loss tensor
        # so behaviour with --artifact-loss-weight=0 is bit-exact equivalent
        # to the version before this feature existed.
        if args.artifact_loss_weight > 0:
            art_loss = artifact_train_loss(
                update_fn,
                train_samples,
                steps=args.steps,
                device=device,
                tile=args.artifact_tile,
            )
            loss = loss + args.artifact_loss_weight * art_loss
            last_artifact_loss = float(art_loss.detach().cpu())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            train_metrics = evaluate(
                model,
                train_samples,
                device,
                steps=args.steps,
                include_artifact_metrics=args.artifact_metrics,
                artifact_tile=args.artifact_tile,
            )
            test_metrics = evaluate(
                model,
                test_samples,
                device,
                steps=args.steps,
                include_artifact_metrics=args.artifact_metrics,
                artifact_tile=args.artifact_tile,
            )
            artifact_tag = (
                f" art_loss={last_artifact_loss:.6f}"
                if args.artifact_loss_weight > 0
                else ""
            )
            print(
                f"epoch={epoch:04d} "
                f"train_loss={train_metrics['loss']:.6f} "
                f"test_loss={test_metrics['loss']:.6f} "
                f"test_after={test_metrics['mean_after']:.6f} "
                f"test_success={test_metrics['success_rate']:.3f}"
                f"{artifact_tag}"
            )

    train_metrics = evaluate(
        model,
        train_samples,
        device,
        steps=args.steps,
        include_artifact_metrics=args.artifact_metrics,
        artifact_tile=args.artifact_tile,
    )
    test_metrics = evaluate(
        model,
        test_samples,
        device,
        steps=args.steps,
        include_artifact_metrics=args.artifact_metrics,
        artifact_tile=args.artifact_tile,
    )

    if args.save:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            checkpoint_payload(
                state_dict=model.state_dict(),
                model_type="SharedPointUpdater",
                train_args=vars(args),
                train_metrics=train_metrics,
                test_metrics=test_metrics,
            ),
            save_path,
        )

    print()
    print("Trainable Minimal VOD Prototype")
    print("===============================")
    print(f"device = {device}")
    print(f"steps  = {args.steps}")
    print()
    print_metrics_block("Train metrics", train_metrics)
    print()
    print_metrics_block("Test metrics", test_metrics)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train shared PyTorch updater for VOD minimal prototype.")
    parser.add_argument("--seed", type=int, default=430)
    parser.add_argument("--train-n", type=int, default=16)
    parser.add_argument("--test-n", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--save", default="D:\\VOD\\prototype\\checkpoints\\shared_point_updater.pt")
    parser.add_argument(
        "--artifact-metrics",
        action="store_true",
        help="include tile-residue artifact diagnostics in train/test metrics (default: off)",
    )
    parser.add_argument(
        "--artifact-tile",
        type=int,
        default=8,
        help="tile period used by tile_residue and the differentiable training penalty",
    )
    parser.add_argument(
        "--artifact-suppression",
        action="store_true",
        help="apply 4/e Orthogonal Compression Noise to BOTH train and test "
             "noisy_views — this changes the training distribution, not just eval (default: off)",
    )
    parser.add_argument(
        "--artifact-scale",
        type=float,
        default=None,
        help="base scale for the suppression noise; defaults to noise_scale when unset",
    )
    parser.add_argument(
        "--artifact-loss-weight",
        type=float,
        default=0.0,
        help="weight for the differentiable artifact regularization penalty added to "
             "projection_loss (default: 0.0 — no change to training)",
    )
    parser.add_argument(
        "--spacetime-video",
        action="store_true",
        help="build the video projection from a 3-D U(t,y,x) Chladni field "
             "instead of the legacy 2-D roll+sin-phase shortcut (default: off)",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=10,
        help="number of frames in the video projection when --spacetime-video is set",
    )
    train(parser.parse_args())


if __name__ == "__main__":
    main()
