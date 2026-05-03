"""Train the Tiny VDiT skeleton on the minimal VOD task.

Wired to the simplified core contract (`vod_minimal/core.py`) for:
    - dataset construction (build_projection_batch)
    - evaluation              (evaluate_projection_error + shared_update_rollout)

The training loss intentionally remains TinyVDiT's own sampled-token
implementation. core.projection_loss runs a full-view rollout per step;
`TinyVDiT.forward_sampled` instead trains on a random token subset to keep
attention compute bounded by `--max-tokens`. Forcing core.projection_loss
here would require running full attention over up to 2048+ tokens per medium
per epoch, which silently changes both training cost and gradient signal.
We keep the sampled path and only adopt the core contract at the
batch/evaluation boundaries — exactly per Codex Step 2 instructions.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from vod_minimal.core import (
    MEDIA,
    ProjectionBatch,
    build_projection_batch,
    evaluate_projection_error,
    shared_update_rollout,
)
from vod_minimal.schema import checkpoint_payload, print_metrics_block
from vod_minimal.vdit import TinyVDiT, VDiTConfig


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
    """Build a ProjectionBatch via the core contract.

    Mirrors the deterministic-seed behavior used elsewhere in the prototype:
    one rng per dataset (so train and test are independent and reproducible).
    Optional artifact_suppression flags activate 4/e Orthogonal Compression
    Noise inside the noisy_views generator — they change the training
    distribution, not the evaluation metric. `spacetime_video=True` builds
    the video projection from a real U(t, y, x) volume; the sampled-token
    forward path in TinyVDiT is shape-agnostic and consumes either the 2-D
    or 3-D video tensor without modification.
    """
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


def to_tensor(array: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(array.astype(np.float32)).to(device)


def make_update_fn(model: TinyVDiT):
    """Wrap the model's no-grad full-view step into a core-compatible update_fn.

    Returned closure has signature (current, target, medium) -> next_view and
    is intended for evaluation (not training). It calls `forward_full`, which
    is already `@torch.no_grad`-decorated and chunks tokens up to
    `chunk_tokens` to bound peak memory.
    """

    def _update(current: torch.Tensor, target: torch.Tensor, medium: str) -> torch.Tensor:
        return model.forward_full(current, target, medium)

    return _update


def train_loss(model: TinyVDiT, batch: ProjectionBatch, device: torch.device) -> torch.Tensor:
    """TinyVDiT's sampled-token loss (preserved verbatim in spirit).

    Iterates the same (sample, medium) pairs but reads them from a
    `ProjectionBatch` instead of a plain list. The per-medium MSE is
    scale-normalized exactly as before.
    """
    losses = []
    for sample in batch.samples:
        for medium in MEDIA:
            current = to_tensor(sample.noisy_views[medium], device)
            target = to_tensor(sample.target_views[medium], device)
            pred, true = model.forward_sampled(current, target, medium)
            denom = true.pow(2).mean().detach().clamp_min(1e-4)
            losses.append(F.mse_loss(pred, true) / denom)
    return torch.stack(losses).mean()


@torch.no_grad()
def evaluate(
    model: TinyVDiT,
    batch: ProjectionBatch,
    device: torch.device,
    steps: int,
    *,
    include_artifact_metrics: bool = False,
    artifact_tile: int = 8,
) -> dict[str, float]:
    """Evaluation goes through `core.evaluate_projection_error`.

    This guarantees the same metric definitions across train_torch_prototype
    and train_vdit_prototype: mean_before / mean_after / mean_improvement /
    success_rate. With include_artifact_metrics=True, tile-residue
    diagnostics are appended (artifact_*); the four base keys do not change.
    """
    model.eval()
    update_fn = make_update_fn(model)

    def _rollout(noisy_views: dict[str, np.ndarray], target_views: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        denoised: dict[str, np.ndarray] = {}
        for medium in MEDIA:
            noisy = to_tensor(noisy_views[medium], device)
            target = to_tensor(target_views[medium], device)
            pred = shared_update_rollout(update_fn, noisy, target, medium, steps=steps)
            denoised[medium] = pred.detach().cpu().numpy()
        return denoised

    return evaluate_projection_error(
        _rollout,
        batch,
        include_artifact_metrics=include_artifact_metrics,
        artifact_tile=artifact_tile,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Tiny VDiT for the VOD minimal prototype.")
    parser.add_argument("--seed", type=int, default=430)
    parser.add_argument("--train-n", type=int, default=12)
    parser.add_argument("--test-n", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--save", default="D:\\VOD\\prototype\\checkpoints\\tiny_vdit.pt")
    parser.add_argument(
        "--artifact-metrics",
        action="store_true",
        help="include tile-residue artifact diagnostics in train/test metrics (default: off)",
    )
    parser.add_argument(
        "--artifact-tile",
        type=int,
        default=8,
        help="tile period used by tile_residue and the suppression operator",
    )
    parser.add_argument(
        "--artifact-suppression",
        action="store_true",
        help="apply 4/e Orthogonal Compression Noise to BOTH train and test "
             "noisy_views — changes the training distribution (default: off). "
             "VDiT does NOT receive a differentiable artifact penalty: the "
             "sampled-token path does not preserve the full spatial grid.",
    )
    parser.add_argument(
        "--artifact-scale",
        type=float,
        default=None,
        help="base scale for the suppression noise; defaults to noise_scale when unset",
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
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    torch.manual_seed(args.seed)
    train_batch = build_dataset(
        args.seed,
        args.train_n,
        artifact_suppression=args.artifact_suppression,
        artifact_scale=args.artifact_scale,
        artifact_tile=args.artifact_tile,
        spacetime_video=args.spacetime_video,
        frames=args.frames,
    )
    test_batch = build_dataset(
        args.seed + 1,
        args.test_n,
        artifact_suppression=args.artifact_suppression,
        artifact_scale=args.artifact_scale,
        artifact_tile=args.artifact_tile,
        spacetime_video=args.spacetime_video,
        frames=args.frames,
    )

    config = VDiTConfig(
        hidden_size=args.hidden,
        depth=args.depth,
        num_heads=args.heads,
        max_tokens=args.max_tokens,
        chunk_tokens=args.max_tokens,
    )
    model = TinyVDiT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best_test: dict[str, float] | None = None
    best_state = None
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = train_loss(model, train_batch, device)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            metrics = evaluate(
                model,
                test_batch,
                device,
                steps=args.steps,
                include_artifact_metrics=args.artifact_metrics,
                artifact_tile=args.artifact_tile,
            )
            if best_test is None or metrics["mean_after"] < best_test["mean_after"]:
                best_test = metrics
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch
            print(
                f"epoch={epoch:04d} loss={float(loss.detach().cpu()):.6f} "
                f"test_after={metrics['mean_after']:.6f} "
                f"test_success={metrics['success_rate']:.3f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    train_metrics = evaluate(
        model,
        train_batch,
        device,
        steps=args.steps,
        include_artifact_metrics=args.artifact_metrics,
        artifact_tile=args.artifact_tile,
    )
    test_metrics = evaluate(
        model,
        test_batch,
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
                model_type="TinyVDiT",
                config=config.__dict__,
                train_args=vars(args),
                best_epoch=best_epoch,
                train_metrics=train_metrics,
                test_metrics=test_metrics,
            ),
            save_path,
        )

    print()
    print("Tiny VDiT Prototype")
    print("===================")
    print(f"device = {device}")
    print(f"hidden = {args.hidden}")
    print(f"depth  = {args.depth}")
    print(f"heads  = {args.heads}")
    print(f"steps  = {args.steps}")
    print(f"best_epoch = {best_epoch}")
    print()
    print_metrics_block("Train metrics", train_metrics)
    print()
    print_metrics_block("Test metrics", test_metrics)


if __name__ == "__main__":
    main()
