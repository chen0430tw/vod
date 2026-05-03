# VOD Minimal Prototype

This folder implements the first Minimal Prototype Order from `D:\VOD\docs\vod_algorithms.md`.

It is intentionally small and NumPy-only. It does not generate real images, videos, audio, or text yet. It validates the first engineering interface:

```text
synthetic Chladni field
  -> projected image/video/audio/text views
  -> independent noise
  -> shared space-dependent update rule
  -> lower target-projection error
```

## Run

```powershell
py -3.13 D:\VOD\prototype\run_minimal_prototype.py --train-n 32 --test-n 32
```

Train the shared PyTorch updater:

```powershell
py -3.13 D:\VOD\prototype\train_torch_prototype.py --train-n 16 --test-n 16 --epochs 60 --steps 8
```

Train Tiny VDiT skeleton:

```powershell
py -3.13 D:\VOD\prototype\train_vdit_prototype.py --train-n 12 --test-n 12 --epochs 60 --steps 1 --hidden 64 --depth 3 --heads 4 --max-tokens 512
```

## Current Result

Saved output:

```text
D:\VOD\docs\vod_minimal_prototype_result.txt
```

Result:

```text
Train mean_before      10.602627
Train mean_after        1.414467
Train mean_improvement  9.188160
Train success_rate      1.000000

Test mean_before       10.743956
Test mean_after         1.228343
Test mean_improvement   9.515613
Test success_rate       1.000000
```

Trainable updater result:

```text
D:\VOD\docs\vod_trainable_prototype_result.txt
```

```text
Test mean_before       9.999230
Test mean_after        1.056039
Test mean_improvement  8.943191
Test success_rate      1.000000
```

`train_torch_prototype.py` now uses the simplified core contract:

```text
build_projection_batch
projection_loss
evaluate_projection_error
shared_update_rollout
```

Checkpoint metadata includes:

```text
model_type = SharedPointUpdater
core_contract_version = vod-minimal-core-v1
train_args
train_metrics
test_metrics
```

`train_vdit_prototype.py` is also wired to the core contract for the
batch/evaluation boundary:

```text
build_projection_batch
evaluate_projection_error  (via shared_update_rollout + model.forward_full)
```

Its training loss intentionally **stays on the sampled-token path**
(`TinyVDiT.forward_sampled`), not `core.projection_loss`. Reason:
projection_loss runs full-view rollouts; the VDiT path bounds attention
compute by `--max-tokens`, which is its whole point. Forcing the full-view
loss here would silently inflate cost and change the gradient signal.

VDiT checkpoint metadata includes:

```text
model_type = TinyVDiT
core_contract_version = vod-minimal-core-v1
config = VDiTConfig.__dict__
args
best_epoch
train_metrics
test_metrics
```

Tiny VDiT result:

```text
D:\VOD\docs\vod_vdit_prototype_result.txt
```

```text
Test mean_before       9.427957
Test mean_after        1.601284
Test mean_improvement  7.826673
Test success_rate      1.000000
best_epoch             50
```

## What This Proves

```text
1. One synthetic Chladni field can be projected into toy image/video/audio/text views.
2. Each view can be corrupted independently.
3. A shared update rule can pull all views toward the same target field under their own projection boundary.
4. The correct metric is target-projection error, not raw cross-media descriptor equality.
5. Modular shrinking can be measured as a path statistic.
```

## What This Does Not Prove

```text
1. It does not prove real media generation quality.
2. It does not train VDiT.
3. It does not yet implement real encoders or decoders.
4. It does not solve text rendering beyond a toy quantized symbolic projection.
5. It uses supervised target projections; real sampling still needs learned generation.
```

## Files

```text
run_minimal_prototype.py
run_core_validation.py       validates the simplified four-interface core
vod_minimal/chladni.py       synthetic Chladni fields and boundaries
vod_minimal/projections.py   toy image/video/audio/text projections
vod_minimal/metrics.py       descriptors, target error, modular shrinking
vod_minimal/core.py          simplified core: build_projection_batch /
                             shared_update_rollout / projection_loss /
                             evaluate_projection_error
vod_minimal/model.py         shared space-dependent field updater
vod_minimal/torch_model.py   trainable shared PyTorch updater
vod_minimal/vdit.py          Tiny VDiT skeleton
vod_minimal/experiment.py    dataset, grid search, evaluation
train_torch_prototype.py     trains the PyTorch updater
train_vdit_prototype.py      trains Tiny VDiT
```

## Simplified Core (vod_minimal/core.py)

`core.py` is the cleaned-up four-interface restatement of the math after
`docs/vod_math_simplification.md`:

```text
build_projection_batch(rng, batch_size, ...) -> ProjectionBatch
shared_update_rollout(update_fn, noisy, target, medium, *, steps, ...) -> view
projection_loss(update_fn, batch, *, steps, device, ...) -> torch.Tensor
evaluate_projection_error(rollout_fn, batch) -> {mean_before, mean_after, ...}
```

The four interfaces are backend-agnostic with respect to the updater:
the analytic `MinimalVOD`, the trainable `SharedPointUpdater`, and the
`TinyVDiT` skeleton can all be plugged in as the `update_fn` callable. No
new dependencies — only `numpy` and `torch`.

Reproducible smoke check using the analytic NumPy step:

```powershell
py -3.13 D:\VOD\prototype\run_core_validation.py
```

This sanity-checks all four interfaces end-to-end with fixed default
hyperparameters (no grid search, no training). Expected behaviour: a
clear drop from `mean_before` to `mean_after`, and `success_rate` well
above 0.5. Reproducing the best historical numbers in
`docs/vod_minimal_prototype_result.txt` still requires
`run_minimal_prototype.py` (which performs grid search).

## Checkpoint Schema

Both trainable prototype scripts save the same metadata schema through:

```text
vod_minimal/schema.py
```

Common fields:

```text
schema_version = vod-checkpoint-schema-v1
core_contract_version = vod-minimal-core-v1
model_type
state_dict
train_args
train_metrics
test_metrics
```

`TinyVDiT` additionally saves:

```text
config
best_epoch
```

## OPU Controller

OPU is integrated only as a controller adapter, not as part of the generation
core:

```text
vod_minimal/opu_adapter.py
run_opu_controller.py
```

It reads checkpoint metrics and suggests runtime controls:

```text
quality low       -> increase steps and quality_strength
resource pressure -> reduce max_tokens
healthy run       -> relax max_tokens
friction/faults   -> reduce step_size
```

Run:

```powershell
py -3.13 D:\VOD\prototype\run_opu_controller.py --checkpoint D:\VOD\prototype\checkpoints\tiny_vdit.pt
```
