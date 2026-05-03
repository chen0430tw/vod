# VOD Backbone Selection

Date: 2026-04-26

## Short Decision

The best source-code skeleton for a future VOD/VDiT backbone is:

```text
D:\VOD\external\Open-Sora\opensora\models\mmdit\model.py
```

Reason:

```text
Open-Sora MMDiT already treats generation as a field-like sequence with time/video-aware positional IDs, timestep conditioning, context conditioning, and transformer blocks.
```

It is closer to VOD than SDXL/NovelAI because VOD must generate video and eventually audio/rhythm-aligned fields.

## Candidate 1: Open-Sora MMDiT

Relevant file:

```text
D:\VOD\external\Open-Sora\opensora\models\mmdit\model.py
```

Useful structure:

```text
MMDiTConfig
MMDiTModel
img_in
time_in
vector_in
txt_in
pe_embedder
double_blocks
single_blocks
final_layer
```

VOD mapping:

```text
img              -> entropy field samples / media projections
img_ids          -> boundary + phase + frequency coordinates
txt              -> condition/context field
timesteps        -> generation tau
y_vec            -> global condition vector
pe_embedder      -> phase/frequency/boundary embedding
DoubleStreamBlock -> condition-field resonance operator
SingleStreamBlock -> unified field interaction operator
final_layer      -> field velocity / update head
```

VDiT replacement target:

```text
MMDiTModel.forward(...)
  becomes
VDiT.forward(field, boundary_ids, condition_tokens, tau, phase_frequency)
```

## Candidate 2: SDXL DiffusionEngine

Relevant file:

```text
D:\VOD\external\generative-models\sgm\models\diffusion.py
```

Useful structure:

```text
DiffusionEngine
first_stage encode/decode
conditioner
denoiser
sampler
loss_fn
EMA
training_step
```

VOD mapping:

```text
DiffusionEngine -> VOD training wrapper
conditioner     -> condition/boundary/phase-frequency builder
denoiser        -> Chladni field denoiser
sampler         -> VOD generation loop
first_stage     -> modality encoder/decoder adapter
```

This is a good training wrapper reference, but it is image/latent-diffusion oriented. It should not be the VOD core.

## Candidate 3: NovelAI / LDM DDPM

Relevant file:

```text
D:\NovelAI\ldm\models\diffusion\ddpm.py
```

Useful structure:

```text
DDPM schedule
q_sample
p_sample
p_sample_loop
conditioning keys
EMA
DDIM sampler compatibility
```

VOD mapping:

```text
noise schedule -> VOD tau schedule
q_sample       -> field corruption
p_sample_loop  -> iterative Chladni field refinement
```

This is useful for a lightweight baseline and sampler logic, but it is older image-first code. Use it as a reference, not the main skeleton.

## Decision

Use this order:

```text
1. Current NumPy/PyTorch minimal prototype validates VOD interface.
2. Replace toy updater with a small trainable PyTorch field updater.
3. Build VDiT skeleton by adapting Open-Sora MMDiT concepts.
4. Borrow SDXL DiffusionEngine only as training/sampling wrapper reference.
5. Borrow NovelAI/DDPM only for simple noise schedule and baseline sampler.
```

## First VDiT File To Create Later

```text
D:\VOD\prototype\vod_minimal\vdit.py
```

Initial VDiT should not import Open-Sora directly. It should copy the architectural pattern:

```text
field input projection
condition input projection
boundary/phase/frequency positional embedding
double-stream blocks
single-stream blocks
field velocity output head
```

## Implementation Status

Tiny VDiT skeleton has been implemented:

```text
D:\VOD\prototype\vod_minimal\vdit.py
D:\VOD\prototype\train_vdit_prototype.py
```

It is not a full Open-Sora port. It is a dependency-light VOD skeleton that copies the useful pattern:

```text
field feature projection
media embedding
phase/position embedding
Transformer blocks
field update head
```

Current validation:

```text
D:\VOD\docs\vod_vdit_prototype_result.txt
```

Result:

```text
Test mean_before       9.427957
Test mean_after        1.601284
Test mean_improvement  7.826673
Test success_rate      1.000000
```

This avoids dependency sprawl while preserving the useful MMDiT design.
