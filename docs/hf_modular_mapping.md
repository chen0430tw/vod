# VOD ↔ HF Modular Diffusers Mapping (Future Phase C Contract)

> Status: **draft contract, no runtime code**.
> This file freezes the input/output shape of the future
> `ModularPipelineBlocks` wrapper so that when Phase C arrives the
> wrap is mechanical (~50 LOC), not a redesign.
>
> Companion files:
> - `vod_minimal/native.py` — current substrate (NativeVOD)
> - `prototype/model_index.json` — classic Diffusers manifest draft
> - `prototype/modular_model_index.json` — Modular Diffusers manifest draft
> - `prototype/STATE_DICT_KEYS.md` — locked parameter naming

---

## 1. Why Modular, not classic Pipeline

Modular Diffusers (HF 2026/3) decomposes a pipeline into independent
blocks with explicit `inputs / intermediate_outputs / expected_components`.
This matches VOD's type B substrate-shared design **better than the
classic `text_encoder + vae + unet + scheduler` pipeline**, because:

- VOD has no separate text encoder / VAE / UNet — they collapse into
  one shared substrate `U(t, y, x, c)`.
- Modular blocks accept arbitrary tensor shapes, so 5-D `(B, T, H, W, C)`
  latent does not need to be flattened to 4-D `(B, C, H, W)`.
- Multi-modality output (image + video + audio + text from a single
  substrate sample) maps naturally to `intermediate_outputs` dict.

## 2. Block decomposition

The future wrapper exposes **two blocks**:

### 2.1 `VODSubstrateSampleBlock`

Pure sampling: random noise → denoised latent U.

```python
class VODSubstrateSampleBlock(ModularPipelineBlocks):
    @property
    def expected_components(self):
        return [
            ComponentSpec("vod", NativeVOD,
                          pretrained_model_name_or_path="<repo>/vod"),
            ComponentSpec("scheduler", NoiseSchedule,
                          pretrained_model_name_or_path="<repo>/scheduler"),
        ]

    @property
    def inputs(self):
        return [
            InputParam("latent_shape",   required=True,
                       description="(B, T, H, W, C) — substrate latent shape"),
            InputParam("num_steps",      required=False, default=50,
                       description="DDIM sampling steps"),
            InputParam("generator",      required=False,
                       description="torch.Generator for reproducibility"),
            InputParam("prediction",     required=False, default="x_0",
                       description="x_0 or epsilon — must match training"),
        ]

    @property
    def intermediate_outputs(self):
        return [
            OutputParam("latent", type_hint=torch.Tensor,
                        description="Denoised substrate U (B, T, H, W, C)"),
        ]

    @torch.no_grad()
    def __call__(self, components, state):
        block_state = self.get_block_state(state)
        block_state.latent = ddim_sample(
            components.vod, block_state.latent_shape, components.scheduler,
            num_steps=block_state.num_steps,
            generator=getattr(block_state, "generator", None),
            prediction=block_state.prediction,
        )
        self.set_block_state(state, block_state)
        return components, state
```

### 2.2 `VODDecodeBlock`

Substrate latent → media views (per-medium dispatch via `requested`).

```python
class VODDecodeBlock(ModularPipelineBlocks):
    @property
    def expected_components(self):
        return [ComponentSpec("vod", NativeVOD, ...)]

    @property
    def inputs(self):
        return [
            InputParam("latent",    required=True,
                       description="Output of VODSubstrateSampleBlock"),
            InputParam("requested", required=False, default=None,
                       description="Tuple of media to decode; None = all active"),
        ]

    @property
    def intermediate_outputs(self):
        # Each medium is a separate output param so downstream blocks
        # can pick what they need (image-only vs. video-only consumers).
        return [
            OutputParam("image", type_hint=torch.Tensor, description="Optional"),
            OutputParam("video", type_hint=torch.Tensor, description="Optional"),
            OutputParam("audio", type_hint=torch.Tensor, description="Optional"),
            OutputParam("text",  type_hint=torch.Tensor, description="Optional"),
        ]

    def __call__(self, components, state):
        block_state = self.get_block_state(state)
        decoded = components.vod.decode(
            block_state.latent, requested=block_state.requested,
        )
        for k, v in decoded.items():
            setattr(block_state, k, v)
        self.set_block_state(state, block_state)
        return components, state
```

## 3. Pipeline composition examples

### 3.1 Image-only generation

```python
blocks = SequentialPipelineBlocks([
    VODSubstrateSampleBlock(),
    VODDecodeBlock(),
])
pipe = blocks.init_pipeline("<repo>/vod-substrate-v1")
pipe.load_components(torch_dtype=torch.bfloat16)
out = pipe(
    latent_shape=(1, 8, 32, 32, 4),
    requested=("image",),
).image
```

### 3.2 Video + image

```python
out = pipe(latent_shape=(1, 8, 32, 32, 4), requested=("image", "video"))
out.image, out.video
```

## 4. What the contract guarantees

- Substrate sampling cost is **media-agnostic**. `requested` only
  saves the per-medium 1×1 head. Substrate-level cost reduction is
  Phase ≥ C.2 (T-shape randomized training, see §5).
- `latent` shape is always 5-D `(B, T, H, W, C)`. Wrappers MUST NOT
  reshape to 4-D — this preserves type B claim.
- All `expected_components` load via VOD's existing `from_pretrained`
  (Phase A A3). No new serialization layer is introduced in Phase C.

## 5. Open items — defer to Phase C

| Item | Why deferred |
|---|---|
| `VODConditioningBlock` (text / class / image conditioning) | VOD is unconditional today; conditioning hook intentionally not pre-baked (YAGNI) |
| `_no_split_modules = ["UNetDenoiser"]` (group offload) | Only meaningful when VOD scales past current 524K toy size |
| `_skip_layerwise_casting_patterns = (".*enc_.*", ".*dec_.*")` (fp8 fidelity) | Same — needs scale to matter; class-level placeholder fine for Phase A |
| T-shape randomized training (substrate cost reduction) | Requires retraining; wait for unconditional sample fidelity to pass first |

## 6. Out of scope (will NOT do, now or in Phase C)

- ❌ Wrap as classic `StableDiffusionPipeline` lookalike
- ❌ Reshape 5-D substrate to 4-D image latent
- ❌ Split substrate into separate `image_decoder` / `video_decoder`
  components (would break type B substrate sharing)
- ❌ Mock a `text_encoder` / `tokenizer` field for unconditional model
  to look "Pipeline-shaped"
