# NativeVOD state_dict Key Contract

> **NEVER rename a key listed below without going through the alias-hook
> migration procedure (§3).** Once a checkpoint is published with a
> given key set, renaming is a breaking change — old checkpoints become
> unloadable, and downstream code that names parameters explicitly
> (LR groups, weight decay exclusion, FSDP wrap policies) will silently
> mismatch.

This file is the single source of truth for state_dict key naming.
A unit test (`tests/test_state_dict_contract.py`) enforces this list
against the runtime model.

---

## 1. Frozen keys (NativeVOD top-level)

NativeVOD with `backbone="unet"`, `time_dim=0`, all media disabled
or enabled — the same 10 top-level entries always exist.

| Key prefix | Module | Notes |
|---|---|---|
| `enc_image.weight`, `enc_image.bias` | `nn.Linear(1, channels)` | Always present |
| `enc_video.weight`, `enc_video.bias` | `nn.Linear(1, channels)` | Always present |
| `enc_audio.weight`, `enc_audio.bias` | `nn.Linear(1, channels)` | Present even when `enable_audio=False` (so checkpoint shape is config-independent) |
| `enc_text.weight`, `enc_text.bias` | `nn.Linear(1, channels)` | Present even when `enable_text=False` |
| `dec_image.weight`, `dec_image.bias` | `nn.Linear(channels, 1)` | Always present |
| `dec_video.weight`, `dec_video.bias` | `nn.Linear(channels, 1)` | Always present |
| `dec_audio.weight`, `dec_audio.bias` | `nn.Linear(channels, 1)` | Always present |
| `dec_text.weight`, `dec_text.bias` | `nn.Linear(channels, 1)` | Always present |
| `step_logit` | `nn.Parameter` (scalar) | Sigmoid-activated denoise step size |
| `denoiser.*` | `UNetDenoiser` or `PointwiseMLPDenoiser` | Subtree namespaced under `denoiser.` |

## 2. Backbone subtree (`denoiser.*`)

### 2.1 `backbone="unet"` → UNetDenoiser

| Key prefix | Module |
|---|---|
| `denoiser.down1.conv1.{weight,bias}` | `_ConvBlock.conv1` (`nn.Conv2d 3×3`) |
| `denoiser.down1.conv2.{weight,bias}` | `_ConvBlock.conv2` |
| `denoiser.down2.conv1.{weight,bias}` | |
| `denoiser.down2.conv2.{weight,bias}` | |
| `denoiser.bot.conv1.{weight,bias}` | bottleneck conv block |
| `denoiser.bot.conv2.{weight,bias}` | |
| `denoiser.temporal_conv.{weight,bias}` | `nn.Conv1d` along T at bottleneck |
| `denoiser.up2.conv1.{weight,bias}` | |
| `denoiser.up2.conv2.{weight,bias}` | |
| `denoiser.up1.conv1.{weight,bias}` | |
| `denoiser.up1.conv2.{weight,bias}` | |
| `denoiser.out_conv.{weight,bias}` | `nn.Conv2d 1×1` to channels |

`pool1`, `pool2`, `act`, `temporal_act` have no parameters.

### 2.2 `backbone="mlp"` → PointwiseMLPDenoiser

| Key prefix | Module |
|---|---|
| `denoiser.net.0.{weight,bias}` | `nn.Linear` |
| `denoiser.net.2.{weight,bias}` | `nn.Linear` |
| `denoiser.net.4.{weight,bias}` | `nn.Linear` |

(SiLU at indices 1, 3 has no parameters.)

## 3. Migration procedure for renames

If a parameter name MUST change (e.g. refactoring `denoiser` →
`unet_denoiser`):

1. Update this file with the new name and a `→ legacy: <old name>`
   annotation.
2. Override `_load_from_state_dict` on the parent module to remap
   `prefix + old_name → prefix + new_name` keys when loading old
   checkpoints. Example skeleton:

   ```python
   def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
       for old, new in self._key_aliases.items():
           old_key = prefix + old
           new_key = prefix + new
           if old_key in state_dict and new_key not in state_dict:
               state_dict[new_key] = state_dict.pop(old_key)
       return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
   ```

3. Bump `ARCHITECTURE_VERSION` minor (e.g. `1.0` → `1.1`) only if the
   change is additive; bump major (`2.0`) if old checkpoints cannot be
   loaded with the alias.
4. Update `tests/test_state_dict_contract.py` to include both old and
   new key sets.

## 4. Adding new modules

When adding a new submodule (e.g. a text encoder for conditioning):

- Pick a name that won't collide with existing prefixes.
- Add to this file in §1 or §2 with the new key set.
- Bump `ARCHITECTURE_VERSION` minor.
- The unit test will fail until this file is updated — that is intentional.

## 5. What is NOT in the state_dict

These attributes hold no learnable parameters and never appear in
`state_dict()`:

- `config: NativeVODConfig` — a `@dataclass(frozen=True)`, persisted to
  `config.json` separately by `save_pretrained`.
- Smoothing tap operators (`_smooth_taps`, `_position_grid`) — pure
  functions on input tensors.
- AvgPool / SiLU layers in UNetDenoiser.
