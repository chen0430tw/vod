# VOD Related Research Notes

Date: 2026-04-26

This note compares VOD's entropy-texture idea against current unified multimodal generation research.

## Short Answer

There are related attempts to unify media generation, but most of them unify at one of these levels:

1. shared discrete tokens
2. shared transformer backbone
3. shared latent embedding space
4. synchronized multi-stream diffusion
5. any-to-any instruction interface

VOD's current proposal is different because it tries to unify media at the level of **compressive information density**, represented as entropy texture and entrons.

So the idea is not isolated, but the proposed primitive is still distinct.

## Relevant Work

### Unified-IO 2

Unified-IO 2 is an autoregressive multimodal model for image, text, audio, and action. It tokenizes inputs and outputs into a shared semantic space and processes them with a single encoder-decoder transformer.

Useful for VOD:

- proves that broad modality unification is a serious research direction
- useful training recipe: multimodal mixture of denoisers
- useful benchmark strategy: train one model across many task families

Limit for VOD:

- still token-centric
- more instruction/task unification than physical information unification
- audio/video generation is not the same as a single entropy field generator

VOD optimization:

```text
Keep Unified-IO style task mixture, but replace shared token space with entropy-texture field supervision.
```

### ImageBind

ImageBind learns one embedding space across image, text, audio, depth, thermal, and IMU, using image-paired data as a binding anchor.

Useful for VOD:

- strong evidence that modalities can be bound through shared latent geometry
- image/video can serve as a natural anchor for other modalities
- useful for contrastive alignment loss

Limit for VOD:

- primarily embedding/retrieval/alignment, not a full native generator
- does not define a generative substrate like entropy texture

VOD optimization:

```text
Use ImageBind-style contrastive binding as a pretraining stage for Phi_m encoders.
Image/video may be the first anchor modality for entropy texture alignment.
```

### Chameleon

Chameleon is an early-fusion mixed-modal foundation model that can process and generate interleaved image and text using a unified token-based architecture.

Useful for VOD:

- early fusion is important; late fusion tends to become a pipeline
- arbitrary interleaving is useful for media documents, storyboards, KV, and generated design docs

Limit for VOD:

- image/text only in the core formulation
- relies on discrete image tokenization

VOD optimization:

```text
Adopt early fusion at the entron-field level, not at discrete token level.
```

### Transfusion

Transfusion combines next-token prediction for text with diffusion for images in one transformer over mixed-modality sequences. It is important because it avoids forcing every modality into the same discrete-token loss.

Useful for VOD:

- validates mixed objective training
- text can keep language-model objective while visual/audio fields use diffusion/flow objective
- modality-specific encoding/decoding layers can coexist with one shared transformer

Limit for VOD:

- demonstrated mainly for text and image
- still arranged as mixed-modality sequences

VOD optimization:

```text
Use hybrid losses:
- flow matching for entropy texture
- semantic prediction for symbolic/discourse structure
- reconstruction for modality decoders
```

### UniDisc

UniDisc explores unified multimodal discrete diffusion for text and image. It performs joint denoising/inpainting over masked discrete tokens.

Useful for VOD:

- diffusion is useful beyond pixels
- joint inpainting across modalities is a good target task
- confidence-based iterative generation can become a VOD sampling policy

Limit for VOD:

- discrete diffusion over tokens
- text/image scope, not full image/video/music/text

VOD optimization:

```text
Generalize mask denoising from token mask to entropy-manifold mask M over Omega.
```

### 3MDiT

3MDiT is a unified tri-modal diffusion transformer for synchronized audio-video generation from text. It models video, audio, and text as jointly evolving streams and uses tri-modal fusion blocks.

Useful for VOD:

- directly relevant to video + audio synchronization
- dynamic text conditioning is important: text should evolve as evidence from audio/video evolves
- tri-modal fusion block is a practical architecture donor

Limit for VOD:

- still stream-based: video stream, audio stream, text stream
- not a single substrate; more like synchronized streams

VOD optimization:

```text
Borrow dynamic text conditioning and tri-modal fusion, but rewrite streams as views over entropy texture.
```

### AudioGen-Omni

AudioGen-Omni is a unified MMDiT for video-synchronized audio, speech, and song generation. It uses joint training over video-text-audio data and frame-level representations for speech/song text.

Useful for VOD:

- very relevant to music/speech/song side
- frame-aligned text/audio representation is useful for lyrics and subtitles
- phase-aware position handling is useful for rhythm and temporal alignment

Limit for VOD:

- mainly audio generation conditioned by video/text
- does not generate all media as equal decoded views of one field

VOD optimization:

```text
Add rhythm/phase coordinates into Omega:
Omega_audio_video = semantic_time + beat_phase + visual_time + scene_layer
```

### JavisDiT / ProAV-DiT / Joint Audio-Video Diffusion

Recent synchronized audio-video diffusion models focus on generating audio and video together with better temporal alignment.

Useful for VOD:

- synchronized audio-video generation is an active frontier
- hierarchical spatio-temporal priors are useful
- audio/video structural mismatch must be handled explicitly

Limit for VOD:

- most are dual-stream or projected-latent designs
- they solve synchronization, not universal media substrate

VOD optimization:

```text
Introduce hierarchy in entropy texture:
E = {E_scene, E_motion, E_rhythm, E_surface, E_symbolic}
```

## Main Gap In Existing Work

Most existing research says:

```text
Different modalities can share tokens, latents, embeddings, or transformers.
```

VOD says:

```text
Different modalities are measurements/views of compressive information structure.
```

This is the key difference.

## Recommended VOD Model Revision

The first VOD math draft should be upgraded in four ways.

### 1. Separate Entropy Texture From Raw Modality Coordinates

Do not define Omega as image/video/audio/text coordinates directly. Define Omega as an abstract event-structure manifold.

```text
Omega = semantic event manifold
omega = (entity, relation, time, phase, salience, layer)
```

Modality coordinates are projections:

```text
pi_image  : Omega -> image plane
pi_video  : Omega -> image plane x time
pi_audio  : Omega -> acoustic time x phase/frequency
pi_text   : Omega -> discourse order
pi_layout : Omega -> design plane x hierarchy
```

### 2. Define Entron As Compression Differential

Current entron definition is close, but should become differential:

```text
e(omega) = Delta I(omega) = I_raw(omega) - I_context(omega)
```

where `I` can be estimated by code length, negative log likelihood, or reconstruction residual.

This makes entron compatible with:

- information theory
- diffusion denoising
- semantic prediction
- lossy compression
- multimodal alignment

### 3. Use Multi-Resolution Entropy Texture

One entropy field may be too flat. Use layered entropy texture:

```text
E = {E_global, E_event, E_object, E_motion, E_rhythm, E_surface, E_symbolic}
```

This avoids forcing music beat, video motion, and image texture into the same scale.

### 4. Train With Cross-View Reconstruction

Instead of only decoding each modality, require generated views to re-encode back to the same entropy field.

```text
Phi_m(D_m(E)) ~= E
```

This is the strongest practical bridge between VOD theory and existing methods like ImageBind, Unified-IO, and 3MDiT.

## Practical First Prototype

The smallest credible VOD prototype should not try full all-media generation immediately.

Start with:

```text
image + short video + audio rhythm
```

Target task:

```text
prompt -> 4 second animated visual + synchronized simple music/rhythm bed
```

Why:

- includes image, video, and music
- avoids full language/song complexity at first
- lets entropy texture prove cross-modal time/phase alignment
- can reuse Open-Sora/YuE/AnyText2 ideas without becoming a router

## Sources

- Unified-IO 2: https://unified-io-2.allenai.org/
- ImageBind: https://huggingface.co/papers/2305.05665
- Chameleon: https://huggingface.co/papers/2405.09818
- Transfusion: https://www.isi.edu/results/publications/15537/transfusion-predict-the-next-token-and-diffuse-images-with-one-multi-modal-model/
- UniDisc: https://unidisc.github.io/
- 3MDiT: https://huggingface.co/papers/2511.21780
- AudioGen-Omni: https://huggingface.co/papers/2508.00733
- JavisDiT: https://openreview.net/forum?id=y7HV7KT3Bd
