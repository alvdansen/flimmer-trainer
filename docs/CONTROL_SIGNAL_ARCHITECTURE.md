# Control Signal Architecture

> Last updated: 2026-02-22 (v2 — added image input taxonomy)
> Status: Design phase — not yet implemented

This document defines Flimmer's core architectural innovation: treating every training input as a registered control signal with a uniform interface.

## The Core Insight

In video diffusion models, there is a fundamental distinction between:
- **Targets**: What the model learns to PRODUCE (denoised video)
- **Controls**: What the model learns to OBEY (text captions, reference images, depth maps, etc.)

Existing trainers blur this distinction. Flimmer makes it explicit and architectural.

## Image Inputs: Not Just "Reference Images"

A common simplification in training tools is treating image inputs as a monolithic category — "the reference image." In practice, users bring many kinds of images into training, each with a fundamentally different ROLE:

### I2V Reference Image
- **What it is**: A frame extracted from or closely matching the video clip (usually first frame)
- **What it tells the model**: "Generate video that starts from this exact visual state"
- **How it's used**: VAE-encoded and channel-concatenated with noisy latents
- **Typical source**: Extracted from the training clip itself (Phase 3 tooling)

### Subject Reference
- **What it is**: A character sheet, product photo, face crop, or other depiction of a specific subject
- **What it tells the model**: "This is what the subject looks like — learn its identity"
- **How it's used**: Depends on training approach — IP adapter embedding, textual inversion, or subject LoRA conditioning
- **Typical source**: User-provided (drawn, photographed, or composited)

### Style Reference
- **What it is**: A mood board image, film still, painting, or aesthetic sample
- **What it tells the model**: "Match this visual aesthetic, color palette, or artistic style"
- **How it's used**: Style embedding, CLIP-space conditioning, or style LoRA target
- **Typical source**: User-curated from reference material

### Structural Guide
- **What it is**: A depth map, edge map, pose skeleton, or segmentation mask
- **What it tells the model**: "Follow this spatial structure"
- **How it's used**: Channel concatenation (Flux-tools style) or context blocks (VACE style)
- **Typical source**: Computed from video clips using estimation models (MiDaS for depth, Canny for edges, etc.) or rendered from 3D scenes

### Brand / Asset Image
- **What it is**: A logo, UI element, graphic overlay, or other specific visual asset
- **What it tells the model**: "Incorporate this element into the output"
- **How it's used**: Varies — may be composited into training data or used as a conditioning signal
- **Typical source**: Client-provided assets

### Why This Taxonomy Matters
The organizational structure itself is pedagogical. When a user sets up a Flimmer dataset and must classify their images by role, they are forced to think explicitly about what each piece of training data MEANS. This prevents the common failure mode of dumping all images into one folder and hoping the trainer figures it out.

Flimmer's data organization (Phase 4) makes image roles explicit in the directory structure and metadata:
```
controls/
├── reference/       ← I2V first-frame references (auto-extracted or user-provided)
├── subject/         ← character/object identity references
├── style/           ← aesthetic references
├── structural/      ← depth, edge, pose maps
│   ├── depth/
│   ├── edge/
│   └── pose/
└── assets/          ← brand elements, logos, etc.
```

Each subdirectory has its own validation rules, encoding pathway, and injection method. The config surface lets users specify which roles are active and how they're weighted.

## How Control Signals Work in Wan Models

### Wan 2.1/2.2 T2V (Text-to-Video)
Single control signal:
- **Text caption** → T5 encoder → cross-attention into transformer blocks

### Wan 2.2 I2V (Image-to-Video)
Two control signals:
- **Text caption** → T5 encoder → cross-attention into transformer blocks
- **Reference image** → VAE encoder → latent concatenation with noisy video latents (extra channels in the channel dimension)

The reference image is NOT part of the target. It's a conditioning signal that tells the model "make video that looks like this." This is architecturally identical to how ControlNet-style signals work.

### VACE (Wan's Control Framework)
Multiple control signals through a separate pathway:
- **Context blocks**: Dedicated transformer blocks that process control signals (depth, edge, pose, etc.)
- **Context tokens**: Control signal representations that are injected into the main transformer via cross-attention
- Each control type occupies its own "slot" in the context block architecture
- This is why VACE generalizes well to new control types — the infrastructure exists

### Flux-tools Control LoRA Approach
- Control signals (depth, edge) are channel-concatenated with the input
- A LoRA teaches the model to interpret these extra channels as structural guidance
- spacepxl's depth control LoRA for Flux is the reference implementation
- This approach works because Flux already has I2V-style channel concatenation machinery

## Signal Types and Their Properties

Each control signal has these attributes:

| Attribute | Reference Image (I2V) | Subject Reference | Style Reference | Text Caption | Depth Map |
|---|---|---|---|---|---|
| **Name** | `reference_image` | `subject_image` | `style_image` | `caption` | `depth_map` |
| **Data type** | Image (PNG/JPG) | Image (PNG/JPG) | Image (PNG/JPG) | Text string | Image/Tensor |
| **Encoding** | VAE → latent | CLIP/IP-Adapter → embedding | CLIP → embedding | T5 → embedding | Passthrough/normalize |
| **Injection** | Channel concat | Cross-attention | Cross-attention/AdaLN | Cross-attention | Channel concat |
| **Dropout** | 0% (required for I2V) | 5-15% | 10-20% | 10-20% | 0-10% |
| **Validation** | Resolution ≥ target | Min resolution, subject visible | Min resolution | Non-empty string | Matches video dims |
| **Alignment** | Matches first video frame | Depicts same subject as video | Aesthetic match | Describes video content | Frame-by-frame spatial match |

## The Signal Registry Pattern

```python
class ControlSignal:
    """Base class for all control signals in Flimmer."""
    
    name: str                    # unique identifier
    required: bool               # must be present in dataset?
    dropout_rate: float          # probability of omitting during training
    
    def prepare(self, raw_data) -> PreparedData:
        """Load and preprocess raw data."""
        ...
    
    def validate(self, prepared_data, video_metadata) -> ValidationResult:
        """Check that this signal is valid for the given video."""
        ...
    
    def encode(self, prepared_data, model) -> Tensor:
        """Transform into model-ready representation."""
        ...
    
    def inject(self, encoded_data, model_inputs) -> ModelInputs:
        """Add this signal to the model's input dict."""
        ...
```

### Registration
```python
# In config or code:
signal_registry.register("caption", TextCaptionSignal(
    dropout_rate=0.15,
    encoder="t5",
))

signal_registry.register("reference_image", ReferenceImageSignal(
    dropout_rate=0.0,
    encoder="vae",
    injection="channel_concat",
))

signal_registry.register("depth_map", DepthMapSignal(
    dropout_rate=0.1,
    encoder="passthrough",  # already a tensor
    injection="channel_concat",
))
```

### Config Surface
```yaml
controls:
  caption:
    enabled: true
    dropout_rate: 0.15
    
  reference_image:
    enabled: true    # false for T2V, true for I2V
    dropout_rate: 0.0
    
  depth_map:
    enabled: false   # toggle on when needed
    source: "precomputed"  # or "generate_on_fly"
    dropout_rate: 0.1
```

## Signal Injection Methods

### Cross-Attention (Text, Audio Embeddings)
The signal is encoded into a sequence of tokens/embeddings. During the forward pass, the transformer's cross-attention layers attend to these embeddings alongside or in place of other cross-attention signals.

### Channel Concatenation (Reference Image, Depth, Edge, Pose)
The signal is encoded into a spatial tensor matching the video latent dimensions. It's concatenated along the channel dimension with the noisy video latents before entering the transformer. The model learns that the extra channels contain conditioning information.

### AdaLN Modulation (Timestep, Style Embeddings)
The signal modulates the layer normalization parameters. This is how timestep information typically enters DiT models, and how some audio conditioning works (MMAudio's sync module).

### Context Blocks (VACE Controls)
The signal is processed through dedicated context blocks that produce context tokens. These tokens are then injected into the main transformer. This is VACE-specific and provides the most structured control pathway.

## Why Extensibility Matters

The control signal architecture is designed so that adding a new control type requires:
1. Writing a new signal class (prepare/validate/encode/inject)
2. Registering it in the config schema
3. Adding it to the defaults YAML

It does NOT require:
- Modifying the training loop
- Changing the dataset loader (beyond adding a new data reader)
- Touching the orchestrator
- Rebuilding existing functionality

This is critical for future-proofing. When new control types emerge (audio, motion vectors, optical flow, semantic masks), Flimmer should absorb them without architectural surgery.

## Audio Control Signal (Designed, Not Implemented)

Audio maps into the control signal framework:
```
controls/
├── audio_track
│   ├── foley_hints     ← "sounds of footsteps, door closing"
│   ├── music_track     ← rhythm/energy signal
│   ├── speech_dialogue ← lip sync driving signal
│   └── audio_embedding ← encoded features (CLAP/Synchformer/mel spectrogram)
```

Preparation → Validation → Encoding pipeline:
- **Prepare**: Extract audio features (CLAP embeddings for semantics, Synchformer features for temporal sync, mel spectrograms for raw signal)
- **Validate**: Check sample rate, duration matches video, detect silence/corruption
- **Encode**: Transform into model-expected representation (method depends on target model)

Implementation trigger: Wan 2.5 open release with native audio, OR MMAudio pipeline integration.

## Interaction with Differential MoE

Control signals may affect the high-noise and low-noise experts differently:
- **Text captions**: May be more important for the high-noise expert (guides global composition)
- **Reference images**: May be more important for the low-noise expert (guides fine detail preservation)
- **Depth maps**: Likely important for both (structural guidance at all noise levels)

This is an open research question. Flimmer's architecture should support per-expert control signal weighting:
```yaml
controls:
  caption:
    weight:
      high_noise: 1.0
      low_noise: 0.8
  reference_image:
    weight:
      high_noise: 0.8
      low_noise: 1.0
```

Whether this actually improves results is TBD — but the architecture should support it from the start.
