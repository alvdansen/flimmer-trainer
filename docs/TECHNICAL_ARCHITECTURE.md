# Technical Architecture — How Video LoRA Training Works

> Last updated: 2026-03-03
> Status: Beta — all major modules built, Wan 2.2 MoE hyperparameters are experimental

This document explains the mechanics of video LoRA training from first principles, common limitations in the current finetuning ecosystem, and how Flimmer's architecture approaches things differently.

## How Existing Trainers Are Structured

Every training framework follows the same three-layer pattern:

### Layer 1: Config Surface (what the user touches)
A YAML/TOML file with a curated subset of parameters. The author makes editorial choices about what to surface (learning rate, rank, dataset path) vs what gets sensible defaults (optimizer type, scheduler, precision) vs what's completely hidden (internal tensor operations, memory management).

### Layer 2: Orchestrator (the manager)
A Python script that reads config, validates it, builds all PyTorch objects, and runs the training loop. Translates "what you want" into actual operations.

### Layer 3: Utilities (the workers)
Isolated modules for specific jobs — dataset loading, noise scheduling, LoRA injection, checkpoint saving. The orchestrator calls them; they don't know about each other.

### How They Differ
- **Kohya (sd-scripts)**: TOML configs + argparse. `train_util.py` defines hundreds of arguments with defaults. The TOML only overrides what you change. Massive codebase supporting every model and training type.
- **Ostris (ai-toolkit)**: YAML configs with factory pattern. `type` fields map to Python classes. More modular, object-oriented.

## The Training Loop (Plain English)

```python
for epoch in range(num_epochs):
    for batch of video clips from dataset:
        1. Encode video through VAE -> latent representation
        2. Sample random noise level (timestep)
        3. Add noise to latents at that level
        4. Prepare conditioning (text embeddings, first frame if I2V)
        5. Ask model to predict the noise (or velocity, for flow matching)
        6. Compare prediction to actual noise -> calculate loss
        7. Backpropagate through LoRA weights only (base model frozen)
        8. Update LoRA weights based on gradients
        9. Log loss for monitoring

    if it's time to save:
        save checkpoint
```

## Four Hard Problems in Video Training

### 1. Noise Scheduling
Every diffusion model has a specific noise schedule that defines how much noise is added at each timestep. Getting this wrong means the model learns to denoise at the wrong levels.

**Wan 2.2 uses flow matching**, not standard DDPM noise scheduling. Flow matching defines a straight-line path from data to noise (or vice versa), parameterized by a continuous time variable t in [0, 1]. The model predicts the velocity field that transports samples along this path.

For Wan 2.2's MoE architecture, this has a critical implication: the high-noise expert handles timesteps where the signal-to-noise ratio is low (early denoising, t near 1), and the low-noise expert handles timesteps where SNR is high (late denoising, t near 0). The boundary is at approximately SNR = 875/1000 of the schedule.

**What trainers do**: They look up the model's scheduler configuration (usually stored in the model's config files) and use matching noise sampling during training. Kohya reads this from the model's `scheduler_config.json` or hardcodes it per model type.

### 2. LoRA Injection
LoRA works by adding small trainable matrices to specific layers of the frozen base model. The choice of which layers to target matters.

**Standard targets**: The query (Q), key (K), and value (V) projection matrices in attention layers, and sometimes the MLP layers. In Wan's DiT architecture, these are within the transformer blocks.

**For Wan 2.2 MoE**: Each expert is a separate transformer. A LoRA can target one expert, the other, or both with different configurations.

**Weight format**: LoRA weights must be saved in a format compatible with inference tools. For Wan, this means safetensors files with keys that match what ComfyUI/diffusers expect. Getting the key naming wrong means the LoRA loads but does nothing (or crashes).

### 3. Memory Management
Video models are enormous. Wan 2.2 I2V is 14B active parameters. Training requires:
- The frozen base model weights in VRAM
- The LoRA adapter weights (trainable)
- Optimizer states (2x the LoRA weights for AdamW)
- Activation cache for backpropagation
- The video latents themselves

**Key techniques**:
- **Gradient checkpointing**: Trade compute for memory — don't store all activations, recompute them during backward pass. Saves ~60% VRAM at ~20% speed cost.
- **Mixed precision (bf16)**: Keep weights in bfloat16 instead of float32. Halves memory for model weights.
- **Model offloading**: Move the text encoder to CPU after encoding captions. Move the VAE to CPU after encoding video latents.
- **Gradient accumulation**: Process one video at a time but accumulate gradients over N steps before updating. Simulates larger batch sizes without the memory cost.

### 4. Video-Specific Considerations
**Temporal compression**: Wan-VAE is a 3D causal VAE with ~4x temporal compression. 81 frames -> ~21 temporal tokens. This means clip preparation directly affects what the model sees — frame rate, clip length, and temporal alignment all matter.

**Frame count constraints**: Wan requires frame counts satisfying `(frames - 1) % 4 == 0`. Valid counts: 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77, 81.

**Resolution buckets**: Like image training, video training benefits from bucketing — grouping clips by resolution to minimize padding waste. But video adds the temporal dimension, so buckets are 3D (width x height x frames).

### Conditioning treated as an afterthought
In most training setups, captions are treated as a metadata file — something you add because the training script expects a `.txt` next to your `.mp4`. But captions are actually a conditioning signal that teaches the model what to associate with the visual content. Thinking of them as an input signal with its own quality requirements (rather than just a sidecar file) changes how you approach dataset preparation and how much weight you give to caption quality, structure, and consistency.

### Manual setup for MoE training
Wan 2.2's dual-expert architecture can be trained with different hyperparameters per expert — anyone can set this up manually by running separate training sessions, managing checkpoint handoffs, and tracking which settings go with which expert. It works, but there's no integrated workflow that handles the unified warmup, forking, and per-expert specialization in a single training run.

## Flimmer's Approach

### MoE Phased Training (Experimental)

For Wan 2.2's dual-expert architecture, Flimmer provides a phased training workflow: start with a unified LoRA across both experts, optionally fork into per-expert copies, then specialize each independently — all configured in one YAML file and executed as a single training run.

```yaml
# Standard unified training (works like any other trainer)
moe:
  fork_enabled: false    # one LoRA on both experts, no forking

# Phased training (experimental)
moe:
  fork_enabled: true
  high_noise:
    learning_rate: 1e-4
    max_epochs: 30
  low_noise:
    learning_rate: 8e-5
    max_epochs: 50
```

**What is this?** Wan 2.2 has two transformer experts — a high-noise expert (handles early denoising: global composition, motion planning) and a low-noise expert (handles late denoising: fine detail, texture). In standard training, both get the same LoRA with the same hyperparameters. Phased training lets you warm up both together (unified phase), then fork the LoRA into two copies and train each expert independently with its own learning rate, epoch count, and other settings.

**This is theoretical.** The idea that each expert benefits from different training intensities is plausible given their architectural specialization, but it hasn't been fully validated yet. We're actively running experiments to compare results. The unified-only mode (`fork_enabled: false`) works identically to standard training in other tools.

**Override everything.** Every MoE hyperparameter can be overridden in YAML — fork on/off, boundary ratio, per-expert learning rate, epochs, dropout, batch size, caption dropout, weight decay, optimizer type, scheduler type. See [Training Config Walkthrough](TRAINING_CONFIG_WALKTHROUGH.md) for the full list.

> **All Wan 2.2 MoE hyperparameters are experimental.** This is a beta release — defaults will evolve as we validate results.

### Video-Native Dataset Pipeline
Before anything touches the training loop:
1. **Temporal validation**: Verify no scene cuts within clips, consistent frame rate, appropriate duration for the model's temporal window
2. **Resolution bucketing**: 3D buckets (W x H x frames) with intelligent grouping
3. **Latent pre-encoding**: Pre-encode videos through VAE and captions through T5, cache to disk to avoid re-encoding every epoch

### Data Preparation as Standalone Tooling
The biggest practical gap in video LoRA training isn't the training loop — it's everything BEFORE the training loop. Existing trainers assume you show up with clean clips, good captions, and properly formatted data. In reality, the data journey looks like:

1. Raw video (footage, downloads, client assets)
2. Scene detection and cutting into clean clips
3. Caption generation
4. First frame extraction (for I2V)
5. Validation and organization
6. Latent pre-encoding
7. THEN training

Steps 1-5 are where most of the time and failure happens. Flimmer's data tools solve these steps as standalone utilities that produce standard output formats, usable with any trainer. This makes them valuable even if you never use Flimmer's training loop.

### Model-Agnostic Training Infrastructure
The training loop doesn't know about Wan or any specific model. Instead, Flimmer defines interfaces that every model backend must implement:

- **Noise schedule**: How noise is added/removed. Wan uses flow matching. Other models may use different approaches. The interface is the same; the implementation differs.
- **Layer mapping**: Which transformer layers get LoRA adapters. Every model has different layer names and structure. The LoRA injection code asks the model backend "which layers should I target?"
- **Forward pass**: How the model processes inputs. T2V models take noisy latents + text embeddings. I2V models also take a first frame. The model backend defines its own forward pass.
- **Checkpoint format**: How to save weights so inference tools can load them. ComfyUI expects specific key names; diffusers expects different ones. The model backend knows its own format.

Adding a new model means implementing these four interfaces. The training loop, optimizer, logging, and everything else stays the same.

## File Structure

```
flimmer-kit/
├── pyproject.toml                 # Package config (9 optional dep groups)
├── examples/                      # Example YAML configs (data/, training/, projects/)
│
├── flimmer/
│   ├── config/                    # Data + training config schemas
│   │   ├── data_schema.py         # Pydantic v2 data config models
│   │   ├── loader.py              # YAML loading, path resolution
│   │   ├── defaults.py            # Constants (fps, frame counts, etc.)
│   │   ├── training_loader.py     # Training config YAML loader + validation
│   │   └── wan22_training_master.py  # Wan 2.2 defaults and validation rules
│   │
│   ├── video/                     # Video ingestion & processing
│   │   ├── __main__.py            # CLI: python -m flimmer.video {...}
│   │   ├── probe.py               # ffprobe wrapper (metadata extraction)
│   │   ├── validate.py            # Structural validation
│   │   ├── scene.py               # PySceneDetect wrapper
│   │   ├── split.py               # ffmpeg wrapper (normalize, split)
│   │   ├── extract.py             # First frame extraction
│   │   └── image_quality.py       # Laplacian sharpness scoring
│   │
│   ├── caption/                   # VLM captioning pipeline
│   │   ├── gemini.py              # Google Gemini backend
│   │   ├── replicate.py           # Replicate backend
│   │   ├── openai_compat.py       # OpenAI-compatible (Ollama, vLLM)
│   │   ├── captioner.py           # Batch orchestrator with retry
│   │   ├── prompts.py             # Use-case prompt templates
│   │   └── scoring.py             # Caption quality scoring
│   │
│   ├── triage/                    # CLIP-based scene matching
│   │   ├── concepts.py            # Reference image loading
│   │   ├── embeddings.py          # CLIP embedding computation
│   │   ├── filters.py             # Similarity threshold filtering
│   │   └── triage.py              # Orchestrator (discover, embed, match)
│   │
│   ├── dataset/                   # Dataset validation & organization
│   │   ├── __main__.py            # CLI: python -m flimmer.dataset {...}
│   │   ├── discover.py            # Sample discovery
│   │   ├── validate.py            # Completeness + structural validation
│   │   ├── organize.py            # Organize into trainer layouts
│   │   ├── bucketing.py           # 3D resolution/temporal bucketing
│   │   └── quality.py             # Quality checks (blur, exposure, duplicates)
│   │
│   ├── encoding/                  # Latent pre-encoding pipeline
│   │   ├── __main__.py            # CLI: python -m flimmer.encoding {...}
│   │   ├── vae_encoder.py         # Wan VAE encoder (video/image -> latents)
│   │   ├── text_encoder.py        # T5 text encoder (captions -> embeddings)
│   │   ├── cache.py               # Cache manifest (build, load, stale detection)
│   │   ├── dataset.py             # CachedLatentDataset for training
│   │   └── bucket.py              # Bucket grouping for batch encoding
│   │
│   └── training/                  # Training infrastructure
│       ├── __main__.py            # CLI: python -m flimmer.training {...}
│       ├── protocols.py           # ModelBackend protocol
│       ├── loop.py                # TrainingOrchestrator (phase-aware loop)
│       ├── phase.py               # Phase resolution (unified, high_noise, low_noise)
│       ├── noise.py               # Flow matching noise schedule
│       ├── lora.py                # LoRA injection + per-expert config
│       ├── optimizer.py           # Optimizer + scheduler creation
│       ├── checkpoint.py          # Save/resume (LoRA + optimizer state)
│       ├── sampler.py             # Video sampling during training
│       ├── metrics.py             # Loss tracking, per-expert metrics
│       ├── logger.py              # W&B logging
│       │
│       └── wan/                   # Wan model backend
│           ├── backend.py         # Forward pass, LoRA targets, expert masks
│           ├── inference.py       # Sampling pipeline
│           ├── checkpoint_io.py   # Checkpoint I/O (diffusers/ComfyUI format)
│           └── registry.py        # Variant registry (2.1 T2V, 2.2 T2V, 2.2 I2V)
│
└── tests/
```

## Remaining Planned Work

- **Phased training architecture** — let users define arbitrary training phase sequences in YAML config. Instead of hardcoded unified/high/low, each phase specifies its expert, learning rate, epochs, dataset, and scheduler. This enables easy swap-in/swap-out during training runs.
- **Flex captioning** — support multiple caption versions per dataset sample in the target folders. The training loop randomly selects a caption variant per epoch, adding natural augmentation without duplicating video data.
- **UI** — a user interface for managing training configs, monitoring runs, and reviewing results.
- **Additional model backends** — LTX-2, SkyReels, and future diffusion transformer models. Adding a new model means implementing the four-interface backend (noise schedule, layer mapping, forward pass, checkpoint format).
- **VACE support** — context block architecture for Wan's VACE variant, enabling control signal conditioning (depth, pose, edge maps) during training.
