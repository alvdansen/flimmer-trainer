# I2V Training Guide

Flimmer supports Image-to-Video (I2V) training for Wan 2.1 and Wan 2.2 models. This guide covers how I2V differs from T2V, how to configure it, and what to expect.

If you're already familiar with T2V training in Flimmer, I2V is the same workflow with one extra step (extracting reference images) and a few config changes. Most settings are inherited from T2V.

---

## What Makes I2V Different

In T2V (Text-to-Video), the model generates video from text alone. A caption is encoded through T5 and injected via cross-attention -- the model learns to produce video that matches the text description.

In I2V (Image-to-Video), a reference image -- called the "first frame" -- is VAE-encoded and channel-concatenated with the noisy video latents. The model learns to generate video that **starts from the reference image's visual state** and evolves according to the text prompt.

This changes the input tensor shape:

- **T2V:** 16 channels (noisy video latents only)
- **I2V:** 36 channels (16 video latent + 20 reference image latent)

The first frame is typically extracted from the training clip itself using `python -m flimmer.video extract`. During training, the VAE encodes the reference image at the video's bucket resolution and concatenates it with the noisy latents before they enter the transformer.

Because the reference image carries significant visual conditioning, I2V training uses different hyperparameter defaults to balance how much the model relies on the image vs the text prompt.

---

## I2V Config Differences

Most settings are identical between T2V and I2V. Only a handful of parameters change:

| Parameter | T2V | I2V | Why |
|-----------|-----|-----|-----|
| `variant` | `2.2_t2v` | `2.2_i2v`, `2.1_i2v_480p`, `2.1_i2v_720p` | Selects the I2V architecture |
| `in_channels` | 16 | 36 | 16 video + 20 reference image latent channels |
| `flow_shift` | 5.0 | 3.0 | Flow matching parameter tuned for I2V |
| `boundary_ratio` (2.2) | 0.875 | 0.900 | Noise boundary between experts is shifted for I2V |
| `caption_dropout_rate` | 0.10 | 0.15 | Higher dropout pushes the model to rely more on the reference image |
| `unified_epochs` | 10 | 15 | More shared training -- the reference image adds conditioning signal |
| `first_frame_dropout_rate` | N/A | 0.05 (optional) | Occasionally drops the reference image during training |

Setting `variant` to an I2V variant auto-fills the correct defaults for `in_channels`, `flow_shift`, and `boundary_ratio`. You don't need to set these manually unless you want to override them.

**Key callouts:**

- **`caption_dropout_rate: 0.15`** -- higher than T2V's 0.10. The reference image already tells the model what the scene looks like, so we drop the caption more often to prevent the model from ignoring the image conditioning.
- **`boundary_ratio: 0.900`** for Wan 2.2 I2V (vs 0.875 for T2V). This shifts where the high-noise expert hands off to the low-noise expert.
- **`flow_shift: 3.0`** for I2V (vs 5.0 for T2V). The flow matching parameter is tuned differently because I2V has stronger conditioning from the reference image.

---

## Wan 2.1 I2V vs Wan 2.2 I2V

### Wan 2.1 I2V

Non-MoE architecture with a single transformer. Simpler setup -- no expert fork, no MoE configuration needed. Available in two resolution subtypes:

| Variant | Resolution | Use case |
|---------|------------|----------|
| `2.1_i2v_480p` | 480p | Faster training, lower VRAM (~44 GB model weights) |
| `2.1_i2v_720p` | 720p | Higher quality output, more VRAM |

Use `variant: 2.1_i2v_480p` or `variant: 2.1_i2v_720p` in your config. A good starting point if you're new to I2V training.

### Wan 2.2 I2V

MoE architecture with dual experts (high-noise and low-noise). Supports fork-and-specialize training where each expert gets independently tuned hyperparameters after a shared warmup phase.

| Variant | Resolution | MoE | Experts |
|---------|------------|-----|---------|
| `2.2_i2v` | 480p/720p | Yes | 2 (high-noise + low-noise) |

Use `variant: 2.2_i2v`. Supports multi-phase project workflows (see `examples/project_moe.yaml`).

### Setup Commands

```bash
# Wan 2.1 I2V (~44 GB model weights)
bash scripts/setup.sh --variant 2.1_i2v_480p

# Wan 2.2 I2V (~69 GB model weights)
bash scripts/setup.sh --variant 2.2_i2v
```

---

## Preparing Reference Images

Reference images are the first frame extracted from each training clip. The model learns to generate video that continues from this frame.

### Extracting Reference Images

```bash
# Extract first frame from each clip as PNG
python -m flimmer.video extract clips/ -o clips/references
```

This pulls the first frame of each clip and saves it as a PNG in the output directory. One reference image per video clip.

### Requirements

- **Format:** PNG or JPG
- **Resolution:** Reference images are encoded at the video's bucket resolution during latent caching. You don't need to manually resize them -- the encoder handles this.
- **One per clip:** Each video clip needs exactly one corresponding reference image.

### When to Provide Custom Reference Images

The extract command uses the clip's actual first frame. This is correct for most training scenarios. You might provide custom reference images when:

- The first frame has artifacts (black frames, transition frames)
- You want the model to learn from a specific keyframe that isn't the first
- You're using externally sourced reference images

Place custom reference images in the references directory with filenames matching their corresponding clips.

---

## Example Configs

Flimmer includes ready-to-use I2V example configs:

- **`examples/i2v_wan21.yaml`** -- Wan 2.1 I2V (non-MoE, single transformer)
- **`examples/i2v_wan22.yaml`** -- Wan 2.2 I2V (MoE, dual experts with fork-and-specialize)
- **`examples/project_moe.yaml`** -- Multi-phase project workflow with phase management

### Annotated Model Section (Wan 2.2 I2V)

```yaml
model:
  variant: 2.2_i2v                               # MoE I2V model

  # Two expert DiT files (MoE = separate high/low noise transformers)
  dit_high: ./models/wan2.2_i2v_high_noise_14B_fp16.safetensors
  dit_low: ./models/wan2.2_i2v_low_noise_14B_fp16.safetensors

  # Shared components (same across all Wan models)
  vae: ./models/wan_2.1_vae.safetensors
  t5: ./models/umt5_xxl_fp16.safetensors

  # Architecture auto-fills from variant:
  # is_moe: true
  # in_channels: 36          # 16 video + 20 reference image
  # boundary_ratio: 0.900    # I2V noise boundary
  # flow_shift: 3.0
```

### Annotated Model Section (Wan 2.1 I2V)

```yaml
model:
  variant: 2.1_i2v_480p                          # Non-MoE I2V model

  # Single DiT file (non-MoE = one unified transformer)
  dit: ./models/wan2.1_i2v_480p_14B_fp16.safetensors

  # Shared components
  vae: ./models/wan_2.1_vae.safetensors
  t5: ./models/umt5_xxl_fp16.safetensors

  # Architecture auto-fills from variant:
  # is_moe: false
  # in_channels: 36          # 16 video + 20 reference image
  # flow_shift: 3.0
```

---

## First-Frame Dropout

### What It Does

First-frame dropout occasionally replaces the reference image with zeros during training. Instead of always seeing the first frame, the model sometimes has to generate video from text alone -- even though it's an I2V model.

### Why Use It

If the model becomes overly dependent on the reference image and starts ignoring text prompts, first-frame dropout forces it to also learn from text conditioning. This improves prompt adherence without sacrificing image conditioning quality.

### How to Configure

```yaml
training:
  # Add this to your I2V config
  first_frame_dropout_rate: 0.05    # Drop reference image 5% of the time
```

### Details

- **When applied:** After caption dropout, with an independent random roll. Both can be active simultaneously -- caption dropout and first_frame_dropout_rate operate independently.
- **Default:** 0 (disabled). Most I2V training doesn't need it.
- **Suggested value:** 0.05 (5%) if you notice the model ignoring prompts.
- **When to use:** Only if the model is following the reference image too closely and not responding to text variations in your prompts.

---

## Quick Start

Minimal steps to get I2V training running:

```bash
# 1. Setup with I2V variant
bash scripts/setup.sh --variant 2.1_i2v_480p

# 2. Prepare your clips (scene detect, split, caption)
python -m flimmer.video ingest "path/to/video.mp4" -o clips
python -m flimmer.video caption clips -p gemini -a "anchor word"

# 3. Extract reference images from your clips
python -m flimmer.video extract clips/ -o clips/references

# 4. Validate dataset
python -m flimmer.dataset validate clips

# 5. Prepare (encode latents + text)
bash scripts/prepare.sh --config i2v_train.yaml

# 6. Train
bash scripts/train.sh --config i2v_train.yaml
```

For detailed pipeline workflows, see [Pipelines](PIPELINES.md). For training config details, see [Training Config Walkthrough](TRAINING_CONFIG_WALKTHROUGH.md).
