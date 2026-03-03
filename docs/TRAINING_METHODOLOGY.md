# Training Methodology & Experimental Findings

> Last updated: 2026-02-22 (v2 — added data preparation barrier-to-entry section)
> Status: Active — update as new findings emerge

This document captures Minta's accumulated training knowledge, experimental findings, and methodology decisions that Flimmer should encode as defaults and opinions.

## Core Methodology: Curator-First

Minta's approach to LoRA training emphasizes **craft over mathematics**. The quality of training results depends primarily on:
1. **Dataset curation** — what goes in determines what comes out
2. **Aesthetic judgment** — evaluating output quality requires trained eyes, not just loss curves
3. **Production context** — LoRAs must work for enterprise clients (Sony, Netflix, A24, Blumhouse, ILM)

Technical hyperparameters matter, but they're in service of curatorial decisions, not the other way around.

## Data Preparation: The #1 Barrier to Entry

The actual journey of a training run starts long before the training loop:

1. **Raw video** — a Blu-ray rip, YouTube download, client footage, or pre-cut clips
2. **Scene detection & cutting** — splitting raw material into clean, continuous shots
3. **Caption generation** — describing what's happening in each clip
4. **Reference image extraction** — pulling first frames or best frames for I2V
5. **Control signal computation** — generating depth maps, edge maps, pose data
6. **Validation & organization** — making sure everything is correctly formatted and aligned
7. **Latent pre-encoding** — compressing everything into the mathematical representation the model uses
8. **Training** — the actual LoRA training run
9. **Inference testing** — verifying the output works in ComfyUI/diffusers

Steps 1–6 are where 80% of the time and failure happens. Existing trainers basically say "figure it out yourself." Every guide, every tutorial, assumes you've already solved these steps. This is the barrier to entry that Flimmer's standalone data tools address.

Flimmer's data tools (scene detection, captioning, frame extraction, validation) work independently of Flimmer's training pipeline. They produce standard output formats compatible with musubi-tuner, ostris, or any trainer. This makes them valuable even for users who never use Flimmer's training loop.

## Differential MoE Training (Key Discovery)

### The Finding
Wan 2.2's dual MoE architecture has two 14B-parameter experts that specialize in different denoising stages:
- **High-noise expert**: Early denoising. Responsible for global composition, motion trajectory, camera movement.
- **Low-noise expert**: Late denoising. Responsible for fine detail, identity preservation, texture.

Training both experts with identical hyperparameters produces suboptimal results. The experts have fundamentally different convergence characteristics.

### What Works
| Parameter | High-Noise Expert | Low-Noise Expert |
|---|---|---|
| LoRA Rank | 16 | 24 |
| Learning Rate | 1e-4 | 8e-5 |
| Epochs | 30 | 50 |

### Why This Works
- High noise experts converge faster because coarse compositional features are "easier" to learn — the model is working with blurry, global structure
- Low noise experts need more capacity (higher rank) and more time because fine details are information-dense
- Low noise experts are extremely sensitive to aggressive training — they overfit rapidly

### The "Strawberryman" Discovery
Training a character ("strawberryman") with aggressive parameters on the low-noise expert produced:
- Washed-out artifacts in fine detail regions
- Failure on dynamic facial expressions
- Loss of texture fidelity
- The character's identity partially collapsed

This revealed that low-noise experts require a fundamentally gentler training approach — lower learning rate AND more epochs, not just more epochs at the same rate.

### Implication for Flimmer
Differential MoE training must be a first-class config feature, not a hack. The config surface should make it natural to set per-expert parameters and should include warnings/validation if the user sets aggressive parameters for the low-noise expert.

## Caption Strategy

### The Problem with Caption-Centric Training
Most tutorials and guides for video LoRA training treat captions as the primary conditioning mechanism. "Write better captions" is the universal advice. But in practice:
- Captions capture semantic content (what's happening) but poorly capture visual style, motion quality, and aesthetic properties
- Over-reliance on captions makes LoRAs fragile — they only work well with prompts that closely match the training captions
- For I2V models, the reference image carries far more conditioning information than the caption

### Flimmer's Approach
- Captions are ONE control signal among many, not THE control signal
- Caption dropout rate is a first-class config parameter (default: 10-20%)
- Higher dropout rates force the model to learn from visual signals rather than text, producing more robust LoRAs
- For I2V training, reference images should be weighted as the primary control signal

## Dataset Preparation Standards

### Video Clips
- **Temporal coherence**: No scene cuts within a clip. Every clip must be a continuous shot.
- **Frame rate**: Consistent within each clip. Wan trains at 16 FPS — source material should be at or above this.
- **Duration**: Must match model's temporal window. Wan: up to 81 frames (5 seconds at 16 FPS). Frame counts must satisfy (frames - 1) % 4 == 0.
- **Resolution**: At or above target training resolution. Wan supports 480P and 720P.
- **Quality**: No compression artifacts, no watermarks, no letterboxing.

### What Minta Looks For
- Consistent lighting within clips
- Meaningful motion (not static shots)
- Diverse angles/perspectives for subject LoRAs
- Consistent aesthetic for style LoRAs
- 15-30 clips for character/subject LoRAs
- 30-60 clips for style LoRAs
- Quality over quantity — 10 excellent clips beat 30 mediocre ones

### Dataset Scripts Location
- Scripts: `C:\Users\minta\Training\`
- Output: `C:\Users\minta\Training\datasets\{name}\{video_clips,image_stills}\`
- Blu-ray drive for ripping: Pioneer BDR-X13U-S

## Production Workflow Context

### Client Delivery Requirements
Minta's LoRAs are delivered to enterprise clients for use in production pipelines. This means:
- **Compatibility**: LoRAs must work with ComfyUI and diffusers without modification
- **Reliability**: Results must be consistent across different prompts and settings
- **Documentation**: Each LoRA ships with recommended settings (strength, prompt guidance, resolution)
- **Quality gates**: Every LoRA goes through visual quality assessment before delivery

### Current Production Stack
- **Active trainer**: musubi-tuner (Minta's fork) — handles day-to-day client work
- **Inference**: ComfyUI with Wan 2.2 I2V
- **Base model**: Wan2.2-I2V-A14B (with high-noise and low-noise expert checkpoints)
- **Flimmer relationship**: Experimental/parallel — does NOT replace musubi-tuner until proven

## Comparison Baselines

Every Flimmer component must be validated against known-good output. The reference implementations are:

| Component | Reference | What to Compare |
|---|---|---|
| Noise scheduling | musubi-tuner's flow matching implementation | Noise distribution across timesteps |
| LoRA injection | musubi-tuner's network setup for Wan | Which layers get LoRA, key naming |
| Checkpoint format | musubi-tuner safetensors output | Weight keys, shapes, compatibility with ComfyUI |
| Dataset loading | musubi-tuner's video dataset loader | How clips are preprocessed, bucketed, batched |
| Training loss | musubi-tuner training run on identical dataset | Loss curves should roughly match for same hyperparams |

## Open Questions (Research Needed)

1. **VACE training**: Why hasn't anyone set up LoRA training for VACE's context blocks? Is it a technical limitation or just a gap in tooling?
2. **Control LoRA methodology**: What's the best approach for teaching a model to respond to new control signals? spacepxl's depth control LoRA for Flux is the reference — does the same approach transfer to video?
3. **Caption dropout interaction with MoE**: Does caption dropout affect the high-noise and low-noise experts differently? The high-noise expert works with coarse features that might be more text-dependent.
4. **Optimal expert boundary**: Is the default SNR boundary for Wan 2.2's expert switching optimal for LoRA training, or should it be tunable?
5. **Audio signal interaction**: When audio is present in training data (LTX-2 case), how does it affect video LoRA quality even if audio isn't explicitly trained?
