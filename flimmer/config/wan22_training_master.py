"""Wan 2.2 training master — T2V and I2V.

Master config for Wan 2.2 LoRA training. Covers both model variants
(T2V and I2V) and all three training modes (fork-and-specialize,
unified-only, expert-only). Users pick their variant and mode via YAML.

This is INFRASTRUCTURE — users never edit this file. They edit a YAML
config that sets a variant (2.2_t2v or 2.2_i2v), and the loader
applies the right defaults from here.

Structure:
  Part 1 — Valid options (shared vocabulary for both T2V and I2V)
  Part 2 — Default values (T2V section, then I2V section, then variant map)
  Part 3 — Pydantic schema (shared validation rules for both variants)
"""

from __future__ import annotations

import warnings
from pydantic import BaseModel, Field, field_validator, model_validator


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PART 1: VALID OPTIONS (vocabulary)
# These are what names are allowed in config fields. They happen to be
# the same across models, but they live here so this file is fully
# self-contained.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

VALID_OPTIMIZERS: set[str] = {
    "adamw",
    "adamw8bit",
    "adafactor",
    "came",
    "prodigy",
    "ademamix",
    "schedule_free_adamw",
}
"""Supported optimizer types.

- adamw: standard AdamW, full precision. Best training quality, highest VRAM.
- adamw8bit: 8-bit AdamW (bitsandbytes). ~60% community usage, best validated.
- adafactor: memory-efficient, no momentum. Good for very large models.
- came: confidence-aware memory-efficient optimizer. Like Adafactor but with momentum.
- prodigy: learning-rate-free optimizer. Sets LR automatically (must use lr=1.0).
- ademamix: dual-EMA momentum (Apple). Untested on Wan, needs long runs to benefit.
- schedule_free_adamw: schedule-free Adam (Meta). Eliminates LR scheduler. Untested on Wan.
"""

VALID_SCHEDULERS: set[str] = {
    "constant",
    "constant_with_warmup",
    "cosine",
    "cosine_with_min_lr",
    "cosine_with_restarts",
    "inverse_sqrt",
    "linear",
    "polynomial",
    "rex",
    "step",
    "warmup_stable_decay",
}
"""Supported learning rate schedulers.

- constant: fixed LR throughout. Simple, predictable.
- constant_with_warmup: linear warmup then constant. Common with Prodigy.
- cosine: classic cosine decay to zero. Smooth convergence.
- cosine_with_min_lr: cosine decay with a floor. Prevents LR from going to zero.
- cosine_with_restarts: cosine annealing with periodic resets. Multiple cycles.
- inverse_sqrt: inverse square root decay. More common in LLM training.
- linear: linear decay to zero. Predictable and easy to reason about.
- polynomial: polynomial decay. Configurable via power parameter.
- rex: reciprocal-exponential schedule. Aggressive early, stable late.
- step: step-based decay at fixed intervals.
- warmup_stable_decay: warmup ramp, stable plateau, then decay.
"""

VALID_MIXED_PRECISION: set[str] = {"bf16", "fp16", "no"}
"""Mixed precision modes for training computation.

- bf16: bfloat16. Best for modern GPUs (Ampere+). Wider dynamic range than fp16.
- fp16: float16. Works on older GPUs. Requires careful loss scaling.
- no: full fp32. Maximum precision, maximum VRAM. Rarely needed for LoRA.
"""

VALID_BASE_PRECISION: set[str] = {"fp8", "fp8_scaled", "bf16", "fp16", "fp32"}
"""Precision for the frozen base model weights.

- fp8: 8-bit float. Cuts base model VRAM roughly in half. Some quality trade-off.
- fp8_scaled: fp8 with per-tensor scaling. Better quality than plain fp8.
- bf16: full bfloat16. No quantization artifacts. Standard quality.
- fp16: float16. Similar VRAM to bf16 on modern GPUs.
- fp32: full precision. Maximum VRAM, rarely needed.
"""

VALID_TIMESTEP_SAMPLING: set[str] = {"uniform", "shift", "logit_normal", "sigmoid"}
"""How to sample noise timesteps during training.

- uniform: equal probability for all timesteps. Simple baseline.
- shift: shifted distribution favoring mid-to-high noise. Wan default.
- logit_normal: logit-normal distribution. Concentrates around the center.
- sigmoid: sigmoid-based sampling. Similar to logit_normal, different tails.
"""

VALID_LOG_BACKENDS: set[str] = {"console", "tensorboard", "wandb"}
"""Supported logging backends.

- console: print to terminal. Always available, no setup needed.
- tensorboard: TensorBoard log files. Good for local experiment tracking.
- wandb: Weights & Biases. Best for comparing runs across experiments.
"""

VALID_CHECKPOINT_FORMATS: set[str] = {"safetensors", "diffusers"}
"""Checkpoint output formats.

- safetensors: single .safetensors file. ComfyUI-compatible. Default.
- diffusers: HuggingFace diffusers directory format. For diffusers pipeline loading.
"""

VALID_FORK_TARGETS: set[str] = {
    # Component-level: targets all projections in the component
    "ffn",
    "self_attn",
    "cross_attn",
    # Projection-level: targets a specific projection within a component
    "ffn.up_proj",
    "ffn.down_proj",
    "self_attn.to_q",
    "self_attn.to_k",
    "self_attn.to_v",
    "self_attn.to_out",
    "cross_attn.to_q",
    "cross_attn.to_k",
    "cross_attn.to_v",
    "cross_attn.to_out",
}
"""Valid targets for per-expert training after fork.

Two levels of granularity:
- Component-level (e.g. "ffn") = shortcut for all projections in that component
- Projection-level (e.g. "cross_attn.to_v") = just that one projection

Rules:
- Components listed at component-level: all projections train per-expert
- Components with specific projections listed: only those projections train,
  the rest of that component is frozen
- Components not listed at all: fully frozen (keeps unified weights)

Example: ["ffn", "self_attn", "cross_attn.to_v"]
  → all FFN trains, all self-attn trains, only cross-attn V trains,
    cross-attn Q/K/O stay frozen
"""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PART 2: T2V DEFAULT VALUES
# Every constant is prefixed T2V_ and has a docstring explaining WHY.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


# ─── Wan 2.2 T2V: Architecture ───

T2V_MODEL_FAMILY: str = "wan"
T2V_MODEL_VARIANT: str = "2.2_t2v"
T2V_IS_MOE: bool = True
T2V_IN_CHANNELS: int = 16
"""Latent input channels. T2V is pure noise → video (16 channels).
I2V would be 36 (16 noise + 20 image conditioning)."""

T2V_NUM_LAYERS: int = 40
"""Number of transformer blocks. All Wan models use 40."""

T2V_BOUNDARY_RATIO: float = 0.875
"""SNR boundary for expert routing during training. Timesteps >= this value
train the high-noise expert; timesteps below train the low-noise expert.
0.875 matches the official Wan 2.2 inference config and is used by every
community trainer (musubi, ai-toolkit, diffusion-pipe, DiffSynth-Studio).
Override via moe.boundary_ratio in your training config to experiment."""

T2V_FLOW_SHIFT: float = 5.0
"""Flow shift parameter. Controls the noise schedule curve — higher values
mean more noise early in the denoising process.

- 480p: use 3.0 (default)
- 720p: change to 5.0
"""

T2V_NUM_TRAIN_TIMESTEPS: int = 1000
"""Total noise schedule steps. 1000 for all Wan models."""


# ─── Wan 2.2 T2V: Training Strategy ───
# This master file covers all three training modes. Fork-and-specialize:
#   1. Unified phase: both experts share one LoRA for unified_epochs
#   2. Fork: LoRA is copied, one per expert
#   3. Expert phase: each copy trains independently with its own overrides
#
# For unified-only: set moe.fork_enabled: false in YAML
# For expert-from-scratch: set training.unified_epochs: 0 in YAML

T2V_FORK_ENABLED: bool = True
"""Master switch for fork-and-specialize training. This is the primary
training strategy for Wan 2.2 MoE — both experts share a unified LoRA,
then fork into per-expert copies with independent training."""


# ─── Wan 2.2 T2V: Fixed Settings ───
# These stay constant for the entire training run — unified phase and
# expert phase alike. Users set them once. LoRA structure, optimizer
# internals, precision, and noise sampling.

T2V_LORA_RANK: int = 16
"""LoRA rank (same as network_dim in musubi/kohya). Controls the capacity
of the adapter. Locked for the entire run — determines matrix dimensions
(A: d×r, B: r×d). Cannot change after creation."""

T2V_LORA_ALPHA: int = 16
"""LoRA alpha (same as network_alpha in musubi/kohya). Scaling factor.
Effective strength = alpha / rank. alpha = rank gives 1.0x (neutral)
scaling. Locked for the entire run like rank."""

T2V_LORAPLUS_LR_RATIO: float = 4.0
"""LoRA+ B-matrix learning rate multiplier. The B matrix (output projection)
starts from zero and needs to learn faster. 4.0 is validated on Wan."""

T2V_OPTIMIZER: str = "adamw8bit"
"""AdamW 8-bit is the standard for Wan LoRA. Cuts optimizer VRAM in half
with negligible quality loss."""

T2V_BETAS: list[float] = [0.9, 0.999]
"""Adam momentum terms. PyTorch default, works well for LoRA."""

T2V_EPS: float = 1e-8
"""Adam epsilon for numerical stability. PyTorch default."""

T2V_MAX_GRAD_NORM: float = 1.0
"""Gradient clipping threshold. Prevents training instability from
gradient spikes without throttling normal updates."""

T2V_WARMUP_STEPS: int = 0
"""Wan LoRA typically needs no warmup. The pretrained model is already
well-conditioned, and LoRA adapters start near zero. Warmup happens once
at the start of training — by the time experts fork, you're already warm."""

T2V_MIXED_PRECISION: str = "bf16"
"""BFloat16 for training computation. Enough dynamic range to avoid the
loss scaling issues that plague fp16."""

T2V_BASE_MODEL_PRECISION: str = "bf16"
"""Full bf16 for the frozen base model. No quantization artifacts.
fp8 is available if VRAM-constrained (cuts ~28 GB to ~14 GB) but
quality comes first — don't quantize unless you have to."""

T2V_TIMESTEP_SAMPLING: str = "shift"
"""Wan models were trained with shifted timestep sampling. Using the same
distribution during fine-tuning maintains consistency with the pretrained
model's learned noise-level expectations."""


# ─── Wan 2.2 T2V: Unified Foundation ───
# Starting values for the unified training phase. Each expert can
# override these after fork — see expert override sections below.
# When no expert override is set, the expert inherits these values.

T2V_UNIFIED_EPOCHS: int = 10
"""Epochs for the unified (shared) training phase before forking.
Both experts train with the same LoRA during this phase.
10 epochs is a starting hypothesis — needs experimental validation."""

T2V_UNIFIED_TARGETS: list[str] | None = None
"""Component targeting during the unified phase. None = all standard
LoRA targets (all attention + FFN). Override to train only specific
components during unified — e.g. ["ffn", "cross_attn"] to skip
self-attention during unified and only add it per-expert."""

T2V_UNIFIED_BLOCK_TARGETS: str | None = None
"""Block targeting during the unified phase. None = all 40 blocks.
Format: "0-39" for all, "0-11" for early blocks only, "0-11,25-34"
for specific ranges. Unspecified blocks are not trained during unified."""

T2V_LEARNING_RATE: float = 5e-5
"""Conservative default for Wan 2.2 T2V LoRA. Community recommendations
(2e-4) are too aggressive — especially for the low-noise expert which
overfits rapidly at high LR. 5e-5 is a safe starting point. Overridable
per expert after fork."""

T2V_WEIGHT_DECAY: float = 0.01
"""Standard weight decay. Helps prevent overfitting without impacting
convergence. Low-noise expert may benefit from higher weight decay
to counteract its tendency to overfit. Overridable per expert."""

T2V_SCHEDULER: str = "cosine_with_min_lr"
"""Cosine decay with a minimum LR floor. Prevents the learning rate from
going to zero in late epochs, which matters for fine detail refinement.
Overridable per expert — high-noise might want faster decay, low-noise
might want constant LR."""

T2V_MIN_LR_RATIO: float = 0.01
"""Minimum LR as fraction of peak LR. 1% of peak keeps the optimizer
active in late training without destabilizing. Overridable per expert."""

T2V_LORA_DROPOUT: float = 0.0
"""No LoRA dropout by default. Most video LoRA datasets are small (10-30
clips), so dropout would just slow learning. Consider 0.05 for >20 clips.
Overridable per expert — may want higher dropout after introducing new data."""

T2V_BATCH_SIZE: int = 1
"""Batch size 1 is the practical default for video LoRA training.
Use gradient accumulation for effective larger batches.
Overridable per expert after fork."""

T2V_GRADIENT_ACCUMULATION: int = 1
"""Steps between optimizer updates. Increase to simulate larger batch
sizes without additional VRAM. Overridable per expert after fork."""

T2V_CAPTION_DROPOUT_RATE: float = 0.1
"""Probability of dropping the entire caption for a training sample.
Forces the model to rely on visual signals rather than text alone.
T2V uses 10% — lower than I2V because text is the only conditioning.
Overridable per expert after fork."""


# ─── Wan 2.2 T2V: High-Noise Expert Overrides ───
# Per-expert settings for the fork phase. None = inherit from unified
# foundation above. Set a value to override for this expert only.

T2V_HIGH_NOISE_LEARNING_RATE: float | None = None
"""High-noise expert LR override. None = keep unified LR."""

T2V_HIGH_NOISE_DROPOUT: float | None = None
"""High-noise expert LoRA dropout override. None = keep unified dropout."""

T2V_HIGH_NOISE_MAX_EPOCHS: int = 50
"""Max epochs for high-noise expert training after fork."""

T2V_HIGH_NOISE_FORK_TARGETS: list[str] | None = None
"""Components to train for high-noise expert after fork. None = same as
unified_targets. Override to focus on specific components — e.g.
["ffn", "self_attn"] based on divergence analysis (FFN 0.872-0.894,
self-attn 0.868-0.914 cosine between experts)."""

T2V_HIGH_NOISE_BLOCK_TARGETS: str | None = None
"""Block targeting for high-noise expert. None = all blocks.
Format: "0-11,25-34" to target specific block ranges.
High-noise divergence was highest in blocks 0-11 (55% movement)."""

T2V_HIGH_NOISE_RESUME_FROM: str | None = None
"""Path to an existing LoRA file to use as starting point for
high-noise expert training. None = start from unified weights (or
fresh if unified_epochs=0)."""

T2V_HIGH_NOISE_BATCH_SIZE: int | None = None
"""Batch size override. None = keep unified batch size."""

T2V_HIGH_NOISE_GRADIENT_ACCUMULATION: int | None = None
"""Gradient accumulation override. None = keep unified value."""

T2V_HIGH_NOISE_CAPTION_DROPOUT: float | None = None
"""Caption dropout override. None = keep unified caption dropout."""

T2V_HIGH_NOISE_WEIGHT_DECAY: float | None = None
"""Weight decay override. None = keep unified weight decay."""

T2V_HIGH_NOISE_MIN_LR_RATIO: float | None = None
"""Min LR ratio override. None = keep unified min LR ratio."""


# ─── Wan 2.2 T2V: Low-Noise Expert Overrides ───

T2V_LOW_NOISE_LEARNING_RATE: float | None = None
"""Low-noise expert LR override. None = keep unified LR."""

T2V_LOW_NOISE_DROPOUT: float | None = None
"""Low-noise expert LoRA dropout override. None = keep unified dropout.
Consider higher dropout for low-noise to prevent overfitting on detail."""

T2V_LOW_NOISE_MAX_EPOCHS: int = 50
"""Max epochs for low-noise expert training after fork."""

T2V_LOW_NOISE_FORK_TARGETS: list[str] | None = None
"""Components to train for low-noise expert after fork. None = same as
unified_targets. Override to focus on specific components."""

T2V_LOW_NOISE_BLOCK_TARGETS: str | None = None
"""Block targeting for low-noise expert. None = all blocks.
Format: "0-11,25-34" to target specific block ranges."""

T2V_LOW_NOISE_RESUME_FROM: str | None = None
"""Path to an existing LoRA file to use as starting point for
low-noise expert training. None = start from unified weights (or
fresh if unified_epochs=0)."""

T2V_LOW_NOISE_BATCH_SIZE: int | None = None
"""Batch size override. None = keep unified batch size."""

T2V_LOW_NOISE_GRADIENT_ACCUMULATION: int | None = None
"""Gradient accumulation override. None = keep unified value."""

T2V_LOW_NOISE_CAPTION_DROPOUT: float | None = None
"""Caption dropout override. None = keep unified caption dropout."""

T2V_LOW_NOISE_WEIGHT_DECAY: float | None = None
"""Weight decay override. None = keep unified weight decay."""

T2V_LOW_NOISE_MIN_LR_RATIO: float | None = None
"""Min LR ratio override. None = keep unified min LR ratio."""


# ─── Wan 2.2 T2V: Save ───

T2V_SAVE_EVERY_N_EPOCHS: int = 5
"""Save a checkpoint every 5 epochs. Balances storage space against
the ability to pick the best checkpoint."""

T2V_CHECKPOINT_FORMAT: str = "safetensors"
"""Safetensors is the most widely supported format across inference
tools (ComfyUI, A1111, diffusers)."""


# ─── Wan 2.2 T2V: Sampling ───
# Follows ai-toolkit conventions: prompts list, neg string, walk_seed,
# sample_steps. Samples are generated every N epochs during training
# for visual progress tracking. OFF by default — inference is expensive.

T2V_SAMPLING_ENABLED: bool = False
"""Sample generation is OFF by default. Generating sample videos during
training requires a full inference pass per prompt per checkpoint, which
can significantly slow down training. Enable explicitly when you want
visual progress tracking."""

T2V_SAMPLING_EVERY_N_EPOCHS: int = 5
"""Generate sample videos every N epochs. Matches save interval by
default so you can see what each checkpoint looks like."""

T2V_SAMPLING_PROMPTS: list[str] = [
    "A person stands still and looks directly at the camera, neutral expression, soft lighting",
    "A cat walks across a wooden table, sunlight streaming through a window",
    "Ocean waves crash against rocks on a rugged coastline, misty spray",
    "A close-up of hands arranging colorful flowers in a glass vase",
]
"""Default prompts for sample generation. One video per prompt.
Four scenarios covering key training quality dimensions:
  1. Static person -- tests identity preservation and facial detail
  2. Animal motion -- tests temporal coherence and natural movement
  3. Scenic landscape -- tests composition and environment rendering
  4. Close-up detail -- tests fine detail and texture quality

Override with your own prompts in the YAML config. Empty list = no samples.

Example:
  - "Annika walks through a sunlit garden"
  - "Annika looks into the camera with a playful smile"
"""

T2V_SAMPLING_NEG: str = ""
"""Negative prompt applied to ALL sample generations. Single string,
not per-prompt. Empty string = no negative conditioning.

Example: "blurry, low quality, distorted"
"""

T2V_SAMPLING_SEED: int = 42
"""Base seed for sample generation. Same seed across epochs makes
it easy to compare progress visually."""

T2V_SAMPLING_WALK_SEED: bool = True
"""Increment seed by 1 for each prompt in the list. Gives variety
across prompts while keeping each individual prompt reproducible.
Prompt 0 = seed, prompt 1 = seed+1, etc."""

T2V_SAMPLING_STEPS: int = 30
"""Denoising steps for sample generation. 30 is a good balance between
quality and speed for progress monitoring."""

T2V_SAMPLING_GUIDANCE: float = 4.0
"""Guidance scale for sample generation. 4.0 is the validated default for
balanced text adherence without over-saturation."""

T2V_SAMPLING_DIR: str | None = None
"""Directory for sample video output. None = {save.output_dir}/samples/.
Relative to config file location."""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PART 2b: I2V DEFAULT VALUES
# Only values that DIFFER from T2V are listed here. Everything else
# (optimizer, scheduler, precision, sampling, etc.) is shared.
#
# Key differences from T2V:
# - Reference image adds 20 conditioning channels (16 → 36 total)
# - Experts moved in SIMILAR direction (TV cosine 0.58 vs ~0 for T2V)
# - Both experts changed substantially (no 1,100x asymmetry like T2V)
# - Unified phase carries more signal → longer unified, shorter per-expert
# - Higher caption dropout (reference image carries more conditioning)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


# ─── Wan 2.2 I2V: Architecture ───

I2V_MODEL_VARIANT: str = "2.2_i2v"
I2V_IN_CHANNELS: int = 36
"""Latent input channels. I2V = 36 (16 noise + 20 VAE-encoded reference image).
The reference image is channel-concatenated with the noisy latents."""

I2V_BOUNDARY_RATIO: float = 0.900
"""SNR boundary for I2V expert routing. Matches official Wan 2.2 I2V config.
Override via moe.boundary_ratio to experiment."""


# ─── Wan 2.2 I2V: Unified Foundation (differences from T2V) ───

I2V_UNIFIED_EPOCHS: int = 15
"""Longer unified phase than T2V (10). Both I2V experts moved in similar
directions (TV cosine 0.58), so the unified phase captures a meaningful
shared signal about 'how this subject looks in the reference image.'"""

I2V_CAPTION_DROPOUT_RATE: float = 0.15
"""Higher than T2V (0.10). The reference image carries substantial
conditioning, so dropping captions more often forces the model to rely
on the visual signal — which is what I2V is about."""


# ─── Wan 2.2 I2V: Expert Overrides (differences from T2V) ───
# I2V low-noise expert changed 36.4% from Wan 2.1 (vs 0.04% for T2V).
# It is NOT 'already done' like T2V low-noise — real training needed.

I2V_LOW_NOISE_MAX_EPOCHS: int = 50
"""I2V low-noise needs real training — 36.4% movement from 2.1 base,
unlike T2V low-noise which barely moved (0.04%). Not a refinement job."""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# VARIANT DEFAULTS MAP
# The loader uses this to apply the right defaults when a user sets
# variant: 2.2_t2v or variant: 2.2_i2v in their YAML.
# Only values that differ from the Pydantic schema defaults (which use
# T2V constants) need to be listed for I2V.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

VARIANT_DEFAULTS: dict[str, dict] = {
    "2.2_t2v": {
        # T2V uses the Pydantic defaults directly — nothing to override.
        "model": {
            "family": T2V_MODEL_FAMILY,
            "variant": T2V_MODEL_VARIANT,
            "is_moe": T2V_IS_MOE,
            "in_channels": T2V_IN_CHANNELS,
            "num_layers": T2V_NUM_LAYERS,
            "boundary_ratio": T2V_BOUNDARY_RATIO,
            "flow_shift": T2V_FLOW_SHIFT,
        },
    },
    "2.2_i2v": {
        # I2V overrides — everything else inherited from T2V/Pydantic defaults.
        "model": {
            "family": T2V_MODEL_FAMILY,
            "variant": I2V_MODEL_VARIANT,
            "is_moe": T2V_IS_MOE,
            "in_channels": I2V_IN_CHANNELS,
            "num_layers": T2V_NUM_LAYERS,
            "boundary_ratio": I2V_BOUNDARY_RATIO,
            "flow_shift": T2V_FLOW_SHIFT,
        },
        "training": {
            "unified_epochs": I2V_UNIFIED_EPOCHS,
            "caption_dropout_rate": I2V_CAPTION_DROPOUT_RATE,
        },
        "moe": {
            "low_noise": {
                "max_epochs": I2V_LOW_NOISE_MAX_EPOCHS,
            },
        },
    },
}
"""Variant defaults map. The loader deep-merges these under the user's
YAML values (user wins on any conflict)."""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PART 3: PYDANTIC SCHEMA
# These models define the YAML structure users write. Default values
# come from the T2V constants above. For I2V, the loader overrides
# the relevant defaults before validation.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


# ─── Helpers ───


def _validate_block_targets(value: str | None) -> str | None:
    """Validate block target string format.

    Accepts formats like "0-11", "0-11,25-34", "5". Checks that each
    range has start <= end and all values are non-negative integers.
    Does NOT validate against num_layers (that happens at training time
    when we know the model architecture).
    """
    if value is None:
        return None
    # Split on commas, validate each range
    for part in value.split(","):
        part = part.strip()
        if not part:
            raise ValueError(
                f"Empty range in block_targets: '{value}'. "
                f"Format: '0-11' or '0-11,25-34'."
            )
        if "-" in part:
            pieces = part.split("-")
            if len(pieces) != 2:
                raise ValueError(
                    f"Invalid range '{part}' in block_targets. "
                    f"Use 'start-end' format (e.g. '0-11')."
                )
            try:
                start, end = int(pieces[0]), int(pieces[1])
            except ValueError:
                raise ValueError(
                    f"Non-integer in block range '{part}'. "
                    f"Block targets must be integers."
                )
            if start < 0 or end < 0:
                raise ValueError(
                    f"Negative block index in '{part}'. "
                    f"Block indices must be >= 0."
                )
            if start > end:
                raise ValueError(
                    f"Invalid range '{part}': start ({start}) > end ({end})."
                )
        else:
            try:
                idx = int(part)
            except ValueError:
                raise ValueError(
                    f"Non-integer block index '{part}'. "
                    f"Block targets must be integers or ranges."
                )
            if idx < 0:
                raise ValueError(
                    f"Negative block index '{part}'. "
                    f"Block indices must be >= 0."
                )
    return value


def _validate_fork_targets(targets: list[str]) -> list[str]:
    """Validate a list of fork target strings against VALID_FORK_TARGETS."""
    invalid = set(targets) - VALID_FORK_TARGETS
    if invalid:
        valid_list = ", ".join(sorted(VALID_FORK_TARGETS))
        invalid_list = ", ".join(sorted(invalid))
        raise ValueError(
            f"Invalid fork target(s): {invalid_list}. "
            f"Valid targets: {valid_list}."
        )
    return targets


# ─── Model Config ───


class ModelConfig(BaseModel):
    """What model to train and where to find it.

    The variant field identifies the model architecture. The loader uses it
    to fill in architecture defaults (channel counts, MoE settings, etc.).
    User fields override auto-filled values.

    Two ways to point at model weights:

    1. **Individual files** (recommended): Set dit_high/dit_low (MoE) or dit
       (non-MoE), plus vae and t5. Each points to a single .safetensors or
       .pth file. This is how models are distributed by Comfy-Org and how
       setup.sh downloads them on RunPod.

    2. **Diffusers directory**: Set path to a local directory or HuggingFace
       ID (e.g. 'Wan-AI/Wan2.2-T2V-14B-Diffusers'). Components are loaded
       from standard subdirectories (transformer/, vae/, text_encoder/).

    When both are set, individual files take priority over the directory.
    """

    # --- Individual weight files (primary) ---
    dit: str | None = Field(
        default=None,
        description=(
            "Path to transformer weights (.safetensors). For non-MoE models "
            "(Wan 2.1). Ignored when dit_high/dit_low are set."
        ),
    )
    dit_high: str | None = Field(
        default=None,
        description=(
            "Path to high-noise expert weights (.safetensors). For MoE models "
            "(Wan 2.2). Used together with dit_low."
        ),
    )
    dit_low: str | None = Field(
        default=None,
        description=(
            "Path to low-noise expert weights (.safetensors). For MoE models "
            "(Wan 2.2). Used together with dit_high."
        ),
    )
    vae: str | None = Field(
        default=None,
        description=(
            "Path to VAE weights (.safetensors). When set, the VAE is loaded "
            "from this file via from_single_file() instead of from a "
            "Diffusers directory."
        ),
    )
    t5: str | None = Field(
        default=None,
        description=(
            "Path to T5 text encoder weights (.pth or .safetensors). When set, "
            "weights are loaded from this file and the tokenizer is downloaded "
            "from HuggingFace (google/umt5-xxl, which is tiny)."
        ),
    )

    # --- Diffusers directory (fallback) ---
    path: str | None = Field(
        default=None,
        description=(
            "Path to a Diffusers model directory or HuggingFace ID "
            "(e.g. 'Wan-AI/Wan2.2-T2V-14B-Diffusers'). Used as fallback when "
            "individual file paths are not set. Local paths are resolved "
            "relative to the config file's location."
        ),
    )
    family: str | None = Field(
        default=None,
        description=(
            "Model family name (e.g. 'wan'). Auto-filled from variant. "
            "Override only if using a custom model."
        ),
    )
    variant: str | None = Field(
        default=None,
        description=(
            "Model variant. Identifies the architecture so defaults can auto-fill. "
            "Options: '2.2_t2v', '2.2_i2v'."
        ),
    )
    is_moe: bool | None = Field(
        default=None,
        description=(
            "Whether this model uses Mixture of Experts architecture. "
            "Auto-set by template."
        ),
    )
    in_channels: int | None = Field(
        default=None,
        gt=0,
        description=(
            "Input channel count for the transformer. Auto-set by template. "
            "T2V = 16 (latent only), I2V = 36 (latent + VAE-encoded reference)."
        ),
    )
    num_layers: int | None = Field(
        default=None,
        gt=0,
        description=(
            "Number of transformer blocks. Auto-set by template. "
            "Wan models use 40 layers."
        ),
    )
    boundary_ratio: float | None = Field(
        default=None,
        description=(
            "MoE expert routing boundary (SNR threshold). "
            "T2V: 0.875, I2V: 0.900. Only relevant for MoE models."
        ),
    )
    flow_shift: float | None = Field(
        default=None,
        description=(
            "Flow matching shift parameter. Controls the noise schedule curve. "
            "480p: 3.0, 720p: 5.0. Auto-set by template."
        ),
    )
    num_train_timesteps: int = Field(
        default=T2V_NUM_TRAIN_TIMESTEPS,
        gt=0,
        description="Total noise schedule steps. 1000 for all Wan models.",
    )
    quantization: str | None = Field(
        default=None,
        description=(
            "Weight quantization mode for model loading. Options: 'int8', 'fp8'. "
            "Reduces VRAM usage at the cost of some precision. "
            "Uses optimum-quanto for LTX models."
        ),
    )

    # Variant validation is handled by the loader, which knows
    # which variants are supported (checked against VARIANT_DEFAULTS keys).


# ─── LoRA Config ───


class LoraConfig(BaseModel):
    """Unified LoRA settings (shared foundation).

    These settings define the LoRA adapter that both experts share during
    the unified training phase. After fork, each expert gets a copy of
    this LoRA — per-expert overrides (learning rate, dropout, fork targets)
    can then modify training behavior per expert.

    Rank and alpha are locked at creation and cannot change after fork
    (they determine matrix dimensions).
    """

    rank: int = Field(
        default=T2V_LORA_RANK,
        gt=0,
        description=(
            "LoRA rank (same as network_dim in musubi/kohya). Controls the "
            "capacity of the adapter. Higher = more capacity, more VRAM. "
            "32 is the community standard for Wan 14B. This value is locked "
            "for the entire training run — it determines matrix dimensions."
        ),
    )
    alpha: int = Field(
        default=T2V_LORA_ALPHA,
        gt=0,
        description=(
            "LoRA alpha (same as network_alpha in musubi/kohya). Scaling factor. "
            "Effective scaling = alpha/rank. "
            "alpha = rank/2 is conservative (0.5x). alpha = rank is neutral (1.0x)."
        ),
    )
    dropout: float = Field(
        default=T2V_LORA_DROPOUT,
        ge=0.0,
        le=1.0,
        description=(
            "LoRA dropout rate (0.0-1.0). Randomly drops LoRA outputs during "
            "training to prevent overfitting. 0.0 = no dropout. "
            "Consider 0.05 for datasets with >20 clips."
        ),
    )
    loraplus_lr_ratio: float = Field(
        default=T2V_LORAPLUS_LR_RATIO,
        ge=1.0,
        description=(
            "LoRA+ B-matrix learning rate multiplier (>= 1.0). The B matrix "
            "(output projection) starts from zero and needs to learn faster. "
            "4.0 is validated on Wan models."
        ),
    )
    target_modules: list[str] | None = Field(
        default=None,
        description=(
            "Which model modules to apply LoRA to. null = use model defaults "
            "(typically all attention projections + FFN layers). "
            "Example: ['to_q', 'to_k', 'to_v', 'to_out.0']"
        ),
    )
    block_rank_overrides: dict[str, int] | None = Field(
        default=None,
        description=(
            "Per-block rank overrides. Keys are block ranges (e.g. '0-9', '10-29'). "
            "Values are rank for those blocks. Unspecified blocks use the base rank. "
            "Example: {'0-9': 48, '30-39': 16} — more capacity early, less late."
        ),
    )
    use_mua_init: bool = Field(
        default=False,
        description=(
            "Enable muA (mu-parameterization aligned) LoRA initialization. "
            "Uses special init for the B matrix and automatically sets alpha = rank. "
            "Research-grade feature — improves scaling behavior at high ranks."
        ),
    )


# ─── Optimizer Config ───


class OptimizerConfig(BaseModel):
    """Optimizer settings for training."""

    type: str = Field(
        default=T2V_OPTIMIZER,
        description=(
            "Optimizer type. adamw8bit is the standard for Wan LoRA — "
            "cuts optimizer VRAM in half with negligible quality loss."
        ),
    )
    learning_rate: float = Field(
        default=T2V_LEARNING_RATE,
        gt=0.0,
        description=(
            "Peak learning rate. 5e-5 is a conservative default for Wan 2.2 T2V. "
            "Prodigy optimizer requires lr=1.0 (it sets the real LR automatically)."
        ),
    )
    weight_decay: float = Field(
        default=T2V_WEIGHT_DECAY,
        ge=0.0,
        description="Weight decay for regularization. 0.01 is the standard default.",
    )
    betas: list[float] = Field(
        default_factory=lambda: T2V_BETAS.copy(),
        description=(
            "Momentum parameters. [beta1, beta2] for Adam variants (length 2). "
            "[beta1, beta2, beta3] for CAME (length 3). "
            "Default [0.9, 0.999] is the PyTorch standard."
        ),
    )
    eps: float | list[float] = Field(
        default=T2V_EPS,
        description=(
            "Epsilon for numerical stability. Scalar for Adam variants. "
            "CAME can use a list of two epsilons [eps1, eps2]. "
            "Default 1e-8 is the PyTorch standard."
        ),
    )
    max_grad_norm: float | None = Field(
        default=T2V_MAX_GRAD_NORM,
        description=(
            "Maximum gradient norm for clipping. Prevents training instability "
            "from gradient spikes. null = no clipping. 1.0 is safe default."
        ),
    )
    optimizer_args: dict = Field(
        default_factory=dict,
        description=(
            "Extra keyword arguments passed directly to the optimizer constructor. "
            "Use for optimizer-specific settings not covered by the fields above. "
            "Example: {'amsgrad': true} for AMSGrad variant of Adam."
        ),
    )

    @field_validator("type", mode="before")
    @classmethod
    def validate_type(cls, v: str) -> str:
        """Validate optimizer type against supported options."""
        if v not in VALID_OPTIMIZERS:
            valid_list = ", ".join(sorted(VALID_OPTIMIZERS))
            raise ValueError(
                f"Unknown optimizer '{v}'. "
                f"Valid optimizers: {valid_list}."
            )
        return v

    @field_validator("betas", mode="before")
    @classmethod
    def validate_betas(cls, v: list[float]) -> list[float]:
        """Betas must be a list of 2-3 floats in [0, 1)."""
        if not isinstance(v, list):
            raise ValueError(
                f"betas must be a list, got {type(v).__name__}."
            )
        if len(v) < 2 or len(v) > 3:
            raise ValueError(
                f"betas must have 2 elements (Adam) or 3 elements (CAME), "
                f"got {len(v)}."
            )
        for i, beta in enumerate(v):
            if not (0.0 <= beta < 1.0):
                raise ValueError(
                    f"betas[{i}] = {beta} is out of range. "
                    f"Each beta must be >= 0.0 and < 1.0."
                )
        return v


# ─── Scheduler Config ───


class SchedulerConfig(BaseModel):
    """Learning rate scheduler settings.

    The scheduler controls how the learning rate changes over the course
    of training. cosine_with_min_lr is the default — it decays smoothly
    but keeps a floor so late-training detail refinement still gets updates.
    """

    type: str = Field(
        default=T2V_SCHEDULER,
        description=(
            "Scheduler type. cosine_with_min_lr is recommended — decays smoothly "
            "with a floor to prevent zero-LR stagnation in late training."
        ),
    )
    warmup_steps: int = Field(
        default=T2V_WARMUP_STEPS,
        ge=0,
        description=(
            "Steps of linear warmup from 0 to peak LR. 0 = no warmup. "
            "Wan LoRA typically needs no warmup."
        ),
    )
    min_lr: float | None = Field(
        default=None,
        description=(
            "Absolute minimum learning rate. Takes precedence over min_lr_ratio "
            "when set. null = use min_lr_ratio instead."
        ),
    )
    min_lr_ratio: float = Field(
        default=T2V_MIN_LR_RATIO,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum LR as fraction of peak LR (0.0-1.0). Used when min_lr is null. "
            "0.01 = LR floor at 1% of peak. Only applies to schedulers that decay."
        ),
    )
    rex_alpha: float = Field(
        default=0.1,
        description="Rex scheduler alpha parameter. Controls early-phase aggressiveness.",
    )
    rex_beta: float = Field(
        default=0.9,
        description="Rex scheduler beta parameter. Controls late-phase stability.",
    )

    @field_validator("type", mode="before")
    @classmethod
    def validate_type(cls, v: str) -> str:
        """Validate scheduler type against supported options."""
        if v not in VALID_SCHEDULERS:
            valid_list = ", ".join(sorted(VALID_SCHEDULERS))
            raise ValueError(
                f"Unknown scheduler '{v}'. "
                f"Valid schedulers: {valid_list}."
            )
        return v


# ─── MoE Expert Overrides ───


class MoeExpertOverrides(BaseModel):
    """Per-expert hyperparameter overrides for MoE differential training.

    All fields are optional — null means "inherit from the base config."
    This is the core Flimmer innovation: high-noise and low-noise experts
    need fundamentally different training regimes.

    High-noise expert: converges faster on coarse compositional features.
    Higher LR, fewer epochs.

    Low-noise expert: needs longer, gentler training for fine detail.
    Lower LR, more epochs. Overfits rapidly when trained aggressively.

    Note: rank and alpha are NOT overridable per-expert. LoRA rank
    determines matrix dimensions (A: d×r, B: r×d) and is locked at
    creation during the unified phase. Both experts share the same
    rank throughout training.
    """

    enabled: bool = Field(
        default=True,
        description=(
            "Whether to train this expert. Set to false to skip training "
            "for this expert entirely — useful for training only one expert."
        ),
    )
    learning_rate: float | None = Field(
        default=None,
        gt=0.0,
        description="Learning rate for this expert after fork. null = inherit from optimizer.learning_rate.",
    )
    dropout: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "LoRA dropout for this expert after fork. null = inherit from lora.dropout. "
            "Consider higher dropout for low-noise expert to prevent overfitting."
        ),
    )
    max_epochs: int | None = Field(
        default=None,
        gt=0,
        description="Max training epochs for this expert after fork. Required — each expert needs its own duration.",
    )
    fork_targets: list[str] | None = Field(
        default=None,
        description=(
            "Components to train for this expert after fork. null = same as "
            "training.unified_targets (keep training what you were training). "
            "Override to focus on specific components per expert."
        ),
    )
    block_targets: str | None = Field(
        default=None,
        description=(
            "Block targeting for this expert after fork. null = all blocks. "
            "Format: '0-11' or '0-11,25-34'. "
            "High-noise divergence was highest in early blocks (0-11)."
        ),
    )
    resume_from: str | None = Field(
        default=None,
        description=(
            "Path to an existing LoRA file as starting point for this expert. "
            "null = start from unified weights (or fresh if unified_epochs=0). "
            "Useful for continuing from a previous run."
        ),
    )
    batch_size: int | None = Field(
        default=None,
        gt=0,
        description="Batch size for this expert after fork. null = inherit from training.batch_size.",
    )
    gradient_accumulation_steps: int | None = Field(
        default=None,
        gt=0,
        description="Gradient accumulation for this expert. null = inherit from training.gradient_accumulation_steps.",
    )
    caption_dropout_rate: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Caption dropout for this expert. null = inherit from training.caption_dropout_rate.",
    )
    weight_decay: float | None = Field(
        default=None,
        ge=0.0,
        description=(
            "Weight decay for this expert. null = inherit from optimizer.weight_decay. "
            "Low-noise expert may benefit from higher weight decay to prevent overfitting."
        ),
    )
    min_lr_ratio: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Min LR ratio for this expert's scheduler. null = inherit from scheduler.min_lr_ratio.",
    )
    optimizer_type: str | None = Field(
        default=None,
        description="Optimizer type for this expert. null = inherit from optimizer.type.",
    )
    scheduler_type: str | None = Field(
        default=None,
        description="Scheduler type for this expert. null = inherit from scheduler.type.",
    )

    @field_validator("fork_targets", mode="before")
    @classmethod
    def validate_fork_targets(cls, v: list[str] | None) -> list[str] | None:
        """Validate per-expert fork targets if specified."""
        if v is not None:
            _validate_fork_targets(v)
        return v

    @field_validator("block_targets", mode="before")
    @classmethod
    def validate_block_targets(cls, v: str | None) -> str | None:
        """Validate per-expert block target format."""
        return _validate_block_targets(v)

    @field_validator("optimizer_type", mode="before")
    @classmethod
    def validate_optimizer_type(cls, v: str | None) -> str | None:
        """Validate per-expert optimizer if specified."""
        if v is not None and v not in VALID_OPTIMIZERS:
            valid_list = ", ".join(sorted(VALID_OPTIMIZERS))
            raise ValueError(
                f"Unknown optimizer '{v}'. Valid optimizers: {valid_list}."
            )
        return v

    @field_validator("scheduler_type", mode="before")
    @classmethod
    def validate_scheduler_type(cls, v: str | None) -> str | None:
        """Validate per-expert scheduler if specified."""
        if v is not None and v not in VALID_SCHEDULERS:
            valid_list = ", ".join(sorted(VALID_SCHEDULERS))
            raise ValueError(
                f"Unknown scheduler '{v}'. Valid schedulers: {valid_list}."
            )
        return v


# ─── MoE Config ───


class MoeConfig(BaseModel):
    """Container for all Mixture of Experts settings.

    MoE differential training is the core differentiator of Flimmer.
    Wan 2.2 uses a dual-expert design where different noise levels are
    handled by different experts — each needing different training treatment.

    Fork-and-specialize is the PRIMARY training strategy:
    1. Unified phase: both experts share one LoRA
    2. Fork: LoRA is copied, one per expert
    3. Expert phase: each copy trains independently with its own settings
    """

    enabled: bool = Field(
        default=T2V_IS_MOE,
        description=(
            "Enable MoE differential training. True by default for Wan 2.2 T2V "
            "(Wan 2.2 T2V is an MoE model). Set to False only if you want to "
            "treat both experts as one (not recommended)."
        ),
    )

    # ── Fork strategy ──

    fork_enabled: bool = Field(
        default=T2V_FORK_ENABLED,
        description=(
            "Enable fork-and-specialize. When False, trains a single unified "
            "LoRA on both experts — no forking, no per-expert specialization. "
            "Useful for simple training or baseline comparison."
        ),
    )
    expert_order: list[str] = Field(
        default_factory=lambda: ["low_noise", "high_noise"],
        description=(
            "Order in which experts are trained after fork. The shorter-training "
            "expert goes first so the longer-training expert can sample previews "
            "using a trained partner. Default: low_noise first (converges faster "
            "based on task vector analysis)."
        ),
    )

    preload_experts: bool = Field(
        default=False,
        description=(
            "Pre-load both expert weight sets into CPU RAM at startup for "
            "faster expert switching (~3s swap vs ~30s disk reload). "
            "Requires ~27GB extra CPU RAM for the inactive expert. "
            "Default: false (reload from disk each switch — slower but "
            "works on machines with limited RAM). Set true on cloud pods "
            "or machines with 64GB+ RAM."
        ),
    )

    # ── Per-expert overrides ──

    high_noise: MoeExpertOverrides = Field(
        default_factory=MoeExpertOverrides,
        description=(
            "Hyperparameter overrides for the high-noise expert. "
            "Handles early denoising — global composition and motion."
        ),
    )
    low_noise: MoeExpertOverrides = Field(
        default_factory=MoeExpertOverrides,
        description=(
            "Hyperparameter overrides for the low-noise expert. "
            "Handles late denoising — fine detail and texture."
        ),
    )

    # ── Routing ──

    boundary_ratio: float | None = Field(
        default=None,
        description=(
            "Override the model's default expert routing boundary. "
            "null = use model.boundary_ratio (from template or user config). "
            "Only set this if you want to experiment with different boundaries."
        ),
    )

    @field_validator("expert_order", mode="before")
    @classmethod
    def validate_expert_order(cls, v: list[str]) -> list[str]:
        """Validate expert order entries."""
        if not isinstance(v, list):
            raise ValueError(
                f"expert_order must be a list, got {type(v).__name__}."
            )
        valid_experts = {"high_noise", "low_noise"}
        for name in v:
            if name not in valid_experts:
                valid_list = ", ".join(sorted(valid_experts))
                raise ValueError(
                    f"Invalid expert name '{name}' in expert_order. "
                    f"Valid names: {valid_list}."
                )
        return v


# ─── Training Loop Config ───


class TrainingLoopConfig(BaseModel):
    """Unified training loop parameters.

    These control the unified phase — the shared foundation that both
    experts train on before fork. Epochs, batch size, precision, noise
    sampling, and unified-phase targeting all live here.

    Fork-specific settings (fork targets, per-expert overrides) are in
    MoeConfig.
    """

    unified_epochs: int = Field(
        default=T2V_UNIFIED_EPOCHS,
        ge=0,
        description=(
            "Epochs for the unified (shared) training phase before forking. "
            "Both experts train with the same LoRA during this phase. "
            "10 epochs is a starting hypothesis — needs experimental validation."
        ),
    )
    unified_targets: list[str] | None = Field(
        default=None,
        description=(
            "Component targeting during the unified phase. null = all standard "
            "LoRA targets (all attention + FFN). Override to train only specific "
            "components during unified."
        ),
    )
    unified_block_targets: str | None = Field(
        default=None,
        description=(
            "Block targeting during unified phase. null = all blocks. "
            "Format: '0-39' for all, '0-11' for early blocks only."
        ),
    )
    batch_size: int = Field(
        default=T2V_BATCH_SIZE,
        gt=0,
        description=(
            "Training batch size. Batch size 1 is the practical default for "
            "video LoRA training. Use gradient_accumulation_steps for "
            "effective larger batches."
        ),
    )
    gradient_accumulation_steps: int = Field(
        default=T2V_GRADIENT_ACCUMULATION,
        gt=0,
        description=(
            "Steps between optimizer updates. Simulates larger batch sizes "
            "without additional VRAM. Effective batch = batch_size * this value."
        ),
    )
    gradient_checkpointing: bool = Field(
        default=True,
        description=(
            "Recompute activations during backward pass to save VRAM. "
            "Always on for video training — the memory savings are essential."
        ),
    )
    mixed_precision: str = Field(
        default=T2V_MIXED_PRECISION,
        description=(
            "Mixed precision mode for training computation. "
            "bf16 is recommended for modern GPUs (Ampere+)."
        ),
    )
    base_model_precision: str = Field(
        default=T2V_BASE_MODEL_PRECISION,
        description=(
            "Precision for the frozen base model weights. bf16 is the quality-first "
            "default. fp8 uses 8-bit quantization (saves ~14GB for a 14B model). "
            "fp8_scaled uses 4-bit NF4 quantization (saves ~21GB, more quality loss). "
            "Quantized modes require bitsandbytes and a CUDA GPU."
        ),
    )
    caption_dropout_rate: float = Field(
        default=T2V_CAPTION_DROPOUT_RATE,
        ge=0.0,
        le=1.0,
        description=(
            "Probability of dropping the entire caption for a sample (0.0-1.0). "
            "Forces the model to rely on visual control signals. "
            "T2V uses 10% — lower than I2V because text is the only conditioning."
        ),
    )
    timestep_sampling: str = Field(
        default=T2V_TIMESTEP_SAMPLING,
        description=(
            "How to sample noise timesteps during training. 'shift' matches "
            "Wan's pretraining distribution."
        ),
    )
    discrete_flow_shift: float | None = Field(
        default=None,
        description=(
            "Override the flow shift parameter. null = use model default "
            "(from template or model.flow_shift). "
            "Only set this to experiment with different noise curves."
        ),
    )
    seed: int | None = Field(
        default=None,
        description="Random seed for reproducibility. null = random seed each run.",
    )
    max_train_steps: int | None = Field(
        default=None,
        gt=0,
        description=(
            "Stop training after this many total optimizer steps across all phases. "
            "Used for checkpoint resume testing — triggers a clean stop with "
            "checkpoint save mid-training. null = train to completion (default)."
        ),
    )
    resume_from: str | None = Field(
        default=None,
        description=(
            "Path to a checkpoint to resume training from. "
            "null = start fresh. Relative to config file location."
        ),
    )

    @field_validator("mixed_precision", mode="before")
    @classmethod
    def validate_mixed_precision(cls, v: str) -> str:
        """Validate mixed precision mode."""
        if v not in VALID_MIXED_PRECISION:
            valid_list = ", ".join(sorted(VALID_MIXED_PRECISION))
            raise ValueError(
                f"Invalid mixed_precision '{v}'. "
                f"Valid options: {valid_list}."
            )
        return v

    @field_validator("base_model_precision", mode="before")
    @classmethod
    def validate_base_model_precision(cls, v: str) -> str:
        """Validate base model precision."""
        if v not in VALID_BASE_PRECISION:
            valid_list = ", ".join(sorted(VALID_BASE_PRECISION))
            raise ValueError(
                f"Invalid base_model_precision '{v}'. "
                f"Valid options: {valid_list}."
            )
        return v

    @field_validator("timestep_sampling", mode="before")
    @classmethod
    def validate_timestep_sampling(cls, v: str) -> str:
        """Validate timestep sampling strategy."""
        if v not in VALID_TIMESTEP_SAMPLING:
            valid_list = ", ".join(sorted(VALID_TIMESTEP_SAMPLING))
            raise ValueError(
                f"Invalid timestep_sampling '{v}'. "
                f"Valid options: {valid_list}."
            )
        return v

    @field_validator("unified_block_targets", mode="before")
    @classmethod
    def validate_unified_block_targets(cls, v: str | None) -> str | None:
        """Validate unified block target format."""
        return _validate_block_targets(v)


# ─── Save Config ───


class SaveConfig(BaseModel):
    """Checkpoint saving settings.

    Controls where and how often checkpoints are saved. The output_dir
    is resolved relative to the config file's location.

    During fork-and-specialize, checkpoints are organized automatically::

        {output_dir}/
          unified/
            {name}_epoch005.safetensors
            {name}_epoch010.safetensors
          high_noise/
            {name}_high_epoch015.safetensors
          low_noise/
            {name}_low_epoch020.safetensors
    """

    output_dir: str = Field(
        default="./output",
        description=(
            "Directory for checkpoint output. Relative to config file location. "
            "Created automatically if it doesn't exist."
        ),
    )
    name: str = Field(
        default="flimmer_lora",
        description=(
            "Base name for checkpoint files. Final filename includes epoch number "
            "and phase: {name}_epoch{n}.safetensors (unified) or "
            "{name}_high_epoch{n}.safetensors / {name}_low_epoch{n}.safetensors (expert)."
        ),
    )
    save_every_n_epochs: int = Field(
        default=T2V_SAVE_EVERY_N_EPOCHS,
        gt=0,
        description="Save a checkpoint every N epochs. 5 is a good balance.",
    )
    save_last: bool = Field(
        default=True,
        description="Always save a checkpoint at the end of training.",
    )
    max_checkpoints: int | None = Field(
        default=None,
        gt=0,
        description=(
            "Maximum number of checkpoints to keep per phase. Oldest are deleted "
            "first. null = keep all checkpoints."
        ),
    )
    format: str = Field(
        default=T2V_CHECKPOINT_FORMAT,
        description=(
            "Checkpoint format. safetensors is the most widely supported "
            "(ComfyUI, A1111, diffusers). diffusers = HuggingFace directory format."
        ),
    )

    @field_validator("format", mode="before")
    @classmethod
    def validate_format(cls, v: str) -> str:
        """Validate checkpoint format."""
        if v not in VALID_CHECKPOINT_FORMATS:
            valid_list = ", ".join(sorted(VALID_CHECKPOINT_FORMATS))
            raise ValueError(
                f"Invalid checkpoint format '{v}'. "
                f"Valid formats: {valid_list}."
            )
        return v


# ─── Logging Config ───


class LoggingConfig(BaseModel):
    """Logging and experiment tracking settings.

    Multiple backends can be active simultaneously (e.g. console + wandb).
    Console logging is always recommended — it's the only backend that
    shows progress in your terminal.
    """

    backends: list[str] = Field(
        default=["console"],
        description=(
            "Active logging backends. console = terminal output. "
            "tensorboard = local log files. wandb = Weights & Biases cloud."
        ),
    )
    log_every_n_steps: int = Field(
        default=10,
        gt=0,
        description="Log metrics every N training steps.",
    )
    wandb_project: str | None = Field(
        default=None,
        description=(
            "Weights & Biases project name. Required if 'wandb' is in backends. "
            "Example: 'flimmer-training'"
        ),
    )
    wandb_entity: str | None = Field(
        default=None,
        description=(
            "Weights & Biases team or organization name. "
            "null = your personal account."
        ),
    )
    wandb_run_name: str | None = Field(
        default=None,
        description=(
            "Weights & Biases run name. null = auto-generated. "
            "Example: 'annika_r16_e50'"
        ),
    )
    wandb_group: str | None = Field(
        default=None,
        description=(
            "W&B run group for clustering related runs (e.g. same dataset, "
            "different hyperparameters). Runs in the same group appear "
            "together in the W&B dashboard."
        ),
    )
    wandb_tags: list[str] = Field(
        default_factory=list,
        description=(
            "W&B tags for filtering runs. Auto-populated with model variant "
            "and training mode if empty."
        ),
    )
    vram_sample_every_n_steps: int = Field(
        default=50,
        gt=0,
        description=(
            "Sample GPU VRAM usage every N training steps. Logged to W&B "
            "as system/vram_allocated_gb. Default 50 (~1 sample per minute)."
        ),
    )

    @field_validator("backends", mode="before")
    @classmethod
    def validate_backends(cls, v: list[str]) -> list[str]:
        """Validate all logging backends."""
        if not isinstance(v, list):
            raise ValueError(
                f"backends must be a list, got {type(v).__name__}."
            )
        invalid = set(v) - VALID_LOG_BACKENDS
        if invalid:
            valid_list = ", ".join(sorted(VALID_LOG_BACKENDS))
            invalid_list = ", ".join(sorted(invalid))
            raise ValueError(
                f"Invalid logging backend(s): {invalid_list}. "
                f"Valid backends: {valid_list}."
            )
        return v


# ─── Sampling Config ───


class SamplingConfig(BaseModel):
    """Sample video generation for monitoring training progress.

    OFF by default — generating sample videos requires a full inference
    pass per prompt per checkpoint, which can significantly slow down
    training. Enable explicitly when you want visual progress tracking.

    Samples are saved to {save.output_dir}/samples/ by default.

    Example YAML::

        sampling:
          enabled: true
          every_n_epochs: 5
          prompts:
            - "Annika walks through a sunlit garden"
            - "Annika looks into the camera with a playful smile"
          neg: "blurry, low quality, distorted"
          seed: 42
          walk_seed: true
          guidance_scale: 5.0
          sample_steps: 30
    """

    enabled: bool = Field(
        default=T2V_SAMPLING_ENABLED,
        description=(
            "Generate sample videos during training. OFF by default — "
            "inference passes significantly slow down training."
        ),
    )
    every_n_epochs: int = Field(
        default=T2V_SAMPLING_EVERY_N_EPOCHS,
        gt=0,
        description="Generate samples every N epochs.",
    )
    prompts: list[str] = Field(
        default_factory=lambda: T2V_SAMPLING_PROMPTS.copy(),
        description=(
            "Positive prompts for sample generation. One video per prompt. "
            "Empty list = no samples even if enabled=True."
        ),
    )
    neg: str = Field(
        default=T2V_SAMPLING_NEG,
        description=(
            "Negative prompt applied to all sample generations. "
            "Empty string = no negative conditioning."
        ),
    )
    seed: int = Field(
        default=T2V_SAMPLING_SEED,
        description="Base seed for sample generation.",
    )
    walk_seed: bool = Field(
        default=T2V_SAMPLING_WALK_SEED,
        description=(
            "Increment seed by 1 for each prompt in the list. "
            "Gives variety across prompts while keeping each reproducible."
        ),
    )
    sample_steps: int = Field(
        default=T2V_SAMPLING_STEPS,
        gt=0,
        description="Denoising steps for sample generation.",
    )
    guidance_scale: float = Field(
        default=T2V_SAMPLING_GUIDANCE,
        gt=0.0,
        description="Classifier-free guidance scale for sample generation.",
    )
    sample_dir: str | None = Field(
        default=T2V_SAMPLING_DIR,
        description=(
            "Directory for sample video output. null = {save.output_dir}/samples/. "
            "Relative to config file location."
        ),
    )
    lora_override: str | None = Field(
        default=None,
        description=(
            "Path to a specific LoRA checkpoint (.safetensors) to use for "
            "sample generation instead of the live training weights. "
            "Useful when you have a favorite checkpoint and want to see "
            "how prompts look with it. Relative to config file location."
        ),
    )
    skip_phases: list[str] = Field(
        default_factory=list,
        description=(
            "List of phase types to skip sampling during. "
            "Valid values: 'unified', 'high_noise', 'low_noise'. "
            "Example: ['unified'] to skip sampling during unified, "
            "sample during expert phases only."
        ),
    )

    @field_validator("skip_phases", mode="before")
    @classmethod
    def validate_skip_phases(cls, v: list[str]) -> list[str]:
        """Validate skip_phases entries."""
        if not isinstance(v, list):
            raise ValueError(
                f"skip_phases must be a list, got {type(v).__name__}."
            )
        valid_phases = {"full_noise", "unified", "high_noise", "low_noise"}
        for name in v:
            if name not in valid_phases:
                valid_list = ", ".join(sorted(valid_phases))
                raise ValueError(
                    f"Invalid phase name '{name}' in skip_phases. "
                    f"Valid names: {valid_list}."
                )
        return v


# ─── Cache Config ───


class CacheConfig(BaseModel):
    """Latent pre-encoding cache settings.

    Controls where and how encoded tensors (VAE latents, T5 text embeddings,
    reference encodings) are cached to disk. Pre-encoding separates the
    expensive encoding step from training — encode once, train many times.

    The cache directory is SEPARATE from the dataset directory. Cache files
    depend on the MODEL (which VAE, which text encoder), so the same dataset
    can have multiple caches for different models.

    Two-step caching: run 'cache-latents' (VAE) and 'cache-text' (T5)
    separately so they never compete for VRAM.
    """

    cache_dir: str = Field(
        default="./cache",
        description=(
            "Directory for cached encodings. Relative to config file location. "
            "Created automatically. Separate from dataset directory because "
            "cache depends on the model, not just the data."
        ),
    )
    dtype: str = Field(
        default="bf16",
        description=(
            "Tensor dtype for cached encodings. bf16 is recommended — "
            "matches training precision with minimal disk usage."
        ),
    )
    target_frames: list[int] = Field(
        default_factory=lambda: [17, 33, 49, 81],
        description=(
            "Frame counts to cache for each video sample. Must all be 4n+1 "
            "(Wan VAE temporal compression requirement). Each video produces "
            "one cached latent per frame count that fits within its duration. "
            "Default: [17, 33, 49, 81] (~1s to ~5s at 16fps). "
            "Use include_head_frame for single-frame image samples."
        ),
    )
    frame_extraction: str = Field(
        default="head",
        description=(
            "How to extract frames from videos. 'head' = first N frames "
            "(matches Wan pretraining). 'uniform' = evenly spaced across duration."
        ),
    )
    include_head_frame: bool = Field(
        default=False,
        description=(
            "Also extract a 1-frame image sample from each video. "
            "Useful for mixed image+video training."
        ),
    )
    reso_step: int = Field(
        default=16,
        gt=0,
        description=(
            "Pixel alignment step for resolution bucketing. 16 matches "
            "Wan VAE's spatial compression factor."
        ),
    )

    @field_validator("dtype", mode="before")
    @classmethod
    def validate_dtype(cls, v: str) -> str:
        """Validate cache dtype."""
        valid = {"bf16", "fp16", "fp32"}
        if v not in valid:
            valid_list = ", ".join(sorted(valid))
            raise ValueError(
                f"Invalid cache dtype '{v}'. Valid options: {valid_list}."
            )
        return v

    @field_validator("target_frames", mode="before")
    @classmethod
    def validate_target_frames(cls, v: list[int]) -> list[int]:
        """Validate that all target frame counts satisfy 4n+1."""
        if not isinstance(v, list):
            raise ValueError(
                f"target_frames must be a list, got {type(v).__name__}."
            )
        if not v:
            raise ValueError(
                "target_frames cannot be empty. "
                "Provide at least one frame count (e.g. [17, 33, 49, 81])."
            )
        for fc in v:
            if not isinstance(fc, int) or fc < 1:
                raise ValueError(
                    f"target_frames contains invalid value {fc}. "
                    f"All frame counts must be positive integers."
                )
            if (fc - 1) % 4 != 0:
                lower = ((fc - 1) // 4) * 4 + 1
                upper = lower + 4
                raise ValueError(
                    f"Frame count {fc} does not satisfy the 4n+1 constraint "
                    f"(required by Wan's 3D causal VAE). "
                    f"Nearest valid values: {lower} or {upper}."
                )
        return v

    @field_validator("frame_extraction", mode="before")
    @classmethod
    def validate_frame_extraction(cls, v: str) -> str:
        """Validate frame extraction method."""
        valid = {"head", "uniform"}
        if v not in valid:
            valid_list = ", ".join(sorted(valid))
            raise ValueError(
                f"Invalid frame_extraction '{v}'. Valid options: {valid_list}."
            )
        return v


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ROOT CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class FlimmerTrainingConfig(BaseModel):
    """Root config for a Flimmer training run.

    Minimum viable config: model variant + model path + data_config.
    Everything else has sensible defaults for fork-and-specialize training.

    Example minimal::

        model:
          variant: 2.2_t2v
          path: C:/path/to/Wan2.2-T2V-14B-Diffusers
        data_config: ./my_dataset/flimmer_data.yaml

    Example with per-expert overrides::

        model:
          variant: 2.2_t2v
          path: C:/path/to/Wan2.2-T2V-14B-Diffusers
          family: wan
        data_config: ./annika/flimmer_data.yaml
        training:
          unified_epochs: 10
        moe:
          high_noise:
            learning_rate: 1e-4
            max_epochs: 30
          low_noise:
            max_epochs: 50
        save:
          output_dir: ./output/annika_v1
          name: annika_lora
    """

    data_config: str = Field(
        description=(
            "Path to the Flimmer data config (flimmer_data.yaml). "
            "Relative to this config file's location."
        ),
    )
    model: ModelConfig = Field(
        description="Model selection and architecture settings.",
    )
    lora: LoraConfig = Field(
        default_factory=LoraConfig,
        description="Unified LoRA adapter settings (shared foundation).",
    )
    optimizer: OptimizerConfig = Field(
        default_factory=OptimizerConfig,
        description="Optimizer settings.",
    )
    scheduler: SchedulerConfig = Field(
        default_factory=SchedulerConfig,
        description="Learning rate scheduler settings.",
    )
    moe: MoeConfig = Field(
        default_factory=MoeConfig,
        description="Mixture of Experts settings (Wan 2.2 only).",
    )
    training: TrainingLoopConfig = Field(
        default_factory=TrainingLoopConfig,
        description="Training loop parameters.",
    )
    save: SaveConfig = Field(
        default_factory=SaveConfig,
        description="Checkpoint saving settings.",
    )
    logging: LoggingConfig = Field(
        default_factory=LoggingConfig,
        description="Logging and experiment tracking.",
    )
    sampling: SamplingConfig = Field(
        default_factory=SamplingConfig,
        description="Sample video generation settings.",
    )
    cache: CacheConfig = Field(
        default_factory=CacheConfig,
        description="Latent pre-encoding cache settings.",
    )

    @model_validator(mode="after")
    def check_moe_consistency(self) -> FlimmerTrainingConfig:
        """Error if MoE is enabled on a non-MoE model."""
        is_moe = self.model.is_moe
        if is_moe is False and self.moe.enabled:
            raise ValueError(
                "MoE is enabled but model.is_moe is false. "
                "Set moe.enabled to false, or set model.is_moe to true."
            )
        return self

    @model_validator(mode="after")
    def check_prodigy_lr(self) -> FlimmerTrainingConfig:
        """Prodigy optimizer requires lr=1.0 — it sets the real LR automatically.

        This is a hard requirement from the Prodigy algorithm. Using any other
        LR with Prodigy produces silently wrong results.
        """
        if self.optimizer.type == "prodigy" and self.optimizer.learning_rate != 1.0:
            raise ValueError(
                f"Prodigy optimizer requires learning_rate=1.0, "
                f"got {self.optimizer.learning_rate}. "
                f"Prodigy determines the optimal learning rate automatically — "
                f"setting it to anything other than 1.0 breaks this mechanism."
            )
        return self

    @model_validator(mode="after")
    def check_wandb_project(self) -> FlimmerTrainingConfig:
        """W&B backend requires a project name to know where to log."""
        if "wandb" in self.logging.backends and not self.logging.wandb_project:
            raise ValueError(
                "wandb is in logging.backends but logging.wandb_project is not set. "
                "Add a project name, e.g.:\n"
                "  logging:\n"
                "    backends: [console, wandb]\n"
                "    wandb_project: flimmer-training"
            )
        return self

    @model_validator(mode="after")
    def check_mua_alpha(self) -> FlimmerTrainingConfig:
        """When muA init is enabled, alpha must equal rank.

        muA-parameterization requires alpha=rank for correct scaling behavior.
        We auto-set this rather than erroring, since it's a deterministic fix.
        """
        if self.lora.use_mua_init and self.lora.alpha != self.lora.rank:
            self.lora.alpha = self.lora.rank
        return self

    @model_validator(mode="after")
    def check_fork_without_moe(self) -> FlimmerTrainingConfig:
        """Fork-and-specialize requires MoE to be enabled.

        Forking only makes sense when there are multiple experts to
        specialize. Without MoE, there's nothing to fork into.
        """
        if self.moe.fork_enabled and not self.moe.enabled:
            raise ValueError(
                "moe.fork_enabled is true but moe.enabled is false. "
                "Fork-and-specialize requires MoE differential training. "
                "Either enable MoE (moe.enabled: true) or disable fork "
                "(moe.fork_enabled: false)."
            )
        return self

    @model_validator(mode="after")
    def warn_aggressive_low_noise(self) -> FlimmerTrainingConfig:
        """Warn if low-noise expert LR is aggressively high.

        Low-noise experts overfit rapidly when trained too aggressively,
        producing washed-out artifacts and failing on dynamic expressions.
        This is a soft warning, not an error — the user might know what
        they're doing.
        """
        if self.moe.enabled and self.moe.low_noise.learning_rate is not None:
            if self.moe.low_noise.learning_rate > 2e-4:
                warnings.warn(
                    f"Low-noise expert learning_rate={self.moe.low_noise.learning_rate} "
                    f"is above 2e-4. Low-noise experts overfit rapidly with high "
                    f"learning rates — this can produce washed-out artifacts and "
                    f"failed dynamic expressions. Consider using 8e-5 to 1e-4.",
                    UserWarning,
                    stacklevel=2,
                )
        return self
