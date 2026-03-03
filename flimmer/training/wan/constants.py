"""Wan architecture constants — pure data, no torch dependency.

Defines which transformer modules get LoRA adapters, block counts,
channel dimensions, and VAE compression ratios for all Wan variants.

These constants are the source of truth for:
    - modules.py: resolving target modules for PEFT
    - checkpoint_io.py: validating state dict keys
    - backend.py: architecture-aware model loading

Why separate from the training config:
    The training config (wan22_training_master.py) defines USER-FACING
    settings. These constants define ARCHITECTURE — they come from the
    model itself and never change regardless of training settings.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Transformer architecture
# ---------------------------------------------------------------------------

WAN_NUM_BLOCKS: int = 40
"""Number of transformer blocks in all Wan models (2.1 and 2.2)."""

WAN_HIDDEN_DIM: int = 5120
"""Hidden dimension of the Wan 14B transformer."""

WAN_NUM_HEADS: int = 40
"""Number of attention heads in the Wan 14B transformer."""

WAN_HEAD_DIM: int = 128
"""Dimension per attention head (hidden_dim / num_heads)."""

WAN_T5_DIM: int = 4096
"""Dimension of the T5 text encoder output (UMT5-XXL)."""


# ---------------------------------------------------------------------------
# VAE compression ratios
# ---------------------------------------------------------------------------

WAN_VAE_TEMPORAL_COMPRESSION: int = 4
"""Temporal compression factor of Wan-VAE (3D causal VAE).
81 frames → ceil(81/4) = 21 temporal tokens in latent space."""

WAN_VAE_SPATIAL_COMPRESSION: int = 8
"""Spatial compression factor per dimension.
480px → 60 latent, 848px → 106 latent."""

WAN_VAE_LATENT_CHANNELS: int = 16
"""Number of channels in the VAE latent space."""


# ---------------------------------------------------------------------------
# Input channel configurations
# ---------------------------------------------------------------------------

WAN_T2V_IN_CHANNELS: int = 16
"""T2V input channels: just the noisy latent (16 channels)."""

WAN_I2V_IN_CHANNELS: int = 36
"""I2V input channels: noisy latent (16) + VAE-encoded reference image (16)
+ binary mask (4) = 36 total. The reference image is channel-concatenated
with the noisy latents before entering the transformer."""

WAN_I2V_REFERENCE_CHANNELS: int = 20
"""Extra channels for I2V conditioning (36 - 16 = 20).
These carry the VAE-encoded reference image and mask."""


# ---------------------------------------------------------------------------
# LoRA target modules
# ---------------------------------------------------------------------------

# These are the module name suffixes that PEFT uses for target_modules.
# They match the actual nn.Module names inside WanTransformer3DModel.
# The full path is like: blocks.0.attn1.to_q — PEFT matches on the suffix.

T2V_LORA_TARGETS: list[str] = [
    # Self-attention (attn1) — within-video temporal/spatial relationships
    "attn1.to_q",
    "attn1.to_k",
    "attn1.to_v",
    "attn1.to_out.0",
    # Cross-attention (attn2) — text conditioning injection
    "attn2.to_q",
    "attn2.to_k",
    "attn2.to_v",
    "attn2.to_out.0",
    # Feed-forward network (ffn)
    "ffn.net.0.proj",   # gate projection (SiLU gate in GEGLU)
    "ffn.net.2",        # output projection
]
"""Standard LoRA targets for T2V models.

10 module patterns per block × 40 blocks = 400 total LoRA modules.
These match diffusers WanTransformer3DModel naming conventions.

Why these specific modules:
- Self-attention (attn1): controls temporal coherence, motion quality,
  spatial composition. The MOST changed layers between experts.
- Cross-attention (attn2): controls text-to-video alignment.
  Changes less between experts (text conditioning is shared).
- FFN: controls feature transformation capacity. High divergence
  between experts (0.872-0.894 cosine in T2V analysis).
"""

I2V_EXTRA_TARGETS: list[str] = [
    # Additional cross-attention projections for I2V reference image
    "attn2.add_k_proj",
    "attn2.add_v_proj",
]
"""Additional LoRA targets for I2V models.

I2V models have extra key/value projections in cross-attention that
process the reference image. These are separate from the text K/V
projections and handle the image-to-video conditioning.

Total I2V targets: 10 (T2V base) + 2 (I2V extra) = 12 per block.
"""


# ---------------------------------------------------------------------------
# Fork target mapping
# ---------------------------------------------------------------------------

# Maps abstract fork target names (from user config) to the actual
# module suffixes used in diffusers. This is the bridge between
# the user-facing config vocabulary (VALID_FORK_TARGETS in the master)
# and the model-specific module names.

FORK_TARGET_TO_MODULES: dict[str, list[str]] = {
    # Component-level targets → all projections in that component
    "ffn": ["ffn.net.0.proj", "ffn.net.2"],
    "self_attn": ["attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0"],
    "cross_attn": ["attn2.to_q", "attn2.to_k", "attn2.to_v", "attn2.to_out.0"],

    # Projection-level targets → single projection
    "ffn.up_proj": ["ffn.net.0.proj"],
    "ffn.down_proj": ["ffn.net.2"],
    "self_attn.to_q": ["attn1.to_q"],
    "self_attn.to_k": ["attn1.to_k"],
    "self_attn.to_v": ["attn1.to_v"],
    "self_attn.to_out": ["attn1.to_out.0"],
    "cross_attn.to_q": ["attn2.to_q"],
    "cross_attn.to_k": ["attn2.to_k"],
    "cross_attn.to_v": ["attn2.to_v"],
    "cross_attn.to_out": ["attn2.to_out.0"],
}
"""Maps user-facing fork target names to diffusers module suffixes.

Used by modules.resolve_target_modules() to convert the user's fork_targets
list into the actual PEFT target_modules list.

Example:
    User writes: fork_targets: ["ffn", "self_attn", "cross_attn.to_v"]
    Resolves to: ["ffn.net.0.proj", "ffn.net.2",
                  "attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0",
                  "attn2.to_v"]
"""


# ---------------------------------------------------------------------------
# Expert model subfolder conventions
# ---------------------------------------------------------------------------

# In Wan 2.2 diffusers format, the two experts are stored in separate
# subdirectories under the model root. The "transformer" subfolder
# contains the first expert, and "transformer_2" contains the second.

WAN_EXPERT_SUBFOLDERS: dict[str, str] = {
    "high_noise": "transformer",
    "low_noise": "transformer_2",
}
"""Maps expert names to diffusers subfolder paths.

For Wan 2.2 MoE models, each expert is a separate WanTransformer3DModel
stored in its own subdirectory. The training loop loads one at a time.

For Wan 2.1 (single transformer), only "transformer" exists — there's
no expert switching.
"""

WAN_SINGLE_SUBFOLDER: str = "transformer"
"""Subfolder for single-expert models (Wan 2.1)."""


# ---------------------------------------------------------------------------
# Musubi format key conversion
# ---------------------------------------------------------------------------

# Musubi-tuner (kohya/sd-scripts lineage) uses a different naming convention
# for LoRA state dict keys. Understanding the conversion is essential for
# checkpoint compatibility.

MUSUBI_PREFIX: str = "lora_unet_"
"""Prefix for all LoRA keys in musubi/kohya format."""

# Musubi key structure: lora_unet_blocks_0_attn1_to_q.lora_down.weight
# Flimmer key structure: blocks.0.attn1.to_q.lora_A.weight
#
# Differences:
#   1. Musubi prefixes with "lora_unet_"
#   2. Musubi uses underscores where diffusers uses dots (for the module path)
#   3. Musubi uses "lora_down"/"lora_up" where PEFT uses "lora_A"/"lora_B"
#
# The separator between module path and LoRA suffix is always a dot in both.

FLIMMER_TO_MUSUBI_LORA_SUFFIX: dict[str, str] = {
    "lora_A.weight": "lora_down.weight",
    "lora_B.weight": "lora_up.weight",
}
"""Maps PEFT/diffusers LoRA suffixes to musubi/kohya suffixes."""

MUSUBI_TO_FLIMMER_LORA_SUFFIX: dict[str, str] = {
    "lora_down.weight": "lora_A.weight",
    "lora_up.weight": "lora_B.weight",
}
"""Maps musubi/kohya LoRA suffixes to PEFT/diffusers suffixes."""
