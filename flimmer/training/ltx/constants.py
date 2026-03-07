"""LTX-2.3 architecture constants — pure data, no torch dependency.

Defines transformer dimensions, VAE compression ratios, input channels,
and LoRA target patterns for the LTX-2.3 model family.

These constants are the source of truth for:
    - registry.py: variant definitions
    - backend.py: architecture-aware model loading and noise schedule
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Transformer architecture
# ---------------------------------------------------------------------------

LTX_NUM_BLOCKS: int = 48
"""Number of transformer blocks in LTX-2.3."""

LTX_HIDDEN_DIM: int = 4096
"""Hidden dimension of the LTX-2.3 transformer."""

LTX_NUM_HEADS: int = 32
"""Number of attention heads in the LTX-2.3 transformer."""

LTX_HEAD_DIM: int = 128
"""Dimension per attention head (hidden_dim / num_heads)."""

LTX_TEXT_DIM: int = 3072
"""Dimension of the Gemma 3-12B text encoder output."""


# ---------------------------------------------------------------------------
# VAE compression ratios
# ---------------------------------------------------------------------------

LTX_VAE_TEMPORAL_COMPRESSION: int = 8
"""Temporal compression factor of the LTX video VAE.
e.g. 33 frames → (33-1)/8 + 1 = 5 temporal tokens in latent space."""

LTX_VAE_SPATIAL_COMPRESSION: int = 32
"""Spatial compression factor per dimension.
e.g. 512px → 16 latent tokens."""

LTX_VAE_LATENT_CHANNELS: int = 128
"""Number of channels in the video VAE latent space."""

LTX_AUDIO_VAE_CHANNELS: int = 8
"""Number of channels in the audio VAE latent space."""

LTX_AUDIO_MEL_BINS: int = 16
"""Number of mel frequency bins for audio encoding."""


# ---------------------------------------------------------------------------
# Input channel configurations
# ---------------------------------------------------------------------------

LTX_T2V_IN_CHANNELS: int = 128
"""T2V input channels: the video VAE latent (128 channels)."""


# ---------------------------------------------------------------------------
# Frame constraint
# ---------------------------------------------------------------------------

LTX_FRAME_MODULUS: int = 8
"""Frame count constraint: (frames - 1) must be divisible by 8.
Valid frame counts: 1, 9, 17, 25, 33, 41, 49, ..."""


# ---------------------------------------------------------------------------
# LoRA target modules
# ---------------------------------------------------------------------------

# These are the module name suffixes that PEFT uses for target_modules.
# They match the actual nn.Module names inside the LTX transformer.
# The full path is like: blocks.0.attn1.to_q — PEFT matches on the suffix.

LTX_VIDEO_LORA_TARGETS: list[str] = [
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
]
"""Standard LoRA targets for LTX-2.3 video-only training.

8 module patterns per block × 48 blocks = 384 total LoRA modules.
Video-only MVP — FFN and audio projections excluded for now.

Why these specific modules:
- Self-attention (attn1): controls temporal coherence, motion quality,
  spatial composition.
- Cross-attention (attn2): controls text-to-video alignment.
"""

LTX_AUDIO_LORA_TARGETS: list[str] = [
    # Audio self-attention
    "audio_attn1.to_q",
    "audio_attn1.to_k",
    "audio_attn1.to_v",
    "audio_attn1.to_out.0",
    # Audio cross-attention
    "audio_attn2.to_q",
    "audio_attn2.to_k",
    "audio_attn2.to_v",
    "audio_attn2.to_out.0",
]
"""Audio LoRA targets for future audio+video training.

Not used in the video-only MVP. Reserved for audio LoRA support.
"""


# ---------------------------------------------------------------------------
# Sequence length computation
# ---------------------------------------------------------------------------

def compute_sequence_length(
    height: int,
    width: int,
    frames: int,
) -> int:
    """Compute the latent sequence length for LTX-2.3.

    The sequence length determines the noise schedule shift parameter.
    LTX-2.3 uses spatial compression of 32x and temporal compression of 8x.

    Args:
        height: Video height in pixels.
        width: Video width in pixels.
        frames: Number of frames (must satisfy frames % 8 == 1).

    Returns:
        Total number of latent tokens (spatial × temporal).
    """
    spatial_h = height // LTX_VAE_SPATIAL_COMPRESSION
    spatial_w = width // LTX_VAE_SPATIAL_COMPRESSION
    temporal = (frames - 1) // LTX_VAE_TEMPORAL_COMPRESSION + 1
    return spatial_h * spatial_w * temporal
