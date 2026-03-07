"""Tests for flimmer.training.ltx.constants — architecture constants and helpers.

This module is GPU-free: constants.py contains only pure data and a simple
arithmetic helper function.

Coverage:
    - Transformer architecture constants (blocks, dims, heads)
    - VAE compression ratios and channel counts
    - Frame constraint modulus
    - LoRA target lists: non-empty, correct count, expected suffixes
    - Sequence length computation helper
"""

from __future__ import annotations

from flimmer.training.ltx.constants import (
    LTX_AUDIO_LORA_TARGETS,
    LTX_AUDIO_MEL_BINS,
    LTX_AUDIO_VAE_CHANNELS,
    LTX_FRAME_MODULUS,
    LTX_HEAD_DIM,
    LTX_HIDDEN_DIM,
    LTX_NUM_BLOCKS,
    LTX_NUM_HEADS,
    LTX_T2V_IN_CHANNELS,
    LTX_TEXT_DIM,
    LTX_VAE_LATENT_CHANNELS,
    LTX_VAE_SPATIAL_COMPRESSION,
    LTX_VAE_TEMPORAL_COMPRESSION,
    LTX_VIDEO_LORA_TARGETS,
    compute_sequence_length,
)


# ---------------------------------------------------------------------------
# Transformer architecture constants
# ---------------------------------------------------------------------------

class TestTransformerConstants:
    """LTX-2.3 transformer architecture values."""

    def test_num_blocks(self):
        assert LTX_NUM_BLOCKS == 48

    def test_hidden_dim(self):
        assert LTX_HIDDEN_DIM == 4096

    def test_num_heads(self):
        assert LTX_NUM_HEADS == 32

    def test_head_dim(self):
        assert LTX_HEAD_DIM == 128

    def test_head_dim_matches_hidden_div_heads(self):
        """head_dim = hidden_dim / num_heads."""
        assert LTX_HEAD_DIM == LTX_HIDDEN_DIM // LTX_NUM_HEADS

    def test_text_dim(self):
        """Gemma 3-12B text encoder output dimension."""
        assert LTX_TEXT_DIM == 3072


# ---------------------------------------------------------------------------
# VAE constants
# ---------------------------------------------------------------------------

class TestVaeConstants:
    """Video and audio VAE compression and channel counts."""

    def test_temporal_compression(self):
        assert LTX_VAE_TEMPORAL_COMPRESSION == 8

    def test_spatial_compression(self):
        assert LTX_VAE_SPATIAL_COMPRESSION == 32

    def test_latent_channels(self):
        assert LTX_VAE_LATENT_CHANNELS == 128

    def test_audio_vae_channels(self):
        assert LTX_AUDIO_VAE_CHANNELS == 8

    def test_audio_mel_bins(self):
        assert LTX_AUDIO_MEL_BINS == 16

    def test_t2v_in_channels_matches_vae_channels(self):
        """T2V input channels equal VAE latent channels."""
        assert LTX_T2V_IN_CHANNELS == LTX_VAE_LATENT_CHANNELS


# ---------------------------------------------------------------------------
# Frame constraint
# ---------------------------------------------------------------------------

class TestFrameConstraint:
    """Frame modulus constraint: (frames - 1) % modulus == 0."""

    def test_frame_modulus(self):
        assert LTX_FRAME_MODULUS == 8


# ---------------------------------------------------------------------------
# LoRA targets
# ---------------------------------------------------------------------------

class TestVideoLoraTargets:
    """Video LoRA target module patterns."""

    def test_is_list(self):
        assert isinstance(LTX_VIDEO_LORA_TARGETS, list)

    def test_has_eight_targets(self):
        """8 targets: 4 self-attn + 4 cross-attn."""
        assert len(LTX_VIDEO_LORA_TARGETS) == 8

    def test_contains_self_attention_projections(self):
        assert "attn1.to_q" in LTX_VIDEO_LORA_TARGETS
        assert "attn1.to_k" in LTX_VIDEO_LORA_TARGETS
        assert "attn1.to_v" in LTX_VIDEO_LORA_TARGETS
        assert "attn1.to_out.0" in LTX_VIDEO_LORA_TARGETS

    def test_contains_cross_attention_projections(self):
        assert "attn2.to_q" in LTX_VIDEO_LORA_TARGETS
        assert "attn2.to_k" in LTX_VIDEO_LORA_TARGETS
        assert "attn2.to_v" in LTX_VIDEO_LORA_TARGETS
        assert "attn2.to_out.0" in LTX_VIDEO_LORA_TARGETS

    def test_no_audio_targets_mixed_in(self):
        """Video targets must not include any audio attention modules."""
        for target in LTX_VIDEO_LORA_TARGETS:
            assert "audio" not in target


class TestAudioLoraTargets:
    """Audio LoRA target module patterns (reserved for future use)."""

    def test_is_list(self):
        assert isinstance(LTX_AUDIO_LORA_TARGETS, list)

    def test_has_eight_targets(self):
        assert len(LTX_AUDIO_LORA_TARGETS) == 8

    def test_all_targets_contain_audio(self):
        """All audio targets must reference audio attention modules."""
        for target in LTX_AUDIO_LORA_TARGETS:
            assert "audio" in target

    def test_no_overlap_with_video_targets(self):
        """Audio and video targets must be disjoint."""
        overlap = set(LTX_VIDEO_LORA_TARGETS) & set(LTX_AUDIO_LORA_TARGETS)
        assert len(overlap) == 0


# ---------------------------------------------------------------------------
# Sequence length computation
# ---------------------------------------------------------------------------

class TestComputeSequenceLength:
    """compute_sequence_length() arithmetic correctness."""

    def test_basic_512x512x33(self):
        """512x512 at 33 frames.
        spatial: 512/32 = 16, temporal: (33-1)/8 + 1 = 5
        total: 16 * 16 * 5 = 1280
        """
        assert compute_sequence_length(512, 512, 33) == 1280

    def test_single_frame(self):
        """Single frame: temporal = (1-1)/8 + 1 = 1."""
        result = compute_sequence_length(256, 256, 1)
        spatial = (256 // 32) ** 2  # 8 * 8 = 64
        assert result == spatial * 1

    def test_1024x1024x17(self):
        """1024x1024 at 17 frames.
        spatial: 1024/32 = 32, temporal: (17-1)/8 + 1 = 3
        total: 32 * 32 * 3 = 3072
        """
        assert compute_sequence_length(1024, 1024, 17) == 3072

    def test_non_square(self):
        """Non-square: 768x512 at 9 frames.
        spatial_h: 768/32 = 24, spatial_w: 512/32 = 16
        temporal: (9-1)/8 + 1 = 2
        total: 24 * 16 * 2 = 768
        """
        assert compute_sequence_length(768, 512, 9) == 768

    def test_result_is_int(self):
        result = compute_sequence_length(512, 512, 33)
        assert isinstance(result, int)

    def test_49_frames(self):
        """49 frames: temporal = (49-1)/8 + 1 = 7."""
        result = compute_sequence_length(512, 512, 49)
        spatial = (512 // 32) ** 2  # 16 * 16 = 256
        assert result == spatial * 7
