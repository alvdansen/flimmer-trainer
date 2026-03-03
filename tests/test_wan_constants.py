"""Tests for flimmer.training.wan.constants — pure-data architecture constants.

No torch, no fixtures, no mocking. These constants define the Wan model
architecture and never change at runtime, so tests are simple assertions
against known-correct values derived from the diffusers implementation.
"""

from flimmer.training.wan.constants import (
    FLIMMER_TO_MUSUBI_LORA_SUFFIX,
    FORK_TARGET_TO_MODULES,
    I2V_EXTRA_TARGETS,
    MUSUBI_PREFIX,
    MUSUBI_TO_FLIMMER_LORA_SUFFIX,
    T2V_LORA_TARGETS,
    WAN_EXPERT_SUBFOLDERS,
    WAN_I2V_IN_CHANNELS,
    WAN_I2V_REFERENCE_CHANNELS,
    WAN_NUM_BLOCKS,
    WAN_T2V_IN_CHANNELS,
    WAN_VAE_LATENT_CHANNELS,
    WAN_VAE_SPATIAL_COMPRESSION,
    WAN_VAE_TEMPORAL_COMPRESSION,
)


# ---------------------------------------------------------------------------
# T2V_LORA_TARGETS
# ---------------------------------------------------------------------------


class TestT2VLoraTargets:
    """T2V_LORA_TARGETS covers all standard transformer projections."""

    def test_exactly_ten_entries(self):
        assert len(T2V_LORA_TARGETS) == 10

    def test_all_entries_are_nonempty_strings(self):
        for entry in T2V_LORA_TARGETS:
            assert isinstance(entry, str)
            assert len(entry) > 0

    def test_includes_all_self_attention_projections(self):
        # attn1 = self-attention; all four projection matrices must be present
        assert "attn1.to_q" in T2V_LORA_TARGETS
        assert "attn1.to_k" in T2V_LORA_TARGETS
        assert "attn1.to_v" in T2V_LORA_TARGETS
        assert "attn1.to_out.0" in T2V_LORA_TARGETS

    def test_includes_all_cross_attention_projections(self):
        # attn2 = cross-attention for text conditioning
        assert "attn2.to_q" in T2V_LORA_TARGETS
        assert "attn2.to_k" in T2V_LORA_TARGETS
        assert "attn2.to_v" in T2V_LORA_TARGETS
        assert "attn2.to_out.0" in T2V_LORA_TARGETS

    def test_includes_ffn_projections(self):
        # GEGLU feed-forward: gate proj + output proj
        assert "ffn.net.0.proj" in T2V_LORA_TARGETS
        assert "ffn.net.2" in T2V_LORA_TARGETS

    def test_no_duplicate_entries(self):
        assert len(T2V_LORA_TARGETS) == len(set(T2V_LORA_TARGETS))


# ---------------------------------------------------------------------------
# I2V_EXTRA_TARGETS
# ---------------------------------------------------------------------------


class TestI2VExtraTargets:
    """I2V_EXTRA_TARGETS adds the reference-image cross-attention projections."""

    def test_exactly_two_entries(self):
        assert len(I2V_EXTRA_TARGETS) == 2

    def test_includes_add_k_proj(self):
        assert "attn2.add_k_proj" in I2V_EXTRA_TARGETS

    def test_includes_add_v_proj(self):
        assert "attn2.add_v_proj" in I2V_EXTRA_TARGETS

    def test_all_entries_are_nonempty_strings(self):
        for entry in I2V_EXTRA_TARGETS:
            assert isinstance(entry, str)
            assert len(entry) > 0

    def test_no_overlap_with_t2v_targets(self):
        # I2V extras are *additional* — they must not duplicate T2V base targets
        for entry in I2V_EXTRA_TARGETS:
            assert entry not in T2V_LORA_TARGETS


# ---------------------------------------------------------------------------
# Transformer architecture constants
# ---------------------------------------------------------------------------


class TestTransformerConstants:
    """Block count and VAE compression values from the Wan architecture."""

    def test_wan_num_blocks(self):
        assert WAN_NUM_BLOCKS == 40

    def test_vae_temporal_compression(self):
        # 81 frames → ~21 temporal tokens
        assert WAN_VAE_TEMPORAL_COMPRESSION == 4

    def test_vae_spatial_compression(self):
        # 480px → 60 latent per spatial dim
        assert WAN_VAE_SPATIAL_COMPRESSION == 8

    def test_vae_latent_channels(self):
        assert WAN_VAE_LATENT_CHANNELS == 16


# ---------------------------------------------------------------------------
# Input channel configurations
# ---------------------------------------------------------------------------


class TestInputChannels:
    """T2V and I2V differ by exactly 20 extra reference-image channels."""

    def test_t2v_in_channels(self):
        assert WAN_T2V_IN_CHANNELS == 16

    def test_i2v_in_channels(self):
        assert WAN_I2V_IN_CHANNELS == 36

    def test_i2v_equals_t2v_plus_reference_channels(self):
        # I2V = noisy latent (16) + VAE reference (16) + mask (4) = 36
        assert WAN_I2V_IN_CHANNELS == WAN_T2V_IN_CHANNELS + WAN_I2V_REFERENCE_CHANNELS

    def test_i2v_reference_channels(self):
        assert WAN_I2V_REFERENCE_CHANNELS == 20


# ---------------------------------------------------------------------------
# FORK_TARGET_TO_MODULES
# ---------------------------------------------------------------------------


class TestForkTargetToModules:
    """FORK_TARGET_TO_MODULES bridges user config names to diffusers suffixes."""

    def test_ffn_maps_to_two_modules(self):
        assert len(FORK_TARGET_TO_MODULES["ffn"]) == 2

    def test_self_attn_maps_to_four_modules(self):
        assert len(FORK_TARGET_TO_MODULES["self_attn"]) == 4

    def test_cross_attn_maps_to_four_modules(self):
        assert len(FORK_TARGET_TO_MODULES["cross_attn"]) == 4

    def test_projection_level_targets_map_to_one_module(self):
        # Every dot-qualified key is a projection-level target → single module
        projection_level_keys = [k for k in FORK_TARGET_TO_MODULES if "." in k]
        for key in projection_level_keys:
            assert len(FORK_TARGET_TO_MODULES[key]) == 1, (
                f"Projection-level key '{key}' should map to exactly 1 module, "
                f"got {FORK_TARGET_TO_MODULES[key]}"
            )

    def test_all_mapped_modules_are_known_targets(self):
        # Every module listed in the dict must exist in T2V_LORA_TARGETS or I2V_EXTRA_TARGETS
        all_known = set(T2V_LORA_TARGETS) | set(I2V_EXTRA_TARGETS)
        for key, modules in FORK_TARGET_TO_MODULES.items():
            for module in modules:
                assert module in all_known, (
                    f"Fork target '{key}' maps to unknown module '{module}'"
                )


# ---------------------------------------------------------------------------
# WAN_EXPERT_SUBFOLDERS
# ---------------------------------------------------------------------------


class TestExpertSubfolders:
    """Expert names map to diffusers-format subfolder paths."""

    def test_high_noise_maps_to_transformer(self):
        assert WAN_EXPERT_SUBFOLDERS["high_noise"] == "transformer"

    def test_low_noise_maps_to_transformer_2(self):
        assert WAN_EXPERT_SUBFOLDERS["low_noise"] == "transformer_2"

    def test_exactly_two_experts(self):
        assert len(WAN_EXPERT_SUBFOLDERS) == 2


# ---------------------------------------------------------------------------
# Musubi checkpoint key conversion
# ---------------------------------------------------------------------------


class TestMusubiKeyConversion:
    """FLIMMER_TO_MUSUBI and MUSUBI_TO_FLIMMER suffix dicts are exact inverses."""

    def test_musubi_prefix_value(self):
        assert MUSUBI_PREFIX == "lora_unet_"

    def test_flimmer_to_musubi_lora_a(self):
        assert FLIMMER_TO_MUSUBI_LORA_SUFFIX["lora_A.weight"] == "lora_down.weight"

    def test_flimmer_to_musubi_lora_b(self):
        assert FLIMMER_TO_MUSUBI_LORA_SUFFIX["lora_B.weight"] == "lora_up.weight"

    def test_musubi_to_flimmer_lora_down(self):
        assert MUSUBI_TO_FLIMMER_LORA_SUFFIX["lora_down.weight"] == "lora_A.weight"

    def test_musubi_to_flimmer_lora_up(self):
        assert MUSUBI_TO_FLIMMER_LORA_SUFFIX["lora_up.weight"] == "lora_B.weight"

    def test_dicts_are_inverses_of_each_other(self):
        # Round-trip: flimmer → musubi → flimmer must recover the original key
        for flimmer_key, musubi_key in FLIMMER_TO_MUSUBI_LORA_SUFFIX.items():
            assert MUSUBI_TO_FLIMMER_LORA_SUFFIX[musubi_key] == flimmer_key

    def test_same_number_of_entries(self):
        assert len(FLIMMER_TO_MUSUBI_LORA_SUFFIX) == len(MUSUBI_TO_FLIMMER_LORA_SUFFIX)
