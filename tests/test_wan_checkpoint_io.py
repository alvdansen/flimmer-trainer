"""Tests for flimmer.training.wan.checkpoint_io — LoRA format conversion.

Covers public functions:
    flimmer_to_musubi(state_dict)    — PEFT/diffusers → musubi/kohya key format
    musubi_to_flimmer(state_dict)    — musubi/kohya → PEFT/diffusers key format
    validate_state_dict_keys(...)    — structural validation of flimmer-format keys
    add_diffusers_prefix(...)        — add transformer. prefix for diffusers loading
    strip_diffusers_prefix(...)      — remove transformer. prefix for training use
    has_diffusers_prefix(...)        — detect whether keys have component prefixes

All tests are GPU-free. Tensor values are represented by simple numpy arrays
or plain Python objects — the conversion functions never touch values, only keys.
"""

from __future__ import annotations

import numpy as np
import pytest

from flimmer.training.wan.checkpoint_io import (
    add_diffusers_prefix,
    flimmer_to_musubi,
    has_diffusers_prefix,
    musubi_to_flimmer,
    strip_diffusers_prefix,
    validate_state_dict_keys,
)


# ---------------------------------------------------------------------------
# Test helper
# ---------------------------------------------------------------------------

def _make_state_dict(
    blocks: tuple[int, ...] = (0,),
    targets: tuple[str, ...] = ("attn1.to_q",),
    rank: int = 16,
) -> dict[str, np.ndarray]:
    """Build a minimal valid flimmer-format state dict for testing.

    Each (block, target) pair produces two keys: lora_A and lora_B.
    Shapes follow the standard convention:
        lora_A: (rank, in_features)   — the down-projection
        lora_B: (out_features, rank)  — the up-projection

    Args:
        blocks:  Block indices to generate keys for.
        targets: Module suffixes (e.g. "attn1.to_q", "ffn.net.2").
        rank:    LoRA rank — sets the contraction dimension of both matrices.

    Returns:
        Dict mapping flimmer key strings to numpy zero arrays.
    """
    sd: dict[str, np.ndarray] = {}
    for b in blocks:
        for t in targets:
            sd[f"blocks.{b}.{t}.lora_A.weight"] = np.zeros((rank, 128))
            sd[f"blocks.{b}.{t}.lora_B.weight"] = np.zeros((128, rank))
    return sd


# ---------------------------------------------------------------------------
# flimmer_to_musubi — forward conversion
# ---------------------------------------------------------------------------

class TestFlimmerToMusubi:
    """flimmer_to_musubi converts PEFT/diffusers keys to musubi/kohya format."""

    def test_attn_lora_A_becomes_lora_down(self):
        """lora_A suffix → lora_down suffix, with musubi prefix and underscored path."""
        sd = {"blocks.0.attn1.to_q.lora_A.weight": object()}
        result = flimmer_to_musubi(sd)
        assert "lora_unet_blocks_0_attn1_to_q.lora_down.weight" in result

    def test_attn_lora_B_becomes_lora_up(self):
        """lora_B suffix → lora_up suffix."""
        sd = {"blocks.0.attn1.to_q.lora_B.weight": object()}
        result = flimmer_to_musubi(sd)
        assert "lora_unet_blocks_0_attn1_to_q.lora_up.weight" in result

    def test_ffn_net_0_proj_key(self):
        """FFN gate projection path converts correctly.

        'ffn.net.0.proj' has dots between numeric and named segments;
        all of them should become underscores in the musubi module path.
        """
        sd = {"blocks.3.ffn.net.0.proj.lora_A.weight": object()}
        result = flimmer_to_musubi(sd)
        assert "lora_unet_blocks_3_ffn_net_0_proj.lora_down.weight" in result

    def test_ffn_net_2_key(self):
        """FFN output projection (net.2) converts correctly."""
        sd = {"blocks.7.ffn.net.2.lora_B.weight": object()}
        result = flimmer_to_musubi(sd)
        assert "lora_unet_blocks_7_ffn_net_2.lora_up.weight" in result

    def test_multiple_keys_at_once(self):
        """A full pair (A + B) converts in a single call, both keys present."""
        sd = _make_state_dict(blocks=(0,), targets=("attn1.to_q",), rank=16)
        result = flimmer_to_musubi(sd)
        assert len(result) == 2
        assert "lora_unet_blocks_0_attn1_to_q.lora_down.weight" in result
        assert "lora_unet_blocks_0_attn1_to_q.lora_up.weight" in result

    def test_empty_state_dict(self):
        """Empty input produces empty output without raising."""
        result = flimmer_to_musubi({})
        assert result == {}

    def test_tensor_values_are_preserved(self):
        """Tensor objects pass through unchanged — only keys are rewritten."""
        sentinel = object()
        sd = {"blocks.0.attn1.to_q.lora_A.weight": sentinel}
        result = flimmer_to_musubi(sd)
        converted_key = "lora_unet_blocks_0_attn1_to_q.lora_down.weight"
        assert result[converted_key] is sentinel

    def test_numpy_array_values_are_preserved(self):
        """Numpy arrays are not copied or modified during key conversion."""
        arr = np.ones((16, 128), dtype=np.float32)
        sd = {"blocks.0.attn1.to_q.lora_A.weight": arr}
        result = flimmer_to_musubi(sd)
        converted_key = "lora_unet_blocks_0_attn1_to_q.lora_down.weight"
        assert result[converted_key] is arr


# ---------------------------------------------------------------------------
# musubi_to_flimmer — reverse conversion
# ---------------------------------------------------------------------------

class TestMusubiToFlimmer:
    """musubi_to_flimmer converts musubi/kohya keys back to PEFT/diffusers format."""

    def test_lora_down_becomes_lora_A(self):
        """lora_down suffix → lora_A suffix, musubi prefix removed, underscores → dots."""
        sd = {"lora_unet_blocks_0_attn1_to_q.lora_down.weight": object()}
        result = musubi_to_flimmer(sd)
        assert "blocks.0.attn1.to_q.lora_A.weight" in result

    def test_lora_up_becomes_lora_B(self):
        """lora_up suffix → lora_B suffix."""
        sd = {"lora_unet_blocks_0_attn1_to_q.lora_up.weight": object()}
        result = musubi_to_flimmer(sd)
        assert "blocks.0.attn1.to_q.lora_B.weight" in result

    def test_ffn_net_0_proj_reverse(self):
        """FFN gate projection reverses correctly.

        The numeric segment '0' in 'net.0.proj' must become a dot-separated
        segment, not collapse into 'net0proj'.
        """
        sd = {"lora_unet_blocks_3_ffn_net_0_proj.lora_down.weight": object()}
        result = musubi_to_flimmer(sd)
        assert "blocks.3.ffn.net.0.proj.lora_A.weight" in result

    def test_to_out_underscore_preserved_as_segment(self):
        """'to_out' is a single module name containing an underscore.

        The reversal must not split 'to_out' into 'to' and 'out'.
        Correct:   blocks.0.attn1.to_out.0.lora_A.weight
        Wrong:     blocks.0.attn1.to.out.0.lora_A.weight
        """
        sd = {"lora_unet_blocks_0_attn1_to_out_0.lora_down.weight": object()}
        result = musubi_to_flimmer(sd)
        assert "blocks.0.attn1.to_out.0.lora_A.weight" in result

    def test_add_k_proj_underscore_preserved(self):
        """'add_k_proj' is a single I2V module name containing underscores.

        Correct:   blocks.5.attn2.add_k_proj.lora_A.weight
        Wrong:     blocks.5.attn2.add.k.proj.lora_A.weight
        """
        sd = {"lora_unet_blocks_5_attn2_add_k_proj.lora_down.weight": object()}
        result = musubi_to_flimmer(sd)
        assert "blocks.5.attn2.add_k_proj.lora_A.weight" in result

    def test_roundtrip_produces_original_keys(self):
        """flimmer → musubi → flimmer gives back exactly the original key set.

        This is the core contract of the two conversion functions. If this
        fails, checkpoint interoperability breaks.
        """
        original = _make_state_dict(
            blocks=(0, 1, 39),
            targets=("attn1.to_q", "attn2.to_k", "ffn.net.0.proj", "ffn.net.2"),
            rank=16,
        )
        musubi = flimmer_to_musubi(original)
        restored = musubi_to_flimmer(musubi)

        assert set(restored.keys()) == set(original.keys())

    def test_roundtrip_values_survive(self):
        """Tensor values pass through both conversion directions unchanged."""
        sentinel = object()
        original = {"blocks.2.attn2.to_v.lora_A.weight": sentinel}
        musubi = flimmer_to_musubi(original)
        restored = musubi_to_flimmer(musubi)
        assert restored["blocks.2.attn2.to_v.lora_A.weight"] is sentinel


# ---------------------------------------------------------------------------
# validate_state_dict_keys — structural validation
# ---------------------------------------------------------------------------

class TestValidateStateDictKeys:
    """validate_state_dict_keys reports issues with flimmer-format state dicts."""

    def test_valid_t2v_returns_no_issues(self):
        """A well-formed T2V state dict with standard targets passes validation."""
        sd = _make_state_dict(
            blocks=(0, 20, 39),
            targets=("attn1.to_q", "attn2.to_k", "ffn.net.0.proj"),
            rank=16,
        )
        issues = validate_state_dict_keys(sd, variant="2.2_t2v", rank=16)
        assert issues == []

    def test_valid_i2v_with_add_k_proj_returns_no_issues(self):
        """I2V state dict including add_k_proj / add_v_proj passes I2V validation.

        These projections are I2V-only — they would fail T2V validation but
        are valid when variant='2.2_i2v'.
        """
        sd = _make_state_dict(
            blocks=(0,),
            targets=("attn2.add_k_proj", "attn2.add_v_proj"),
            rank=16,
        )
        issues = validate_state_dict_keys(sd, variant="2.2_i2v", rank=16)
        assert issues == []

    def test_add_k_proj_in_t2v_variant_is_an_issue(self):
        """I2V-only modules in a T2V state dict should be flagged as unknown."""
        sd = _make_state_dict(
            blocks=(0,),
            targets=("attn2.add_k_proj",),
            rank=16,
        )
        issues = validate_state_dict_keys(sd, variant="2.2_t2v", rank=16)
        assert len(issues) > 0
        assert any("add_k_proj" in issue for issue in issues)

    def test_invalid_key_format_returns_issue(self):
        """Keys that don't match the expected pattern are flagged.

        The expected pattern is: blocks.N.module_suffix.lora_[A|B].weight
        Anything else (missing prefix, wrong suffix, random string) is an issue.
        """
        sd = {"totally_wrong_key": object()}
        issues = validate_state_dict_keys(sd, variant="2.2_t2v")
        assert len(issues) == 1
        assert "totally_wrong_key" in issues[0]

    def test_block_number_out_of_range_returns_issue(self):
        """Block indices outside 0-39 are flagged.

        Wan models have exactly 40 blocks (indices 0-39). Block 45 does not
        exist — loading such a key would silently skip during training.
        """
        sd = {
            "blocks.45.attn1.to_q.lora_A.weight": object(),
            "blocks.45.attn1.to_q.lora_B.weight": object(),
        }
        issues = validate_state_dict_keys(sd, variant="2.2_t2v")
        assert any("45" in issue and "range" in issue for issue in issues)

    def test_unknown_module_suffix_returns_issue(self):
        """Module names not in the known target list are flagged.

        'attn3.to_q' is not a real Wan module — catching this prevents
        silently-invalid LoRA files from entering the training pipeline.
        """
        sd = {
            "blocks.0.attn3.to_q.lora_A.weight": object(),
            "blocks.0.attn3.to_q.lora_B.weight": object(),
        }
        issues = validate_state_dict_keys(sd, variant="2.2_t2v")
        assert any("attn3.to_q" in issue for issue in issues)

    def test_missing_lora_B_is_flagged(self):
        """A lora_A without a matching lora_B is an incomplete pair.

        Unpaired weights mean the LoRA module is partially defined — this
        would cause shape errors when the LoRA is applied to the model.
        """
        sd = {"blocks.0.attn1.to_q.lora_A.weight": object()}
        issues = validate_state_dict_keys(sd, variant="2.2_t2v")
        assert any("lora_B" in issue and "attn1.to_q" in issue for issue in issues)

    def test_missing_lora_A_is_flagged(self):
        """A lora_B without a matching lora_A is an incomplete pair."""
        sd = {"blocks.0.attn1.to_q.lora_B.weight": object()}
        issues = validate_state_dict_keys(sd, variant="2.2_t2v")
        assert any("lora_A" in issue and "attn1.to_q" in issue for issue in issues)

    def test_correct_rank_returns_no_issues(self):
        """Tensors with the correct rank dimension pass shape validation.

        lora_A shape: (rank, in_features) — rank is dim 0.
        lora_B shape: (out_features, rank) — rank is dim 1.
        """
        sd = _make_state_dict(blocks=(0,), targets=("attn1.to_q",), rank=16)
        issues = validate_state_dict_keys(sd, variant="2.2_t2v", rank=16)
        assert issues == []

    def test_wrong_rank_in_lora_A_returns_issue(self):
        """If lora_A.shape[0] != expected rank, a shape issue is reported.

        This catches the common error of loading a rank-32 LoRA into a
        rank-16 training run — which would silently broadcast or error.
        """
        sd = {
            "blocks.0.attn1.to_q.lora_A.weight": np.zeros((32, 128)),  # rank=32 in dim 0
            "blocks.0.attn1.to_q.lora_B.weight": np.zeros((128, 32)),  # rank=32 in dim 1
        }
        # Validate against expected rank 16 — both tensors are actually rank 32
        issues = validate_state_dict_keys(sd, variant="2.2_t2v", rank=16)
        assert any("Rank mismatch" in issue for issue in issues)

    def test_wrong_rank_in_lora_B_returns_issue(self):
        """If lora_B.shape[1] != expected rank, a shape issue is reported."""
        sd = {
            "blocks.0.attn1.to_q.lora_A.weight": np.zeros((16, 128)),  # correct rank
            "blocks.0.attn1.to_q.lora_B.weight": np.zeros((128, 8)),   # wrong rank (8 vs 16)
        }
        issues = validate_state_dict_keys(sd, variant="2.2_t2v", rank=16)
        assert any("Rank mismatch" in issue and "lora_B" in issue for issue in issues)

    def test_rank_check_skipped_when_rank_is_none(self):
        """Passing rank=None disables shape validation entirely.

        Shape validation requires the caller to know the expected rank.
        If rank is not provided, keys-only validation still runs.
        """
        sd = {
            "blocks.0.attn1.to_q.lora_A.weight": np.zeros((32, 128)),  # any rank
            "blocks.0.attn1.to_q.lora_B.weight": np.zeros((128, 32)),
        }
        # No rank argument — shape issues should NOT appear
        issues = validate_state_dict_keys(sd, variant="2.2_t2v")
        shape_issues = [i for i in issues if "Rank mismatch" in i]
        assert shape_issues == []

    def test_tensors_without_shape_attribute_skipped_for_rank_check(self):
        """Plain Python objects without .shape skip rank validation gracefully.

        The function must not crash when values are non-tensor sentinels.
        This matters for unit tests and for metadata keys that hold scalars.
        """
        sd = {
            "blocks.0.attn1.to_q.lora_A.weight": object(),  # no .shape
            "blocks.0.attn1.to_q.lora_B.weight": object(),
        }
        # rank=16 provided but values have no .shape — no crash expected
        issues = validate_state_dict_keys(sd, variant="2.2_t2v", rank=16)
        shape_issues = [i for i in issues if "Rank mismatch" in i]
        assert shape_issues == []

    def test_empty_state_dict_returns_no_issues(self):
        """An empty state dict is trivially valid — no keys to check."""
        issues = validate_state_dict_keys({}, variant="2.2_t2v")
        assert issues == []

    def test_multiple_issues_all_reported(self):
        """All issues in one state dict are collected, not short-circuited.

        A single bad file might have multiple problems (wrong format AND
        out-of-range block AND unpaired key). The validator must report all.
        """
        sd = {
            "totally_wrong": object(),                           # bad format
            "blocks.99.attn1.to_q.lora_A.weight": object(),    # out-of-range + unpaired A
        }
        issues = validate_state_dict_keys(sd, variant="2.2_t2v")
        assert len(issues) >= 3  # bad format, out-of-range, missing B


# ---------------------------------------------------------------------------
# add_diffusers_prefix / strip_diffusers_prefix / has_diffusers_prefix
# ---------------------------------------------------------------------------

class TestDiffusersPrefix:
    """Diffusers component prefix operations for pipeline.load_lora_weights()."""

    def test_add_prefix_default(self):
        """Default prefix is 'transformer' (high-noise expert)."""
        sd = {"blocks.0.attn1.to_q.lora_A.weight": object()}
        result = add_diffusers_prefix(sd)
        assert "transformer.blocks.0.attn1.to_q.lora_A.weight" in result

    def test_add_prefix_transformer_2(self):
        """Low-noise expert gets 'transformer_2' prefix."""
        sd = {"blocks.0.attn1.to_q.lora_A.weight": object()}
        result = add_diffusers_prefix(sd, prefix="transformer_2")
        assert "transformer_2.blocks.0.attn1.to_q.lora_A.weight" in result

    def test_add_prefix_preserves_values(self):
        """Tensor values are the same objects after prefixing."""
        val = object()
        sd = {"blocks.0.attn1.to_q.lora_A.weight": val}
        result = add_diffusers_prefix(sd)
        assert result["transformer.blocks.0.attn1.to_q.lora_A.weight"] is val

    def test_strip_prefix_transformer(self):
        """Strips 'transformer.' prefix back to clean keys."""
        sd = {"transformer.blocks.0.attn1.to_q.lora_A.weight": object()}
        result = strip_diffusers_prefix(sd)
        assert "blocks.0.attn1.to_q.lora_A.weight" in result

    def test_strip_prefix_transformer_2(self):
        """Strips 'transformer_2.' prefix back to clean keys."""
        sd = {"transformer_2.blocks.0.attn1.to_q.lora_A.weight": object()}
        result = strip_diffusers_prefix(sd)
        assert "blocks.0.attn1.to_q.lora_A.weight" in result

    def test_strip_prefix_mixed(self):
        """Mixed keys with both prefixes are correctly stripped."""
        sd = {
            "transformer.blocks.0.attn1.to_q.lora_A.weight": object(),
            "transformer_2.blocks.0.attn1.to_q.lora_A.weight": object(),
        }
        result = strip_diffusers_prefix(sd)
        assert len(result) == 1  # Both map to same clean key
        assert "blocks.0.attn1.to_q.lora_A.weight" in result

    def test_strip_prefix_passthrough(self):
        """Keys without a known prefix are passed through unchanged."""
        sd = {"blocks.0.attn1.to_q.lora_A.weight": object()}
        result = strip_diffusers_prefix(sd)
        assert "blocks.0.attn1.to_q.lora_A.weight" in result

    def test_roundtrip_add_strip(self):
        """add → strip produces original keys."""
        sd = _make_state_dict(blocks=(0, 5), targets=("attn1.to_q", "ffn.net.2"))
        prefixed = add_diffusers_prefix(sd, prefix="transformer")
        restored = strip_diffusers_prefix(prefixed)
        assert set(restored.keys()) == set(sd.keys())

    def test_has_diffusers_prefix_true(self):
        """Detects transformer. prefix."""
        sd = {"transformer.blocks.0.attn1.to_q.lora_A.weight": object()}
        assert has_diffusers_prefix(sd) is True

    def test_has_diffusers_prefix_false(self):
        """Clean keys have no prefix."""
        sd = {"blocks.0.attn1.to_q.lora_A.weight": object()}
        assert has_diffusers_prefix(sd) is False

    def test_has_diffusers_prefix_transformer_2(self):
        """Detects transformer_2. prefix."""
        sd = {"transformer_2.blocks.0.attn1.to_q.lora_A.weight": object()}
        assert has_diffusers_prefix(sd) is True

    def test_empty_state_dict(self):
        """Empty state dicts are handled gracefully by all functions."""
        assert add_diffusers_prefix({}) == {}
        assert strip_diffusers_prefix({}) == {}
        assert has_diffusers_prefix({}) is False
