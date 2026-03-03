"""Tests for flimmer.training.lora — LoRA state management."""

import numpy as np
import pytest

from flimmer.training.errors import LoRAError
from flimmer.training.lora import (
    LoRAState,
    build_parameter_groups,
    merge_experts,
)
from flimmer.training.phase import PhaseType


def _make_state_dict(prefix="blocks.0.attn1"):
    """Create a small mock state dict with numpy arrays."""
    return {
        f"{prefix}.to_q.lora_A.weight": np.random.randn(16, 64).astype(np.float32),
        f"{prefix}.to_q.lora_B.weight": np.zeros((64, 16), dtype=np.float32),
        f"{prefix}.to_k.lora_A.weight": np.random.randn(16, 64).astype(np.float32),
        f"{prefix}.to_k.lora_B.weight": np.zeros((64, 16), dtype=np.float32),
    }


class TestLoRAState:
    """Basic LoRAState operations."""

    def test_create(self):
        state = LoRAState(
            state_dict=_make_state_dict(),
            rank=16,
            alpha=16,
            phase_type=PhaseType.UNIFIED,
        )
        assert state.rank == 16
        assert state.alpha == 16
        assert state.phase_type == PhaseType.UNIFIED
        assert len(state.state_dict) == 4


class TestFork:
    """LoRA forking — deep copy into two independent states."""

    def test_fork_produces_two(self):
        state = LoRAState(
            state_dict=_make_state_dict(),
            rank=16, alpha=16, phase_type=PhaseType.UNIFIED,
        )
        copy1, copy2 = state.fork()
        assert copy1.phase_type == PhaseType.HIGH_NOISE
        assert copy2.phase_type == PhaseType.LOW_NOISE

    def test_fork_is_deep_copy(self):
        state = LoRAState(
            state_dict=_make_state_dict(),
            rank=16, alpha=16, phase_type=PhaseType.UNIFIED,
        )
        copy1, copy2 = state.fork()
        # Modify copy1 — copy2 and original should be unaffected
        key = list(copy1.state_dict.keys())[0]
        copy1.state_dict[key][:] = 999.0
        assert not np.allclose(copy1.state_dict[key], copy2.state_dict[key])
        assert not np.allclose(copy1.state_dict[key], state.state_dict[key])

    def test_fork_preserves_rank_alpha(self):
        state = LoRAState(
            state_dict=_make_state_dict(),
            rank=32, alpha=16, phase_type=PhaseType.UNIFIED,
        )
        copy1, copy2 = state.fork()
        assert copy1.rank == 32
        assert copy1.alpha == 16
        assert copy2.rank == 32
        assert copy2.alpha == 16

    def test_fork_preserves_metadata(self):
        state = LoRAState(
            state_dict=_make_state_dict(),
            rank=16, alpha=16, phase_type=PhaseType.UNIFIED,
            metadata={"epoch": "10"},
        )
        copy1, copy2 = state.fork()
        assert copy1.metadata == {"epoch": "10"}
        assert copy2.metadata == {"epoch": "10"}


class TestSaveLoad:
    """Save/load round-trip via safetensors."""

    def test_save_and_load(self, tmp_path):
        state = LoRAState(
            state_dict=_make_state_dict(),
            rank=16, alpha=16, phase_type=PhaseType.HIGH_NOISE,
            metadata={"epoch": "5"},
        )
        path = tmp_path / "test_lora.safetensors"
        state.save(path)
        assert path.is_file()

        loaded = LoRAState.load(path)
        assert loaded.rank == 16
        assert loaded.alpha == 16
        assert loaded.phase_type == PhaseType.HIGH_NOISE
        assert len(loaded.state_dict) == 4

    def test_save_with_diffusers_prefix(self, tmp_path):
        """Saving with prefix adds transformer. to keys; loading strips it."""
        state = LoRAState(
            state_dict=_make_state_dict(),
            rank=16, alpha=16, phase_type=PhaseType.HIGH_NOISE,
        )
        path = tmp_path / "prefixed_lora.safetensors"
        state.save(path, diffusers_prefix="transformer")

        # Load with strip_prefix=True (default) → clean keys
        loaded = LoRAState.load(path)
        for key in loaded.state_dict:
            assert not key.startswith("transformer.")
            assert key.startswith("blocks.")

        # Load with strip_prefix=False → keeps prefixed keys
        loaded_raw = LoRAState.load(path, strip_prefix=False)
        for key in loaded_raw.state_dict:
            assert key.startswith("transformer.")

    def test_save_creates_parent_dirs(self, tmp_path):
        state = LoRAState(
            state_dict=_make_state_dict(),
            rank=16, alpha=16, phase_type=PhaseType.UNIFIED,
        )
        path = tmp_path / "nested" / "dir" / "lora.safetensors"
        state.save(path)
        assert path.is_file()

    def test_load_missing_file(self, tmp_path):
        with pytest.raises(LoRAError, match="not found"):
            LoRAState.load(tmp_path / "nonexistent.safetensors")

    def test_round_trip_values(self, tmp_path):
        sd = _make_state_dict()
        state = LoRAState(state_dict=sd, rank=16, alpha=16, phase_type=PhaseType.UNIFIED)
        path = tmp_path / "rt.safetensors"
        state.save(path)
        loaded = LoRAState.load(path)
        for key in sd:
            np.testing.assert_allclose(
                np.array(loaded.state_dict[key]),
                sd[key],
                atol=1e-6,
            )


class TestMergeExperts:
    """Merging two expert LoRAs."""

    def test_basic_merge(self):
        high = LoRAState(
            state_dict=_make_state_dict("blocks.0.attn1"),
            rank=16, alpha=16, phase_type=PhaseType.HIGH_NOISE,
        )
        low = LoRAState(
            state_dict=_make_state_dict("blocks.0.attn1"),
            rank=16, alpha=16, phase_type=PhaseType.LOW_NOISE,
        )
        merged = merge_experts(high, low)
        # Keys should have diffusers component prefixes:
        # transformer. = high-noise, transformer_2. = low-noise
        for key in merged.state_dict:
            assert key.startswith("transformer.") or key.startswith("transformer_2.")

    def test_merge_key_count(self):
        high = LoRAState(
            state_dict=_make_state_dict(), rank=16, alpha=16,
            phase_type=PhaseType.HIGH_NOISE,
        )
        low = LoRAState(
            state_dict=_make_state_dict(), rank=16, alpha=16,
            phase_type=PhaseType.LOW_NOISE,
        )
        merged = merge_experts(high, low)
        assert len(merged.state_dict) == 8  # 4 + 4

    def test_merge_rank_mismatch(self):
        high = LoRAState(state_dict={}, rank=16, alpha=16, phase_type=PhaseType.HIGH_NOISE)
        low = LoRAState(state_dict={}, rank=32, alpha=16, phase_type=PhaseType.LOW_NOISE)
        with pytest.raises(LoRAError, match="different ranks"):
            merge_experts(high, low)

    def test_merge_alpha_mismatch(self):
        high = LoRAState(state_dict={}, rank=16, alpha=16, phase_type=PhaseType.HIGH_NOISE)
        low = LoRAState(state_dict={}, rank=16, alpha=8, phase_type=PhaseType.LOW_NOISE)
        with pytest.raises(LoRAError, match="different alpha"):
            merge_experts(high, low)

    def test_merge_metadata(self):
        high = LoRAState(state_dict={}, rank=16, alpha=16, phase_type=PhaseType.HIGH_NOISE)
        low = LoRAState(state_dict={}, rank=16, alpha=16, phase_type=PhaseType.LOW_NOISE)
        merged = merge_experts(high, low)
        assert merged.metadata["merged_from"] == "high_noise+low_noise"
        assert merged.phase_type == PhaseType.UNIFIED


class TestFilterByTargets:
    """Parameter filtering by fork_targets and block_targets."""

    def test_no_filters_all_trainable(self):
        state = LoRAState(
            state_dict=_make_state_dict(),
            rank=16, alpha=16, phase_type=PhaseType.UNIFIED,
        )
        mask = state.filter_by_targets(fork_targets=None, block_targets=None)
        assert all(mask.values())

    def test_block_filter(self):
        sd = {}
        sd.update({f"blocks.0.{k}": v for k, v in _make_state_dict("x").items()})
        sd.update({f"blocks.5.{k}": v for k, v in _make_state_dict("y").items()})
        state = LoRAState(state_dict=sd, rank=16, alpha=16, phase_type=PhaseType.UNIFIED)
        mask = state.filter_by_targets(fork_targets=None, block_targets="0-2")
        for key, trainable in mask.items():
            if "blocks.0" in key:
                assert trainable
            elif "blocks.5" in key:
                assert not trainable

    def test_fork_target_filter_ffn(self):
        sd = {
            "blocks.0.ffn.up.lora_A.weight": np.zeros((4, 4)),
            "blocks.0.attn1.to_q.lora_A.weight": np.zeros((4, 4)),
        }
        state = LoRAState(state_dict=sd, rank=16, alpha=16, phase_type=PhaseType.UNIFIED)
        mask = state.filter_by_targets(fork_targets=["ffn"], block_targets=None)
        assert mask["blocks.0.ffn.up.lora_A.weight"] is True
        assert mask["blocks.0.attn1.to_q.lora_A.weight"] is False

    def test_combined_filter(self):
        sd = {
            "blocks.0.ffn.up.lora_A.weight": np.zeros((4, 4)),
            "blocks.0.attn1.to_q.lora_A.weight": np.zeros((4, 4)),
            "blocks.5.ffn.up.lora_A.weight": np.zeros((4, 4)),
        }
        state = LoRAState(state_dict=sd, rank=16, alpha=16, phase_type=PhaseType.UNIFIED)
        mask = state.filter_by_targets(fork_targets=["ffn"], block_targets="0-2")
        assert mask["blocks.0.ffn.up.lora_A.weight"] is True
        assert mask["blocks.0.attn1.to_q.lora_A.weight"] is False
        assert mask["blocks.5.ffn.up.lora_A.weight"] is False


class TestBuildParameterGroups:
    """LoRA+ parameter group construction."""

    def test_basic_groups(self):
        sd = _make_state_dict()
        mask = {k: True for k in sd}
        groups = build_parameter_groups(sd, mask, learning_rate=5e-5, loraplus_lr_ratio=4.0)
        assert len(groups) == 2  # A group + B group

    def test_loraplus_lr_ratio(self):
        sd = _make_state_dict()
        mask = {k: True for k in sd}
        groups = build_parameter_groups(sd, mask, learning_rate=1e-4, loraplus_lr_ratio=4.0)
        a_group = [g for g in groups if g["lr"] == 1e-4]
        b_group = [g for g in groups if g["lr"] == 4e-4]
        assert len(a_group) == 1
        assert len(b_group) == 1

    def test_frozen_params_excluded(self):
        sd = _make_state_dict()
        mask = {k: False for k in sd}
        groups = build_parameter_groups(sd, mask, learning_rate=5e-5)
        assert len(groups) == 0  # All frozen

    def test_no_loraplus(self):
        sd = _make_state_dict()
        mask = {k: True for k in sd}
        groups = build_parameter_groups(sd, mask, learning_rate=5e-5, loraplus_lr_ratio=1.0)
        # Both groups should have the same LR
        for g in groups:
            assert g["lr"] == 5e-5
