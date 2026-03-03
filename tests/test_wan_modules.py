"""Tests for flimmer.training.wan.modules — PEFT bridge.

Covers the GPU-free resolve_target_modules() function in full, plus
existence and signature checks for the GPU-required functions
(create_lora_on_model, extract_lora_state_dict, inject_lora_state_dict,
remove_lora_from_model).

GPU-required function tests use monkeypatch to simulate a missing 'peft'
installation, or inspect.signature to verify the function contracts
without actually calling them.
"""

from __future__ import annotations

import inspect
import sys

import pytest

from flimmer.training.wan.constants import (
    FORK_TARGET_TO_MODULES,
    I2V_EXTRA_TARGETS,
    T2V_LORA_TARGETS,
)
from flimmer.training.wan.modules import (
    create_lora_on_model,
    extract_lora_state_dict,
    inject_lora_state_dict,
    remove_lora_from_model,
    resolve_target_modules,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _i2v_targets() -> list[str]:
    """Return the full I2V target list (T2V base + I2V extras)."""
    return list(T2V_LORA_TARGETS) + list(I2V_EXTRA_TARGETS)


# ---------------------------------------------------------------------------
# resolve_target_modules — variant targets only
# ---------------------------------------------------------------------------


class TestResolveVariantOnly:
    """Baseline: resolve with only variant_targets, no overrides or fork."""

    def test_t2v_targets_returns_same_list(self):
        """Passing T2V_LORA_TARGETS with no other args returns them unchanged."""
        result = resolve_target_modules(T2V_LORA_TARGETS)
        assert result == list(T2V_LORA_TARGETS)

    def test_t2v_targets_returns_correct_count(self):
        """Ten T2V target modules must all survive the round-trip."""
        result = resolve_target_modules(T2V_LORA_TARGETS)
        assert len(result) == 10

    def test_returns_new_list_not_same_object(self):
        """The returned list is a new object, not the original reference."""
        result = resolve_target_modules(T2V_LORA_TARGETS)
        assert result is not T2V_LORA_TARGETS

    def test_deduplicates_variant_targets(self):
        """Duplicate entries in variant_targets are collapsed to one."""
        duped = list(T2V_LORA_TARGETS) + list(T2V_LORA_TARGETS)
        result = resolve_target_modules(duped)
        assert len(result) == len(T2V_LORA_TARGETS)
        # Deduplication must preserve the first-seen order
        assert result == list(T2V_LORA_TARGETS)

    def test_preserves_order(self):
        """Order of the original list is preserved after deduplication."""
        custom = ["ffn.net.2", "attn1.to_q", "attn2.to_v"]
        result = resolve_target_modules(custom)
        assert result == custom

    def test_none_fork_targets_returns_all(self):
        """Explicitly passing fork_targets=None is equivalent to omitting it."""
        result_default = resolve_target_modules(T2V_LORA_TARGETS)
        result_explicit = resolve_target_modules(T2V_LORA_TARGETS, fork_targets=None)
        assert result_default == result_explicit


# ---------------------------------------------------------------------------
# resolve_target_modules — user_overrides
# ---------------------------------------------------------------------------


class TestResolveUserOverrides:
    """user_overrides replaces the variant defaults entirely."""

    def test_user_overrides_replaces_variant_targets(self):
        """When user_overrides is set it completely replaces variant_targets."""
        overrides = ["attn1.to_q", "attn1.to_k"]
        result = resolve_target_modules(T2V_LORA_TARGETS, user_overrides=overrides)
        assert result == overrides

    def test_user_overrides_excludes_all_variant_targets(self):
        """No variant target leaks through when user_overrides is active."""
        overrides = ["attn1.to_q"]
        result = resolve_target_modules(T2V_LORA_TARGETS, user_overrides=overrides)
        for t in T2V_LORA_TARGETS:
            if t not in overrides:
                assert t not in result

    def test_user_overrides_accepts_non_standard_names(self):
        """User overrides are not validated against any known list."""
        custom = ["my_custom_layer", "some.other.projection"]
        result = resolve_target_modules(T2V_LORA_TARGETS, user_overrides=custom)
        assert result == custom

    def test_user_overrides_deduplicates(self):
        """Duplicates in user_overrides are also collapsed."""
        overrides = ["attn1.to_q", "attn1.to_q", "ffn.net.2"]
        result = resolve_target_modules(T2V_LORA_TARGETS, user_overrides=overrides)
        assert result == ["attn1.to_q", "ffn.net.2"]

    def test_user_overrides_none_falls_back_to_variant(self):
        """user_overrides=None (the default) uses variant_targets."""
        result = resolve_target_modules(T2V_LORA_TARGETS, user_overrides=None)
        assert result == list(T2V_LORA_TARGETS)


# ---------------------------------------------------------------------------
# resolve_target_modules — fork_targets filtering
# ---------------------------------------------------------------------------


class TestResolveForkTargets:
    """fork_targets filters the active base to a subset of modules."""

    def test_fork_ffn_returns_only_ffn_modules(self):
        """fork_targets=['ffn'] filters to exactly the two FFN projections."""
        result = resolve_target_modules(T2V_LORA_TARGETS, fork_targets=["ffn"])
        assert set(result) == set(FORK_TARGET_TO_MODULES["ffn"])

    def test_fork_ffn_count(self):
        """FFN fork yields exactly two modules: gate proj + output proj."""
        result = resolve_target_modules(T2V_LORA_TARGETS, fork_targets=["ffn"])
        assert len(result) == 2

    def test_fork_self_attn_returns_attn1_modules(self):
        """fork_targets=['self_attn'] filters to the four attn1 projections."""
        result = resolve_target_modules(T2V_LORA_TARGETS, fork_targets=["self_attn"])
        assert set(result) == set(FORK_TARGET_TO_MODULES["self_attn"])

    def test_fork_self_attn_count(self):
        """Self-attention fork yields exactly four modules (q, k, v, out)."""
        result = resolve_target_modules(T2V_LORA_TARGETS, fork_targets=["self_attn"])
        assert len(result) == 4

    def test_fork_cross_attn_returns_attn2_modules(self):
        """fork_targets=['cross_attn'] filters to the four attn2 projections."""
        result = resolve_target_modules(T2V_LORA_TARGETS, fork_targets=["cross_attn"])
        assert set(result) == set(FORK_TARGET_TO_MODULES["cross_attn"])

    def test_fork_cross_attn_excludes_self_attn(self):
        """cross_attn fork must not include any attn1 modules."""
        result = resolve_target_modules(T2V_LORA_TARGETS, fork_targets=["cross_attn"])
        for module in result:
            assert not module.startswith("attn1"), (
                f"Self-attention module '{module}' leaked into cross_attn fork"
            )

    def test_fork_ffn_and_self_attn_combines_both(self):
        """fork_targets=['ffn', 'self_attn'] returns the union of both groups."""
        result = resolve_target_modules(
            T2V_LORA_TARGETS, fork_targets=["ffn", "self_attn"]
        )
        expected = set(FORK_TARGET_TO_MODULES["ffn"]) | set(FORK_TARGET_TO_MODULES["self_attn"])
        assert set(result) == expected

    def test_fork_ffn_and_self_attn_no_cross_attn_leakage(self):
        """Combining ffn + self_attn must not include any cross-attention module."""
        result = resolve_target_modules(
            T2V_LORA_TARGETS, fork_targets=["ffn", "self_attn"]
        )
        for module in result:
            assert "attn2" not in module, (
                f"Cross-attention module '{module}' leaked into ffn+self_attn fork"
            )

    def test_fork_projection_level_cross_attn_to_v(self):
        """fork_targets=['cross_attn.to_v'] resolves to exactly one module."""
        result = resolve_target_modules(
            T2V_LORA_TARGETS, fork_targets=["cross_attn.to_v"]
        )
        assert result == ["attn2.to_v"]

    def test_fork_projection_level_self_attn_to_q(self):
        """fork_targets=['self_attn.to_q'] resolves to exactly one module."""
        result = resolve_target_modules(
            T2V_LORA_TARGETS, fork_targets=["self_attn.to_q"]
        )
        assert result == ["attn1.to_q"]

    def test_fork_projection_level_ffn_up_proj(self):
        """fork_targets=['ffn.up_proj'] resolves to the gate projection only."""
        result = resolve_target_modules(
            T2V_LORA_TARGETS, fork_targets=["ffn.up_proj"]
        )
        assert result == ["ffn.net.0.proj"]

    def test_fork_unknown_literal_used_directly(self):
        """An unrecognised fork target is treated as a literal module suffix.

        If the literal happens to be in variant_targets it appears in the
        output; if it is NOT in variant_targets it is silently absent because
        the filter step only keeps modules that are both allowed by fork_targets
        AND present in the base list.
        """
        # "attn1.to_q" is a real module suffix that exists in T2V_LORA_TARGETS.
        # Passing it directly as a fork_target (without going through the alias
        # map) should still allow it through.
        result = resolve_target_modules(
            T2V_LORA_TARGETS, fork_targets=["attn1.to_q"]
        )
        assert "attn1.to_q" in result

    def test_fork_unknown_literal_not_in_base_returns_empty(self):
        """An unrecognised fork target that isn't in variant_targets gives []."""
        result = resolve_target_modules(
            T2V_LORA_TARGETS, fork_targets=["totally_unknown_layer"]
        )
        assert result == []

    def test_empty_fork_targets_returns_empty(self):
        """An empty fork_targets list produces an empty result — no modules match."""
        result = resolve_target_modules(T2V_LORA_TARGETS, fork_targets=[])
        assert result == []

    def test_fork_result_is_deduplicated(self):
        """Combining two fork targets that expand to the same module deduplicates."""
        # 'cross_attn.to_v' and 'cross_attn' both include 'attn2.to_v'.
        # The result must contain 'attn2.to_v' only once.
        result = resolve_target_modules(
            T2V_LORA_TARGETS,
            fork_targets=["cross_attn", "cross_attn.to_v"],
        )
        assert result.count("attn2.to_v") == 1

    def test_fork_preserves_order_from_base(self):
        """Filtered modules appear in the same order they appear in the base list."""
        result = resolve_target_modules(
            T2V_LORA_TARGETS, fork_targets=["ffn", "self_attn"]
        )
        # Collect the expected items from T2V_LORA_TARGETS in their original order
        allowed = set(FORK_TARGET_TO_MODULES["ffn"]) | set(FORK_TARGET_TO_MODULES["self_attn"])
        expected_order = [t for t in T2V_LORA_TARGETS if t in allowed]
        assert result == expected_order


# ---------------------------------------------------------------------------
# resolve_target_modules — user_overrides + fork_targets combined
# ---------------------------------------------------------------------------


class TestResolveOverridesPlusFork:
    """user_overrides are applied first; fork_targets then filter the result."""

    def test_overrides_then_fork(self):
        """Override to a custom set, then fork-filter to a subset of that."""
        # Override to self-attention only, then fork to just to_q
        overrides = list(FORK_TARGET_TO_MODULES["self_attn"])
        result = resolve_target_modules(
            T2V_LORA_TARGETS,
            user_overrides=overrides,
            fork_targets=["self_attn.to_q"],
        )
        assert result == ["attn1.to_q"]

    def test_fork_on_override_set_not_variant_set(self):
        """Fork filtering applies to user_overrides, NOT to variant_targets."""
        # User overrides to FFN only; fork to self_attn (which isn't in overrides)
        # → result should be empty because self_attn modules aren't in the base
        overrides = list(FORK_TARGET_TO_MODULES["ffn"])
        result = resolve_target_modules(
            T2V_LORA_TARGETS,
            user_overrides=overrides,
            fork_targets=["self_attn"],
        )
        assert result == []

    def test_overrides_plus_ffn_fork(self):
        """Override with a mixed list; FFN fork keeps only the FFN entries."""
        overrides = ["ffn.net.0.proj", "ffn.net.2", "attn1.to_q"]
        result = resolve_target_modules(
            T2V_LORA_TARGETS,
            user_overrides=overrides,
            fork_targets=["ffn"],
        )
        assert set(result) == {"ffn.net.0.proj", "ffn.net.2"}


# ---------------------------------------------------------------------------
# resolve_target_modules — I2V targets
# ---------------------------------------------------------------------------


class TestResolveI2VTargets:
    """I2V-specific target list behaviour."""

    def test_i2v_targets_include_add_k_proj_and_add_v_proj(self):
        """Full I2V target list (T2V + extras) includes both add_* projections."""
        i2v = _i2v_targets()
        result = resolve_target_modules(i2v)
        assert "attn2.add_k_proj" in result
        assert "attn2.add_v_proj" in result

    def test_i2v_targets_total_count(self):
        """I2V list = 10 T2V targets + 2 I2V extras = 12."""
        result = resolve_target_modules(_i2v_targets())
        assert len(result) == 12

    def test_cross_attn_fork_on_i2v_excludes_add_projections(self):
        """fork_targets=['cross_attn'] on I2V targets does NOT include add_k/add_v.

        FORK_TARGET_TO_MODULES['cross_attn'] maps to the four standard attn2
        projections (to_q, to_k, to_v, to_out.0). The I2V-specific add_k_proj
        and add_v_proj are NOT in that mapping — they would need their own
        fork target (e.g. 'cross_attn.add_k_proj') to be selected.
        """
        result = resolve_target_modules(
            _i2v_targets(), fork_targets=["cross_attn"]
        )
        assert "attn2.add_k_proj" not in result
        assert "attn2.add_v_proj" not in result

    def test_i2v_add_projections_selectable_as_literals(self):
        """I2V add_* projections are reachable via literal fork targets."""
        result = resolve_target_modules(
            _i2v_targets(),
            fork_targets=["attn2.add_k_proj", "attn2.add_v_proj"],
        )
        assert set(result) == {"attn2.add_k_proj", "attn2.add_v_proj"}


# ---------------------------------------------------------------------------
# GPU-required functions — ImportError when peft is missing
# ---------------------------------------------------------------------------


class TestCreateLoraImportError:
    """create_lora_on_model raises ImportError when peft is not installed."""

    def test_raises_import_error_without_peft(self, monkeypatch):
        """Simulate a missing peft install and confirm the helpful error is raised."""
        # Hide peft from sys.modules so the import inside the function fails
        monkeypatch.setitem(sys.modules, "peft", None)  # type: ignore[arg-type]
        with pytest.raises(ImportError, match="peft"):
            create_lora_on_model(
                model=object(),
                target_modules=["attn1.to_q"],
                rank=16,
                alpha=16,
            )

    def test_error_message_includes_install_hint(self, monkeypatch):
        """The ImportError message tells the user how to fix it."""
        monkeypatch.setitem(sys.modules, "peft", None)  # type: ignore[arg-type]
        with pytest.raises(ImportError) as exc_info:
            create_lora_on_model(
                model=object(),
                target_modules=["attn1.to_q"],
                rank=16,
                alpha=16,
            )
        assert "pip install peft" in str(exc_info.value)


# ---------------------------------------------------------------------------
# GPU-required functions — signature contracts
# ---------------------------------------------------------------------------


class TestGpuFunctionSignatures:
    """Verify that GPU functions exist and carry the expected signatures.

    These tests never call the functions — they only inspect the function
    objects. This catches renames, removed parameters, and type-annotation
    drift without needing a GPU or torch installation.
    """

    def test_create_lora_on_model_exists(self):
        assert callable(create_lora_on_model)

    def test_create_lora_on_model_parameters(self):
        sig = inspect.signature(create_lora_on_model)
        params = list(sig.parameters.keys())
        assert "model" in params
        assert "target_modules" in params
        assert "rank" in params
        assert "alpha" in params
        assert "dropout" in params
        assert "adapter_name" in params

    def test_create_lora_dropout_default_is_zero(self):
        sig = inspect.signature(create_lora_on_model)
        assert sig.parameters["dropout"].default == 0.0

    def test_create_lora_adapter_name_default(self):
        sig = inspect.signature(create_lora_on_model)
        assert sig.parameters["adapter_name"].default == "default"

    def test_extract_lora_state_dict_exists(self):
        assert callable(extract_lora_state_dict)

    def test_extract_lora_state_dict_parameters(self):
        sig = inspect.signature(extract_lora_state_dict)
        params = list(sig.parameters.keys())
        assert "model" in params

    def test_extract_lora_state_dict_single_required_param(self):
        """extract_lora_state_dict takes exactly one parameter: model."""
        sig = inspect.signature(extract_lora_state_dict)
        required = [
            p for p in sig.parameters.values()
            if p.default is inspect.Parameter.empty
        ]
        assert len(required) == 1
        assert required[0].name == "model"

    def test_inject_lora_state_dict_exists(self):
        assert callable(inject_lora_state_dict)

    def test_inject_lora_state_dict_parameters(self):
        sig = inspect.signature(inject_lora_state_dict)
        params = list(sig.parameters.keys())
        assert "model" in params
        assert "state_dict" in params

    def test_remove_lora_from_model_exists(self):
        assert callable(remove_lora_from_model)

    def test_remove_lora_from_model_parameters(self):
        sig = inspect.signature(remove_lora_from_model)
        params = list(sig.parameters.keys())
        assert "model" in params

    def test_remove_lora_returns_model_passthrough_without_peft(self):
        """Without peft installed, remove_lora_from_model returns the model unchanged.

        The function has a bare try/except ImportError — if peft isn't present
        it falls through and returns the original model object.
        """
        sentinel = object()
        # Ensure peft is not importable for this call
        original = sys.modules.get("peft", ...)
        sys.modules["peft"] = None  # type: ignore[assignment]
        try:
            result = remove_lora_from_model(sentinel)
        finally:
            if original is ...:
                del sys.modules["peft"]
            else:
                sys.modules["peft"] = original
        assert result is sentinel
