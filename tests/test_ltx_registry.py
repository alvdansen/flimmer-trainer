"""Tests for flimmer.training.ltx.registry — variant registry and backend factory.

This module is GPU-free: registry.py only defines the variant mapping and a
factory function. The factory does a late import of LtxModelBackend (which
requires torch/ltx-core), so we mock that import to keep tests torch-free.

Coverage:
    - LTX_VARIANTS structure and completeness
    - Per-variant property assertions (is_moe, is_i2v, channels, etc.)
    - get_variant_info() — copy guarantee, unknown-variant error
    - get_ltx_backend() — happy path with mocked backend, variant=None error,
      unknown-variant error, config override behaviour (lora target_modules)
"""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from flimmer.training.ltx.registry import LTX_VARIANTS, get_variant_info


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_config(
    variant: str | None = "2.3_t2v",
    path: str = "/fake/model",
    lora_target_modules: list[str] | None = None,
    quantization: str | None = None,
) -> MagicMock:
    """Build a minimal mock FlimmerTrainingConfig for factory tests.

    Mirrors the shape that get_ltx_backend() actually accesses:
        config.model.variant
        config.model.path
        config.model.quantization
        config.lora.target_modules
    """
    model_cfg = MagicMock()
    model_cfg.variant = variant
    model_cfg.path = path
    model_cfg.quantization = quantization

    lora_cfg = MagicMock()
    lora_cfg.target_modules = lora_target_modules
    lora_cfg.rank = 32
    lora_cfg.alpha = 32

    cfg = MagicMock()
    cfg.model = model_cfg
    cfg.lora = lora_cfg
    return cfg


def _make_mock_backend_module(mock_cls: MagicMock) -> ModuleType:
    """Build a fake flimmer.training.ltx.backend module with a mock class.

    get_ltx_backend() does:
        from flimmer.training.ltx.backend import LtxModelBackend

    By inserting a fake module into sys.modules before the call, that import
    resolves to our mock without ever loading torch or ltx-core.
    """
    mod = ModuleType("flimmer.training.ltx.backend")
    mod.LtxModelBackend = mock_cls  # type: ignore[attr-defined]
    return mod


# ---------------------------------------------------------------------------
# LTX_VARIANTS structure
# ---------------------------------------------------------------------------

class TestLtxVariantsStructure:
    """Registry must contain the expected variant keys."""

    def test_variants_has_one_entry(self):
        """MVP ships with a single LTX variant."""
        assert len(LTX_VARIANTS) == 1

    def test_expected_key_present(self):
        assert "2.3_t2v" in LTX_VARIANTS

    def test_all_variants_have_required_keys(self):
        """Every variant must expose all required architecture keys."""
        required = {
            "model_id",
            "is_moe",
            "is_i2v",
            "in_channels",
            "num_blocks",
            "lora_targets",
            "hf_repo",
            "pipeline_class",
        }
        for name, variant in LTX_VARIANTS.items():
            missing = required - set(variant.keys())
            assert not missing, f"Variant '{name}' is missing keys: {missing}"

    def test_lora_targets_are_lists(self):
        """lora_targets must be plain lists (not tuples or other sequences)."""
        for name, variant in LTX_VARIANTS.items():
            assert isinstance(variant["lora_targets"], list), (
                f"Variant '{name}': lora_targets should be list, "
                f"got {type(variant['lora_targets'])}"
            )


# ---------------------------------------------------------------------------
# Per-variant property assertions
# ---------------------------------------------------------------------------

class TestT2vVariant:
    """LTX 2.3 T2V — single transformer, text-only conditioning."""

    def test_is_not_moe(self):
        assert LTX_VARIANTS["2.3_t2v"]["is_moe"] is False

    def test_is_not_i2v(self):
        assert LTX_VARIANTS["2.3_t2v"]["is_i2v"] is False

    def test_in_channels(self):
        # LTX: 128 channels (VAE latent)
        assert LTX_VARIANTS["2.3_t2v"]["in_channels"] == 128

    def test_num_blocks(self):
        assert LTX_VARIANTS["2.3_t2v"]["num_blocks"] == 48

    def test_pipeline_class(self):
        assert LTX_VARIANTS["2.3_t2v"]["pipeline_class"] == "LTXPipeline"

    def test_model_id(self):
        assert "2.3" in LTX_VARIANTS["2.3_t2v"]["model_id"]
        assert "t2v" in LTX_VARIANTS["2.3_t2v"]["model_id"]

    def test_hf_repo(self):
        assert LTX_VARIANTS["2.3_t2v"]["hf_repo"] == "Lightricks/LTX-2.3"

    def test_lora_target_count(self):
        # 8 video-only targets (4 self-attn + 4 cross-attn)
        assert len(LTX_VARIANTS["2.3_t2v"]["lora_targets"]) == 8


# ---------------------------------------------------------------------------
# get_variant_info()
# ---------------------------------------------------------------------------

class TestGetVariantInfo:
    """get_variant_info() returns a copy and raises for unknown variants."""

    def test_returns_dict(self):
        info = get_variant_info("2.3_t2v")
        assert isinstance(info, dict)

    def test_returns_copy_not_same_object(self):
        info1 = get_variant_info("2.3_t2v")
        info2 = get_variant_info("2.3_t2v")
        assert info1 is not info2
        assert info1 is not LTX_VARIANTS["2.3_t2v"]

    def test_mutation_does_not_affect_registry(self):
        """Modifying the returned dict must not mutate the global registry."""
        info = get_variant_info("2.3_t2v")
        original_model_id = LTX_VARIANTS["2.3_t2v"]["model_id"]
        info["model_id"] = "mutated"
        assert LTX_VARIANTS["2.3_t2v"]["model_id"] == original_model_id

    def test_all_valid_variants_succeed(self):
        for name in ("2.3_t2v",):
            info = get_variant_info(name)
            assert info is not None

    def test_unknown_variant_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown LTX variant"):
            get_variant_info("3.0_unknown")

    def test_unknown_variant_error_mentions_valid_options(self):
        """Error message should name the valid variants so users know what to use."""
        with pytest.raises(ValueError) as exc_info:
            get_variant_info("bad_variant")
        msg = str(exc_info.value)
        # At least one valid variant name must appear in the message
        assert any(key in msg for key in LTX_VARIANTS), (
            "Error message should list valid variants"
        )

    def test_empty_string_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown LTX variant"):
            get_variant_info("")

    def test_case_sensitive(self):
        """Variant names are case-sensitive ('2.3_T2V' is not valid)."""
        with pytest.raises(ValueError):
            get_variant_info("2.3_T2V")


# ---------------------------------------------------------------------------
# get_ltx_backend() — mocked to avoid torch/ltx-core dependency
# ---------------------------------------------------------------------------

class TestGetLtxBackend:
    """Factory tests — LtxModelBackend is mocked so torch is never imported."""

    def test_variant_none_raises_value_error(self):
        """config.model.variant=None must raise before touching the backend."""
        config = _make_mock_config(variant=None)
        from flimmer.training.ltx.registry import get_ltx_backend
        with pytest.raises(ValueError, match="config.model.variant is required"):
            get_ltx_backend(config)

    def test_unknown_variant_raises_value_error(self):
        """Unknown variant string propagates as ValueError from get_variant_info."""
        config = _make_mock_config(variant="99.0_unk")
        from flimmer.training.ltx.registry import get_ltx_backend
        with pytest.raises(ValueError, match="Unknown LTX variant"):
            get_ltx_backend(config)

    def test_happy_path_calls_backend_constructor(self):
        """A valid config produces an LtxModelBackend (mocked) instance."""
        from flimmer.training.ltx.registry import get_ltx_backend

        mock_backend_cls = MagicMock(name="LtxModelBackend")
        mock_instance = MagicMock(name="backend_instance")
        mock_backend_cls.return_value = mock_instance

        config = _make_mock_config(variant="2.3_t2v", path="/my/model")

        with patch.dict(
            sys.modules,
            {"flimmer.training.ltx.backend": _make_mock_backend_module(mock_backend_cls)},
        ):
            result = get_ltx_backend(config)

        assert result is mock_instance

    def test_happy_path_uses_variant_defaults(self):
        """Without config overrides, the backend receives variant-default values."""
        from flimmer.training.ltx.registry import get_ltx_backend

        mock_backend_cls = MagicMock(name="LtxModelBackend")
        mock_backend_cls.return_value = MagicMock()

        config = _make_mock_config(
            variant="2.3_t2v",
            lora_target_modules=None,
        )

        with patch.dict(
            sys.modules,
            {"flimmer.training.ltx.backend": _make_mock_backend_module(mock_backend_cls)},
        ):
            get_ltx_backend(config)

        _, kwargs = mock_backend_cls.call_args
        assert kwargs["is_moe"] is False
        assert kwargs["is_i2v"] is False
        assert kwargs["in_channels"] == 128
        assert kwargs["num_blocks"] == 48
        assert kwargs["model_id"] == "ltx-2.3-t2v"
        assert kwargs["hf_repo"] == "Lightricks/LTX-2.3"

    def test_lora_target_modules_override(self):
        """config.lora.target_modules replaces the variant's default lora_targets."""
        from flimmer.training.ltx.registry import get_ltx_backend

        mock_backend_cls = MagicMock(name="LtxModelBackend")
        mock_backend_cls.return_value = MagicMock()

        custom_targets = ["attn1.to_q", "attn1.to_k"]
        config = _make_mock_config(
            variant="2.3_t2v",
            lora_target_modules=custom_targets,
        )

        with patch.dict(
            sys.modules,
            {"flimmer.training.ltx.backend": _make_mock_backend_module(mock_backend_cls)},
        ):
            get_ltx_backend(config)

        _, kwargs = mock_backend_cls.call_args
        assert kwargs["lora_targets"] == custom_targets

    def test_model_path_forwarded_to_backend(self):
        """config.model.path must be passed as model_path to the backend."""
        from flimmer.training.ltx.registry import get_ltx_backend

        mock_backend_cls = MagicMock(name="LtxModelBackend")
        mock_backend_cls.return_value = MagicMock()

        config = _make_mock_config(variant="2.3_t2v", path="/custom/path/to/model")

        with patch.dict(
            sys.modules,
            {"flimmer.training.ltx.backend": _make_mock_backend_module(mock_backend_cls)},
        ):
            get_ltx_backend(config)

        _, kwargs = mock_backend_cls.call_args
        assert kwargs["model_path"] == "/custom/path/to/model"

    def test_quantization_forwarded(self):
        """config.model.quantization is passed to the backend."""
        from flimmer.training.ltx.registry import get_ltx_backend

        mock_backend_cls = MagicMock(name="LtxModelBackend")
        mock_backend_cls.return_value = MagicMock()

        config = _make_mock_config(variant="2.3_t2v", quantization="int8")

        with patch.dict(
            sys.modules,
            {"flimmer.training.ltx.backend": _make_mock_backend_module(mock_backend_cls)},
        ):
            get_ltx_backend(config)

        _, kwargs = mock_backend_cls.call_args
        assert kwargs["quantization"] == "int8"

    def test_no_quantization_passes_none(self):
        """Without quantization config, None is forwarded."""
        from flimmer.training.ltx.registry import get_ltx_backend

        mock_backend_cls = MagicMock(name="LtxModelBackend")
        mock_backend_cls.return_value = MagicMock()

        config = _make_mock_config(variant="2.3_t2v", quantization=None)

        with patch.dict(
            sys.modules,
            {"flimmer.training.ltx.backend": _make_mock_backend_module(mock_backend_cls)},
        ):
            get_ltx_backend(config)

        _, kwargs = mock_backend_cls.call_args
        assert kwargs["quantization"] is None
