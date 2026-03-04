"""Tests for flimmer.training.wan.registry — variant registry and backend factory.

This module is GPU-free: registry.py only defines the variant mapping and a
factory function. The factory does a late import of WanModelBackend (which
requires torch/diffusers), so we mock that import to keep tests torch-free.

Coverage:
    - WAN_VARIANTS structure and completeness
    - Per-variant property assertions (is_moe, is_i2v, channels, pipeline, etc.)
    - get_variant_info() — copy guarantee, unknown-variant error
    - get_wan_backend() — happy path with mocked backend, variant=None error,
      unknown-variant error, config override behaviour (boundary_ratio,
      flow_shift, lora target_modules)
"""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from flimmer.training.wan.registry import WAN_VARIANTS, get_variant_info


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_config(
    variant: str | None = "2.2_t2v",
    path: str = "/fake/model",
    boundary_ratio: float | None = None,
    flow_shift: float | None = None,
    lora_target_modules: list[str] | None = None,
) -> MagicMock:
    """Build a minimal mock FlimmerTrainingConfig for factory tests.

    Mirrors the shape that get_wan_backend() actually accesses:
        config.model.variant
        config.model.path
        config.model.boundary_ratio
        config.model.flow_shift
        config.lora.target_modules   (only when hasattr(config, "lora") is True)
    """
    model_cfg = MagicMock()
    model_cfg.variant = variant
    model_cfg.path = path
    model_cfg.boundary_ratio = boundary_ratio
    model_cfg.flow_shift = flow_shift
    model_cfg.resolution = None  # Explicit None so MagicMock doesn't return a mock
    model_cfg.dit = None
    model_cfg.dit_high = None
    model_cfg.dit_low = None

    lora_cfg = MagicMock()
    lora_cfg.target_modules = lora_target_modules

    cfg = MagicMock()
    cfg.model = model_cfg
    cfg.lora = lora_cfg
    return cfg


# ---------------------------------------------------------------------------
# WAN_VARIANTS structure
# ---------------------------------------------------------------------------

class TestWanVariantsStructure:
    """Registry must contain exactly the four expected variant keys."""

    def test_variants_has_four_entries(self):
        assert len(WAN_VARIANTS) == 4

    def test_expected_keys_present(self):
        assert "2.2_t2v" in WAN_VARIANTS
        assert "2.2_i2v" in WAN_VARIANTS
        assert "2.1_t2v" in WAN_VARIANTS
        assert "2.1_i2v" in WAN_VARIANTS

    def test_all_variants_have_required_keys(self):
        """Every variant must expose all required architecture keys."""
        required = {
            "model_id",
            "is_moe",
            "is_i2v",
            "in_channels",
            "num_blocks",
            "boundary_ratio",
            "flow_shift",
            "lora_targets",
            "expert_subfolders",
            "pipeline_class",
            "subtypes",
            "default_subtype",
        }
        for name, variant in WAN_VARIANTS.items():
            missing = required - set(variant.keys())
            assert not missing, f"Variant '{name}' is missing keys: {missing}"

    def test_lora_targets_are_lists(self):
        """lora_targets must be plain lists (not tuples or other sequences)."""
        for name, variant in WAN_VARIANTS.items():
            assert isinstance(variant["lora_targets"], list), (
                f"Variant '{name}': lora_targets should be list, "
                f"got {type(variant['lora_targets'])}"
            )

    def test_expert_subfolders_are_dicts(self):
        """expert_subfolders must be plain dicts."""
        for name, variant in WAN_VARIANTS.items():
            assert isinstance(variant["expert_subfolders"], dict), (
                f"Variant '{name}': expert_subfolders should be dict, "
                f"got {type(variant['expert_subfolders'])}"
            )


# ---------------------------------------------------------------------------
# Per-variant property assertions
# ---------------------------------------------------------------------------

class TestT2vVariant:
    """Wan 2.2 T2V — MoE, text-only conditioning."""

    def test_is_moe(self):
        assert WAN_VARIANTS["2.2_t2v"]["is_moe"] is True

    def test_is_not_i2v(self):
        assert WAN_VARIANTS["2.2_t2v"]["is_i2v"] is False

    def test_in_channels(self):
        # T2V: 16 channels (noisy latent only)
        assert WAN_VARIANTS["2.2_t2v"]["in_channels"] == 16

    def test_num_blocks(self):
        assert WAN_VARIANTS["2.2_t2v"]["num_blocks"] == 40

    def test_boundary_ratio(self):
        assert WAN_VARIANTS["2.2_t2v"]["boundary_ratio"] == 0.875

    def test_pipeline_class(self):
        assert WAN_VARIANTS["2.2_t2v"]["pipeline_class"] == "WanPipeline"

    def test_model_id(self):
        assert "2.2" in WAN_VARIANTS["2.2_t2v"]["model_id"]
        assert "t2v" in WAN_VARIANTS["2.2_t2v"]["model_id"]

    def test_lora_target_count(self):
        # 10 base T2V targets
        assert len(WAN_VARIANTS["2.2_t2v"]["lora_targets"]) == 10


class TestI2vVariant:
    """Wan 2.2 I2V — MoE, reference image conditioning."""

    def test_is_moe(self):
        assert WAN_VARIANTS["2.2_i2v"]["is_moe"] is True

    def test_is_i2v(self):
        assert WAN_VARIANTS["2.2_i2v"]["is_i2v"] is True

    def test_in_channels(self):
        # I2V: 36 channels (16 noisy + 16 reference + 4 mask)
        assert WAN_VARIANTS["2.2_i2v"]["in_channels"] == 36

    def test_num_blocks(self):
        assert WAN_VARIANTS["2.2_i2v"]["num_blocks"] == 40

    def test_boundary_ratio(self):
        # I2V boundary is slightly higher than T2V
        assert WAN_VARIANTS["2.2_i2v"]["boundary_ratio"] == 0.900

    def test_pipeline_class(self):
        assert WAN_VARIANTS["2.2_i2v"]["pipeline_class"] == "WanImageToVideoPipeline"

    def test_model_id(self):
        assert "2.2" in WAN_VARIANTS["2.2_i2v"]["model_id"]
        assert "i2v" in WAN_VARIANTS["2.2_i2v"]["model_id"]

    def test_lora_target_count(self):
        # 10 base T2V targets + 2 I2V extra = 12
        assert len(WAN_VARIANTS["2.2_i2v"]["lora_targets"]) == 12

    def test_i2v_extra_targets_present(self):
        """I2V must include the extra add_k_proj and add_v_proj targets."""
        targets = WAN_VARIANTS["2.2_i2v"]["lora_targets"]
        assert "attn2.add_k_proj" in targets
        assert "attn2.add_v_proj" in targets

    def test_i2v_has_more_targets_than_t2v(self):
        t2v_count = len(WAN_VARIANTS["2.2_t2v"]["lora_targets"])
        i2v_count = len(WAN_VARIANTS["2.2_i2v"]["lora_targets"])
        assert i2v_count > t2v_count


class TestWan21Variant:
    """Wan 2.1 T2V — single transformer, no MoE."""

    def test_is_not_moe(self):
        assert WAN_VARIANTS["2.1_t2v"]["is_moe"] is False

    def test_is_not_i2v(self):
        assert WAN_VARIANTS["2.1_t2v"]["is_i2v"] is False

    def test_in_channels(self):
        # Same latent layout as T2V (16 channels)
        assert WAN_VARIANTS["2.1_t2v"]["in_channels"] == 16

    def test_num_blocks(self):
        # Same block count as 2.2
        assert WAN_VARIANTS["2.1_t2v"]["num_blocks"] == 40

    def test_boundary_ratio_is_none(self):
        # Non-MoE models have no expert boundary
        assert WAN_VARIANTS["2.1_t2v"]["boundary_ratio"] is None

    def test_model_id(self):
        assert "2.1" in WAN_VARIANTS["2.1_t2v"]["model_id"]

    def test_pipeline_class(self):
        # 2.1 uses the same pipeline class as 2.2 T2V
        assert WAN_VARIANTS["2.1_t2v"]["pipeline_class"] == "WanPipeline"


class TestWan21I2vVariant:
    """Wan 2.1 I2V -- non-MoE, reference image conditioning."""

    def test_is_not_moe(self):
        assert WAN_VARIANTS["2.1_i2v"]["is_moe"] is False

    def test_is_i2v(self):
        assert WAN_VARIANTS["2.1_i2v"]["is_i2v"] is True

    def test_in_channels(self):
        assert WAN_VARIANTS["2.1_i2v"]["in_channels"] == 36

    def test_num_blocks(self):
        assert WAN_VARIANTS["2.1_i2v"]["num_blocks"] == 40

    def test_boundary_ratio_is_none(self):
        assert WAN_VARIANTS["2.1_i2v"]["boundary_ratio"] is None

    def test_pipeline_class(self):
        assert WAN_VARIANTS["2.1_i2v"]["pipeline_class"] == "WanImageToVideoPipeline"

    def test_model_id(self):
        assert "2.1" in WAN_VARIANTS["2.1_i2v"]["model_id"]
        assert "i2v" in WAN_VARIANTS["2.1_i2v"]["model_id"]

    def test_lora_target_count(self):
        # 10 base T2V targets + 2 I2V extra = 12
        assert len(WAN_VARIANTS["2.1_i2v"]["lora_targets"]) == 12

    def test_i2v_extra_targets_present(self):
        targets = WAN_VARIANTS["2.1_i2v"]["lora_targets"]
        assert "attn2.add_k_proj" in targets
        assert "attn2.add_v_proj" in targets

    def test_has_default_subfolder(self):
        subs = WAN_VARIANTS["2.1_i2v"]["expert_subfolders"]
        assert "default" in subs
        assert "high_noise" not in subs


class TestSubtypes:
    """All variants must have resolution subtypes."""

    def test_all_variants_have_subtypes(self):
        for name, variant in WAN_VARIANTS.items():
            assert "subtypes" in variant, f"'{name}' missing subtypes"
            assert "default_subtype" in variant, f"'{name}' missing default_subtype"

    def test_subtypes_have_hf_repo(self):
        for name, variant in WAN_VARIANTS.items():
            for res, sub in variant["subtypes"].items():
                assert "hf_repo" in sub, f"'{name}' subtype '{res}' missing hf_repo"

    def test_21_i2v_has_separate_resolution_repos(self):
        subs = WAN_VARIANTS["2.1_i2v"]["subtypes"]
        assert subs["480p"]["hf_repo"] != subs["720p"]["hf_repo"]

    def test_22_i2v_has_same_resolution_repos(self):
        subs = WAN_VARIANTS["2.2_i2v"]["subtypes"]
        assert subs["480p"]["hf_repo"] == subs["720p"]["hf_repo"]


# ---------------------------------------------------------------------------
# Expert subfolders
# ---------------------------------------------------------------------------

class TestExpertSubfolders:
    """MoE variants must map both experts; non-MoE maps only 'default'."""

    def test_moe_variants_have_high_and_low_noise(self):
        for name in ("2.2_t2v", "2.2_i2v"):
            subs = WAN_VARIANTS[name]["expert_subfolders"]
            assert "high_noise" in subs, f"'{name}' missing 'high_noise' subfolder"
            assert "low_noise" in subs, f"'{name}' missing 'low_noise' subfolder"

    def test_moe_subfolder_values_are_strings(self):
        for name in ("2.2_t2v", "2.2_i2v"):
            for key, val in WAN_VARIANTS[name]["expert_subfolders"].items():
                assert isinstance(val, str), (
                    f"Variant '{name}' subfolder '{key}' value should be str"
                )

    def test_non_moe_has_default_subfolder(self):
        subs = WAN_VARIANTS["2.1_t2v"]["expert_subfolders"]
        assert "default" in subs

    def test_non_moe_has_no_expert_keys(self):
        subs = WAN_VARIANTS["2.1_t2v"]["expert_subfolders"]
        assert "high_noise" not in subs
        assert "low_noise" not in subs

    def test_moe_high_noise_is_transformer(self):
        # high_noise expert lives in the primary transformer subfolder
        subs = WAN_VARIANTS["2.2_t2v"]["expert_subfolders"]
        assert subs["high_noise"] == "transformer"

    def test_moe_low_noise_is_transformer_2(self):
        # low_noise expert lives in the secondary subfolder
        subs = WAN_VARIANTS["2.2_t2v"]["expert_subfolders"]
        assert subs["low_noise"] == "transformer_2"


# ---------------------------------------------------------------------------
# get_variant_info()
# ---------------------------------------------------------------------------

class TestGetVariantInfo:
    """get_variant_info() returns a copy and raises for unknown variants."""

    def test_returns_dict(self):
        info = get_variant_info("2.2_t2v")
        assert isinstance(info, dict)

    def test_returns_copy_not_same_object(self):
        info1 = get_variant_info("2.2_t2v")
        info2 = get_variant_info("2.2_t2v")
        assert info1 is not info2
        assert info1 is not WAN_VARIANTS["2.2_t2v"]

    def test_mutation_does_not_affect_registry(self):
        """Modifying the returned dict must not mutate the global registry."""
        info = get_variant_info("2.2_t2v")
        original_model_id = WAN_VARIANTS["2.2_t2v"]["model_id"]
        info["model_id"] = "mutated"
        assert WAN_VARIANTS["2.2_t2v"]["model_id"] == original_model_id

    def test_all_valid_variants_succeed(self):
        for name in ("2.2_t2v", "2.2_i2v", "2.1_t2v", "2.1_i2v"):
            info = get_variant_info(name)
            assert info is not None

    def test_unknown_variant_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown Wan variant"):
            get_variant_info("3.0_unknown")

    def test_unknown_variant_error_mentions_valid_options(self):
        """Error message should name the valid variants so users know what to use."""
        with pytest.raises(ValueError) as exc_info:
            get_variant_info("bad_variant")
        msg = str(exc_info.value)
        # At least one valid variant name must appear in the message
        assert any(key in msg for key in WAN_VARIANTS), (
            "Error message should list valid variants"
        )

    def test_empty_string_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown Wan variant"):
            get_variant_info("")

    def test_case_sensitive(self):
        """Variant names are case-sensitive ('2.2_T2V' is not valid)."""
        with pytest.raises(ValueError):
            get_variant_info("2.2_T2V")


# ---------------------------------------------------------------------------
# get_wan_backend() — mocked to avoid torch dependency
# ---------------------------------------------------------------------------

class TestGetWanBackend:
    """Factory tests — WanModelBackend is mocked so torch is never imported."""

    def _patch_backend(self):
        """Return a context manager that replaces WanModelBackend with a mock.

        get_wan_backend() does 'from flimmer.training.wan.backend import
        WanModelBackend' inside the function body (late import). We patch
        the name where it's imported, i.e. inside the registry module's
        execution context, by patching the backend module itself.
        """
        mock_backend_cls = MagicMock(name="WanModelBackend")
        mock_backend_cls.return_value = MagicMock(name="backend_instance")
        return patch(
            "flimmer.training.wan.backend.WanModelBackend",
            mock_backend_cls,
        ), mock_backend_cls

    def test_variant_none_raises_value_error(self):
        """config.model.variant=None must raise before touching the backend."""
        config = _make_mock_config(variant=None)
        from flimmer.training.wan.registry import get_wan_backend
        with pytest.raises(ValueError, match="config.model.variant is required"):
            get_wan_backend(config)

    def test_unknown_variant_raises_value_error(self):
        """Unknown variant string propagates as ValueError from get_variant_info."""
        config = _make_mock_config(variant="99.0_unk")
        from flimmer.training.wan.registry import get_wan_backend
        with pytest.raises(ValueError, match="Unknown Wan variant"):
            get_wan_backend(config)

    def test_happy_path_calls_backend_constructor(self):
        """A valid config produces a WanModelBackend (mocked) instance."""
        from flimmer.training.wan.registry import get_wan_backend

        mock_backend_cls = MagicMock(name="WanModelBackend")
        mock_instance = MagicMock(name="backend_instance")
        mock_backend_cls.return_value = mock_instance

        config = _make_mock_config(variant="2.2_t2v", path="/my/model")

        with patch("flimmer.training.wan.registry.WanModelBackend", mock_backend_cls, create=True):
            # Patch at the module level where the name is resolved at call time
            import flimmer.training.wan.backend as backend_mod
            with patch.object(backend_mod, "WanModelBackend", mock_backend_cls):
                # The factory does a late import, so patch sys.modules too
                with patch.dict(
                    sys.modules,
                    {"flimmer.training.wan.backend": _make_mock_backend_module(mock_backend_cls)},
                ):
                    result = get_wan_backend(config)

        # The result is whatever the mocked constructor returned
        assert result is mock_instance

    def test_happy_path_uses_variant_defaults(self):
        """Without config overrides, the backend receives variant-default values."""
        from flimmer.training.wan.registry import get_wan_backend

        mock_backend_cls = MagicMock(name="WanModelBackend")
        mock_backend_cls.return_value = MagicMock()

        config = _make_mock_config(
            variant="2.2_t2v",
            boundary_ratio=None,   # no override → use variant default
            flow_shift=None,       # no override → use variant default
            lora_target_modules=None,
        )

        with patch.dict(
            sys.modules,
            {"flimmer.training.wan.backend": _make_mock_backend_module(mock_backend_cls)},
        ):
            get_wan_backend(config)

        _, kwargs = mock_backend_cls.call_args
        assert kwargs["boundary_ratio"] == 0.875  # 2.2_t2v default
        assert kwargs["flow_shift"] == 5.0       # validated T2V default
        assert kwargs["is_moe"] is True
        assert kwargs["is_i2v"] is False
        assert kwargs["in_channels"] == 16

    def test_boundary_ratio_override(self):
        """config.model.boundary_ratio overrides the variant default."""
        from flimmer.training.wan.registry import get_wan_backend

        mock_backend_cls = MagicMock(name="WanModelBackend")
        mock_backend_cls.return_value = MagicMock()

        config = _make_mock_config(variant="2.2_t2v", boundary_ratio=0.800)

        with patch.dict(
            sys.modules,
            {"flimmer.training.wan.backend": _make_mock_backend_module(mock_backend_cls)},
        ):
            get_wan_backend(config)

        _, kwargs = mock_backend_cls.call_args
        assert kwargs["boundary_ratio"] == 0.800

    def test_flow_shift_override(self):
        """config.model.flow_shift overrides the variant default."""
        from flimmer.training.wan.registry import get_wan_backend

        mock_backend_cls = MagicMock(name="WanModelBackend")
        mock_backend_cls.return_value = MagicMock()

        config = _make_mock_config(variant="2.2_t2v", flow_shift=5.0)

        with patch.dict(
            sys.modules,
            {"flimmer.training.wan.backend": _make_mock_backend_module(mock_backend_cls)},
        ):
            get_wan_backend(config)

        _, kwargs = mock_backend_cls.call_args
        assert kwargs["flow_shift"] == 5.0

    def test_lora_target_modules_override(self):
        """config.lora.target_modules replaces the variant's default lora_targets."""
        from flimmer.training.wan.registry import get_wan_backend

        mock_backend_cls = MagicMock(name="WanModelBackend")
        mock_backend_cls.return_value = MagicMock()

        custom_targets = ["attn1.to_q", "attn1.to_k"]
        config = _make_mock_config(
            variant="2.2_t2v",
            lora_target_modules=custom_targets,
        )

        with patch.dict(
            sys.modules,
            {"flimmer.training.wan.backend": _make_mock_backend_module(mock_backend_cls)},
        ):
            get_wan_backend(config)

        _, kwargs = mock_backend_cls.call_args
        assert kwargs["lora_targets"] == custom_targets

    def test_i2v_variant_passes_correct_channels(self):
        """I2V config must set in_channels=36 on the backend."""
        from flimmer.training.wan.registry import get_wan_backend

        mock_backend_cls = MagicMock(name="WanModelBackend")
        mock_backend_cls.return_value = MagicMock()

        config = _make_mock_config(variant="2.2_i2v")

        with patch.dict(
            sys.modules,
            {"flimmer.training.wan.backend": _make_mock_backend_module(mock_backend_cls)},
        ):
            get_wan_backend(config)

        _, kwargs = mock_backend_cls.call_args
        assert kwargs["in_channels"] == 36
        assert kwargs["is_i2v"] is True

    def test_21_t2v_passes_non_moe(self):
        """Wan 2.1 T2V must forward is_moe=False and boundary_ratio=None."""
        from flimmer.training.wan.registry import get_wan_backend

        mock_backend_cls = MagicMock(name="WanModelBackend")
        mock_backend_cls.return_value = MagicMock()

        config = _make_mock_config(variant="2.1_t2v")

        with patch.dict(
            sys.modules,
            {"flimmer.training.wan.backend": _make_mock_backend_module(mock_backend_cls)},
        ):
            get_wan_backend(config)

        _, kwargs = mock_backend_cls.call_args
        assert kwargs["is_moe"] is False
        assert kwargs["boundary_ratio"] is None

    def test_21_i2v_passes_correct_params(self):
        """Wan 2.1 I2V must forward is_moe=False, is_i2v=True, in_channels=36."""
        from flimmer.training.wan.registry import get_wan_backend

        mock_backend_cls = MagicMock(name="WanModelBackend")
        mock_backend_cls.return_value = MagicMock()

        config = _make_mock_config(variant="2.1_i2v")

        with patch.dict(
            sys.modules,
            {"flimmer.training.wan.backend": _make_mock_backend_module(mock_backend_cls)},
        ):
            get_wan_backend(config)

        _, kwargs = mock_backend_cls.call_args
        assert kwargs["is_moe"] is False
        assert kwargs["is_i2v"] is True
        assert kwargs["in_channels"] == 36
        assert kwargs["boundary_ratio"] is None
        # Config repo should be set from subtypes
        assert kwargs["config_repo"] is not None
        assert "2.1" in kwargs["config_repo"]
        assert "I2V" in kwargs["config_repo"]

    def test_model_path_forwarded_to_backend(self):
        """config.model.path must be passed as model_path to the backend."""
        from flimmer.training.wan.registry import get_wan_backend

        mock_backend_cls = MagicMock(name="WanModelBackend")
        mock_backend_cls.return_value = MagicMock()

        config = _make_mock_config(variant="2.2_t2v", path="/custom/path/to/model")

        with patch.dict(
            sys.modules,
            {"flimmer.training.wan.backend": _make_mock_backend_module(mock_backend_cls)},
        ):
            get_wan_backend(config)

        _, kwargs = mock_backend_cls.call_args
        assert kwargs["model_path"] == "/custom/path/to/model"


# ---------------------------------------------------------------------------
# Helper for mocking the backend module
# ---------------------------------------------------------------------------

def _make_mock_backend_module(mock_cls: MagicMock) -> ModuleType:
    """Build a fake flimmer.training.wan.backend module with a mock class.

    get_wan_backend() does:
        from flimmer.training.wan.backend import WanModelBackend

    By inserting a fake module into sys.modules before the call, that import
    resolves to our mock without ever loading torch or diffusers.
    """
    mod = ModuleType("flimmer.training.wan.backend")
    mod.WanModelBackend = mock_cls  # type: ignore[attr-defined]
    return mod
