"""Tests for Wan 2.1 model definitions and multi-stage training.

Verifies:
- Wan 2.1 T2V and I2V register as non-MoE with unified-only phase type
- Wan 2.1 models have NO boundary_ratio param
- I2V variant has reference_image signal and higher caption_dropout_rate
- Expert phase types fail validate_against() for non-MoE models
- Multi-stage training via multiple unified PhaseConfigs with different overrides
- All four model definitions appear in list_models()
"""

import pytest

from flimmer.phases import (
    PhaseConfig,
    PhaseConfigError,
    get_model_definition,
    list_models,
    resolve_phase,
)
from flimmer.phases.registry import clear_registry, register_model


@pytest.fixture(autouse=True)
def _clean_registry():
    """Isolate each test from registry side effects."""
    clear_registry()
    yield
    clear_registry()


@pytest.fixture
def _register_all():
    """Register all four model definitions."""
    from flimmer.phases.models.moe import WAN_22_T2V
    from flimmer.phases.models.wan21_i2v import WAN_21_I2V
    from flimmer.phases.models.wan21_t2v import WAN_21_T2V
    from flimmer.phases.models.wan22_i2v import WAN_22_I2V

    register_model(WAN_22_T2V, replace=True)
    register_model(WAN_22_I2V, replace=True)
    register_model(WAN_21_T2V, replace=True)
    register_model(WAN_21_I2V, replace=True)


# ---------------------------------------------------------------------------
# Wan 2.1 T2V Registration Tests
# ---------------------------------------------------------------------------


class TestWan21T2vRegistration:
    """Tests that Wan 2.1 T2V registers correctly as non-MoE."""

    def test_registers_and_retrievable(self, _register_all):
        model = get_model_definition("wan-2.1-t2v-14b")
        assert model.model_id == "wan-2.1-t2v-14b"
        assert model.family == "wan"
        assert model.variant == "2.1_t2v"

    def test_is_not_moe(self, _register_all):
        model = get_model_definition("wan-2.1-t2v-14b")
        assert model.is_moe is False

    def test_has_only_unified_phase_type(self, _register_all):
        model = get_model_definition("wan-2.1-t2v-14b")
        phase_type_names = [pt.name for pt in model.phase_types]
        assert phase_type_names == ["unified"]

    def test_no_boundary_ratio_param(self, _register_all):
        model = get_model_definition("wan-2.1-t2v-14b")
        param = model.get_param("boundary_ratio")
        assert param is None

    def test_supported_signals_text_and_video(self, _register_all):
        model = get_model_definition("wan-2.1-t2v-14b")
        modalities = {s.modality for s in model.supported_signals}
        assert modalities == {"text", "video"}

    def test_caption_dropout_rate_default(self, _register_all):
        model = get_model_definition("wan-2.1-t2v-14b")
        param = model.get_param("caption_dropout_rate")
        assert param is not None
        assert param.default == 0.10


# ---------------------------------------------------------------------------
# Wan 2.1 I2V Registration Tests
# ---------------------------------------------------------------------------


class TestWan21I2vRegistration:
    """Tests that Wan 2.1 I2V registers correctly with reference_image."""

    def test_registers_with_reference_image(self, _register_all):
        model = get_model_definition("wan-2.1-i2v-14b")
        modalities = {s.modality for s in model.supported_signals}
        assert "reference_image" in modalities

    def test_higher_caption_dropout_default(self, _register_all):
        model = get_model_definition("wan-2.1-i2v-14b")
        param = model.get_param("caption_dropout_rate")
        assert param is not None
        assert param.default == 0.15

    def test_is_not_moe(self, _register_all):
        model = get_model_definition("wan-2.1-i2v-14b")
        assert model.is_moe is False


# ---------------------------------------------------------------------------
# Cross-model Validation Tests
# ---------------------------------------------------------------------------


class TestCrossModelValidation:
    """Tests that phase types are correctly enforced across models."""

    def test_high_noise_fails_for_wan21_t2v(self, _register_all):
        config = PhaseConfig(
            phase_type="high_noise",
            extras={"boundary_ratio": 0.875},
        )
        model = get_model_definition("wan-2.1-t2v-14b")
        with pytest.raises(PhaseConfigError, match="high_noise"):
            config.validate_against(model)

    def test_high_noise_fails_for_wan21_i2v(self, _register_all):
        config = PhaseConfig(
            phase_type="high_noise",
            extras={"boundary_ratio": 0.875},
        )
        model = get_model_definition("wan-2.1-i2v-14b")
        with pytest.raises(PhaseConfigError, match="high_noise"):
            config.validate_against(model)


# ---------------------------------------------------------------------------
# Multi-stage Training Tests
# ---------------------------------------------------------------------------


class TestMultiStageTraining:
    """Tests multi-stage training via multiple unified PhaseConfigs."""

    def test_two_unified_stages_resolve_independently(self, _register_all):
        model = get_model_definition("wan-2.1-t2v-14b")
        base_params = {p.name: p.default for p in model.phase_params}

        # Stage 1: warmup with higher LR
        stage1_config = PhaseConfig(
            phase_type="unified",
            display_name="Warmup",
            overrides={"learning_rate": 1e-4},
            dataset="./warmup.yaml",
        )

        # Stage 2: finetune with lower LR
        stage2_config = PhaseConfig(
            phase_type="unified",
            display_name="Finetune",
            overrides={"learning_rate": 5e-5},
            dataset="./finetune.yaml",
        )

        resolved1 = resolve_phase("wan-2.1-t2v-14b", stage1_config, base_params)
        resolved2 = resolve_phase("wan-2.1-t2v-14b", stage2_config, base_params)

        # Both resolve successfully
        assert resolved1.phase_type == "unified"
        assert resolved2.phase_type == "unified"

        # Different LR overrides
        assert resolved1.params["learning_rate"] == 1e-4
        assert resolved2.params["learning_rate"] == 5e-5

        # Different datasets
        assert resolved1.dataset == "./warmup.yaml"
        assert resolved2.dataset == "./finetune.yaml"

        # Different display names
        assert resolved1.display_name == "Warmup"
        assert resolved2.display_name == "Finetune"


# ---------------------------------------------------------------------------
# All Models in Registry Test
# ---------------------------------------------------------------------------


class TestAllModelsRegistered:
    """Tests that all four models appear after importing models package."""

    def test_all_four_models_in_list(self, _register_all):
        models = list_models()
        assert "wan-2.2-t2v-14b" in models
        assert "wan-2.2-i2v-14b" in models
        assert "wan-2.1-t2v-14b" in models
        assert "wan-2.1-i2v-14b" in models
        assert len(models) == 4
