"""End-to-end tests for MoE (Wan 2.2 T2V) and Wan 2.2 I2V model definitions.

Tests the full flow: model registration -> PhaseConfig creation ->
validate_against() -> resolve_phase() -> ResolvedPhase correctness.
"""

import pytest

from flimmer.phases import (
    PhaseConfig,
    PhaseConfigError,
    get_model_definition,
    list_models,
    resolve_phase,
)
from flimmer.phases.registry import clear_registry


@pytest.fixture(autouse=True)
def _clean_registry():
    """Isolate each test from registry side effects."""
    clear_registry()
    yield
    clear_registry()


@pytest.fixture
def _register_models():
    """Register MoE and I2V model definitions into the clean registry.

    Imports the module-level constants and re-registers them, avoiding
    issues with Python module caching and import side effects.
    """
    from flimmer.phases.models.moe import WAN_22_T2V
    from flimmer.phases.models.wan22_i2v import WAN_22_I2V
    from flimmer.phases.registry import register_model

    register_model(WAN_22_T2V, replace=True)
    register_model(WAN_22_I2V, replace=True)


# ---------------------------------------------------------------------------
# MoE (Wan 2.2 T2V) Registration Tests
# ---------------------------------------------------------------------------


class TestMoeRegistration:
    """Tests that the MoE model registers correctly."""

    def test_moe_model_registers_and_retrievable(self, _register_models):
        model = get_model_definition("wan-2.2-t2v-14b")
        assert model.model_id == "wan-2.2-t2v-14b"
        assert model.family == "wan"
        assert model.variant == "2.2_t2v"
        assert model.is_moe is True

    def test_moe_model_has_correct_phase_types(self, _register_models):
        model = get_model_definition("wan-2.2-t2v-14b")
        phase_type_names = [pt.name for pt in model.phase_types]
        assert "full_noise" in phase_type_names
        assert "high_noise" in phase_type_names
        assert "low_noise" in phase_type_names
        assert len(phase_type_names) == 3

    def test_moe_expert_phases_require_boundary_ratio(self, _register_models):
        model = get_model_definition("wan-2.2-t2v-14b")
        for pt in model.phase_types:
            if pt.name in ("high_noise", "low_noise"):
                assert "boundary_ratio" in pt.required_fields
            elif pt.name == "full_noise":
                assert "boundary_ratio" not in pt.required_fields

    def test_moe_has_phase_level_params(self, _register_models):
        model = get_model_definition("wan-2.2-t2v-14b")
        phase_param_names = {p.name for p in model.phase_params}
        expected = {
            "learning_rate",
            "weight_decay",
            "batch_size",
            "gradient_accumulation_steps",
            "caption_dropout_rate",
            "lora_dropout",
            "max_epochs",
            "min_lr_ratio",
            "optimizer_type",
            "scheduler_type",
            "boundary_ratio",
        }
        assert expected.issubset(phase_param_names)

    def test_moe_has_run_level_params(self, _register_models):
        model = get_model_definition("wan-2.2-t2v-14b")
        run_param_names = {p.name for p in model.run_level_params}
        expected = {"lora_rank", "lora_alpha", "mixed_precision", "base_model_precision"}
        assert expected.issubset(run_param_names)

    def test_moe_supported_signals(self, _register_models):
        model = get_model_definition("wan-2.2-t2v-14b")
        modalities = {s.modality for s in model.supported_signals}
        assert modalities == {"text", "video"}
        # Both should be required
        for sig in model.supported_signals:
            assert sig.required is True


# ---------------------------------------------------------------------------
# MoE Phase Resolution Tests
# ---------------------------------------------------------------------------


class TestMoeResolution:
    """Tests resolve_phase() with MoE model definitions."""

    @pytest.fixture
    def base_params(self, _register_models):
        """Base params from model defaults."""
        model = get_model_definition("wan-2.2-t2v-14b")
        return {p.name: p.default for p in model.phase_params}

    def test_full_noise_resolves_without_boundary_ratio_in_extras(self, base_params):
        config = PhaseConfig(phase_type="full_noise")
        resolved = resolve_phase("wan-2.2-t2v-14b", config, base_params)
        assert resolved.phase_type == "full_noise"
        # boundary_ratio should be in params (it's phase-level) but not required in extras
        assert "boundary_ratio" in resolved.params
        assert resolved.extras == {}

    def test_high_noise_resolves_with_boundary_ratio_in_extras(self, base_params):
        config = PhaseConfig(
            phase_type="high_noise",
            extras={"boundary_ratio": 0.875},
        )
        resolved = resolve_phase("wan-2.2-t2v-14b", config, base_params)
        assert resolved.phase_type == "high_noise"
        assert resolved.extras["boundary_ratio"] == 0.875

    def test_high_noise_fails_without_boundary_ratio(self, base_params):
        config = PhaseConfig(phase_type="high_noise")
        with pytest.raises(PhaseConfigError, match="boundary_ratio"):
            resolve_phase("wan-2.2-t2v-14b", config, base_params)

    def test_per_expert_lr_override(self, base_params):
        config = PhaseConfig(
            phase_type="high_noise",
            overrides={"learning_rate": 1e-4},
            extras={"boundary_ratio": 0.875},
        )
        resolved = resolve_phase("wan-2.2-t2v-14b", config, base_params)
        assert resolved.params["learning_rate"] == 1e-4

    def test_per_expert_dataset(self, base_params):
        config = PhaseConfig(
            phase_type="high_noise",
            dataset="./high_noise_data.yaml",
            extras={"boundary_ratio": 0.875},
        )
        resolved = resolve_phase("wan-2.2-t2v-14b", config, base_params)
        assert resolved.dataset == "./high_noise_data.yaml"


# ---------------------------------------------------------------------------
# Wan 2.2 I2V Registration and Signal Tests
# ---------------------------------------------------------------------------


class TestWan22I2v:
    """Tests for Wan 2.2 I2V model definition."""

    def test_i2v_registers_with_first_frame(self, _register_models):
        model = get_model_definition("wan-2.2-i2v-14b")
        modalities = {s.modality for s in model.supported_signals}
        assert "first_frame" in modalities
        assert "text" in modalities
        assert "video" in modalities

    def test_i2v_first_frame_is_required(self, _register_models):
        model = get_model_definition("wan-2.2-i2v-14b")
        ref_signal = next(
            s for s in model.supported_signals if s.modality == "first_frame"
        )
        assert ref_signal.required is True

    def test_i2v_higher_caption_dropout_default(self, _register_models):
        model = get_model_definition("wan-2.2-i2v-14b")
        caption_param = model.get_param("caption_dropout_rate")
        assert caption_param is not None
        assert caption_param.default == 0.15

    def test_i2v_is_moe(self, _register_models):
        model = get_model_definition("wan-2.2-i2v-14b")
        assert model.is_moe is True
        phase_type_names = [pt.name for pt in model.phase_types]
        assert "high_noise" in phase_type_names
        assert "low_noise" in phase_type_names

    def test_i2v_per_phase_signal_disable(self, _register_models):
        model = get_model_definition("wan-2.2-i2v-14b")
        base_params = {p.name: p.default for p in model.phase_params}
        config = PhaseConfig(
            phase_type="high_noise",
            signals={"first_frame": False},
            extras={"boundary_ratio": 0.875},
        )
        resolved = resolve_phase("wan-2.2-i2v-14b", config, base_params)
        assert resolved.signals["first_frame"] is False
        assert resolved.signals["text"] is True
        assert resolved.signals["video"] is True
