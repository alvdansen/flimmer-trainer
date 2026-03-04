"""Tests for PhaseConfig validation and ResolvedPhase immutability."""

import dataclasses

import pytest

from flimmer.phases import ModelDefinition, PhaseConfigError
from flimmer.phases.phase_model import PhaseConfig, ResolvedPhase


class TestPhaseConfigCreation:
    """Test PhaseConfig creation with valid data."""

    def test_minimal_creation(self) -> None:
        config = PhaseConfig(phase_type="full_noise")
        assert config.phase_type == "full_noise"
        assert config.display_name == ""
        assert config.enabled is True
        assert config.overrides == {}
        assert config.extras == {}
        # New fields default to None (backward compatible)
        assert config.dataset is None
        assert config.signals is None

    def test_full_creation(self) -> None:
        config = PhaseConfig(
            phase_type="high_noise",
            display_name="High Noise Phase",
            enabled=True,
            overrides={"learning_rate": 1e-4},
            extras={"boundary_ratio": 0.875},
        )
        assert config.phase_type == "high_noise"
        assert config.display_name == "High Noise Phase"
        assert config.overrides == {"learning_rate": 1e-4}
        assert config.extras == {"boundary_ratio": 0.875}

    def test_disabled_phase(self) -> None:
        config = PhaseConfig(phase_type="full_noise", enabled=False)
        assert config.enabled is False


class TestPhaseConfigValidation:
    """Test PhaseConfig.validate_against method."""

    def test_valid_config_passes(
        self, sample_model_definition: ModelDefinition
    ) -> None:
        config = PhaseConfig(phase_type="full_noise")
        # Should not raise
        config.validate_against(sample_model_definition)

    def test_valid_config_with_override(
        self, sample_model_definition: ModelDefinition
    ) -> None:
        config = PhaseConfig(
            phase_type="full_noise",
            overrides={"learning_rate": 1e-4},
        )
        config.validate_against(sample_model_definition)

    def test_valid_config_with_required_extras(
        self, sample_model_definition: ModelDefinition
    ) -> None:
        config = PhaseConfig(
            phase_type="high_noise",
            overrides={"learning_rate": 1e-4},
            extras={"boundary_ratio": 0.875},
        )
        config.validate_against(sample_model_definition)

    def test_unknown_phase_type_raises(
        self, sample_model_definition: ModelDefinition
    ) -> None:
        config = PhaseConfig(phase_type="nonexistent")
        with pytest.raises(PhaseConfigError, match="Unknown phase type"):
            config.validate_against(sample_model_definition)

    def test_unknown_phase_type_includes_valid_types(
        self, sample_model_definition: ModelDefinition
    ) -> None:
        config = PhaseConfig(phase_type="nonexistent")
        with pytest.raises(PhaseConfigError, match="full_noise"):
            config.validate_against(sample_model_definition)

    def test_invalid_override_key_raises(
        self, sample_model_definition: ModelDefinition
    ) -> None:
        config = PhaseConfig(
            phase_type="full_noise",
            overrides={"no_such_param": 42},
        )
        with pytest.raises(PhaseConfigError, match="no_such_param"):
            config.validate_against(sample_model_definition)

    def test_override_run_level_param_raises(
        self, sample_model_definition: ModelDefinition
    ) -> None:
        """Run-level params (phase_level=False) cannot be overridden per-phase."""
        config = PhaseConfig(
            phase_type="full_noise",
            overrides={"model_weights_path": "/some/path"},
        )
        with pytest.raises(PhaseConfigError, match="model_weights_path"):
            config.validate_against(sample_model_definition)

    def test_override_value_below_min_raises(
        self, sample_model_definition: ModelDefinition
    ) -> None:
        config = PhaseConfig(
            phase_type="full_noise",
            overrides={"learning_rate": 1e-9},  # min is 1e-6
        )
        with pytest.raises(PhaseConfigError, match="learning_rate"):
            config.validate_against(sample_model_definition)

    def test_override_value_above_max_raises(
        self, sample_model_definition: ModelDefinition
    ) -> None:
        config = PhaseConfig(
            phase_type="full_noise",
            overrides={"learning_rate": 1.0},  # max is 1e-3
        )
        with pytest.raises(PhaseConfigError, match="learning_rate"):
            config.validate_against(sample_model_definition)

    def test_missing_required_extras_raises(
        self, sample_model_definition: ModelDefinition
    ) -> None:
        """high_noise requires boundary_ratio in extras."""
        config = PhaseConfig(
            phase_type="high_noise",
            overrides={"learning_rate": 1e-4},
            extras={},  # Missing boundary_ratio
        )
        with pytest.raises(PhaseConfigError, match="boundary_ratio"):
            config.validate_against(sample_model_definition)


class TestResolvedPhase:
    """Test ResolvedPhase immutability and get_param."""

    def test_creation(self) -> None:
        resolved = ResolvedPhase(
            phase_type="full_noise",
            display_name="Main Phase",
            enabled=True,
            params={"learning_rate": 5e-5, "batch_size": 4},
            extras={},
            dataset=None,
            signals={"text": True},
        )
        assert resolved.phase_type == "full_noise"
        assert resolved.display_name == "Main Phase"
        assert resolved.enabled is True
        assert resolved.params["learning_rate"] == 5e-5
        assert resolved.dataset is None
        assert resolved.signals == {"text": True}

    def test_frozen_cannot_assign_phase_type(self) -> None:
        resolved = ResolvedPhase(
            phase_type="full_noise",
            display_name="",
            enabled=True,
            params={},
            extras={},
            dataset=None,
            signals={},
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            resolved.phase_type = "different"  # type: ignore[misc]

    def test_frozen_cannot_assign_params(self) -> None:
        resolved = ResolvedPhase(
            phase_type="full_noise",
            display_name="",
            enabled=True,
            params={"learning_rate": 5e-5},
            extras={},
            dataset=None,
            signals={},
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            resolved.params = {}  # type: ignore[misc]

    def test_get_param_returns_value(self) -> None:
        resolved = ResolvedPhase(
            phase_type="full_noise",
            display_name="",
            enabled=True,
            params={"learning_rate": 5e-5, "batch_size": 4},
            extras={},
            dataset=None,
            signals={},
        )
        assert resolved.get_param("learning_rate") == 5e-5
        assert resolved.get_param("batch_size") == 4

    def test_get_param_unknown_raises_key_error(self) -> None:
        resolved = ResolvedPhase(
            phase_type="full_noise",
            display_name="",
            enabled=True,
            params={"learning_rate": 5e-5},
            extras={},
            dataset=None,
            signals={},
        )
        with pytest.raises(KeyError, match="no_such_param"):
            resolved.get_param("no_such_param")

    def test_extras_accessible(self) -> None:
        resolved = ResolvedPhase(
            phase_type="high_noise",
            display_name="HN",
            enabled=True,
            params={"learning_rate": 5e-5},
            extras={"boundary_ratio": 0.875},
            dataset=None,
            signals={},
        )
        assert resolved.extras["boundary_ratio"] == 0.875
