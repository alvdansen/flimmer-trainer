"""Tests for phase resolution engine: get_phase_knobs and resolve_phase."""

import dataclasses

import pytest

from flimmer.phases import (
    ModelDefinition,
    ModelNotFoundError,
    ParamSpec,
    PhaseConfigError,
    PhaseTypeDeclaration,
    SignalDeclaration,
    register_model,
)
from flimmer.phases.phase_model import PhaseConfig, ResolvedPhase
from flimmer.phases.registry import clear_registry
from flimmer.phases.resolution import get_phase_knobs, resolve_phase


@pytest.fixture(autouse=True)
def _clean_registry() -> None:
    """Ensure each test starts with a clean registry."""
    clear_registry()


@pytest.fixture
def registered_model(sample_model_definition: ModelDefinition) -> ModelDefinition:
    """Register the sample model and return it."""
    register_model(sample_model_definition)
    return sample_model_definition


class TestGetPhaseKnobs:
    """Test get_phase_knobs lookup."""

    def test_returns_phase_level_params(
        self, registered_model: ModelDefinition
    ) -> None:
        knobs = get_phase_knobs("wan-2.2-t2v-14b", "unified")
        # Should include all phase_level=True params
        assert "learning_rate" in knobs
        assert "batch_size" in knobs
        assert "dropout" in knobs
        # Should NOT include run-level params
        assert "model_weights_path" not in knobs
        assert "use_gradient_checkpointing" not in knobs

    def test_includes_required_fields_params(
        self, registered_model: ModelDefinition
    ) -> None:
        """high_noise requires boundary_ratio -- it should appear in knobs."""
        knobs = get_phase_knobs("wan-2.2-t2v-14b", "high_noise")
        assert "boundary_ratio" in knobs
        assert isinstance(knobs["boundary_ratio"], ParamSpec)

    def test_unified_phase_includes_phase_params(
        self, registered_model: ModelDefinition
    ) -> None:
        """unified has no required_fields, but still gets all phase-level params."""
        knobs = get_phase_knobs("wan-2.2-t2v-14b", "unified")
        assert "learning_rate" in knobs
        assert "batch_size" in knobs
        assert "boundary_ratio" in knobs  # phase_level=True in sample

    def test_returns_param_spec_objects(
        self, registered_model: ModelDefinition
    ) -> None:
        knobs = get_phase_knobs("wan-2.2-t2v-14b", "unified")
        lr = knobs["learning_rate"]
        assert isinstance(lr, ParamSpec)
        assert lr.type == "float"
        assert lr.min_value == 1e-6
        assert lr.max_value == 1e-3

    def test_unknown_model_raises(self) -> None:
        with pytest.raises(ModelNotFoundError, match="nonexistent"):
            get_phase_knobs("nonexistent", "unified")

    def test_unknown_phase_type_raises(
        self, registered_model: ModelDefinition
    ) -> None:
        with pytest.raises(PhaseConfigError, match="fake_phase"):
            get_phase_knobs("wan-2.2-t2v-14b", "fake_phase")


class TestResolvePhase:
    """Test resolve_phase resolution engine."""

    def test_all_defaults_no_overrides(
        self,
        registered_model: ModelDefinition,
        base_config: dict[str, float | int | str | bool],
    ) -> None:
        config = PhaseConfig(phase_type="unified")
        resolved = resolve_phase("wan-2.2-t2v-14b", config, base_config)
        assert isinstance(resolved, ResolvedPhase)
        assert resolved.phase_type == "unified"
        # All phase-level params should have values from base_config
        assert resolved.params["learning_rate"] == base_config["learning_rate"]
        assert resolved.params["batch_size"] == base_config["batch_size"]

    def test_overrides_replace_base_values(
        self,
        registered_model: ModelDefinition,
        base_config: dict[str, float | int | str | bool],
    ) -> None:
        config = PhaseConfig(
            phase_type="unified",
            overrides={"learning_rate": 2e-4},
        )
        resolved = resolve_phase("wan-2.2-t2v-14b", config, base_config)
        assert resolved.params["learning_rate"] == 2e-4
        # Non-overridden params should keep base values
        assert resolved.params["batch_size"] == base_config["batch_size"]

    def test_none_override_inherits_from_base(
        self,
        registered_model: ModelDefinition,
        base_config: dict[str, float | int | str | bool],
    ) -> None:
        """None in overrides = inherit from base (the _resolve idiom)."""
        config = PhaseConfig(
            phase_type="unified",
            overrides={"learning_rate": None},
        )
        resolved = resolve_phase("wan-2.2-t2v-14b", config, base_config)
        assert resolved.params["learning_rate"] == base_config["learning_rate"]

    def test_base_params_used_when_no_override(
        self,
        registered_model: ModelDefinition,
    ) -> None:
        """Base params dict provides values, not just ParamSpec defaults."""
        custom_base = {
            "learning_rate": 9e-5,
            "batch_size": 8,
            "dropout": 0.2,
            "boundary_ratio": 0.5,
        }
        config = PhaseConfig(phase_type="unified")
        resolved = resolve_phase("wan-2.2-t2v-14b", config, custom_base)
        assert resolved.params["learning_rate"] == 9e-5
        assert resolved.params["batch_size"] == 8

    def test_validates_config_against_model(
        self,
        registered_model: ModelDefinition,
        base_config: dict[str, float | int | str | bool],
    ) -> None:
        """Invalid phase_type should be caught during resolve."""
        config = PhaseConfig(phase_type="nonexistent")
        with pytest.raises(PhaseConfigError, match="Unknown phase type"):
            resolve_phase("wan-2.2-t2v-14b", config, base_config)

    def test_missing_required_param_no_value_source_raises(
        self,
        registered_model: ModelDefinition,
    ) -> None:
        """If a phase-level param has no override, no base value, and no default,
        resolution should fail."""
        # Create a model with a param that has no sensible default
        model_def = ModelDefinition(
            model_id="test-model",
            family="test",
            variant="v1",
            display_name="Test",
            supported_signals=[SignalDeclaration(modality="text")],
            phase_types=[PhaseTypeDeclaration(name="basic")],
            params=[
                ParamSpec(
                    name="required_param",
                    type="float",
                    default=0.0,  # ParamSpec requires a default
                    min_value=0.0,
                    max_value=1.0,
                    phase_level=True,
                ),
            ],
        )
        register_model(model_def)
        # With base_params that don't include the param, resolution
        # should still work because ParamSpec has a default
        config = PhaseConfig(phase_type="basic")
        resolved = resolve_phase("test-model", config, {})
        assert resolved.params["required_param"] == 0.0

    def test_extras_passed_through_to_resolved(
        self,
        registered_model: ModelDefinition,
        base_config: dict[str, float | int | str | bool],
    ) -> None:
        config = PhaseConfig(
            phase_type="high_noise",
            extras={"boundary_ratio": 0.875},
        )
        resolved = resolve_phase("wan-2.2-t2v-14b", config, base_config)
        assert resolved.extras["boundary_ratio"] == 0.875

    def test_resolved_phase_is_frozen(
        self,
        registered_model: ModelDefinition,
        base_config: dict[str, float | int | str | bool],
    ) -> None:
        config = PhaseConfig(phase_type="unified")
        resolved = resolve_phase("wan-2.2-t2v-14b", config, base_config)
        with pytest.raises(dataclasses.FrozenInstanceError):
            resolved.phase_type = "different"  # type: ignore[misc]

    def test_display_name_from_config(
        self,
        registered_model: ModelDefinition,
        base_config: dict[str, float | int | str | bool],
    ) -> None:
        config = PhaseConfig(
            phase_type="unified",
            display_name="My Custom Phase",
        )
        resolved = resolve_phase("wan-2.2-t2v-14b", config, base_config)
        assert resolved.display_name == "My Custom Phase"

    def test_enabled_from_config(
        self,
        registered_model: ModelDefinition,
        base_config: dict[str, float | int | str | bool],
    ) -> None:
        config = PhaseConfig(phase_type="unified", enabled=False)
        resolved = resolve_phase("wan-2.2-t2v-14b", config, base_config)
        assert resolved.enabled is False

    def test_full_round_trip(self) -> None:
        """Full round-trip: register -> config -> resolve -> read params."""
        # Register a simple model
        model_def = ModelDefinition(
            model_id="round-trip-model",
            family="test",
            variant="v1",
            display_name="Round Trip Model",
            supported_signals=[
                SignalDeclaration(modality="video", required=True),
            ],
            phase_types=[
                PhaseTypeDeclaration(name="main"),
                PhaseTypeDeclaration(
                    name="expert",
                    required_fields=["threshold"],
                ),
            ],
            params=[
                ParamSpec(
                    name="lr",
                    type="float",
                    default=1e-4,
                    min_value=1e-6,
                    max_value=1e-2,
                    phase_level=True,
                ),
                ParamSpec(
                    name="steps",
                    type="int",
                    default=1000,
                    min_value=1,
                    max_value=100000,
                    phase_level=True,
                ),
                ParamSpec(
                    name="threshold",
                    type="float",
                    default=0.5,
                    min_value=0.0,
                    max_value=1.0,
                    phase_level=True,
                ),
            ],
        )
        register_model(model_def)

        # Create phase config with overrides
        config = PhaseConfig(
            phase_type="expert",
            display_name="Expert Phase 1",
            overrides={"lr": 5e-5, "steps": 2000},
            extras={"threshold": 0.7},
        )

        # Resolve with base params
        base = {"lr": 1e-4, "steps": 1000, "threshold": 0.5}
        resolved = resolve_phase("round-trip-model", config, base)

        # Verify everything came through
        assert resolved.phase_type == "expert"
        assert resolved.display_name == "Expert Phase 1"
        assert resolved.enabled is True
        assert resolved.get_param("lr") == 5e-5  # overridden
        assert resolved.get_param("steps") == 2000  # overridden
        assert resolved.get_param("threshold") == 0.5  # from base
        assert resolved.extras["threshold"] == 0.7


class TestResolvePhaseDataset:
    """Test resolve_phase dataset pass-through."""

    def test_no_dataset_resolves_to_none(
        self,
        registered_model: ModelDefinition,
        base_config: dict[str, float | int | str | bool],
    ) -> None:
        """resolve_phase with no dataset -> ResolvedPhase.dataset is None."""
        config = PhaseConfig(phase_type="unified")
        resolved = resolve_phase("wan-2.2-t2v-14b", config, base_config)
        assert resolved.dataset is None

    def test_dataset_passed_through(
        self,
        registered_model: ModelDefinition,
        base_config: dict[str, float | int | str | bool],
    ) -> None:
        """resolve_phase with dataset path -> ResolvedPhase.dataset is the path."""
        config = PhaseConfig(
            phase_type="unified",
            dataset="./data.yaml",
        )
        resolved = resolve_phase("wan-2.2-t2v-14b", config, base_config)
        assert resolved.dataset == "./data.yaml"


class TestResolvePhaseSignals:
    """Test resolve_phase signal resolution (selective merge with model defaults)."""

    def test_no_signals_all_enabled(
        self,
        registered_model: ModelDefinition,
        base_config: dict[str, float | int | str | bool],
    ) -> None:
        """signals=None -> all model modalities enabled (True)."""
        config = PhaseConfig(phase_type="unified")
        resolved = resolve_phase("wan-2.2-t2v-14b", config, base_config)
        # Sample model has: text, video, reference_image, custom
        assert resolved.signals == {
            "text": True,
            "video": True,
            "reference_image": True,
            "custom": True,
        }

    def test_selective_disable(
        self,
        registered_model: ModelDefinition,
        base_config: dict[str, float | int | str | bool],
    ) -> None:
        """signals={"reference_image": False} -> merged with all-True defaults."""
        config = PhaseConfig(
            phase_type="unified",
            signals={"reference_image": False},
        )
        resolved = resolve_phase("wan-2.2-t2v-14b", config, base_config)
        assert resolved.signals == {
            "text": True,
            "video": True,
            "reference_image": False,
            "custom": True,
        }

    def test_disable_text(
        self,
        registered_model: ModelDefinition,
        base_config: dict[str, float | int | str | bool],
    ) -> None:
        """signals={"text": False} -> text disabled, others True."""
        config = PhaseConfig(
            phase_type="unified",
            signals={"text": False},
        )
        resolved = resolve_phase("wan-2.2-t2v-14b", config, base_config)
        assert resolved.signals["text"] is False
        assert resolved.signals["video"] is True
        assert resolved.signals["reference_image"] is True
        assert resolved.signals["custom"] is True

    def test_multiple_overrides(
        self,
        registered_model: ModelDefinition,
        base_config: dict[str, float | int | str | bool],
    ) -> None:
        """Multiple signal overrides merged correctly."""
        config = PhaseConfig(
            phase_type="unified",
            signals={"text": False, "custom": False},
        )
        resolved = resolve_phase("wan-2.2-t2v-14b", config, base_config)
        assert resolved.signals == {
            "text": False,
            "video": True,
            "reference_image": True,
            "custom": False,
        }

    def test_empty_signals_dict_all_enabled(
        self,
        registered_model: ModelDefinition,
        base_config: dict[str, float | int | str | bool],
    ) -> None:
        """signals={} (no overrides) -> same as None, all enabled."""
        config = PhaseConfig(
            phase_type="unified",
            signals={},
        )
        resolved = resolve_phase("wan-2.2-t2v-14b", config, base_config)
        assert resolved.signals == {
            "text": True,
            "video": True,
            "reference_image": True,
            "custom": True,
        }

    def test_validation_still_runs(
        self,
        registered_model: ModelDefinition,
        base_config: dict[str, float | int | str | bool],
    ) -> None:
        """resolve_phase still validates config before resolving signals."""
        config = PhaseConfig(
            phase_type="unified",
            signals={"audio": True},  # Invalid modality
        )
        with pytest.raises(PhaseConfigError, match="audio"):
            resolve_phase("wan-2.2-t2v-14b", config, base_config)
