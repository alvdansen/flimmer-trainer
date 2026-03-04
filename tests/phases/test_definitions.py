"""Tests for definition types: ParamSpec, SignalDeclaration, PhaseTypeDeclaration, ModelDefinition."""

import pytest
from pydantic import ValidationError

from flimmer.phases import (
    FlimmerPhasesError,
    ModelDefinition,
    ModelNotFoundError,
    ParamSpec,
    PhaseConfigError,
    PhaseTypeDeclaration,
    SignalDeclaration,
)


# ---------------------------------------------------------------------------
# ParamSpec validation
# ---------------------------------------------------------------------------


class TestParamSpec:
    """ParamSpec validates type, range, and default consistency."""

    def test_valid_float_param(self) -> None:
        p = ParamSpec(
            name="learning_rate",
            type="float",
            default=5e-5,
            min_value=1e-6,
            max_value=1e-3,
        )
        assert p.name == "learning_rate"
        assert p.type == "float"
        assert p.default == 5e-5
        assert p.phase_level is True  # default

    def test_valid_int_param(self) -> None:
        p = ParamSpec(name="batch_size", type="int", default=4, min_value=1, max_value=64)
        assert p.default == 4

    def test_valid_str_param(self) -> None:
        p = ParamSpec(name="model_path", type="str", default="/weights")
        assert p.default == "/weights"

    def test_valid_bool_param(self) -> None:
        p = ParamSpec(name="grad_ckpt", type="bool", default=True)
        assert p.default is True

    def test_phase_level_default_true(self) -> None:
        p = ParamSpec(name="lr", type="float", default=1e-4)
        assert p.phase_level is True

    def test_phase_level_explicit_false(self) -> None:
        p = ParamSpec(name="weights", type="str", default="", phase_level=False)
        assert p.phase_level is False

    def test_rejects_min_greater_than_max(self) -> None:
        with pytest.raises(ValidationError, match="min_value.*max_value|max_value.*min_value"):
            ParamSpec(name="bad", type="float", default=0.5, min_value=1.0, max_value=0.1)

    def test_rejects_default_below_min(self) -> None:
        with pytest.raises(ValidationError, match="default"):
            ParamSpec(name="bad", type="float", default=0.0, min_value=0.1, max_value=1.0)

    def test_rejects_default_above_max(self) -> None:
        with pytest.raises(ValidationError, match="default"):
            ParamSpec(name="bad", type="float", default=2.0, min_value=0.1, max_value=1.0)

    def test_allows_default_at_min_boundary(self) -> None:
        p = ParamSpec(name="p", type="float", default=0.0, min_value=0.0, max_value=1.0)
        assert p.default == 0.0

    def test_allows_default_at_max_boundary(self) -> None:
        p = ParamSpec(name="p", type="float", default=1.0, min_value=0.0, max_value=1.0)
        assert p.default == 1.0

    def test_allows_no_range(self) -> None:
        """When no min/max are provided, any default is accepted."""
        p = ParamSpec(name="p", type="float", default=999.0)
        assert p.min_value is None
        assert p.max_value is None

    def test_allows_only_min(self) -> None:
        p = ParamSpec(name="p", type="float", default=5.0, min_value=0.0)
        assert p.max_value is None

    def test_allows_only_max(self) -> None:
        p = ParamSpec(name="p", type="float", default=5.0, max_value=10.0)
        assert p.min_value is None

    def test_rejects_default_below_min_when_only_min(self) -> None:
        with pytest.raises(ValidationError, match="default"):
            ParamSpec(name="bad", type="float", default=-1.0, min_value=0.0)

    def test_rejects_default_above_max_when_only_max(self) -> None:
        with pytest.raises(ValidationError, match="default"):
            ParamSpec(name="bad", type="float", default=11.0, max_value=10.0)

    def test_rejects_invalid_type_literal(self) -> None:
        with pytest.raises(ValidationError):
            ParamSpec(name="bad", type="complex", default=1.0)

    def test_description_default_empty(self) -> None:
        p = ParamSpec(name="p", type="float", default=1.0)
        assert p.description == ""


# ---------------------------------------------------------------------------
# SignalDeclaration
# ---------------------------------------------------------------------------


class TestSignalDeclaration:
    """SignalDeclaration accepts arbitrary string modality (NOT an enum)."""

    def test_standard_modality(self) -> None:
        s = SignalDeclaration(modality="text")
        assert s.modality == "text"
        assert s.required is False

    def test_custom_modality(self) -> None:
        s = SignalDeclaration(modality="my_custom_signal", required=True)
        assert s.modality == "my_custom_signal"
        assert s.required is True

    def test_arbitrary_string_modality(self) -> None:
        """Ensures modality is NOT an enum -- any string works."""
        s = SignalDeclaration(modality="something_never_seen_before")
        assert s.modality == "something_never_seen_before"

    def test_description_default_empty(self) -> None:
        s = SignalDeclaration(modality="video")
        assert s.description == ""


# ---------------------------------------------------------------------------
# PhaseTypeDeclaration
# ---------------------------------------------------------------------------


class TestPhaseTypeDeclaration:
    """PhaseTypeDeclaration has name, description, required_fields."""

    def test_basic_creation(self) -> None:
        pt = PhaseTypeDeclaration(name="unified", description="All noise levels")
        assert pt.name == "unified"
        assert pt.required_fields == []

    def test_with_required_fields(self) -> None:
        pt = PhaseTypeDeclaration(
            name="high_noise",
            required_fields=["boundary_ratio"],
        )
        assert pt.required_fields == ["boundary_ratio"]


# ---------------------------------------------------------------------------
# ModelDefinition
# ---------------------------------------------------------------------------


class TestModelDefinition:
    """ModelDefinition validates cross-references and provides filtering properties."""

    def test_full_creation(self, sample_model_definition: ModelDefinition) -> None:
        m = sample_model_definition
        assert m.model_id == "wan-2.2-t2v-14b"
        assert m.family == "wan"
        assert m.is_moe is True
        assert len(m.supported_signals) == 4
        assert len(m.phase_types) == 3
        assert len(m.params) == 6

    def test_run_level_params(self, sample_model_definition: ModelDefinition) -> None:
        """run_level_params returns only params where phase_level=False."""
        run_params = sample_model_definition.run_level_params
        names = [p.name for p in run_params]
        assert "model_weights_path" in names
        assert "use_gradient_checkpointing" in names
        assert "learning_rate" not in names
        assert len(run_params) == 2

    def test_phase_params(self, sample_model_definition: ModelDefinition) -> None:
        """phase_params returns only params where phase_level=True."""
        phase_params = sample_model_definition.phase_params
        names = [p.name for p in phase_params]
        assert "learning_rate" in names
        assert "batch_size" in names
        assert "dropout" in names
        assert "boundary_ratio" in names
        assert "model_weights_path" not in names
        assert len(phase_params) == 4

    def test_get_param_found(self, sample_model_definition: ModelDefinition) -> None:
        p = sample_model_definition.get_param("learning_rate")
        assert p is not None
        assert p.name == "learning_rate"

    def test_get_param_not_found(self, sample_model_definition: ModelDefinition) -> None:
        p = sample_model_definition.get_param("nonexistent")
        assert p is None

    def test_validates_required_fields_reference_existing_params(self) -> None:
        """required_fields in PhaseTypeDeclarations must reference existing param names."""
        with pytest.raises(ValidationError, match="boundary_ratio"):
            ModelDefinition(
                model_id="bad-model",
                family="test",
                variant="v1",
                display_name="Bad Model",
                supported_signals=[],
                phase_types=[
                    PhaseTypeDeclaration(
                        name="expert",
                        required_fields=["boundary_ratio"],  # not in params!
                    ),
                ],
                params=[
                    ParamSpec(name="learning_rate", type="float", default=1e-4),
                ],
            )

    def test_defaults_dict(self, sample_model_definition: ModelDefinition) -> None:
        assert sample_model_definition.defaults["learning_rate"] == 2e-4

    def test_defaults_empty_by_default(self) -> None:
        m = ModelDefinition(
            model_id="minimal",
            family="test",
            variant="v1",
            display_name="Minimal",
            supported_signals=[],
            phase_types=[],
            params=[],
        )
        assert m.defaults == {}


# ---------------------------------------------------------------------------
# Error hierarchy
# ---------------------------------------------------------------------------


class TestErrorHierarchy:
    """Custom exception hierarchy inherits correctly."""

    def test_base_error_is_exception(self) -> None:
        assert issubclass(FlimmerPhasesError, Exception)

    def test_model_not_found_inherits_base(self) -> None:
        assert issubclass(ModelNotFoundError, FlimmerPhasesError)

    def test_phase_config_error_inherits_base(self) -> None:
        assert issubclass(PhaseConfigError, FlimmerPhasesError)

    def test_model_not_found_has_model_id(self) -> None:
        err = ModelNotFoundError("test-model", ["model-a", "model-b"])
        assert err.model_id == "test-model"
        assert "test-model" in str(err)
        assert "model-a" in str(err)

    def test_phase_config_error_has_detail(self) -> None:
        err = PhaseConfigError("bad config value")
        assert err.detail == "bad config value"
        assert "bad config value" in str(err)
