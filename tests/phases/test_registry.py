"""Tests for model registry: register_model, get_model_definition, list_models."""

import pytest

from flimmer.phases import (
    ModelDefinition,
    ModelNotFoundError,
    ParamSpec,
    PhaseTypeDeclaration,
    SignalDeclaration,
)
from flimmer.phases.registry import (
    MODEL_REGISTRY,
    clear_registry,
    get_model_definition,
    list_models,
    register_model,
)


@pytest.fixture(autouse=True)
def _clean_registry() -> None:
    """Clear the registry before each test to prevent pollution."""
    clear_registry()


def _make_definition(model_id: str = "test-model") -> ModelDefinition:
    """Helper to create a minimal ModelDefinition for testing."""
    return ModelDefinition(
        model_id=model_id,
        family="test",
        variant="v1",
        display_name=f"Test Model ({model_id})",
        supported_signals=[
            SignalDeclaration(modality="text", required=True),
        ],
        phase_types=[
            PhaseTypeDeclaration(name="default"),
        ],
        params=[
            ParamSpec(name="learning_rate", type="float", default=1e-4),
        ],
    )


class TestRegisterModel:
    """register_model adds definitions and prevents silent overwrites."""

    def test_register_and_retrieve(self) -> None:
        defn = _make_definition("model-a")
        register_model(defn)
        result = get_model_definition("model-a")
        assert result.model_id == "model-a"
        assert result.display_name == "Test Model (model-a)"

    def test_duplicate_registration_raises_value_error(self) -> None:
        defn = _make_definition("model-a")
        register_model(defn)
        with pytest.raises(ValueError, match="already registered"):
            register_model(defn)

    def test_replace_true_allows_overwrite(self) -> None:
        defn1 = _make_definition("model-a")
        register_model(defn1)

        defn2 = ModelDefinition(
            model_id="model-a",
            family="test",
            variant="v2",
            display_name="Updated Model",
            supported_signals=[],
            phase_types=[],
            params=[],
        )
        register_model(defn2, replace=True)

        result = get_model_definition("model-a")
        assert result.display_name == "Updated Model"
        assert result.variant == "v2"

    def test_register_multiple_models(self) -> None:
        register_model(_make_definition("alpha"))
        register_model(_make_definition("beta"))
        register_model(_make_definition("gamma"))
        assert len(MODEL_REGISTRY) == 3


class TestGetModelDefinition:
    """get_model_definition returns definitions or raises clear errors."""

    def test_returns_correct_definition(self) -> None:
        register_model(_make_definition("my-model"))
        result = get_model_definition("my-model")
        assert result.model_id == "my-model"

    def test_raises_model_not_found_error(self) -> None:
        with pytest.raises(ModelNotFoundError) as exc_info:
            get_model_definition("nonexistent")
        assert exc_info.value.model_id == "nonexistent"

    def test_error_includes_valid_ids(self) -> None:
        register_model(_make_definition("alpha"))
        register_model(_make_definition("beta"))

        with pytest.raises(ModelNotFoundError) as exc_info:
            get_model_definition("missing")

        err = exc_info.value
        assert "alpha" in str(err)
        assert "beta" in str(err)
        assert err.valid_ids == ["alpha", "beta"]  # sorted

    def test_error_on_empty_registry(self) -> None:
        with pytest.raises(ModelNotFoundError) as exc_info:
            get_model_definition("anything")
        assert exc_info.value.valid_ids == []


class TestListModels:
    """list_models returns sorted model IDs."""

    def test_empty_registry(self) -> None:
        assert list_models() == []

    def test_returns_sorted(self) -> None:
        register_model(_make_definition("charlie"))
        register_model(_make_definition("alpha"))
        register_model(_make_definition("bravo"))
        assert list_models() == ["alpha", "bravo", "charlie"]

    def test_single_model(self) -> None:
        register_model(_make_definition("solo"))
        assert list_models() == ["solo"]


class TestClearRegistry:
    """clear_registry empties the dict for test isolation."""

    def test_clears_all_entries(self) -> None:
        register_model(_make_definition("a"))
        register_model(_make_definition("b"))
        assert len(MODEL_REGISTRY) == 2
        clear_registry()
        assert len(MODEL_REGISTRY) == 0

    def test_list_empty_after_clear(self) -> None:
        register_model(_make_definition("a"))
        clear_registry()
        assert list_models() == []
