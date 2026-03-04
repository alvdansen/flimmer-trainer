"""Shared pytest fixtures for flimmer.phases tests."""

import pytest

from flimmer.phases import (
    ModelDefinition,
    ParamSpec,
    PhaseTypeDeclaration,
    SignalDeclaration,
)
from flimmer.phases.phase_model import PhaseConfig
from flimmer.phases.registry import clear_registry


@pytest.fixture
def clean_registry():
    """Clear the model registry before and after the test.

    Use this fixture in tests that register real model definitions
    to prevent cross-test pollution.
    """
    clear_registry()
    yield
    clear_registry()


@pytest.fixture
def register_all_models(clean_registry):
    """Register all model definitions into a clean registry.

    Depends on clean_registry for isolation. Uses replace=True because
    Python caches module imports -- the initial import registers models,
    but after clean_registry clears the dict, re-importing is a no-op.
    """
    from flimmer.phases.models.moe import WAN_22_T2V
    from flimmer.phases.models.wan21_i2v import WAN_21_I2V
    from flimmer.phases.models.wan21_t2v import WAN_21_T2V
    from flimmer.phases.models.wan22_i2v import WAN_22_I2V
    from flimmer.phases.registry import register_model

    for model in (WAN_21_T2V, WAN_21_I2V, WAN_22_T2V, WAN_22_I2V):
        register_model(model, replace=True)


@pytest.fixture
def sample_param_specs() -> list[ParamSpec]:
    """Realistic param specs with both phase-level and run-level params."""
    return [
        ParamSpec(
            name="learning_rate",
            type="float",
            default=5e-5,
            min_value=1e-6,
            max_value=1e-3,
            description="Optimizer learning rate",
            phase_level=True,
        ),
        ParamSpec(
            name="batch_size",
            type="int",
            default=4,
            min_value=1,
            max_value=64,
            description="Training batch size",
            phase_level=True,
        ),
        ParamSpec(
            name="dropout",
            type="float",
            default=0.1,
            min_value=0.0,
            max_value=1.0,
            description="Dropout rate",
            phase_level=True,
        ),
        ParamSpec(
            name="model_weights_path",
            type="str",
            default="",
            description="Path to local model weights (run-level only)",
            phase_level=False,
        ),
        ParamSpec(
            name="use_gradient_checkpointing",
            type="bool",
            default=True,
            description="Enable gradient checkpointing (run-level only)",
            phase_level=False,
        ),
        ParamSpec(
            name="boundary_ratio",
            type="float",
            default=0.875,
            min_value=0.0,
            max_value=1.0,
            description="Noise boundary ratio for MoE expert phases",
            phase_level=True,
        ),
    ]


@pytest.fixture
def sample_signal_declarations() -> list[SignalDeclaration]:
    """Realistic signal declarations for a multimodal model."""
    return [
        SignalDeclaration(
            modality="text",
            required=True,
            description="Text prompt conditioning",
        ),
        SignalDeclaration(
            modality="video",
            required=True,
            description="Target video data",
        ),
        SignalDeclaration(
            modality="reference_image",
            required=False,
            description="Reference images for style/content guidance",
        ),
        SignalDeclaration(
            modality="custom",
            required=False,
            description="User-defined signal category",
        ),
    ]


@pytest.fixture
def sample_model_definition(
    sample_param_specs: list[ParamSpec],
    sample_signal_declarations: list[SignalDeclaration],
) -> ModelDefinition:
    """Realistic ModelDefinition modeled after MoE-style model."""
    return ModelDefinition(
        model_id="wan-2.2-t2v-14b",
        family="wan",
        variant="2.2_t2v",
        display_name="Wan 2.2 T2V 14B (MoE)",
        is_moe=True,
        supported_signals=sample_signal_declarations,
        phase_types=[
            PhaseTypeDeclaration(
                name="unified",
                description="Single-phase training (all noise levels)",
                required_fields=[],
            ),
            PhaseTypeDeclaration(
                name="high_noise",
                description="High noise expert phase",
                required_fields=["boundary_ratio"],
            ),
            PhaseTypeDeclaration(
                name="low_noise",
                description="Low noise expert phase",
                required_fields=["boundary_ratio"],
            ),
        ],
        params=sample_param_specs,
        defaults={
            "learning_rate": 2e-4,
            "batch_size": 4,
        },
    )


@pytest.fixture
def sample_phase_config() -> PhaseConfig:
    """PhaseConfig matching sample_model_definition's first phase type (unified),
    with one override (learning_rate)."""
    return PhaseConfig(
        phase_type="unified",
        overrides={"learning_rate": 1e-4},
    )


@pytest.fixture
def base_config(sample_model_definition: ModelDefinition) -> dict[str, float | int | str | bool]:
    """Base parameter values: all phase-level params at their defaults
    from sample_model_definition."""
    return {p.name: p.default for p in sample_model_definition.phase_params}
