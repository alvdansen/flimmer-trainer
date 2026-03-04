"""Model registry: stores and retrieves ModelDefinition instances by model_id.

Follows the proven WAN_VARIANTS plain-dict pattern from dimljus-kit,
upgraded to typed Pydantic values.
"""

from __future__ import annotations

from .definitions import ModelDefinition
from .errors import ModelNotFoundError

MODEL_REGISTRY: dict[str, ModelDefinition] = {}


def register_model(definition: ModelDefinition, *, replace: bool = False) -> None:
    """Register a model definition in the global registry.

    Args:
        definition: The model definition to register.
        replace: If True, allow overwriting an existing registration.
            Defaults to False to prevent silent overwrites.

    Raises:
        ValueError: If model_id is already registered and replace is False.
    """
    if definition.model_id in MODEL_REGISTRY and not replace:
        raise ValueError(
            f"Model '{definition.model_id}' already registered. "
            f"Use replace=True to override."
        )
    MODEL_REGISTRY[definition.model_id] = definition


def get_model_definition(model_id: str) -> ModelDefinition:
    """Retrieve a model definition by its ID.

    Args:
        model_id: The unique model identifier to look up.

    Returns:
        The matching ModelDefinition.

    Raises:
        ModelNotFoundError: If model_id is not in the registry.
            Error message includes a sorted list of valid model IDs.
    """
    if model_id not in MODEL_REGISTRY:
        valid_ids = sorted(MODEL_REGISTRY.keys())
        raise ModelNotFoundError(model_id, valid_ids)
    return MODEL_REGISTRY[model_id]


def list_models() -> list[str]:
    """Return a sorted list of all registered model IDs."""
    return sorted(MODEL_REGISTRY.keys())


def clear_registry() -> None:
    """Clear all entries from the registry. For test isolation only."""
    MODEL_REGISTRY.clear()
