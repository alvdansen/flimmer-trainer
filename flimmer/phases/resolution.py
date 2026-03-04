"""Phase resolution engine: knob discovery and override resolution.

Reads model definitions from the registry to determine available knobs
(ParamSpecs) for a given model + phase type, and resolves PhaseConfig
overrides against base parameter values to produce a fully-specified
ResolvedPhase.

The core resolution uses the proven _resolve(override, base) idiom
from flimmer: override if not None, else base value.
"""

from __future__ import annotations

from .definitions import ModelDefinition, ParamSpec
from .errors import PhaseConfigError
from .phase_model import PhaseConfig, ResolvedPhase
from .registry import get_model_definition


def _resolve(
    override_val: float | int | str | bool | None,
    base_val: float | int | str | bool | None,
) -> float | int | str | bool | None:
    """Use override if not None, else base value.

    This is the proven pattern from flimmer for merging
    per-phase overrides with base/default values.
    """
    return override_val if override_val is not None else base_val


def get_phase_knobs(model_id: str, phase_type: str) -> dict[str, ParamSpec]:
    """Return available overridable params for a model + phase type combination.

    Includes all phase-level params from the model definition, plus any
    additional params referenced in the phase type's required_fields (even
    if those have phase_level=False, since they're structurally required).

    Args:
        model_id: The registered model identifier.
        phase_type: The phase type to look up knobs for.

    Returns:
        Dict mapping param name -> ParamSpec for all overridable params.

    Raises:
        ModelNotFoundError: If model_id is not in the registry.
        PhaseConfigError: If phase_type is not declared by the model.
    """
    model_def = get_model_definition(model_id)

    # Find the phase type declaration
    phase_type_decl = None
    for pt in model_def.phase_types:
        if pt.name == phase_type:
            phase_type_decl = pt
            break

    if phase_type_decl is None:
        valid_types = sorted(pt.name for pt in model_def.phase_types)
        raise PhaseConfigError(
            f"Unknown phase type '{phase_type}' for model '{model_id}'. "
            f"Valid phase types: {valid_types}"
        )

    # Collect all phase-level params
    knobs: dict[str, ParamSpec] = {
        p.name: p for p in model_def.params if p.phase_level
    }

    # Also include required_fields params even if phase_level=False
    for field_name in phase_type_decl.required_fields:
        if field_name not in knobs:
            param = model_def.get_param(field_name)
            if param is not None:
                knobs[field_name] = param

    return knobs


def resolve_phase(
    model_id: str,
    phase_config: PhaseConfig,
    base_params: dict[str, float | int | str | bool],
) -> ResolvedPhase:
    """Resolve a PhaseConfig against base parameters to produce a ResolvedPhase.

    For each phase-level param in the model definition:
    1. Check if there's an override in phase_config.overrides
    2. If override is not None, use it; otherwise fall back to base_params
    3. If not in base_params either, use the ParamSpec default
    4. If still None after all sources, raise PhaseConfigError

    Args:
        model_id: The registered model identifier.
        phase_config: The user's phase configuration with optional overrides.
        base_params: Base parameter values (e.g., run-level defaults).

    Returns:
        A fully-resolved, frozen ResolvedPhase with all params filled.

    Raises:
        ModelNotFoundError: If model_id is not in the registry.
        PhaseConfigError: If config is invalid or a required param has no value.
    """
    model_def = get_model_definition(model_id)

    # Validate config against model definition (fail fast)
    phase_config.validate_against(model_def)

    # Resolve each phase-level param
    resolved_params: dict[str, float | int | str | bool] = {}
    for param in model_def.phase_params:
        override_val = phase_config.overrides.get(param.name)
        base_val = base_params.get(param.name, param.default)
        result = _resolve(override_val, base_val)

        if result is None:
            raise PhaseConfigError(
                f"Phase-level parameter '{param.name}' has no value: "
                f"no override, no base value, and no default"
            )
        resolved_params[param.name] = result

    # Resolve dataset: simple pass-through (None = inherit from run-level)
    resolved_dataset = phase_config.dataset

    # Resolve signals: build all-True base from model's supported_signals,
    # then merge phase_config.signals as selective overrides
    resolved_signals: dict[str, bool] = {
        sig.modality: True for sig in model_def.supported_signals
    }
    if phase_config.signals is not None:
        resolved_signals.update(phase_config.signals)

    return ResolvedPhase(
        phase_type=phase_config.phase_type,
        display_name=phase_config.display_name,
        enabled=phase_config.enabled,
        params=resolved_params,
        extras=dict(phase_config.extras),
        dataset=resolved_dataset,
        signals=resolved_signals,
    )
