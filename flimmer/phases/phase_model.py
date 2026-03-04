"""Phase data models: user input (PhaseConfig) and resolved output (ResolvedPhase).

PhaseConfig is the user-facing Pydantic model for configuring a training phase.
ResolvedPhase is the frozen dataclass output after resolution — all values filled,
no None values for params.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import BaseModel, field_validator

from .errors import PhaseConfigError

# Backward compat: old phase type names → current names
_PHASE_TYPE_ALIASES: dict[str, str] = {
    "unified": "full_noise",
}

if TYPE_CHECKING:
    from .definitions import ModelDefinition


class PhaseConfig(BaseModel):
    """User-facing configuration for a single training phase.

    Attributes:
        phase_type: Must match one of the model's declared phase types.
        display_name: Optional user-friendly name for organization.
        enabled: Whether this phase is active.
        overrides: Param name -> override value. None = inherit from base.
        extras: Phase-type-specific required field values
            (e.g., boundary_ratio for MoE expert phases).
    """

    phase_type: str
    display_name: str = ""
    enabled: bool = True
    overrides: dict[str, float | int | str | bool | None] = {}
    extras: dict[str, object] = {}
    dataset: str | None = None
    signals: dict[str, bool] | None = None

    @field_validator("phase_type", mode="before")
    @classmethod
    def _normalize_phase_type(cls, v: str) -> str:
        """Accept legacy phase type names (e.g. 'unified' → 'full_noise')."""
        return _PHASE_TYPE_ALIASES.get(v, v)

    def validate_against(self, model_def: ModelDefinition) -> None:
        """Validate this config against a model definition.

        Checks:
            (a) phase_type is in model's declared phase types
            (b) override keys are valid phase-level param names
            (c) override values respect ParamSpec constraints (type, min, max)
            (d) required_fields for this phase_type are present in extras
            (e) signal modalities are in model's supported_signals

        Raises:
            PhaseConfigError: With a specific message on any validation failure.
        """
        # (a) Validate phase_type
        valid_phase_types = [pt.name for pt in model_def.phase_types]
        if self.phase_type not in valid_phase_types:
            raise PhaseConfigError(
                f"Unknown phase type '{self.phase_type}' for model "
                f"'{model_def.model_id}'. "
                f"Valid phase types: {sorted(valid_phase_types)}"
            )

        # (b) Validate override keys are phase-level params
        phase_param_names = {p.name for p in model_def.phase_params}
        for key in self.overrides:
            if key not in phase_param_names:
                # Check if it's a run-level param for a more helpful message
                all_param_names = {p.name for p in model_def.params}
                if key in all_param_names:
                    raise PhaseConfigError(
                        f"Parameter '{key}' is run-level only and cannot be "
                        f"overridden per-phase. Phase-level params: "
                        f"{sorted(phase_param_names)}"
                    )
                raise PhaseConfigError(
                    f"Unknown parameter '{key}' in overrides for model "
                    f"'{model_def.model_id}'. "
                    f"Valid phase-level params: {sorted(phase_param_names)}"
                )

        # (c) Validate override values against ParamSpec constraints
        for key, value in self.overrides.items():
            if value is None:
                continue  # None means inherit from base, no validation needed
            param_spec = model_def.get_param(key)
            if param_spec is None:
                continue  # Already caught by (b)

            # Range checks for numeric values
            if param_spec.min_value is not None and isinstance(value, (int, float)):
                if value < param_spec.min_value:
                    raise PhaseConfigError(
                        f"Override value for '{key}' ({value}) is below "
                        f"minimum ({param_spec.min_value})"
                    )
            if param_spec.max_value is not None and isinstance(value, (int, float)):
                if value > param_spec.max_value:
                    raise PhaseConfigError(
                        f"Override value for '{key}' ({value}) is above "
                        f"maximum ({param_spec.max_value})"
                    )

        # (d) Check required_fields for this phase_type are in extras
        phase_type_decl = next(
            pt for pt in model_def.phase_types if pt.name == self.phase_type
        )
        for field_name in phase_type_decl.required_fields:
            if field_name not in self.extras:
                raise PhaseConfigError(
                    f"Phase type '{self.phase_type}' requires field "
                    f"'{field_name}' in extras but it was not provided"
                )

        # (e) Validate signal modalities against model's supported_signals
        if self.signals is not None:
            supported_modalities = {
                s.modality for s in model_def.supported_signals
            }
            for modality in self.signals:
                if modality not in supported_modalities:
                    raise PhaseConfigError(
                        f"Signal modality '{modality}' is not supported by "
                        f"model '{model_def.model_id}'. Supported modalities: "
                        f"{sorted(supported_modalities)}"
                    )


@dataclass(frozen=True)
class ResolvedPhase:
    """Fully-resolved training phase — all values filled, immutable.

    Created by the resolution engine after merging overrides with base
    values. No None values for params.

    Attributes:
        phase_type: The structural phase type.
        display_name: User-friendly name.
        enabled: Whether this phase is active.
        params: All parameter values, fully resolved (no Nones).
        extras: Phase-type-specific field values.
        dataset: Per-phase dataset path, or None to inherit from run-level.
        signals: Fully-specified signal modality -> enabled dict (all model
            modalities present with a bool value after resolution).
    """

    phase_type: str
    display_name: str
    enabled: bool
    params: dict[str, float | int | str | bool]
    extras: dict[str, object]
    dataset: str | None
    signals: dict[str, bool]

    def get_param(self, name: str) -> float | int | str | bool:
        """Return the resolved value for a parameter name.

        Args:
            name: The parameter name to look up.

        Returns:
            The resolved parameter value.

        Raises:
            KeyError: If the parameter name is not found in resolved params.
        """
        if name not in self.params:
            raise KeyError(
                f"Parameter '{name}' not found in resolved phase. "
                f"Available params: {sorted(self.params.keys())}"
            )
        return self.params[name]
