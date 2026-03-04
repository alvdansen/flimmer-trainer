"""Pydantic v2 models for model definitions.

All types that describe what a model supports: parameters, signals,
phase types, and the complete model definition.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class ParamSpec(BaseModel):
    """Specification for a single overridable parameter.

    Declares a parameter's name, type, default value, optional range
    constraints, and whether it is phase-level or run-level.
    """

    name: str
    type: Literal["float", "int", "str", "bool"]
    default: float | int | str | bool
    min_value: float | int | None = None
    max_value: float | int | None = None
    description: str = ""
    phase_level: bool = True

    @model_validator(mode="after")
    def _validate_range_and_default(self) -> ParamSpec:
        # (a) min_value <= max_value when both set
        if self.min_value is not None and self.max_value is not None:
            if self.min_value > self.max_value:
                raise ValueError(
                    f"min_value ({self.min_value}) must not exceed "
                    f"max_value ({self.max_value})"
                )

        # (b) default within [min_value, max_value] range
        if self.min_value is not None and isinstance(self.default, (int, float)):
            if self.default < self.min_value:
                raise ValueError(
                    f"default ({self.default}) is below "
                    f"min_value ({self.min_value})"
                )
        if self.max_value is not None and isinstance(self.default, (int, float)):
            if self.default > self.max_value:
                raise ValueError(
                    f"default ({self.default}) is above "
                    f"max_value ({self.max_value})"
                )

        return self


class SignalDeclaration(BaseModel):
    """Declaration of a signal modality a model supports.

    Modality is a plain string (NOT an enum) to allow user-created
    categories and a "custom" bucket.
    """

    modality: str
    required: bool = False
    description: str = ""


class PhaseTypeDeclaration(BaseModel):
    """Declaration of a phase type a model supports.

    required_fields lists param names that are mandatory for this
    specific phase type (e.g., "boundary_ratio" for MoE expert phases).
    """

    name: str
    description: str = ""
    required_fields: list[str] = Field(default_factory=list)


class ModelDefinition(BaseModel):
    """Complete definition of a model's phase-configurable surface.

    Declares the model's identity, supported signals, phase types,
    and all configurable parameters with their constraints.
    """

    model_id: str
    family: str
    variant: str
    display_name: str
    is_moe: bool = False

    supported_signals: list[SignalDeclaration]
    phase_types: list[PhaseTypeDeclaration]
    params: list[ParamSpec]
    defaults: dict[str, object] = Field(default_factory=dict)

    @property
    def run_level_params(self) -> list[ParamSpec]:
        """Return only params where phase_level=False (run-level only)."""
        return [p for p in self.params if not p.phase_level]

    @property
    def phase_params(self) -> list[ParamSpec]:
        """Return only params where phase_level=True (phase-overridable)."""
        return [p for p in self.params if p.phase_level]

    def get_param(self, name: str) -> ParamSpec | None:
        """Look up a param by name, or return None if not found."""
        for p in self.params:
            if p.name == name:
                return p
        return None

    @model_validator(mode="after")
    def _validate_required_fields_exist_in_params(self) -> ModelDefinition:
        """Ensure every required_field in every PhaseTypeDeclaration
        references a param name that exists in the params list."""
        param_names = {p.name for p in self.params}
        for pt in self.phase_types:
            for field_name in pt.required_fields:
                if field_name not in param_names:
                    raise ValueError(
                        f"Phase type '{pt.name}' declares required field "
                        f"'{field_name}' but no param with that name exists. "
                        f"Available params: {sorted(param_names)}"
                    )
        return self
