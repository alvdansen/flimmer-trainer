"""dimljus_phases: Model-agnostic, registry-driven phase configuration."""

from .definitions import (
    ModelDefinition,
    ParamSpec,
    PhaseTypeDeclaration,
    SignalDeclaration,
)
from .errors import DimljusPhasesError, ModelNotFoundError, PhaseConfigError
from .phase_model import PhaseConfig, ResolvedPhase
from .registry import (
    MODEL_REGISTRY,
    get_model_definition,
    list_models,
    register_model,
)
from .project import PhaseEntry, PhaseStatus, Project
from .resolution import get_phase_knobs, resolve_phase
from .templates import template_moe_standard, template_wan21_finetune
from .validation import ValidationIssue, ValidationResult, validate_project
from .yaml_export import export_yaml

from . import models as _models  # noqa: F401  -- triggers auto-registration of all model definitions

__all__ = [
    "ParamSpec",
    "SignalDeclaration",
    "PhaseTypeDeclaration",
    "ModelDefinition",
    "DimljusPhasesError",
    "ModelNotFoundError",
    "PhaseConfigError",
    "PhaseConfig",
    "ResolvedPhase",
    "MODEL_REGISTRY",
    "register_model",
    "get_model_definition",
    "list_models",
    "get_phase_knobs",
    "resolve_phase",
    "PhaseStatus",
    "PhaseEntry",
    "Project",
    "ValidationIssue",
    "ValidationResult",
    "validate_project",
    "export_yaml",
    "template_moe_standard",
    "template_wan21_finetune",
]
