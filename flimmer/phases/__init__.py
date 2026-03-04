"""flimmer.phases: Model-agnostic, registry-driven phase configuration."""

from .definitions import (
    ModelDefinition,
    ParamSpec,
    PhaseTypeDeclaration,
    SignalDeclaration,
)
from .errors import FlimmerPhasesError, ModelNotFoundError, PhaseConfigError
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
    "FlimmerPhasesError",
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

import logging

_logger = logging.getLogger(__name__)


def _check_registry_sync() -> None:
    """Warn if MODEL_REGISTRY model_ids don't all have WAN_VARIANTS entries."""
    try:
        from flimmer.training.wan.registry import WAN_VARIANTS
    except ImportError:
        return  # training extras not installed
    for model_id, defn in MODEL_REGISTRY.items():
        if defn.variant not in WAN_VARIANTS:
            _logger.warning(
                "Model '%s' registered in phases but no WAN_VARIANTS entry "
                "for variant '%s'. Implement a backend in "
                "flimmer/training/ for this model.",
                model_id, defn.variant,
            )


_check_registry_sync()
