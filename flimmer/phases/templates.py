"""Template factory functions: pre-filled Projects for common training workflows.

Templates are just factory functions that return Project instances with
sensible defaults -- not a separate concept or class hierarchy.

Public API:
    - template_moe_standard(name, model_id) -> Project
    - template_wan21_finetune(name, model_id) -> Project
"""

from __future__ import annotations

from .phase_model import PhaseConfig
from .project import Project


def template_moe_standard(
    name: str,
    model_id: str = "wan-2.2-t2v-14b",
) -> Project:
    """Create a standard MoE training project with 3 phases.

    Phases:
        1. Unified warmup (lower LR, fewer epochs)
        2. High-noise expert (higher LR, more epochs)
        3. Low-noise expert (medium LR, most epochs)

    Args:
        name: Project name.
        model_id: Model ID (default: wan-2.2-t2v-14b).

    Returns:
        A pre-filled Project ready for customization or export.
    """
    project = Project.create(name=name, model_id=model_id)

    project.add_phase(PhaseConfig(
        phase_type="unified",
        display_name="Unified Warmup",
        overrides={"learning_rate": 5e-5, "max_epochs": 10},
    ))

    project.add_phase(PhaseConfig(
        phase_type="high_noise",
        display_name="High Noise Expert",
        overrides={"learning_rate": 1e-4, "max_epochs": 30},
        extras={"boundary_ratio": 0.875},
    ))

    project.add_phase(PhaseConfig(
        phase_type="low_noise",
        display_name="Low Noise Expert",
        overrides={"learning_rate": 8e-5, "max_epochs": 50},
        extras={"boundary_ratio": 0.875},
    ))

    return project


def template_wan21_finetune(
    name: str,
    model_id: str = "wan-2.1-t2v-14b",
) -> Project:
    """Create a simple Wan 2.1 fine-tuning project with 1 unified phase.

    Args:
        name: Project name.
        model_id: Model ID (default: wan-2.1-t2v-14b).

    Returns:
        A pre-filled Project ready for customization or export.
    """
    project = Project.create(name=name, model_id=model_id)

    project.add_phase(PhaseConfig(
        phase_type="unified",
        display_name="Standard Training",
        overrides={"learning_rate": 5e-5, "max_epochs": 100},
    ))

    return project
