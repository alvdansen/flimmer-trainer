"""YAML config export: serialize resolved phases to dimljus trainer format.

Converts a Project's resolved phases into the nested YAML structure expected
by the dimljus trainer config loader. Handles MoE vs non-MoE branching,
conditional signals block, and clean float formatting.

Public API:
    - export_yaml(project, output_path) -> Path
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from .phase_model import ResolvedPhase
from .registry import get_model_definition
from .resolution import resolve_phase

if TYPE_CHECKING:
    from .definitions import ModelDefinition
    from .project import Project

# ---------------------------------------------------------------------------
# Custom YAML Dumper with readable float formatting
# ---------------------------------------------------------------------------

class _DimljusDumper(yaml.SafeDumper):
    """Custom YAML dumper with human-readable float representation.

    Subclasses SafeDumper to avoid mutating the global SafeDumper state,
    which would cause side effects across tests and other code.

    Adds an implicit resolver for scientific notation (e.g. 5e-05, 1e-4)
    so that yaml.dump outputs plain scalars instead of !!float tagged ones.
    """


# Add implicit resolver for scientific notation floats (e.g. 5e-05, 1e-4, 8e-05)
# PyYAML's default float resolver doesn't match scientific notation, causing
# explicit !!float tags in the output. This resolver ensures clean output.
_SCIENTIFIC_FLOAT_RE = re.compile(
    r"^[-+]?(?:\d+\.?\d*|\.\d+)[eE][-+]?\d+$"
)
_DimljusDumper.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    _SCIENTIFIC_FLOAT_RE,
    list("-+0123456789."),
)


def _float_representer(dumper: yaml.Dumper, value: float) -> yaml.ScalarNode:
    """Format floats readably: 5.0e-05 not 4.9999999999999996e-05.

    Uses Python's :g format specifier which picks the shortest
    representation (fixed or scientific) without trailing zeros.

    For scientific notation, ensures a decimal point is present (e.g. 5.0e-05
    not 5e-05) so that PyYAML's safe_load round-trips the value as a float
    rather than a string.
    """
    if value != value:  # NaN
        return dumper.represent_scalar("tag:yaml.org,2002:float", ".nan")
    if value == float("inf"):
        return dumper.represent_scalar("tag:yaml.org,2002:float", ".inf")
    if value == float("-inf"):
        return dumper.represent_scalar("tag:yaml.org,2002:float", "-.inf")

    formatted = f"{value:g}"

    # For scientific notation, ensure a decimal point before the 'e'
    # so PyYAML safe_load parses it as float (5.0e-05 not 5e-05)
    if "e" in formatted or "E" in formatted:
        e_pos = formatted.lower().index("e")
        mantissa = formatted[:e_pos]
        exponent = formatted[e_pos:]
        if "." not in mantissa:
            mantissa += ".0"
        formatted = mantissa + exponent
    elif "." not in formatted:
        # Plain integer-like float: ensure it has a decimal point
        formatted += ".0"

    return dumper.represent_scalar("tag:yaml.org,2002:float", formatted)


_DimljusDumper.add_representer(float, _float_representer)


# ---------------------------------------------------------------------------
# Internal: build the dimljus config dict from resolved phases
# ---------------------------------------------------------------------------

# Params that map with a different key name in the YAML output
_PARAM_KEY_MAP = {
    "optimizer_type": "type",
    "scheduler_type": "type",
    "lora_dropout": "dropout",
}

# Params that belong to specific YAML blocks
_OPTIMIZER_PARAMS = {"optimizer_type", "learning_rate", "weight_decay"}
_SCHEDULER_PARAMS = {"scheduler_type", "min_lr_ratio"}
_TRAINING_PARAMS = {
    "max_epochs", "batch_size", "gradient_accumulation_steps",
    "caption_dropout_rate",
}
_LORA_PHASE_PARAMS = {"lora_dropout"}

# Params that are YAML-structural and should NOT appear in expert overrides
_MOE_SKIP_PARAMS = {"boundary_ratio"}


def _yaml_key(param_name: str) -> str:
    """Map a param name to its YAML key name."""
    return _PARAM_KEY_MAP.get(param_name, param_name)


def _has_non_assumed_signals(resolved: ResolvedPhase) -> bool:
    """Check if the resolved phase has non-text/non-video signals enabled.

    Text and video are always assumed for all models. Signals like
    reference_image indicate the signals block should be emitted.
    """
    assumed = {"text", "video"}
    for modality, enabled in resolved.signals.items():
        if modality not in assumed and enabled:
            return True
    return False


def _build_expert_overrides(
    expert: ResolvedPhase,
    base: ResolvedPhase,
) -> dict:
    """Build a minimal dict of only the params that differ from the base phase.

    Only includes params that are meaningful as per-expert overrides
    (not structural params like boundary_ratio).
    """
    overrides: dict = {}
    for param_name, expert_val in expert.params.items():
        if param_name in _MOE_SKIP_PARAMS:
            continue
        base_val = base.params.get(param_name)
        if expert_val != base_val:
            key = _yaml_key(param_name)
            overrides[key] = expert_val
    return overrides


def _resolved_to_dimljus_dict(
    project: Project,
    resolved_phases: list[ResolvedPhase],
    model_def: ModelDefinition,
) -> dict:
    """Map resolved phases to the dimljus trainer config dict structure.

    Args:
        project: The source project.
        resolved_phases: List of resolved (enabled) phases.
        model_def: The model definition from registry.

    Returns:
        A dict matching the dimljus YAML config structure.
    """
    config: dict = {}

    # Identify the base/unified phase (first resolved phase)
    base_phase = resolved_phases[0]
    base_params = base_phase.params

    # -- model block --
    config["model"] = {"variant": model_def.variant}

    # -- data_config (from base phase dataset) --
    if base_phase.dataset is not None:
        config["data_config"] = base_phase.dataset

    # -- lora block (run-level + phase-level lora_dropout) --
    lora_block: dict = {
        "rank": project.run_level_params.get("lora_rank", 16),
        "alpha": project.run_level_params.get("lora_alpha", 16),
    }
    if "lora_dropout" in base_params:
        lora_block["dropout"] = base_params["lora_dropout"]
    config["lora"] = lora_block

    # -- optimizer block --
    optimizer_block: dict = {}
    if "optimizer_type" in base_params:
        optimizer_block["type"] = base_params["optimizer_type"]
    if "learning_rate" in base_params:
        optimizer_block["learning_rate"] = base_params["learning_rate"]
    if "weight_decay" in base_params:
        optimizer_block["weight_decay"] = base_params["weight_decay"]
    config["optimizer"] = optimizer_block

    # -- scheduler block --
    scheduler_block: dict = {}
    if "scheduler_type" in base_params:
        scheduler_block["type"] = base_params["scheduler_type"]
    if "min_lr_ratio" in base_params:
        scheduler_block["min_lr_ratio"] = base_params["min_lr_ratio"]
    config["scheduler"] = scheduler_block

    # -- training block --
    training_block: dict = {
        "mixed_precision": project.run_level_params.get("mixed_precision", "bf16"),
        "base_model_precision": project.run_level_params.get(
            "base_model_precision", "bf16"
        ),
    }
    if "max_epochs" in base_params:
        training_block["unified_epochs"] = base_params["max_epochs"]
    if "batch_size" in base_params:
        training_block["batch_size"] = base_params["batch_size"]
    if "gradient_accumulation_steps" in base_params:
        training_block["gradient_accumulation_steps"] = base_params[
            "gradient_accumulation_steps"
        ]
    if "caption_dropout_rate" in base_params:
        training_block["caption_dropout_rate"] = base_params["caption_dropout_rate"]
    config["training"] = training_block

    # -- moe block (only for MoE models with expert phases) --
    if model_def.is_moe:
        expert_phases = [
            rp for rp in resolved_phases
            if rp.phase_type in ("high_noise", "low_noise")
        ]
        if expert_phases:
            moe_block: dict = {
                "enabled": True,
                "fork_enabled": True,
            }
            # boundary_ratio from the first expert phase's extras
            for ep in expert_phases:
                if "boundary_ratio" in ep.extras:
                    moe_block["boundary_ratio"] = ep.extras["boundary_ratio"]
                    break

            # Per-expert override blocks (minimal: only differing values)
            for ep in expert_phases:
                overrides = _build_expert_overrides(ep, base_phase)
                if overrides:
                    moe_block[ep.phase_type] = overrides

            config["moe"] = moe_block

    # -- signals block (conditional: omitted for T2V text+video only) --
    if _has_non_assumed_signals(base_phase):
        signals_block: dict = {}
        assumed = {"text", "video"}
        for modality, enabled in base_phase.signals.items():
            if modality not in assumed:
                signals_block[modality] = enabled
        config["signals"] = signals_block

    # -- save block --
    config["save"] = {
        "output_dir": f"./output/{project.name}",
        "name": project.name,
    }

    return config


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def export_yaml(project: Project, output_path: Path) -> Path:
    """Export a project to a dimljus-compatible YAML config file.

    Resolves each enabled phase via resolve_phase(), maps the results
    to the dimljus trainer config structure, and writes to output_path.

    Args:
        project: The project to export.
        output_path: Path where the YAML file will be written.

    Returns:
        The output_path (for chaining / confirmation).
    """
    output_path = Path(output_path)
    model_def = get_model_definition(project.model_id)

    # Build base_params from model's phase-level param defaults
    base_params: dict[str, float | int | str | bool] = {
        p.name: p.default for p in model_def.phase_params
    }

    # Resolve each enabled phase
    resolved_phases: list[ResolvedPhase] = []
    for entry in project.phases:
        if entry.config.enabled:
            resolved = resolve_phase(
                project.model_id, entry.config, base_params
            )
            resolved_phases.append(resolved)

    # Build the dimljus config dict
    config_dict = _resolved_to_dimljus_dict(project, resolved_phases, model_def)

    # Write YAML with custom float formatting
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(
            config_dict,
            f,
            Dumper=_DimljusDumper,
            default_flow_style=False,
            sort_keys=False,
        )

    return output_path
