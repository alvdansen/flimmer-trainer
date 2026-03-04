"""Load a project YAML file and convert to a flimmer.phases.Project.

The project YAML is the user-facing format for defining multi-phase
training projects. It gets converted to a Project object which manages
the PENDING -> RUNNING -> COMPLETED lifecycle.

Key design choice: merge_phase_config() reads the FULL base training
config YAML and patches specific values. It does NOT use export_yaml()
because that function generates config from resolved phases but misses
model paths, data_config, save config, and other base fields.
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

from flimmer.phases import PhaseConfig, Project


def load_project_yaml(path: Path) -> dict:
    """Load and parse a project YAML file.

    Reads the raw YAML without validation -- the caller is responsible
    for interpreting the structure. This keeps parsing separate from
    Project construction.

    Args:
        path: Path to the project YAML file.

    Returns:
        Raw dict from YAML parsing.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data


def project_from_yaml(yaml_path: Path) -> Project:
    """Convert a project YAML to a Project instance.

    If a flimmer_project.json exists alongside the YAML, loads it
    instead (preserves phase status for resume behavior). If the YAML
    has been modified since the JSON was written, warns the user and
    re-creates from the YAML.

    Args:
        yaml_path: Path to the project YAML file.

    Returns:
        Project instance (new or loaded from existing state).
    """
    data = load_project_yaml(yaml_path)
    project_dir = yaml_path.parent

    # Check for existing project state (resume behavior)
    json_path = project_dir / "flimmer_project.json"
    if json_path.exists():
        # Detect stale JSON: if YAML was edited after JSON was written,
        # the user's changes would be silently ignored. Re-create instead.
        yaml_mtime = yaml_path.stat().st_mtime
        json_mtime = json_path.stat().st_mtime
        if yaml_mtime > json_mtime:
            import sys
            print(
                f"Note: {yaml_path.name} is newer than flimmer_project.json. "
                f"Re-reading project from YAML.",
                file=sys.stderr,
            )
            # Remove stale JSON so we recreate from YAML below
            json_path.unlink()
        else:
            return Project.load(project_dir)

    # Create new project from YAML
    project = Project.create(
        name=data["name"],
        model_id=data["model_id"],
        run_level_params=data.get("run_level_params", {}),
    )

    # Add each phase from the YAML
    for phase_data in data.get("phases", []):
        config = PhaseConfig(
            phase_type=phase_data["type"],
            display_name=phase_data.get("name", ""),
            overrides=phase_data.get("overrides", {}),
            extras=phase_data.get("extras", {}),
            dataset=phase_data.get("dataset"),
        )
        project.add_phase(config)

    project.save(project_dir)
    return project


# ── Override mapping ────────────────────────────────────────────────
# Maps phase override keys to their location in the base training config.
# Format: override_key -> (config_section, config_key)

_OPTIMIZER_OVERRIDES = {
    "learning_rate": ("optimizer", "learning_rate"),
    "weight_decay": ("optimizer", "weight_decay"),
}

_TRAINING_OVERRIDES = {
    "batch_size": ("training", "batch_size"),
    "gradient_accumulation_steps": ("training", "gradient_accumulation_steps"),
    "caption_dropout_rate": ("training", "caption_dropout_rate"),
}

# Run-level params that map to config sections
_RUN_LEVEL_MAP = {
    "mixed_precision": ("training", "mixed_precision"),
    "base_model_precision": ("training", "base_model_precision"),
    "lora_rank": ("lora", "rank"),
    "lora_alpha": ("lora", "alpha"),
}

# Phase types that are MoE expert phases
_EXPERT_PHASE_TYPES = {"high_noise", "low_noise"}


def merge_phase_config(
    base_config_path: Path,
    project: Project,
    phase_index: int,
    output_path: Path,
) -> Path:
    """Merge a base training config with phase-specific overrides.

    This is the critical bridge function. It reads the full base training
    config YAML, applies phase overrides and run-level params, and writes
    the merged result. The training CLI then reads this merged config.

    For unified phases: overrides go directly into the relevant config
    sections (optimizer.learning_rate, training.unified_epochs, etc.).

    For MoE expert phases: sets moe.fork_enabled=True, applies
    boundary_ratio from extras, and puts override values in the
    expert-specific section (moe.high_noise or moe.low_noise).

    Args:
        base_config_path: Path to the full base training config YAML.
        project: The Project instance with run_level_params.
        phase_index: Index of the phase to merge.
        output_path: Where to write the merged config YAML.

    Returns:
        The output_path (for chaining / confirmation).
    """
    # Read the base config
    with open(base_config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Deep copy to avoid mutating the original
    config = copy.deepcopy(config)

    phase_entry = project.phases[phase_index]
    phase_config = phase_entry.config
    overrides = phase_config.overrides
    phase_type = phase_config.phase_type

    # Apply run-level params from the project
    for param_name, (section, key) in _RUN_LEVEL_MAP.items():
        if param_name in project.run_level_params:
            config.setdefault(section, {})[key] = project.run_level_params[param_name]

    # Handle MoE expert phases vs unified phases differently
    if phase_type in _EXPERT_PHASE_TYPES:
        _apply_expert_overrides(config, phase_config, overrides)
    else:
        _apply_unified_overrides(config, overrides)

    # Write the merged config
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)

    return output_path


def _apply_unified_overrides(config: dict, overrides: dict) -> None:
    """Apply overrides for a unified (non-expert) phase.

    Maps override keys to their config locations and updates the
    base config in place. Disables MoE fork so resolve_phases()
    produces exactly one unified phase (no leaked expert phases).

    Args:
        config: The mutable base config dict.
        overrides: Phase override key-value pairs.
    """
    # Disable MoE fork — a unified project phase must resolve to
    # exactly one unified training phase, not unified + experts.
    config.setdefault("moe", {})["fork_enabled"] = False

    for key, value in overrides.items():
        if value is None:
            continue

        # Optimizer overrides
        if key in _OPTIMIZER_OVERRIDES:
            section, config_key = _OPTIMIZER_OVERRIDES[key]
            config.setdefault(section, {})[config_key] = value
        # Training overrides
        elif key in _TRAINING_OVERRIDES:
            section, config_key = _TRAINING_OVERRIDES[key]
            config.setdefault(section, {})[config_key] = value
        # max_epochs -> training.unified_epochs for unified phases
        elif key == "max_epochs":
            config.setdefault("training", {})["unified_epochs"] = value


def _apply_expert_overrides(
    config: dict, phase_config: PhaseConfig, overrides: dict
) -> None:
    """Apply overrides for an MoE expert phase.

    Sets moe.fork_enabled=True, applies boundary_ratio from extras,
    and puts per-expert overrides into the expert section. Disables
    the unified phase and the OTHER expert so resolve_phases()
    produces exactly one expert training phase.

    Args:
        config: The mutable base config dict.
        phase_config: The PhaseConfig with phase_type and extras.
        overrides: Phase override key-value pairs.
    """
    moe = config.setdefault("moe", {})
    moe["fork_enabled"] = True

    # Suppress the unified phase — this project phase is expert-only.
    config.setdefault("training", {})["unified_epochs"] = 0

    # Disable the OTHER expert so it doesn't leak through.
    other_expert = "low_noise" if phase_config.phase_type == "high_noise" else "high_noise"
    moe.setdefault(other_expert, {})["enabled"] = False

    # Apply boundary_ratio from extras
    if "boundary_ratio" in phase_config.extras:
        moe["boundary_ratio"] = phase_config.extras["boundary_ratio"]

    # Build expert-specific override dict
    expert_overrides: dict = {}
    for key, value in overrides.items():
        if value is None:
            continue
        expert_overrides[key] = value

    # Place overrides under the expert section (e.g., moe.high_noise)
    # Merge into existing expert dict to preserve defaults, then add overrides.
    expert_section = moe.setdefault(phase_config.phase_type, {})
    expert_section.update(expert_overrides)
    expert_section["enabled"] = True
