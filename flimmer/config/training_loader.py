"""Config loader for Flimmer training configs.

Handles YAML loading, variant defaults merging, path resolution, and
human-readable error formatting. This is the main entry point for
consuming a Flimmer training config.

Load sequence:
  1. Find YAML file (path or directory with flimmer_train.yaml)
  2. Load YAML -> dict
  3. Identify model.variant (if present)
  4. Deep-merge: variant_defaults <- user_config (user wins)
  5. Auto-enable MoE if variant is MoE
  6. Resolve paths: data_config, model.path, save.output_dir, training.resume_from
  7. Parse through Pydantic -> FlimmerTrainingConfig
  8. Check data_config path exists (but don't load/validate it)

Usage::

    from flimmer.config import load_training_config

    # From a YAML file
    config = load_training_config("path/to/flimmer_train.yaml")

    # From a directory (looks for flimmer_train.yaml inside)
    config = load_training_config("path/to/project_folder")
"""

from __future__ import annotations

import copy
import re
from pathlib import Path

import yaml
from pydantic import ValidationError

from flimmer.config.loader import FlimmerConfigError
from flimmer.config.wan22_training_master import VARIANT_DEFAULTS, FlimmerTrainingConfig

# The filename we look for when given a directory
TRAINING_CONFIG_FILENAME = "flimmer_train.yaml"


def load_training_config(path: str | Path) -> FlimmerTrainingConfig:
    """Load and validate a Flimmer training config.

    This is the main entry point. It accepts two kinds of input:

    1. **A YAML file path**: loads and parses it directly.
    2. **A directory containing flimmer_train.yaml**: discovers and loads it.

    Variant defaults are applied as a base layer before user config.
    All relative paths are resolved relative to the config file's location.

    Args:
        path: Path to a YAML config file, or a directory.

    Returns:
        A fully validated FlimmerTrainingConfig with variant defaults
        applied and all relative paths resolved.

    Raises:
        FileNotFoundError: If the path doesn't exist.
        FlimmerConfigError: If the config has validation errors.
    """
    path = Path(path).resolve()

    if not path.exists():
        raise FileNotFoundError(
            f"Path not found: {path}\n"
            f"Check that the path exists and is spelled correctly."
        )

    # Determine the config file and the base directory for path resolution
    if path.is_file():
        config_data = _load_yaml(path)
        base_dir = path.parent
    elif (path / TRAINING_CONFIG_FILENAME).is_file():
        config_data = _load_yaml(path / TRAINING_CONFIG_FILENAME)
        base_dir = path
    else:
        raise FlimmerConfigError(
            f"No training config found at: {path}\n"
            f"Expected a YAML file or a directory containing '{TRAINING_CONFIG_FILENAME}'."
        )

    # Apply variant defaults (variant as base, user config on top)
    config_data = _apply_variant_defaults(config_data)

    # Auto-enable MoE if the variant is an MoE model
    config_data = _auto_enable_moe(config_data)

    # Resolve relative paths against the config file's directory
    config_data = _resolve_paths(config_data, base_dir)

    # Parse and validate through Pydantic
    try:
        config = FlimmerTrainingConfig.model_validate(config_data)
    except ValidationError as e:
        raise FlimmerConfigError(_format_validation_error(e)) from e

    # Check that data_config path exists (but don't load it)
    _check_data_config_path(config)

    return config


def _load_yaml(path: Path) -> dict:
    """Load a YAML file and return its contents as a dict.

    Returns an empty dict if the file is empty or contains only comments.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override dict on top of base dict.

    Rules:
    - Scalars: override wins
    - Dicts: recurse (merge nested dicts)
    - Lists: override replaces entirely (no list merging — avoids confusion)
    - Keys in override not in base: added
    - Keys in base not in override: kept

    Both input dicts are left unmodified — returns a new dict.
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _apply_variant_defaults(data: dict) -> dict:
    """Apply variant defaults as the base layer.

    If model.variant is set and corresponds to a known variant, the
    variant's defaults are deep-merged as the base, with user config
    on top. User values always win.

    The variant name is preserved in the output so Pydantic can
    validate it and cross-validators can check it.
    """
    model_block = data.get("model", {})
    if not isinstance(model_block, dict):
        return data

    variant_name = model_block.get("variant")
    if variant_name is None or variant_name not in VARIANT_DEFAULTS:
        # No variant or unknown variant — Pydantic will validate later
        return data

    # Deep-merge: variant defaults as base, user config on top
    variant = VARIANT_DEFAULTS[variant_name]
    result = _deep_merge(variant, data)

    # Ensure the variant name is preserved in the result
    if "model" not in result:
        result["model"] = {}
    result["model"]["variant"] = variant_name

    return result


def _auto_enable_moe(data: dict) -> dict:
    """Auto-set moe.enabled=True if the model variant is MoE.

    This saves users from having to explicitly set moe.enabled when using
    an MoE variant. The user can still override to False if needed.
    """
    model_block = data.get("model", {})
    if not isinstance(model_block, dict):
        return data

    # Check if the model is MoE (from variant defaults or explicit setting)
    is_moe = model_block.get("is_moe")
    if is_moe is True:
        moe_block = data.get("moe", {})
        if not isinstance(moe_block, dict):
            moe_block = {}
        # Only auto-enable if the user hasn't explicitly set it
        if "enabled" not in moe_block:
            moe_block["enabled"] = True
            data["moe"] = moe_block

    return data


def _is_huggingface_id(path_str: str) -> bool:
    """Detect whether a path string is a HuggingFace model ID.

    HuggingFace IDs look like 'org/model-name' (e.g. 'Wan-AI/Wan2.2-T2V-14B').
    Local paths have backslashes, drive letters, or start with './' or '/'.

    Returns True if the string looks like a HuggingFace ID, False otherwise.
    """
    # Contains backslash -> Windows path
    if "\\" in path_str:
        return False
    # Starts with drive letter (C:, D:, etc.) -> Windows path
    if re.match(r"^[A-Za-z]:", path_str):
        return False
    # Starts with . or / -> local path
    if path_str.startswith(".") or path_str.startswith("/"):
        return False
    # Contains exactly one forward slash and no spaces -> likely HF ID
    parts = path_str.split("/")
    if len(parts) == 2 and all(parts):
        return True
    return False


def _resolve_paths(data: dict, base_dir: Path) -> dict:
    """Resolve all relative paths in the config against base_dir.

    Paths that are already absolute are left as-is. HuggingFace model IDs
    are left as-is (not resolved as local paths). This ensures the config
    works regardless of the current working directory.
    """
    # Resolve data_config path
    if "data_config" in data and isinstance(data["data_config"], str):
        data["data_config"] = str(_resolve_one(data["data_config"], base_dir))

    # Resolve model paths
    model_block = data.get("model", {})
    if isinstance(model_block, dict):
        # Diffusers directory override (skip HuggingFace IDs)
        if "path" in model_block:
            path_str = model_block["path"]
            if isinstance(path_str, str) and not _is_huggingface_id(path_str):
                model_block["path"] = str(_resolve_model_path(path_str, base_dir))
        # Individual weight file paths (with parent-directory fallback)
        for file_key in ("dit", "dit_high", "dit_low", "vae", "t5"):
            if model_block.get(file_key) and isinstance(model_block[file_key], str):
                model_block[file_key] = str(
                    _resolve_model_path(model_block[file_key], base_dir)
                )

    # Resolve save.output_dir
    save_block = data.get("save", {})
    if isinstance(save_block, dict) and "output_dir" in save_block:
        save_block["output_dir"] = str(
            _resolve_one(save_block["output_dir"], base_dir)
        )

    # Resolve training.resume_from
    training_block = data.get("training", {})
    if isinstance(training_block, dict) and training_block.get("resume_from"):
        training_block["resume_from"] = str(
            _resolve_one(training_block["resume_from"], base_dir)
        )

    # Resolve per-expert resume_from paths
    moe_block = data.get("moe", {})
    if isinstance(moe_block, dict):
        for expert_key in ("high_noise", "low_noise"):
            expert = moe_block.get(expert_key, {})
            if isinstance(expert, dict) and expert.get("resume_from"):
                expert["resume_from"] = str(
                    _resolve_one(expert["resume_from"], base_dir)
                )

    # Resolve sampling paths
    sampling_block = data.get("sampling", {})
    if isinstance(sampling_block, dict):
        if sampling_block.get("sample_dir"):
            sampling_block["sample_dir"] = str(
                _resolve_one(sampling_block["sample_dir"], base_dir)
            )
        if sampling_block.get("lora_override"):
            sampling_block["lora_override"] = str(
                _resolve_one(sampling_block["lora_override"], base_dir)
            )

    # Resolve cache.cache_dir (always, even when using default)
    if "cache" not in data:
        data["cache"] = {}
    cache_block = data["cache"]
    if isinstance(cache_block, dict):
        # Use default "./cache" if not specified, then resolve relative to config
        cache_dir = cache_block.get("cache_dir", "./cache")
        cache_block["cache_dir"] = str(
            _resolve_one(cache_dir, base_dir)
        )

    return data


def _resolve_one(path_str: str, base_dir: Path) -> Path:
    """Resolve a single path string against a base directory.

    Absolute paths are returned as-is. Relative paths are resolved
    against base_dir and fully resolved (symlinks, .., etc.).
    """
    p = Path(path_str)
    if p.is_absolute():
        return p.resolve()
    return (base_dir / p).resolve()


def _resolve_model_path(path_str: str, base_dir: Path) -> Path:
    """Resolve a model weight path with parent-directory fallback.

    First tries resolving relative to base_dir (the config file's directory).
    If the file doesn't exist there, walks up parent directories looking for
    a match. Stops at a .git root or after 5 levels. Returns the first match,
    or the original resolution if nothing is found.

    This lets users put configs in subfolders (e.g. my_project/train.yaml)
    while model weights live at the repo root (models/), and ./models/...
    still resolves correctly.
    """
    p = Path(path_str)
    if p.is_absolute():
        return p.resolve()

    # Try the config file's directory first (current behavior)
    candidate = (base_dir / p).resolve()
    if candidate.exists():
        return candidate

    # Walk up parents looking for a match
    search_dir = base_dir.resolve()
    for _ in range(5):
        parent = search_dir.parent
        if parent == search_dir:
            break  # filesystem root
        search_dir = parent
        fallback = (search_dir / p).resolve()
        if fallback.exists():
            return fallback
        # Stop at git root — don't search above the repo
        if (search_dir / ".git").exists():
            break

    # Nothing found — return original resolution (error messages stay the same)
    return candidate


def _check_data_config_path(config: FlimmerTrainingConfig) -> None:
    """Verify that the data_config path exists on disk.

    This is an existence check only — we don't load or validate the data
    config here. That happens at training time when both configs are needed.
    """
    p = Path(config.data_config)
    if not p.exists():
        raise FlimmerConfigError(
            f"Data config not found: {p}\n\n"
            f"The training config points to this data config file, but it "
            f"doesn't exist. Check the 'data_config' path in your training config.\n"
            f"Relative paths are resolved from the training config file's location."
        )


def _format_validation_error(error: ValidationError) -> str:
    """Convert Pydantic's ValidationError into human-readable messages.

    Pydantic's default error output is developer-oriented and can be
    confusing for users who don't know what "value_error" means.
    We reformat each error into: where it is -> what's wrong -> how to fix it.
    """
    lines = ["Training config validation failed:\n"]

    for err in error.errors():
        # Build the location string (e.g., "model > variant")
        loc_parts = [str(part) for part in err["loc"]]
        location = " > ".join(loc_parts) if loc_parts else "(root)"

        # The error message from Pydantic (or our custom validators)
        message = err["msg"]

        # Clean up Pydantic's "Value error, " prefix from custom validators
        if message.startswith("Value error, "):
            message = message[len("Value error, "):]

        lines.append(f"  {location}: {message}")

    lines.append(
        "\nSee config_templates/ in flimmer-kit for valid training config files."
    )
    return "\n".join(lines)
