"""Variant registry — maps config strings to backend constructor args.

Each Wan variant (2.2_t2v, 2.2_i2v, 2.1_t2v, 2.1_i2v) has different
architecture parameters. The registry centralizes this mapping so the
backend constructor doesn't need to hardcode variant-specific logic.

This module is GPU-free — it just defines the mapping. Actual model loading
happens in backend.py.
"""

from __future__ import annotations

from typing import Any

from flimmer.training.wan.constants import (
    I2V_EXTRA_TARGETS,
    T2V_LORA_TARGETS,
    WAN_EXPERT_SUBFOLDERS,
    WAN_I2V_IN_CHANNELS,
    WAN_NUM_BLOCKS,
    WAN_SINGLE_SUBFOLDER,
    WAN_T2V_IN_CHANNELS,
)


# ---------------------------------------------------------------------------
# Variant definitions
# ---------------------------------------------------------------------------

WAN_VARIANTS: dict[str, dict[str, Any]] = {
    "2.2_t2v": {
        "model_id": "wan-2.2-t2v-14b",
        "is_moe": True,
        "is_i2v": False,
        "in_channels": WAN_T2V_IN_CHANNELS,
        "num_blocks": WAN_NUM_BLOCKS,
        "boundary_ratio": 0.875,
        "flow_shift": 5.0,
        "lora_targets": list(T2V_LORA_TARGETS),
        "expert_subfolders": dict(WAN_EXPERT_SUBFOLDERS),
        "pipeline_class": "WanPipeline",
        "subtypes": {
            "480p": {"hf_repo": "Wan-AI/Wan2.2-T2V-A14B-Diffusers"},
            "720p": {"hf_repo": "Wan-AI/Wan2.2-T2V-A14B-Diffusers"},
        },
        "default_subtype": "480p",
    },
    "2.2_i2v": {
        "model_id": "wan-2.2-i2v-14b",
        "is_moe": True,
        "is_i2v": True,
        "in_channels": WAN_I2V_IN_CHANNELS,
        "num_blocks": WAN_NUM_BLOCKS,
        "boundary_ratio": 0.900,
        "flow_shift": 3.0,
        "lora_targets": list(T2V_LORA_TARGETS) + list(I2V_EXTRA_TARGETS),
        "expert_subfolders": dict(WAN_EXPERT_SUBFOLDERS),
        "pipeline_class": "WanImageToVideoPipeline",
        "subtypes": {
            "480p": {"hf_repo": "Wan-AI/Wan2.2-I2V-A14B-Diffusers"},
            "720p": {"hf_repo": "Wan-AI/Wan2.2-I2V-A14B-Diffusers"},
        },
        "default_subtype": "480p",
    },
    "2.1_t2v": {
        "model_id": "wan-2.1-t2v-14b",
        "is_moe": False,
        "is_i2v": False,
        "in_channels": WAN_T2V_IN_CHANNELS,
        "num_blocks": WAN_NUM_BLOCKS,
        "boundary_ratio": None,  # No expert routing for single-transformer
        "flow_shift": 3.0,
        "lora_targets": list(T2V_LORA_TARGETS),
        "expert_subfolders": {
            "default": WAN_SINGLE_SUBFOLDER,
        },
        "pipeline_class": "WanPipeline",
        "subtypes": {
            "480p": {"hf_repo": "Wan-AI/Wan2.1-T2V-14B-Diffusers"},
            "720p": {"hf_repo": "Wan-AI/Wan2.1-T2V-14B-Diffusers"},
        },
        "default_subtype": "480p",
    },
    "2.1_i2v": {
        "model_id": "wan-2.1-i2v-14b",
        "is_moe": False,
        "is_i2v": True,
        "in_channels": WAN_I2V_IN_CHANNELS,
        "num_blocks": WAN_NUM_BLOCKS,
        "boundary_ratio": None,  # No expert routing for non-MoE
        "flow_shift": 3.0,
        "lora_targets": list(T2V_LORA_TARGETS) + list(I2V_EXTRA_TARGETS),
        "expert_subfolders": {
            "default": WAN_SINGLE_SUBFOLDER,
        },
        "pipeline_class": "WanImageToVideoPipeline",
        "subtypes": {
            "480p": {"hf_repo": "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"},
            "720p": {"hf_repo": "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers"},
        },
        "default_subtype": "480p",
    },
}
"""Registry of Wan model variants and their architecture parameters.

Each entry contains everything needed to construct a WanModelBackend:
    - model_id: human-readable identifier for logging/metadata
    - is_moe: whether this variant uses Mixture of Experts
    - is_i2v: whether this variant accepts reference images
    - in_channels: latent input channel count
    - num_blocks: number of transformer blocks
    - boundary_ratio: default SNR boundary for expert routing (None = no MoE)
    - flow_shift: default flow matching shift parameter
    - lora_targets: default module names for LoRA adapter placement
    - expert_subfolders: maps expert names to diffusers subfolder paths
    - pipeline_class: diffusers pipeline class name for inference
    - subtypes: resolution subtypes mapping to HF repo IDs
    - default_subtype: default resolution subtype (e.g. '480p')
"""


def get_variant_info(variant: str) -> dict[str, Any]:
    """Look up variant configuration by name.

    Args:
        variant: Variant string (e.g. '2.2_t2v', '2.2_i2v', '2.1_t2v', '2.1_i2v').

    Returns:
        Dict of variant parameters (copy, safe to modify).

    Raises:
        ValueError: If the variant is not recognized.
    """
    if variant not in WAN_VARIANTS:
        valid = ", ".join(sorted(WAN_VARIANTS.keys()))
        raise ValueError(
            f"Unknown Wan variant '{variant}'. "
            f"Valid variants: {valid}."
        )
    # Return a copy so callers can't mutate the registry
    return dict(WAN_VARIANTS[variant])


def get_wan_backend(config: Any) -> Any:
    """Factory: training config → configured WanModelBackend.

    Creates a WanModelBackend from a FlimmerTrainingConfig. Looks up the
    variant from config.model.variant and merges architecture defaults
    with any user overrides from the config.

    Args:
        config: FlimmerTrainingConfig instance (or any object with
            config.model.variant, config.model.path, etc.).

    Returns:
        WanModelBackend instance configured for the specified variant.

    Raises:
        ValueError: If the variant is unknown.
        ImportError: If torch/diffusers are not installed.
    """
    # Late import to avoid requiring torch at import time
    from flimmer.training.wan.backend import WanModelBackend

    variant = config.model.variant
    if variant is None:
        raise ValueError(
            "config.model.variant is required. "
            "Set variant to '2.2_t2v', '2.2_i2v', '2.1_t2v', or '2.1_i2v'."
        )

    info = get_variant_info(variant)

    # Resolve HF config repo for from_single_file() loading.
    # Uses the resolution subtype from training config, falling back
    # to the variant's default_subtype.
    config_repo = None
    if "subtypes" in info:
        resolution = getattr(config.model, "resolution", None)
        subtype_key = resolution or info.get("default_subtype", "480p")
        subtype = info["subtypes"].get(subtype_key, {})
        config_repo = subtype.get("hf_repo")

    # Allow config to override variant defaults
    boundary_ratio = info["boundary_ratio"]
    if config.model.boundary_ratio is not None:
        boundary_ratio = config.model.boundary_ratio

    flow_shift = info["flow_shift"]
    if config.model.flow_shift is not None:
        flow_shift = config.model.flow_shift

    # Resolve LoRA targets: user override > variant default
    lora_targets = info["lora_targets"]
    if hasattr(config, "lora") and config.lora.target_modules is not None:
        lora_targets = config.lora.target_modules

    # Resolve individual file paths from config
    dit_path = getattr(config.model, "dit", None)
    dit_high_path = getattr(config.model, "dit_high", None)
    dit_low_path = getattr(config.model, "dit_low", None)

    # Resolve preload_experts from MoE config (default: False)
    preload_experts = False
    if hasattr(config, "moe"):
        preload_experts = getattr(config.moe, "preload_experts", False)

    return WanModelBackend(
        model_id=info["model_id"],
        model_path=config.model.path or "",
        is_moe=info["is_moe"],
        is_i2v=info["is_i2v"],
        in_channels=info["in_channels"],
        num_blocks=info["num_blocks"],
        boundary_ratio=boundary_ratio,
        flow_shift=flow_shift,
        lora_targets=lora_targets,
        expert_subfolders=info["expert_subfolders"],
        dit_path=dit_path,
        dit_high_path=dit_high_path,
        dit_low_path=dit_low_path,
        preload_experts=preload_experts,
        config_repo=config_repo,
    )
