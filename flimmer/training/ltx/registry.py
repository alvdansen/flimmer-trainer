"""Variant registry — maps config strings to backend constructor args.

Each LTX variant has different architecture parameters. The registry
centralizes this mapping so the backend constructor doesn't need to
hardcode variant-specific logic.

This module is GPU-free — it just defines the mapping. Actual model loading
happens in backend.py.
"""

from __future__ import annotations

from typing import Any

from flimmer.training.ltx.constants import (
    LTX_NUM_BLOCKS,
    LTX_T2V_IN_CHANNELS,
    LTX_VIDEO_LORA_TARGETS,
)


# ---------------------------------------------------------------------------
# Variant definitions
# ---------------------------------------------------------------------------

LTX_VARIANTS: dict[str, dict[str, Any]] = {
    "2.3_t2v": {
        "model_id": "ltx-2.3-t2v",
        "is_moe": False,
        "is_i2v": False,
        "in_channels": LTX_T2V_IN_CHANNELS,
        "num_blocks": LTX_NUM_BLOCKS,
        "lora_targets": list(LTX_VIDEO_LORA_TARGETS),
        "hf_repo": "Lightricks/LTX-2.3",
        "pipeline_class": "LTXPipeline",
    },
}
"""Registry of LTX model variants and their architecture parameters.

Each entry contains everything needed to construct an LtxModelBackend:
    - model_id: human-readable identifier for logging/metadata
    - is_moe: whether this variant uses Mixture of Experts (always False)
    - is_i2v: whether this variant accepts reference images (always False)
    - in_channels: latent input channel count
    - num_blocks: number of transformer blocks
    - lora_targets: default module names for LoRA adapter placement
    - hf_repo: HuggingFace repository ID
    - pipeline_class: pipeline class name for inference (placeholder)
"""


def get_variant_info(variant: str) -> dict[str, Any]:
    """Look up variant configuration by name.

    Args:
        variant: Variant string (e.g. '2.3_t2v').

    Returns:
        Dict of variant parameters (copy, safe to modify).

    Raises:
        ValueError: If the variant is not recognized.
    """
    if variant not in LTX_VARIANTS:
        valid = ", ".join(sorted(LTX_VARIANTS.keys()))
        raise ValueError(
            f"Unknown LTX variant '{variant}'. "
            f"Valid variants: {valid}."
        )
    # Return a copy so callers can't mutate the registry
    return dict(LTX_VARIANTS[variant])


def get_ltx_backend(config: Any) -> Any:
    """Factory: training config → configured LtxModelBackend.

    Creates an LtxModelBackend from a FlimmerTrainingConfig. Looks up the
    variant from config.model.variant and merges architecture defaults
    with any user overrides from the config.

    Args:
        config: FlimmerTrainingConfig instance (or any object with
            config.model.variant, config.model.path, etc.).

    Returns:
        LtxModelBackend instance configured for the specified variant.

    Raises:
        ValueError: If the variant is unknown.
        ImportError: If torch/ltx-core are not installed.
    """
    # Late import to avoid requiring torch at import time
    from flimmer.training.ltx.backend import LtxModelBackend

    variant = config.model.variant
    if variant is None:
        raise ValueError(
            "config.model.variant is required. "
            "Set variant to '2.3_t2v'."
        )

    info = get_variant_info(variant)

    # Resolve LoRA targets: user override > variant default
    lora_targets = info["lora_targets"]
    if hasattr(config, "lora") and config.lora.target_modules is not None:
        lora_targets = config.lora.target_modules

    # Resolve quantization from config (default: None)
    quantization = None
    if hasattr(config, "model") and hasattr(config.model, "quantization"):
        quantization = getattr(config.model, "quantization", None)

    return LtxModelBackend(
        model_id=info["model_id"],
        model_path=config.model.path or "",
        is_moe=info["is_moe"],
        is_i2v=info["is_i2v"],
        in_channels=info["in_channels"],
        num_blocks=info["num_blocks"],
        lora_targets=lora_targets,
        hf_repo=info["hf_repo"],
        quantization=quantization,
    )
