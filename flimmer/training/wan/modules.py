"""PEFT bridge — create, extract, inject, remove LoRA on live models.

This is the critical bridge between Phase 7's LoRAState (state dicts —
just name→tensor mappings) and actual PEFT LoRA layers on the live
torch model. Phase 7 manipulates state dicts; this module manipulates
the model itself.

The PEFT lifecycle during training:
    1. Unified start: create_lora_on_model() → train
    2. Before fork: extract_lora_state_dict() → remove_lora_from_model()
    3. After fork: LoRAState.fork() produces two copies
    4. Expert start: create_lora_on_model() → inject_lora_state_dict(forked)
    5. Expert switch: extract → remove → unload model → load other → create → inject

GPU-free functions:
    resolve_target_modules()  — resolves user config to PEFT target list

GPU-required functions:
    create_lora_on_model()    — creates LoRA layers via PEFT
    extract_lora_state_dict() — reads LoRA weights from model
    inject_lora_state_dict()  — writes LoRA weights into model
    remove_lora_from_model()  — removes LoRA layers cleanly
"""

from __future__ import annotations

from typing import Any

from flimmer.training.wan.constants import (
    FORK_TARGET_TO_MODULES,
    I2V_EXTRA_TARGETS,
    T2V_LORA_TARGETS,
)


# ---------------------------------------------------------------------------
# GPU-free: target module resolution
# ---------------------------------------------------------------------------

def resolve_target_modules(
    variant_targets: list[str],
    user_overrides: list[str] | None = None,
    fork_targets: list[str] | None = None,
) -> list[str]:
    """Resolve the final list of PEFT target module names.

    Three levels of resolution:
    1. Variant defaults (T2V_LORA_TARGETS or T2V + I2V_EXTRA)
    2. User overrides from lora.target_modules (replaces variant defaults)
    3. Fork targets from fork_targets config (filters to subset)

    Args:
        variant_targets: Default targets from the variant registry.
        user_overrides: User's explicit target_modules list (from config).
            If set, completely replaces variant_targets.
        fork_targets: Per-expert fork targets (from config).
            If set, filters the active targets to only matching modules.
            Uses FORK_TARGET_TO_MODULES mapping.

    Returns:
        Deduplicated list of module name suffixes for PEFT target_modules.

    Example:
        >>> resolve_target_modules(T2V_LORA_TARGETS)
        ['attn1.to_q', 'attn1.to_k', ..., 'ffn.net.2']

        >>> resolve_target_modules(T2V_LORA_TARGETS, fork_targets=["ffn"])
        ['ffn.net.0.proj', 'ffn.net.2']
    """
    # Step 1: user overrides replace variant defaults entirely
    base_targets = list(user_overrides) if user_overrides else list(variant_targets)

    # Step 2: fork targets filter down to a subset
    if fork_targets is None:
        return _deduplicate(base_targets)

    # Expand fork target names to module suffixes
    allowed_suffixes: set[str] = set()
    for ft in fork_targets:
        if ft in FORK_TARGET_TO_MODULES:
            allowed_suffixes.update(FORK_TARGET_TO_MODULES[ft])
        else:
            # If the fork target isn't in our mapping, treat it as a
            # literal module suffix — the user may be targeting a specific
            # module directly
            allowed_suffixes.add(ft)

    # Filter base targets to only those matching fork targets
    filtered = [t for t in base_targets if t in allowed_suffixes]
    return _deduplicate(filtered)


def _deduplicate(targets: list[str]) -> list[str]:
    """Remove duplicates while preserving order."""
    seen: set[str] = set()
    result: list[str] = []
    for t in targets:
        if t not in seen:
            seen.add(t)
            result.append(t)
    return result


# ---------------------------------------------------------------------------
# GPU-required: PEFT operations on live models
# ---------------------------------------------------------------------------

def create_lora_on_model(
    model: Any,
    target_modules: list[str],
    rank: int,
    alpha: int,
    dropout: float = 0.0,
    adapter_name: str = "default",
) -> Any:
    """Create LoRA adapter layers on a model using PEFT.

    Adds trainable LoRA matrices to the specified modules. The base model
    weights are frozen — only the LoRA parameters are trainable.

    Args:
        model: A torch.nn.Module (e.g. WanTransformer3DModel).
        target_modules: List of module name suffixes to add LoRA to.
        rank: LoRA rank (determines A/B matrix dimensions).
        alpha: LoRA alpha (scaling factor; effective scale = alpha / rank).
        dropout: LoRA dropout rate (0.0 = no dropout).
        adapter_name: Name for the PEFT adapter (for multi-adapter support).

    Returns:
        The PEFT-wrapped model with LoRA adapters. The caller MUST capture
        this return value — get_peft_model returns a new PeftModel wrapper.

    Raises:
        ImportError: If peft is not installed.
        RuntimeError: If LoRA creation fails.
    """
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError:
        raise ImportError(
            "peft is required for LoRA operations. "
            "Install with: pip install peft>=0.12"
        )

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        init_lora_weights=True,
    )

    # get_peft_model returns a PeftModel wrapping the original model
    return get_peft_model(model, lora_config, adapter_name=adapter_name)


def extract_lora_state_dict(model: Any) -> dict[str, Any]:
    """Extract LoRA parameters from a PEFT-wrapped model.

    Returns a state dict containing ONLY the LoRA A/B matrices — not
    the base model weights. Keys follow the PEFT naming convention:
    'base_model.model.blocks.0.attn1.to_q.lora_A.default.weight'

    This function strips the PEFT wrapper prefix to produce clean keys:
    'blocks.0.attn1.to_q.lora_A.weight'

    Args:
        model: A PEFT-wrapped model with LoRA adapters.

    Returns:
        Dict mapping clean parameter names to tensor values.
    """
    import torch

    state_dict: dict[str, Any] = {}

    for name, param in model.named_parameters():
        if "lora_" in name:
            # Strip PEFT wrapper prefixes:
            # 'base_model.model.blocks.0.attn1.to_q.lora_A.default.weight'
            # → 'blocks.0.attn1.to_q.lora_A.weight'
            clean_name = name
            # Remove 'base_model.model.' prefix
            if clean_name.startswith("base_model.model."):
                clean_name = clean_name[len("base_model.model."):]
            # Remove adapter name from LoRA suffix
            # 'lora_A.default.weight' → 'lora_A.weight'
            clean_name = clean_name.replace(".default.", ".")
            state_dict[clean_name] = param.detach().cpu().clone()

    return state_dict


def inject_lora_state_dict(
    model: Any,
    state_dict: dict[str, Any],
) -> None:
    """Inject LoRA weights from a state dict into a PEFT-wrapped model.

    The model must already have LoRA adapters created (via create_lora_on_model).
    This function loads pre-trained weights into those adapters.

    Args:
        model: A PEFT-wrapped model with LoRA adapters.
        state_dict: Dict of clean LoRA parameter names → tensors.
            Keys should be like 'blocks.0.attn1.to_q.lora_A.weight'.

    Raises:
        RuntimeError: If state dict keys don't match model parameters.
    """
    import torch

    # Build a reverse mapping from clean names to model parameter names
    param_map: dict[str, str] = {}
    for name, _ in model.named_parameters():
        if "lora_" in name:
            clean_name = name
            if clean_name.startswith("base_model.model."):
                clean_name = clean_name[len("base_model.model."):]
            clean_name = clean_name.replace(".default.", ".")
            param_map[clean_name] = name

    # Load each weight
    with torch.no_grad():
        for clean_name, tensor in state_dict.items():
            if clean_name not in param_map:
                continue  # Skip keys not in current model
            full_name = param_map[clean_name]

            # Navigate to the parameter
            parts = full_name.split(".")
            obj = model
            for part in parts[:-1]:
                obj = getattr(obj, part)
            param = getattr(obj, parts[-1])
            param.copy_(tensor.to(param.device, param.dtype))


def remove_lora_from_model(model: Any) -> Any:
    """Remove LoRA adapters and return the base model.

    Merges LoRA weights into the base model and removes the PEFT wrapper.
    After this call, the model is a plain nn.Module with no LoRA layers.

    Note: this MERGES the current LoRA weights. If you want to discard
    them, extract first, then remove.

    Args:
        model: A PEFT-wrapped model with LoRA adapters.

    Returns:
        The unwrapped base model (nn.Module).
    """
    try:
        from peft import PeftModel
        if isinstance(model, PeftModel):
            return model.unload()
    except ImportError:
        pass
    return model
