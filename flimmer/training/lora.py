"""LoRA state management — fork, merge, save, load.

Manages LoRA weight DICTIONARIES, not actual nn.Module layers (those are
model-specific and live in Phase 8). This module handles:

    - State dict I/O (save/load via safetensors)
    - Fork: deep copy a unified LoRA into two independent expert copies
    - Merge: combine two expert LoRAs into one for inference
    - Parameter filtering: determine which params are trainable based
      on fork_targets and block_targets

The key insight: Phase 7 is model-agnostic. It manipulates state dicts
(name → tensor mappings) without knowing what model architecture the
tensors belong to. Phase 8 creates the actual LoRA layers and injects
them into Wan's transformer blocks.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from flimmer.training.errors import LoRAError
from flimmer.training.phase import PhaseType


@dataclass
class LoRAState:
    """Container for LoRA weight state and metadata.

    Holds the state dict (parameter name → tensor mapping) along with
    the LoRA configuration (rank, alpha) and the phase that produced it.
    This is the unit of save/load/fork/merge operations.

    The state_dict uses the same key format as safetensors files — flat
    string keys mapping to tensor-like values.
    """
    state_dict: dict[str, Any]
    """Parameter name → tensor mapping. Keys are flat strings like
    'blocks.0.attn1.to_q.lora_A.weight'."""

    rank: int
    """LoRA rank — determines matrix dimensions (A: d×r, B: r×d).
    Same for all parameters in this LoRA."""

    alpha: int
    """LoRA alpha — scaling factor. Effective scaling = alpha / rank."""

    phase_type: PhaseType
    """Which training phase produced this state (UNIFIED, HIGH_NOISE, LOW_NOISE)."""

    metadata: dict[str, str] = field(default_factory=dict)
    """Extra metadata stored alongside weights (e.g., epoch, step, loss)."""

    def fork(self) -> tuple[LoRAState, LoRAState]:
        """Deep copy this LoRA state into two independent copies.

        Used at the transition from unified to expert phases. Each copy
        gets its own state_dict so they can diverge independently during
        expert training. The rank, alpha, and metadata are also copied.

        Returns:
            Tuple of (copy_1, copy_2). Both are HIGH_NOISE by default —
            the caller sets the correct phase_type.
        """
        copy_1 = LoRAState(
            state_dict=copy.deepcopy(self.state_dict),
            rank=self.rank,
            alpha=self.alpha,
            phase_type=PhaseType.HIGH_NOISE,
            metadata=copy.deepcopy(self.metadata),
        )
        copy_2 = LoRAState(
            state_dict=copy.deepcopy(self.state_dict),
            rank=self.rank,
            alpha=self.alpha,
            phase_type=PhaseType.LOW_NOISE,
            metadata=copy.deepcopy(self.metadata),
        )
        return copy_1, copy_2

    def save(
        self,
        path: str | Path,
        extra_metadata: dict[str, str] | None = None,
        diffusers_prefix: str | None = None,
    ) -> Path:
        """Save LoRA weights to a safetensors file.

        Optionally adds a diffusers component prefix to all keys, making
        the file directly loadable by pipeline.load_lora_weights().

        Args:
            path: Output file path (should end in .safetensors).
            extra_metadata: Additional metadata to include in the file header.
            diffusers_prefix: If set, prepend this prefix to all keys
                (e.g. 'transformer' or 'transformer_2'). The saved file
                will be directly loadable by diffusers inference pipelines.
                None = save with clean internal keys (for training use).

        Returns:
            The resolved Path where the file was saved.

        Raises:
            LoRAError: If the save fails.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Build metadata dict — safetensors metadata must be string→string
        meta = {
            "rank": str(self.rank),
            "alpha": str(self.alpha),
            "phase_type": self.phase_type.value,
        }
        if diffusers_prefix:
            meta["diffusers_prefix"] = diffusers_prefix
        meta.update(self.metadata)
        if extra_metadata:
            meta.update(extra_metadata)

        # Optionally prefix keys for diffusers compatibility
        save_dict = self.state_dict
        if diffusers_prefix and save_dict:
            save_dict = {
                f"{diffusers_prefix}.{k}": v for k, v in save_dict.items()
            }

        # Detect whether state_dict contains numpy or torch tensors
        has_numpy = any(
            isinstance(v, np.ndarray) for v in save_dict.values()
        ) if save_dict else False

        # Try the matching backend first, then fall back
        try:
            if has_numpy:
                from safetensors.numpy import save_file as save_fn
            else:
                try:
                    from safetensors.torch import save_file as save_fn
                except ImportError:
                    from safetensors.numpy import save_file as save_fn
        except ImportError:
            raise LoRAError(
                "safetensors library not installed. "
                "Install with: pip install safetensors"
            )

        try:
            save_fn(save_dict, str(path), metadata=meta)
        except LoRAError:
            raise
        except Exception as e:
            raise LoRAError(
                f"Failed to save LoRA to '{path}': {e}"
            ) from e

        return path

    @staticmethod
    def load(path: str | Path, strip_prefix: bool = True) -> LoRAState:
        """Load LoRA weights from a safetensors file.

        Reads the state dict and metadata. Extracts rank, alpha, and
        phase_type from metadata if present.

        Auto-detects diffusers component prefixes (transformer., transformer_2.)
        and strips them by default so the state dict is in clean internal
        format for training use. Pass strip_prefix=False to keep original keys.

        Args:
            path: Path to the .safetensors file.
            strip_prefix: If True (default), strip diffusers component prefixes
                from keys. Set False to keep original keys (e.g. for passing
                directly to pipeline.load_lora_weights()).

        Returns:
            LoRAState with loaded weights and metadata.

        Raises:
            LoRAError: If the file is missing, corrupt, or unreadable.
        """
        path = Path(path)
        if not path.is_file():
            raise LoRAError(
                f"LoRA file not found: {path}\n"
                f"Check that the path is correct and the file exists."
            )

        try:
            from safetensors import safe_open
        except ImportError:
            raise LoRAError(
                "safetensors library not installed. "
                "Install with: pip install safetensors"
            )

        try:
            # Determine framework
            framework = "pt"
            try:
                import torch  # noqa: F401
            except ImportError:
                framework = "numpy"

            with safe_open(str(path), framework=framework) as f:
                metadata = f.metadata() or {}
                state_dict = {key: f.get_tensor(key) for key in f.keys()}
        except Exception as e:
            raise LoRAError(
                f"Failed to load LoRA from '{path}': {e}\n"
                f"The file may be corrupted. Try re-downloading or re-saving."
            ) from e

        # Auto-detect and strip diffusers prefixes for training use
        if strip_prefix and state_dict:
            _KNOWN_PREFIXES = ("transformer_2.", "transformer.")
            needs_strip = any(
                key.startswith(pfx) for key in state_dict for pfx in _KNOWN_PREFIXES
            )
            if needs_strip:
                stripped: dict[str, Any] = {}
                for key, value in state_dict.items():
                    clean_key = key
                    for pfx in _KNOWN_PREFIXES:
                        if key.startswith(pfx):
                            clean_key = key[len(pfx):]
                            break
                    stripped[clean_key] = value
                state_dict = stripped

        # Extract structured metadata
        rank = int(metadata.get("rank", "0"))
        alpha = int(metadata.get("alpha", str(rank)))
        phase_str = metadata.get("phase_type", PhaseType.UNIFIED.value)
        # Backward compat: accept "unified" from old safetensors files
        if phase_str == "unified":
            phase_str = "full_noise"
        try:
            phase_type = PhaseType(phase_str)
        except ValueError:
            phase_type = PhaseType.UNIFIED

        # Remove structured keys from extra metadata
        extra_meta = {
            k: v for k, v in metadata.items()
            if k not in ("rank", "alpha", "phase_type", "diffusers_prefix")
        }

        return LoRAState(
            state_dict=state_dict,
            rank=rank,
            alpha=alpha,
            phase_type=phase_type,
            metadata=extra_meta,
        )

    def filter_by_targets(
        self,
        fork_targets: list[str] | None,
        block_targets: str | None,
    ) -> dict[str, bool]:
        """Determine which parameters should be trainable.

        Filters the state dict keys based on fork_targets (which
        components) and block_targets (which blocks). Returns a dict
        mapping parameter names to trainability.

        Args:
            fork_targets: List of component names to train (e.g. ['ffn', 'self_attn']).
                None = all parameters are trainable.
            block_targets: Block range string (e.g. '0-11,25-34').
                None = all blocks are trainable.

        Returns:
            Dict mapping each state_dict key to True (trainable) or
            False (frozen).
        """
        result: dict[str, bool] = {}

        # Parse block ranges
        allowed_blocks: set[int] | None = None
        if block_targets is not None:
            allowed_blocks = _parse_block_ranges(block_targets)

        for key in self.state_dict:
            trainable = True

            # Check block targeting
            if allowed_blocks is not None:
                block_num = _extract_block_number(key)
                if block_num is not None and block_num not in allowed_blocks:
                    trainable = False

            # Check component targeting
            if trainable and fork_targets is not None:
                if not _matches_fork_targets(key, fork_targets):
                    trainable = False

            result[key] = trainable

        return result


def merge_experts(
    high_noise: LoRAState,
    low_noise: LoRAState,
) -> LoRAState:
    """Merge two expert LoRAs into one file for diffusers inference.

    Uses diffusers component prefixes so the merged file is directly
    loadable via pipeline.load_lora_weights():
        - transformer.blocks.0...   → HIGH-noise expert
        - transformer_2.blocks.0... → LOW-noise expert

    If both experts have the same keys (common for fork-and-specialize),
    the merged dict contains both under different prefixes.

    Args:
        high_noise: High-noise expert LoRA state.
        low_noise: Low-noise expert LoRA state.

    Returns:
        Merged LoRAState with combined weights, directly loadable by
        diffusers WanPipeline.load_lora_weights().

    Raises:
        LoRAError: If the experts have incompatible rank/alpha.
    """
    if high_noise.rank != low_noise.rank:
        raise LoRAError(
            f"Cannot merge experts with different ranks: "
            f"high_noise={high_noise.rank}, low_noise={low_noise.rank}. "
            f"Both experts must use the same rank."
        )
    if high_noise.alpha != low_noise.alpha:
        raise LoRAError(
            f"Cannot merge experts with different alpha: "
            f"high_noise={high_noise.alpha}, low_noise={low_noise.alpha}. "
            f"Both experts must use the same alpha."
        )

    # Diffusers convention:
    #   transformer   = HIGH-noise expert
    #   transformer_2 = LOW-noise expert
    TRANSFORMER = "transformer"
    TRANSFORMER_2 = "transformer_2"

    merged: dict[str, Any] = {}

    # Add high-noise weights with diffusers prefix
    for key, tensor in high_noise.state_dict.items():
        merged[f"{TRANSFORMER}.{key}"] = tensor

    # Add low-noise weights with diffusers prefix
    for key, tensor in low_noise.state_dict.items():
        merged[f"{TRANSFORMER_2}.{key}"] = tensor

    return LoRAState(
        state_dict=merged,
        rank=high_noise.rank,
        alpha=high_noise.alpha,
        phase_type=PhaseType.UNIFIED,  # Merged = back to unified for inference
        metadata={
            "merged_from": "high_noise+low_noise",
            "high_noise_keys": str(len(high_noise.state_dict)),
            "low_noise_keys": str(len(low_noise.state_dict)),
        },
    )


def build_parameter_groups(
    state_dict: dict[str, Any],
    trainable_mask: dict[str, bool],
    learning_rate: float,
    loraplus_lr_ratio: float = 1.0,
) -> list[dict[str, Any]]:
    """Build optimizer parameter groups with LoRA+ support.

    LoRA+ uses different learning rates for A-matrix (down projection)
    and B-matrix (up projection). The B matrix starts from zero and
    needs to learn faster.

    Args:
        state_dict: Full parameter state dict.
        trainable_mask: Which parameters are trainable (from filter_by_targets).
        learning_rate: Base learning rate for A-matrix parameters.
        loraplus_lr_ratio: Multiplier for B-matrix learning rate.
            1.0 = same LR for both (standard LoRA). > 1.0 = LoRA+.

    Returns:
        List of optimizer parameter group dicts, each with 'params' and 'lr'.
        Frozen parameters are excluded.
    """
    a_params: list[str] = []
    b_params: list[str] = []

    for key, is_trainable in trainable_mask.items():
        if not is_trainable:
            continue
        if "lora_B" in key or "lora_up" in key:
            b_params.append(key)
        else:
            a_params.append(key)

    groups: list[dict[str, Any]] = []

    if a_params:
        groups.append({
            "params": a_params,
            "lr": learning_rate,
        })

    if b_params:
        groups.append({
            "params": b_params,
            "lr": learning_rate * loraplus_lr_ratio,
        })

    return groups


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_block_ranges(block_targets: str) -> set[int]:
    """Parse a block range string into a set of block indices.

    Accepts formats like '0-11', '0-11,25-34', '5'.

    Args:
        block_targets: Block range string.

    Returns:
        Set of allowed block indices.
    """
    allowed: set[int] = set()
    for part in block_targets.split(","):
        part = part.strip()
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start, end = int(start_s), int(end_s)
            allowed.update(range(start, end + 1))
        else:
            allowed.add(int(part))
    return allowed


def _extract_block_number(key: str) -> int | None:
    """Extract the block/layer number from a parameter key.

    Looks for patterns like 'blocks.5.' or 'layers.12.' in the key.
    Returns the number, or None if no block number is found.

    Args:
        key: Parameter name string.

    Returns:
        Block number, or None.
    """
    match = re.search(r"(?:blocks|layers|block|layer)\.(\d+)\.", key)
    if match:
        return int(match.group(1))
    return None


def _matches_fork_targets(key: str, fork_targets: list[str]) -> bool:
    """Check if a parameter key matches any of the fork targets.

    Fork targets can be component-level ('ffn', 'self_attn') or
    projection-level ('cross_attn.to_v'). A parameter matches if its
    key contains the target string.

    Args:
        key: Parameter name string.
        fork_targets: List of target patterns.

    Returns:
        True if the key matches at least one target.
    """
    key_lower = key.lower()
    for target in fork_targets:
        # Normalize target for matching
        target_lower = target.lower()

        # Direct substring match
        if target_lower in key_lower:
            return True

        # Handle component aliases
        # 'ffn' matches 'feed_forward', 'mlp', 'ffn'
        if target_lower == "ffn" and any(
            alias in key_lower for alias in ("feed_forward", "mlp", "ffn")
        ):
            return True

        # 'self_attn' matches 'attn1', 'self_attn', 'self_attention'
        if target_lower == "self_attn" and any(
            alias in key_lower for alias in ("attn1", "self_attn", "self_attention")
        ):
            return True

        # 'cross_attn' matches 'attn2', 'cross_attn', 'cross_attention'
        if target_lower == "cross_attn" and any(
            alias in key_lower for alias in ("attn2", "cross_attn", "cross_attention")
        ):
            return True

    return False
