"""LTX-2.3 model backend for Flimmer training.

Implements the ModelBackend protocol for the LTX-2.3 model family.

Module structure:
    constants      — LTX architecture constants, LoRA target patterns
    registry       — Variant registry: config string → backend constructor args
    backend        — LtxModelBackend (implements ModelBackend protocol)

GPU-free modules (constants, registry) can be imported without torch/ltx-core.
GPU-dependent modules (backend) require the 'ltx' dependency group.
"""

from flimmer.training.ltx.constants import (
    LTX_NUM_BLOCKS,
    LTX_VIDEO_LORA_TARGETS,
)
from flimmer.training.ltx.registry import LTX_VARIANTS, get_ltx_backend

__all__ = [
    "LTX_NUM_BLOCKS",
    "LTX_VIDEO_LORA_TARGETS",
    "LTX_VARIANTS",
    "get_ltx_backend",
]
