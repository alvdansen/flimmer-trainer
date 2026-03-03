"""Wan model backend for Flimmer training.

Implements the Phase 7 protocols (ModelBackend, InferencePipeline) for
the Wan model family: Wan 2.1 T2V, Wan 2.2 T2V, and Wan 2.2 I2V.

Module structure:
    constants      — Wan architecture constants, LoRA target patterns
    registry       — Variant registry: config string → backend constructor args
    modules        — PEFT bridge: create/extract/inject/remove LoRA on model
    checkpoint_io  — Format conversion: flimmer ↔ musubi ↔ ComfyUI
    backend        — WanModelBackend (implements ModelBackend protocol)
    inference      — WanInferencePipeline (implements InferencePipeline protocol)

GPU-free modules (constants, registry, checkpoint_io, parts of modules)
can be imported without torch/diffusers. GPU-dependent modules (backend,
inference, live PEFT operations) require the 'wan' dependency group.
"""

from flimmer.training.wan.constants import (
    I2V_EXTRA_TARGETS,
    T2V_LORA_TARGETS,
    WAN_NUM_BLOCKS,
)
from flimmer.training.wan.registry import WAN_VARIANTS, get_wan_backend

__all__ = [
    "T2V_LORA_TARGETS",
    "I2V_EXTRA_TARGETS",
    "WAN_NUM_BLOCKS",
    "WAN_VARIANTS",
    "get_wan_backend",
]
