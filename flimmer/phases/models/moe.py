"""Wan 2.2 T2V 14B (MoE) model definition.

Mixture-of-Experts model with three phase types:
- unified: Both experts share one LoRA, no expert masking
- high_noise: High-noise expert (coarse composition, motion)
- low_noise: Low-noise expert (fine detail, texture)

Expert phases require boundary_ratio in extras to define the
SNR boundary for expert routing.
"""

from ..definitions import (
    ModelDefinition,
    ParamSpec,
    PhaseTypeDeclaration,
    SignalDeclaration,
)
from ..registry import register_model

WAN_22_T2V = ModelDefinition(
    model_id="wan-2.2-t2v-14b",
    family="wan",
    variant="2.2_t2v",
    display_name="Wan 2.2 T2V 14B (MoE)",
    is_moe=True,
    supported_signals=[
        SignalDeclaration(
            modality="text",
            required=True,
            description="Text prompt conditioning via UMT5",
        ),
        SignalDeclaration(
            modality="video",
            required=True,
            description="Target video data",
        ),
    ],
    phase_types=[
        PhaseTypeDeclaration(
            name="unified",
            description="Both experts share one LoRA. No expert masking.",
        ),
        PhaseTypeDeclaration(
            name="high_noise",
            description="High-noise expert phase (coarse composition, motion)",
            required_fields=["boundary_ratio"],
        ),
        PhaseTypeDeclaration(
            name="low_noise",
            description="Low-noise expert phase (fine detail, texture)",
            required_fields=["boundary_ratio"],
        ),
    ],
    params=[
        # -- Phase-level (overridable per phase/expert) --
        ParamSpec(
            name="learning_rate",
            type="float",
            default=5e-5,
            min_value=1e-6,
            max_value=1e-2,
            phase_level=True,
            description="Optimizer learning rate",
        ),
        ParamSpec(
            name="weight_decay",
            type="float",
            default=0.01,
            min_value=0.0,
            max_value=1.0,
            phase_level=True,
            description="Weight decay for regularization",
        ),
        ParamSpec(
            name="batch_size",
            type="int",
            default=1,
            min_value=1,
            max_value=64,
            phase_level=True,
            description="Training batch size",
        ),
        ParamSpec(
            name="gradient_accumulation_steps",
            type="int",
            default=1,
            min_value=1,
            max_value=128,
            phase_level=True,
            description="Steps between optimizer updates",
        ),
        ParamSpec(
            name="caption_dropout_rate",
            type="float",
            default=0.10,
            min_value=0.0,
            max_value=1.0,
            phase_level=True,
            description="Probability of dropping entire caption",
        ),
        ParamSpec(
            name="lora_dropout",
            type="float",
            default=0.0,
            min_value=0.0,
            max_value=1.0,
            phase_level=True,
            description="LoRA adapter dropout rate",
        ),
        ParamSpec(
            name="max_epochs",
            type="int",
            default=10,
            min_value=1,
            max_value=10000,
            phase_level=True,
            description="Training epochs for this phase",
        ),
        ParamSpec(
            name="min_lr_ratio",
            type="float",
            default=0.01,
            min_value=0.0,
            max_value=1.0,
            phase_level=True,
            description="Minimum LR as fraction of peak LR",
        ),
        ParamSpec(
            name="optimizer_type",
            type="str",
            default="adamw8bit",
            phase_level=True,
            description="Optimizer type",
        ),
        ParamSpec(
            name="scheduler_type",
            type="str",
            default="cosine_with_min_lr",
            phase_level=True,
            description="LR scheduler type",
        ),
        ParamSpec(
            name="boundary_ratio",
            type="float",
            default=0.875,
            min_value=0.0,
            max_value=1.0,
            phase_level=True,
            description="SNR boundary for expert routing",
        ),
        # -- Run-level (not overridable per phase) --
        ParamSpec(
            name="lora_rank",
            type="int",
            default=16,
            min_value=1,
            max_value=256,
            phase_level=False,
            description="LoRA rank (locked at creation)",
        ),
        ParamSpec(
            name="lora_alpha",
            type="int",
            default=16,
            min_value=1,
            max_value=256,
            phase_level=False,
            description="LoRA alpha scaling factor",
        ),
        ParamSpec(
            name="mixed_precision",
            type="str",
            default="bf16",
            phase_level=False,
            description="Mixed precision mode",
        ),
        ParamSpec(
            name="base_model_precision",
            type="str",
            default="bf16",
            phase_level=False,
            description="Frozen base model precision",
        ),
    ],
    defaults={
        "learning_rate": 5e-5,
        "batch_size": 1,
        "max_epochs": 10,
        "boundary_ratio": 0.875,
    },
)

register_model(WAN_22_T2V)
