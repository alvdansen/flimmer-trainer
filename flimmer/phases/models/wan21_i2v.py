"""Wan 2.1 I2V 14B (non-MoE) model definition.

Image-to-Video variant of Wan 2.1. Non-MoE, full_noise phase type only.
Adds first_frame signal and uses higher caption_dropout_rate default
(0.15) because the first-frame image carries more conditioning.

No boundary_ratio param since there is no MoE expert routing.
"""

from ..definitions import (
    ModelDefinition,
    ParamSpec,
    PhaseTypeDeclaration,
    SignalDeclaration,
)
from ..registry import register_model

WAN_21_I2V = ModelDefinition(
    model_id="wan-2.1-i2v-14b",
    family="wan",
    variant="2.1_i2v",
    display_name="Wan 2.1 I2V 14B",
    is_moe=False,
    supported_signals=[
        SignalDeclaration(
            modality="text",
            required=True,
            description="Text prompt conditioning",
        ),
        SignalDeclaration(
            modality="video",
            required=True,
            description="Target video data",
        ),
        SignalDeclaration(
            modality="first_frame",
            required=True,
            description="First-frame image for I2V conditioning",
        ),
    ],
    phase_types=[
        PhaseTypeDeclaration(
            name="full_noise",
            description="Standard single-phase training",
        ),
    ],
    params=[
        # -- Phase-level (overridable per phase) --
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
            default=0.15,
            min_value=0.0,
            max_value=1.0,
            phase_level=True,
            description="Probability of dropping entire caption (higher for I2V)",
        ),
        ParamSpec(
            name="first_frame_dropout_rate",
            type="float",
            default=0.05,
            min_value=0.0,
            max_value=1.0,
            phase_level=True,
            description="Probability of dropping first-frame conditioning (independent from caption dropout)",
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
    },
)

register_model(WAN_21_I2V)
