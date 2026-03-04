"""Tests for the Flimmer training config schema, loader, and validation.

Rewritten to match the current codebase — everything lives in
wan22_training_master.py (not training_defaults.py / training_schema.py).

Covers:
  - All Pydantic sub-configs: ModelConfig, LoraConfig, OptimizerConfig,
    SchedulerConfig, MoeExpertOverrides, MoeConfig, TrainingLoopConfig,
    SaveConfig, LoggingConfig, SamplingConfig
  - Root-level FlimmerTrainingConfig validators (check_moe_consistency,
    check_prodigy_lr, check_wandb_project, check_mua_alpha,
    check_fork_without_moe, warn_aggressive_low_noise)
  - VARIANT_DEFAULTS and T2V/I2V constant values
  - Loader helpers: _deep_merge, _apply_variant_defaults, _auto_enable_moe,
    _is_huggingface_id, _resolve_paths, _format_validation_error
  - Integration tests: load_training_config with real YAML via tmp_path
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest
import yaml

from flimmer.config.wan22_training_master import (
    FlimmerTrainingConfig,
    ModelConfig,
    LoraConfig,
    OptimizerConfig,
    SchedulerConfig,
    MoeExpertOverrides,
    MoeConfig,
    TrainingLoopConfig,
    SaveConfig,
    LoggingConfig,
    SamplingConfig,
    VALID_OPTIMIZERS,
    VALID_SCHEDULERS,
    VALID_MIXED_PRECISION,
    VALID_BASE_PRECISION,
    VALID_TIMESTEP_SAMPLING,
    VALID_LOG_BACKENDS,
    VALID_CHECKPOINT_FORMATS,
    VALID_FORK_TARGETS,
    VARIANT_DEFAULTS,
    T2V_LORA_RANK,
    T2V_LORA_ALPHA,
    T2V_LEARNING_RATE,
    T2V_OPTIMIZER,
    T2V_SCHEDULER,
    T2V_WEIGHT_DECAY,
    T2V_MIXED_PRECISION,
    T2V_BASE_MODEL_PRECISION,
    T2V_TIMESTEP_SAMPLING,
    T2V_UNIFIED_EPOCHS,
    T2V_BATCH_SIZE,
    T2V_GRADIENT_ACCUMULATION,
    T2V_CAPTION_DROPOUT_RATE,
    T2V_LORAPLUS_LR_RATIO,
    T2V_LORA_DROPOUT,
    T2V_WARMUP_STEPS,
    T2V_MIN_LR_RATIO,
    T2V_SAVE_EVERY_N_EPOCHS,
    T2V_CHECKPOINT_FORMAT,
    T2V_SAMPLING_ENABLED,
    T2V_SAMPLING_SEED,
    T2V_SAMPLING_STEPS,
    T2V_SAMPLING_GUIDANCE,
    T2V_FORK_ENABLED,
    T2V_IS_MOE,
    T2V_BETAS,
    T2V_EPS,
    T2V_MAX_GRAD_NORM,
)
from flimmer.config.training_loader import (
    load_training_config,
    _deep_merge,
    _apply_variant_defaults,
    _auto_enable_moe,
    _is_huggingface_id,
    _resolve_paths,
    _resolve_model_path,
    _format_validation_error,
    TRAINING_CONFIG_FILENAME,
)
from flimmer.config.loader import FlimmerConfigError


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FIXTURES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@pytest.fixture
def tmp_data_config(tmp_path: Path) -> Path:
    """Create a minimal data config file that load_training_config can find."""
    dataset_dir = tmp_path / "video_clips"
    dataset_dir.mkdir()
    (dataset_dir / "clip_001.mp4").touch()

    data_config = tmp_path / "flimmer_data.yaml"
    data_config.write_text(
        yaml.dump({"datasets": [{"path": str(dataset_dir)}]}),
        encoding="utf-8",
    )
    return data_config


@pytest.fixture
def tmp_model_dir(tmp_path: Path) -> Path:
    """Create a fake model directory."""
    model_dir = tmp_path / "models" / "Wan2.2-T2V-14B"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").touch()
    return model_dir


def write_train_config(path: Path, data: dict) -> Path:
    """Write a YAML training config dict to a file and return the file path."""
    config_path = path / "flimmer_train.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False)
    return config_path


def minimal_config_data(
    model_path: str | Path, data_config: str | Path
) -> dict:
    """Return the smallest valid training config dict (uses variant, not template)."""
    return {
        "model": {
            "variant": "2.2_t2v",
            "path": str(model_path),
        },
        "data_config": str(data_config),
    }


def make_root_config(**overrides) -> FlimmerTrainingConfig:
    """Build a FlimmerTrainingConfig with sensible defaults and overrides.

    Always provides model.path and data_config so the root config is valid.
    Caller can override any sub-config by passing keyword arguments matching
    the field name, e.g. make_root_config(lora=LoraConfig(rank=64)).
    """
    defaults = {
        "model": ModelConfig(path="Wan-AI/Wan2.2-T2V-14B-Diffusers"),
        "data_config": "C:/fake/flimmer_data.yaml",
    }
    defaults.update(overrides)
    return FlimmerTrainingConfig(**defaults)


# ====================================================================
# 1. TestModelConfig
# ====================================================================


class TestModelConfig:
    """ModelConfig uses variant (not template) and has architecture fields."""

    def test_minimal_model_config(self):
        """Only path is required — everything else is optional."""
        m = ModelConfig(path="C:/models/Wan")
        assert m.path == "C:/models/Wan"
        assert m.variant is None
        assert m.family is None
        assert m.is_moe is None

    def test_variant_field(self):
        """variant is a free string field (loader checks against VARIANT_DEFAULTS)."""
        m = ModelConfig(path="C:/m", variant="2.2_t2v")
        assert m.variant == "2.2_t2v"

    def test_family_field(self):
        m = ModelConfig(path="C:/m", family="wan")
        assert m.family == "wan"

    def test_is_moe_field(self):
        m = ModelConfig(path="C:/m", is_moe=True)
        assert m.is_moe is True

    def test_in_channels(self):
        m = ModelConfig(path="C:/m", in_channels=36)
        assert m.in_channels == 36

    def test_in_channels_must_be_positive(self):
        with pytest.raises(Exception):
            ModelConfig(path="C:/m", in_channels=0)

    def test_num_layers(self):
        m = ModelConfig(path="C:/m", num_layers=40)
        assert m.num_layers == 40

    def test_num_layers_must_be_positive(self):
        with pytest.raises(Exception):
            ModelConfig(path="C:/m", num_layers=0)

    def test_boundary_ratio(self):
        m = ModelConfig(path="C:/m", boundary_ratio=0.875)
        assert m.boundary_ratio == 0.875

    def test_flow_shift(self):
        m = ModelConfig(path="C:/m", flow_shift=5.0)
        assert m.flow_shift == 5.0

    def test_num_train_timesteps_default(self):
        m = ModelConfig(path="C:/m")
        assert m.num_train_timesteps == 1000

    def test_num_train_timesteps_override(self):
        m = ModelConfig(path="C:/m", num_train_timesteps=500)
        assert m.num_train_timesteps == 500

    def test_no_template_field(self):
        """The old "template" field no longer exists — variant replaced it."""
        m = ModelConfig(path="C:/m")
        assert not hasattr(m, "template")

    def test_hf_id_as_path(self):
        m = ModelConfig(path="Wan-AI/Wan2.2-T2V-14B-Diffusers")
        assert m.path == "Wan-AI/Wan2.2-T2V-14B-Diffusers"


# ====================================================================
# 2. TestLoraConfig
# ====================================================================


class TestLoraConfig:
    """LoRA settings: rank, alpha, dropout, loraplus, target_modules,
    block_rank_overrides, mua_init. NO exclude_modules field."""

    def test_default_rank(self):
        assert LoraConfig().rank == T2V_LORA_RANK == 16

    def test_default_alpha(self):
        assert LoraConfig().alpha == T2V_LORA_ALPHA == 16

    def test_default_dropout(self):
        assert LoraConfig().dropout == T2V_LORA_DROPOUT == 0.0

    def test_default_loraplus(self):
        assert LoraConfig().loraplus_lr_ratio == T2V_LORAPLUS_LR_RATIO == 4.0

    def test_default_target_modules_is_none(self):
        assert LoraConfig().target_modules is None

    def test_default_block_rank_overrides_is_none(self):
        assert LoraConfig().block_rank_overrides is None

    def test_default_mua_init_false(self):
        assert LoraConfig().use_mua_init is False

    def test_rank_override(self):
        assert LoraConfig(rank=64).rank == 64

    def test_alpha_override(self):
        assert LoraConfig(alpha=32).alpha == 32

    def test_rank_must_be_positive(self):
        with pytest.raises(Exception):
            LoraConfig(rank=0)

    def test_alpha_must_be_positive(self):
        with pytest.raises(Exception):
            LoraConfig(alpha=0)

    def test_dropout_range_low(self):
        assert LoraConfig(dropout=0.0).dropout == 0.0

    def test_dropout_range_high(self):
        assert LoraConfig(dropout=1.0).dropout == 1.0

    def test_dropout_below_zero(self):
        with pytest.raises(Exception):
            LoraConfig(dropout=-0.01)

    def test_dropout_above_one(self):
        with pytest.raises(Exception):
            LoraConfig(dropout=1.01)

    def test_loraplus_min_is_one(self):
        assert LoraConfig(loraplus_lr_ratio=1.0).loraplus_lr_ratio == 1.0

    def test_loraplus_below_one(self):
        with pytest.raises(Exception):
            LoraConfig(loraplus_lr_ratio=0.5)

    def test_target_modules_list(self):
        cfg = LoraConfig(target_modules=["to_q", "to_k", "to_v"])
        assert cfg.target_modules == ["to_q", "to_k", "to_v"]

    def test_block_rank_overrides_dict(self):
        cfg = LoraConfig(block_rank_overrides={"0-9": 48, "30-39": 8})
        assert cfg.block_rank_overrides == {"0-9": 48, "30-39": 8}

    def test_mua_init_true(self):
        cfg = LoraConfig(use_mua_init=True)
        assert cfg.use_mua_init is True

    def test_no_exclude_modules_field(self):
        """exclude_modules was removed from LoraConfig."""
        cfg = LoraConfig()
        assert not hasattr(cfg, "exclude_modules")


# ====================================================================
# 3. TestOptimizerConfig
# ====================================================================


class TestOptimizerConfig:
    """Optimizer: type validation, betas validation, prodigy lr check."""

    def test_default_type(self):
        assert OptimizerConfig().type == T2V_OPTIMIZER == "adamw8bit"

    def test_default_learning_rate(self):
        assert OptimizerConfig().learning_rate == T2V_LEARNING_RATE == 5e-5

    def test_default_weight_decay(self):
        assert OptimizerConfig().weight_decay == T2V_WEIGHT_DECAY == 0.01

    def test_default_betas(self):
        assert OptimizerConfig().betas == T2V_BETAS == [0.9, 0.999]

    def test_default_eps(self):
        assert OptimizerConfig().eps == T2V_EPS == 1e-8

    def test_default_max_grad_norm(self):
        assert OptimizerConfig().max_grad_norm == T2V_MAX_GRAD_NORM == 1.0

    def test_default_optimizer_args(self):
        assert OptimizerConfig().optimizer_args == {}

    def test_all_valid_optimizers(self):
        """Every value in VALID_OPTIMIZERS should be accepted."""
        for opt in VALID_OPTIMIZERS:
            cfg = OptimizerConfig(type=opt)
            assert cfg.type == opt

    def test_invalid_optimizer_raises(self):
        with pytest.raises(Exception, match="Unknown optimizer"):
            OptimizerConfig(type="sgd")

    def test_betas_two_elements(self):
        cfg = OptimizerConfig(betas=[0.95, 0.99])
        assert cfg.betas == [0.95, 0.99]

    def test_betas_three_elements_came(self):
        """CAME uses three betas."""
        cfg = OptimizerConfig(betas=[0.9, 0.999, 0.9999])
        assert len(cfg.betas) == 3

    def test_betas_one_element_raises(self):
        with pytest.raises(Exception, match="2 elements"):
            OptimizerConfig(betas=[0.9])

    def test_betas_four_elements_raises(self):
        with pytest.raises(Exception):
            OptimizerConfig(betas=[0.9, 0.99, 0.999, 0.9999])

    def test_betas_out_of_range(self):
        with pytest.raises(Exception, match="out of range"):
            OptimizerConfig(betas=[0.9, 1.0])

    def test_betas_negative(self):
        with pytest.raises(Exception, match="out of range"):
            OptimizerConfig(betas=[-0.1, 0.999])

    def test_learning_rate_must_be_positive(self):
        with pytest.raises(Exception):
            OptimizerConfig(learning_rate=0.0)

    def test_weight_decay_zero_ok(self):
        assert OptimizerConfig(weight_decay=0.0).weight_decay == 0.0

    def test_max_grad_norm_none(self):
        cfg = OptimizerConfig(max_grad_norm=None)
        assert cfg.max_grad_norm is None

    def test_optimizer_args_passthrough(self):
        cfg = OptimizerConfig(optimizer_args={"amsgrad": True})
        assert cfg.optimizer_args == {"amsgrad": True}


# ====================================================================
# 4. TestSchedulerConfig
# ====================================================================


class TestSchedulerConfig:
    """Scheduler: type validation, warmup, min_lr, rex params."""

    def test_default_type(self):
        assert SchedulerConfig().type == T2V_SCHEDULER == "cosine_with_min_lr"

    def test_default_warmup(self):
        assert SchedulerConfig().warmup_steps == T2V_WARMUP_STEPS == 0

    def test_default_min_lr_ratio(self):
        assert SchedulerConfig().min_lr_ratio == T2V_MIN_LR_RATIO == 0.01

    def test_default_min_lr_is_none(self):
        assert SchedulerConfig().min_lr is None

    def test_default_rex_params(self):
        assert SchedulerConfig().rex_alpha == 0.1
        assert SchedulerConfig().rex_beta == 0.9

    def test_all_valid_schedulers(self):
        for s in VALID_SCHEDULERS:
            cfg = SchedulerConfig(type=s)
            assert cfg.type == s

    def test_invalid_scheduler_raises(self):
        with pytest.raises(Exception, match="Unknown scheduler"):
            SchedulerConfig(type="exponential")

    def test_warmup_non_negative(self):
        assert SchedulerConfig(warmup_steps=0).warmup_steps == 0
        assert SchedulerConfig(warmup_steps=500).warmup_steps == 500

    def test_warmup_negative_raises(self):
        with pytest.raises(Exception):
            SchedulerConfig(warmup_steps=-1)

    def test_min_lr_absolute(self):
        cfg = SchedulerConfig(min_lr=1e-6)
        assert cfg.min_lr == 1e-6

    def test_min_lr_ratio_range(self):
        assert SchedulerConfig(min_lr_ratio=0.0).min_lr_ratio == 0.0
        assert SchedulerConfig(min_lr_ratio=1.0).min_lr_ratio == 1.0

    def test_min_lr_ratio_above_one(self):
        with pytest.raises(Exception):
            SchedulerConfig(min_lr_ratio=1.01)

    def test_rex_custom_params(self):
        cfg = SchedulerConfig(type="rex", rex_alpha=0.2, rex_beta=0.8)
        assert cfg.rex_alpha == 0.2
        assert cfg.rex_beta == 0.8


# ====================================================================
# 5. TestMoeExpertOverrides
# ====================================================================


class TestMoeExpertOverrides:
    """All nullable per-expert override fields. Fork targets and block targets validated."""

    def test_all_defaults_are_none_or_true(self):
        """Most fields default to None (inherit). enabled defaults True."""
        eo = MoeExpertOverrides()
        assert eo.enabled is True
        assert eo.learning_rate is None
        assert eo.dropout is None
        assert eo.max_epochs is None
        assert eo.fork_targets is None
        assert eo.block_targets is None
        assert eo.resume_from is None
        assert eo.batch_size is None
        assert eo.gradient_accumulation_steps is None
        assert eo.caption_dropout_rate is None
        assert eo.weight_decay is None
        assert eo.min_lr_ratio is None
        assert eo.optimizer_type is None
        assert eo.scheduler_type is None

    def test_learning_rate_override(self):
        eo = MoeExpertOverrides(learning_rate=1e-4)
        assert eo.learning_rate == 1e-4

    def test_learning_rate_must_be_positive(self):
        with pytest.raises(Exception):
            MoeExpertOverrides(learning_rate=0.0)

    def test_dropout_override(self):
        eo = MoeExpertOverrides(dropout=0.05)
        assert eo.dropout == 0.05

    def test_dropout_range(self):
        with pytest.raises(Exception):
            MoeExpertOverrides(dropout=1.5)

    def test_max_epochs_override(self):
        eo = MoeExpertOverrides(max_epochs=30)
        assert eo.max_epochs == 30

    def test_max_epochs_must_be_positive(self):
        with pytest.raises(Exception):
            MoeExpertOverrides(max_epochs=0)

    def test_fork_targets_valid(self):
        eo = MoeExpertOverrides(fork_targets=["ffn", "self_attn"])
        assert eo.fork_targets == ["ffn", "self_attn"]

    def test_fork_targets_projection_level(self):
        eo = MoeExpertOverrides(fork_targets=["cross_attn.to_v", "ffn.up_proj"])
        assert "cross_attn.to_v" in eo.fork_targets

    def test_fork_targets_invalid_raises(self):
        with pytest.raises(Exception, match="Invalid fork target"):
            MoeExpertOverrides(fork_targets=["banana"])

    def test_block_targets_single_range(self):
        eo = MoeExpertOverrides(block_targets="0-11")
        assert eo.block_targets == "0-11"

    def test_block_targets_multiple_ranges(self):
        eo = MoeExpertOverrides(block_targets="0-11,25-34")
        assert eo.block_targets == "0-11,25-34"

    def test_block_targets_single_index(self):
        eo = MoeExpertOverrides(block_targets="5")
        assert eo.block_targets == "5"

    def test_block_targets_invalid_format(self):
        with pytest.raises(Exception, match="block"):
            MoeExpertOverrides(block_targets="abc")

    def test_block_targets_negative_index(self):
        with pytest.raises(Exception, match="Non-integer"):
            MoeExpertOverrides(block_targets="-1")

    def test_block_targets_reversed_range(self):
        with pytest.raises(Exception, match="start"):
            MoeExpertOverrides(block_targets="11-0")

    def test_resume_from(self):
        eo = MoeExpertOverrides(resume_from="C:/checkpoints/lora.safetensors")
        assert eo.resume_from == "C:/checkpoints/lora.safetensors"

    def test_batch_size_override(self):
        eo = MoeExpertOverrides(batch_size=2)
        assert eo.batch_size == 2

    def test_batch_size_must_be_positive(self):
        with pytest.raises(Exception):
            MoeExpertOverrides(batch_size=0)

    def test_gradient_accumulation_override(self):
        eo = MoeExpertOverrides(gradient_accumulation_steps=4)
        assert eo.gradient_accumulation_steps == 4

    def test_gradient_accumulation_must_be_positive(self):
        with pytest.raises(Exception):
            MoeExpertOverrides(gradient_accumulation_steps=0)

    def test_caption_dropout_rate_override(self):
        eo = MoeExpertOverrides(caption_dropout_rate=0.2)
        assert eo.caption_dropout_rate == 0.2

    def test_caption_dropout_range(self):
        with pytest.raises(Exception):
            MoeExpertOverrides(caption_dropout_rate=-0.1)

    def test_weight_decay_override(self):
        eo = MoeExpertOverrides(weight_decay=0.05)
        assert eo.weight_decay == 0.05

    def test_weight_decay_non_negative(self):
        with pytest.raises(Exception):
            MoeExpertOverrides(weight_decay=-0.01)

    def test_min_lr_ratio_override(self):
        eo = MoeExpertOverrides(min_lr_ratio=0.05)
        assert eo.min_lr_ratio == 0.05

    def test_min_lr_ratio_range(self):
        with pytest.raises(Exception):
            MoeExpertOverrides(min_lr_ratio=1.5)

    def test_optimizer_type_valid(self):
        eo = MoeExpertOverrides(optimizer_type="prodigy")
        assert eo.optimizer_type == "prodigy"

    def test_optimizer_type_invalid(self):
        with pytest.raises(Exception, match="Unknown optimizer"):
            MoeExpertOverrides(optimizer_type="rmsprop")

    def test_scheduler_type_valid(self):
        eo = MoeExpertOverrides(scheduler_type="linear")
        assert eo.scheduler_type == "linear"

    def test_scheduler_type_invalid(self):
        with pytest.raises(Exception, match="Unknown scheduler"):
            MoeExpertOverrides(scheduler_type="exponential_decay")

    def test_enabled_false(self):
        eo = MoeExpertOverrides(enabled=False)
        assert eo.enabled is False


# ====================================================================
# 6. TestMoeConfig
# ====================================================================


class TestMoeConfig:
    """MoE: enabled, fork_enabled (no freeze_shared_after_fork, no fork_criterion)."""

    def test_default_enabled(self):
        assert MoeConfig().enabled == T2V_IS_MOE

    def test_default_fork_enabled(self):
        assert MoeConfig().fork_enabled == T2V_FORK_ENABLED

    def test_default_boundary_ratio_none(self):
        assert MoeConfig().boundary_ratio is None

    def test_high_noise_default(self):
        assert isinstance(MoeConfig().high_noise, MoeExpertOverrides)

    def test_low_noise_default(self):
        assert isinstance(MoeConfig().low_noise, MoeExpertOverrides)

    def test_no_freeze_shared_after_fork(self):
        cfg = MoeConfig()
        assert not hasattr(cfg, "freeze_shared_after_fork")

    def test_no_fork_criterion(self):
        cfg = MoeConfig()
        assert not hasattr(cfg, "fork_criterion")

    def test_boundary_ratio_override(self):
        cfg = MoeConfig(boundary_ratio=0.85)
        assert cfg.boundary_ratio == 0.85

    def test_expert_overrides(self):
        cfg = MoeConfig(
            high_noise=MoeExpertOverrides(learning_rate=1e-4, max_epochs=30),
            low_noise=MoeExpertOverrides(learning_rate=8e-5, max_epochs=50),
        )
        assert cfg.high_noise.learning_rate == 1e-4
        assert cfg.high_noise.max_epochs == 30
        assert cfg.low_noise.learning_rate == 8e-5
        assert cfg.low_noise.max_epochs == 50

    def test_fork_disabled(self):
        cfg = MoeConfig(fork_enabled=False)
        assert cfg.fork_enabled is False

    def test_moe_disabled(self):
        cfg = MoeConfig(enabled=False, fork_enabled=False)
        assert cfg.enabled is False


# ====================================================================
# 7. TestTrainingLoopConfig
# ====================================================================


class TestTrainingLoopConfig:
    """TrainingLoop: unified_epochs, unified_targets, batch_size, precision, etc. NO max_epochs."""

    def test_default_unified_epochs(self):
        assert TrainingLoopConfig().unified_epochs == T2V_UNIFIED_EPOCHS == 10

    def test_default_unified_targets_none(self):
        assert TrainingLoopConfig().unified_targets is None

    def test_default_unified_block_targets_none(self):
        assert TrainingLoopConfig().unified_block_targets is None

    def test_default_batch_size(self):
        assert TrainingLoopConfig().batch_size == T2V_BATCH_SIZE == 1

    def test_default_gradient_accumulation(self):
        assert TrainingLoopConfig().gradient_accumulation_steps == T2V_GRADIENT_ACCUMULATION == 1

    def test_default_mixed_precision(self):
        assert TrainingLoopConfig().mixed_precision == T2V_MIXED_PRECISION == "bf16"

    def test_default_base_model_precision(self):
        assert TrainingLoopConfig().base_model_precision == T2V_BASE_MODEL_PRECISION == "bf16"

    def test_default_caption_dropout(self):
        assert TrainingLoopConfig().caption_dropout_rate == T2V_CAPTION_DROPOUT_RATE == 0.1

    def test_default_timestep_sampling(self):
        assert TrainingLoopConfig().timestep_sampling == T2V_TIMESTEP_SAMPLING == "shift"

    def test_default_gradient_checkpointing(self):
        assert TrainingLoopConfig().gradient_checkpointing is True

    def test_default_seed_none(self):
        assert TrainingLoopConfig().seed is None

    def test_default_resume_from_none(self):
        assert TrainingLoopConfig().resume_from is None

    def test_no_max_epochs_field(self):
        """max_epochs was removed — duration is unified_epochs + per-expert max_epochs."""
        cfg = TrainingLoopConfig()
        assert not hasattr(cfg, "max_epochs")

    def test_unified_epochs_zero(self):
        """Zero unified epochs = expert-from-scratch mode."""
        cfg = TrainingLoopConfig(unified_epochs=0)
        assert cfg.unified_epochs == 0

    def test_unified_epochs_negative_raises(self):
        with pytest.raises(Exception):
            TrainingLoopConfig(unified_epochs=-1)

    def test_unified_block_targets_valid(self):
        cfg = TrainingLoopConfig(unified_block_targets="0-11")
        assert cfg.unified_block_targets == "0-11"

    def test_unified_block_targets_multi_range(self):
        cfg = TrainingLoopConfig(unified_block_targets="0-11,25-34")
        assert cfg.unified_block_targets == "0-11,25-34"

    def test_unified_block_targets_invalid(self):
        with pytest.raises(Exception):
            TrainingLoopConfig(unified_block_targets="xyz")

    def test_unified_block_targets_reversed_range(self):
        with pytest.raises(Exception):
            TrainingLoopConfig(unified_block_targets="11-0")

    def test_mixed_precision_valid(self):
        for mp in VALID_MIXED_PRECISION:
            cfg = TrainingLoopConfig(mixed_precision=mp)
            assert cfg.mixed_precision == mp

    def test_mixed_precision_invalid(self):
        with pytest.raises(Exception, match="mixed_precision"):
            TrainingLoopConfig(mixed_precision="fp8")

    def test_base_model_precision_valid(self):
        for bp in VALID_BASE_PRECISION:
            cfg = TrainingLoopConfig(base_model_precision=bp)
            assert cfg.base_model_precision == bp

    def test_base_model_precision_invalid(self):
        with pytest.raises(Exception, match="base_model_precision"):
            TrainingLoopConfig(base_model_precision="int8")

    def test_timestep_sampling_valid(self):
        for ts in VALID_TIMESTEP_SAMPLING:
            cfg = TrainingLoopConfig(timestep_sampling=ts)
            assert cfg.timestep_sampling == ts

    def test_timestep_sampling_invalid(self):
        with pytest.raises(Exception, match="timestep_sampling"):
            TrainingLoopConfig(timestep_sampling="random")

    def test_caption_dropout_range(self):
        assert TrainingLoopConfig(caption_dropout_rate=0.0).caption_dropout_rate == 0.0
        assert TrainingLoopConfig(caption_dropout_rate=1.0).caption_dropout_rate == 1.0

    def test_caption_dropout_out_of_range(self):
        with pytest.raises(Exception):
            TrainingLoopConfig(caption_dropout_rate=1.1)

    def test_batch_size_must_be_positive(self):
        with pytest.raises(Exception):
            TrainingLoopConfig(batch_size=0)

    def test_gradient_accumulation_must_be_positive(self):
        with pytest.raises(Exception):
            TrainingLoopConfig(gradient_accumulation_steps=0)

    def test_seed_explicit(self):
        cfg = TrainingLoopConfig(seed=42)
        assert cfg.seed == 42

    def test_resume_from_path(self):
        cfg = TrainingLoopConfig(resume_from="C:/checkpoints/epoch010.safetensors")
        assert cfg.resume_from == "C:/checkpoints/epoch010.safetensors"

    def test_discrete_flow_shift_default_none(self):
        assert TrainingLoopConfig().discrete_flow_shift is None

    def test_discrete_flow_shift_override(self):
        cfg = TrainingLoopConfig(discrete_flow_shift=5.0)
        assert cfg.discrete_flow_shift == 5.0


# ====================================================================
# 8. TestSaveConfig
# ====================================================================


class TestSaveConfig:
    """Save: output_dir, name, save_every_n_epochs, format validation."""

    def test_default_output_dir(self):
        assert SaveConfig().output_dir == "./output"

    def test_default_name(self):
        assert SaveConfig().name == "flimmer_lora"

    def test_default_save_every(self):
        assert SaveConfig().save_every_n_epochs == T2V_SAVE_EVERY_N_EPOCHS == 5

    def test_default_format(self):
        assert SaveConfig().format == T2V_CHECKPOINT_FORMAT == "safetensors"

    def test_default_save_last(self):
        assert SaveConfig().save_last is True

    def test_default_max_checkpoints(self):
        assert SaveConfig().max_checkpoints is None

    def test_format_safetensors(self):
        cfg = SaveConfig(format="safetensors")
        assert cfg.format == "safetensors"

    def test_format_diffusers(self):
        cfg = SaveConfig(format="diffusers")
        assert cfg.format == "diffusers"

    def test_format_invalid(self):
        with pytest.raises(Exception, match="checkpoint format"):
            SaveConfig(format="pytorch")

    def test_save_every_must_be_positive(self):
        with pytest.raises(Exception):
            SaveConfig(save_every_n_epochs=0)

    def test_max_checkpoints_positive(self):
        cfg = SaveConfig(max_checkpoints=3)
        assert cfg.max_checkpoints == 3

    def test_max_checkpoints_zero_raises(self):
        with pytest.raises(Exception):
            SaveConfig(max_checkpoints=0)

    def test_custom_name(self):
        cfg = SaveConfig(name="holly_lora")
        assert cfg.name == "holly_lora"


# ====================================================================
# 9. TestLoggingConfig
# ====================================================================


class TestLoggingConfig:
    """Logging: backends validation, wandb_project, wandb_entity, wandb_run_name."""

    def test_default_backends(self):
        assert LoggingConfig().backends == ["console"]

    def test_default_log_every(self):
        assert LoggingConfig().log_every_n_steps == 10

    def test_default_wandb_project_none(self):
        assert LoggingConfig().wandb_project is None

    def test_default_wandb_entity_none(self):
        assert LoggingConfig().wandb_entity is None

    def test_default_wandb_run_name_none(self):
        assert LoggingConfig().wandb_run_name is None

    def test_valid_backends(self):
        for b in VALID_LOG_BACKENDS:
            cfg = LoggingConfig(backends=[b])
            assert b in cfg.backends

    def test_multiple_backends(self):
        cfg = LoggingConfig(backends=["console", "wandb"])
        assert cfg.backends == ["console", "wandb"]

    def test_invalid_backend_raises(self):
        with pytest.raises(Exception, match="Invalid logging backend"):
            LoggingConfig(backends=["console", "mlflow"])

    def test_backends_must_be_list(self):
        with pytest.raises(Exception, match="list"):
            LoggingConfig(backends="console")  # type: ignore[arg-type]

    def test_wandb_project_set(self):
        cfg = LoggingConfig(wandb_project="flimmer-training")
        assert cfg.wandb_project == "flimmer-training"

    def test_wandb_entity_set(self):
        cfg = LoggingConfig(wandb_entity="alvdansen")
        assert cfg.wandb_entity == "alvdansen"

    def test_wandb_run_name_set(self):
        cfg = LoggingConfig(wandb_run_name="holly_i2v_r16")
        assert cfg.wandb_run_name == "holly_i2v_r16"


# ====================================================================
# 10. TestSamplingConfig
# ====================================================================


class TestSamplingConfig:
    """Sampling: ai-toolkit format — enabled, every_n_epochs, prompts, neg,
    seed, walk_seed, sample_steps, guidance_scale, sample_dir."""

    def test_default_enabled(self):
        assert SamplingConfig().enabled == T2V_SAMPLING_ENABLED
        assert SamplingConfig().enabled is False

    def test_default_every_n_epochs(self):
        assert SamplingConfig().every_n_epochs == 5

    def test_default_prompts_populated(self):
        """Default prompts cover 4 key scenarios (updated in Plan 02-02)."""
        prompts = SamplingConfig().prompts
        assert len(prompts) == 4
        # Verify the 4 scenarios: static person, motion, scenic, detail
        assert any("person" in p.lower() for p in prompts)
        assert any("cat" in p.lower() for p in prompts)
        assert any("ocean" in p.lower() or "waves" in p.lower() for p in prompts)
        assert any("close-up" in p.lower() or "hands" in p.lower() for p in prompts)

    def test_default_neg_empty(self):
        assert SamplingConfig().neg == ""

    def test_default_seed(self):
        assert SamplingConfig().seed == T2V_SAMPLING_SEED == 42

    def test_default_walk_seed(self):
        assert SamplingConfig().walk_seed is True

    def test_default_sample_steps(self):
        assert SamplingConfig().sample_steps == T2V_SAMPLING_STEPS == 30

    def test_default_guidance_scale(self):
        assert SamplingConfig().guidance_scale == T2V_SAMPLING_GUIDANCE == 4.0

    def test_default_sample_dir_none(self):
        assert SamplingConfig().sample_dir is None

    def test_enabled_true(self):
        cfg = SamplingConfig(enabled=True)
        assert cfg.enabled is True

    def test_prompts_list(self):
        cfg = SamplingConfig(prompts=["a cat", "a dog"])
        assert cfg.prompts == ["a cat", "a dog"]

    def test_neg_string(self):
        cfg = SamplingConfig(neg="blurry, low quality")
        assert cfg.neg == "blurry, low quality"

    def test_walk_seed_false(self):
        cfg = SamplingConfig(walk_seed=False)
        assert cfg.walk_seed is False

    def test_sample_steps_must_be_positive(self):
        with pytest.raises(Exception):
            SamplingConfig(sample_steps=0)

    def test_guidance_scale_must_be_positive(self):
        with pytest.raises(Exception):
            SamplingConfig(guidance_scale=0.0)

    def test_every_n_epochs_must_be_positive(self):
        with pytest.raises(Exception):
            SamplingConfig(every_n_epochs=0)

    def test_sample_dir_override(self):
        cfg = SamplingConfig(sample_dir="./my_samples")
        assert cfg.sample_dir == "./my_samples"


# ====================================================================
# 11. TestFlimmerTrainingConfig (root validators)
# ====================================================================


class TestFlimmerTrainingConfig:
    """Root-level cross-field validators."""

    def test_check_moe_consistency_ok(self):
        """MoE enabled on an MoE model is fine."""
        cfg = make_root_config(
            model=ModelConfig(path="C:/m", is_moe=True),
            moe=MoeConfig(enabled=True),
        )
        assert cfg.moe.enabled is True

    def test_check_moe_consistency_error(self):
        """MoE enabled but model.is_moe=False raises."""
        with pytest.raises(Exception, match="MoE is enabled but model.is_moe is false"):
            make_root_config(
                model=ModelConfig(path="C:/m", is_moe=False),
                moe=MoeConfig(enabled=True),
            )

    def test_check_moe_consistency_none_is_ok(self):
        """is_moe=None does not trigger the check (variant has not been resolved yet)."""
        cfg = make_root_config(
            model=ModelConfig(path="C:/m", is_moe=None),
            moe=MoeConfig(enabled=True),
        )
        assert cfg.moe.enabled is True

    def test_check_prodigy_lr_ok(self):
        """Prodigy with lr=1.0 is valid."""
        cfg = make_root_config(
            optimizer=OptimizerConfig(type="prodigy", learning_rate=1.0),
        )
        assert cfg.optimizer.type == "prodigy"

    def test_check_prodigy_lr_error(self):
        """Prodigy with lr != 1.0 raises."""
        with pytest.raises(Exception, match="Prodigy optimizer requires learning_rate=1.0"):
            make_root_config(
                optimizer=OptimizerConfig(type="prodigy", learning_rate=5e-5),
            )

    def test_check_wandb_project_ok(self):
        """W&B with a project name is valid."""
        cfg = make_root_config(
            logging=LoggingConfig(backends=["console", "wandb"], wandb_project="test"),
        )
        assert "wandb" in cfg.logging.backends

    def test_check_wandb_project_error(self):
        """W&B without project name raises."""
        with pytest.raises(Exception, match="wandb_project"):
            make_root_config(
                logging=LoggingConfig(backends=["wandb"]),
            )

    def test_check_wandb_project_no_wandb_ok(self):
        """No wandb backend => no project requirement."""
        cfg = make_root_config(
            logging=LoggingConfig(backends=["console"]),
        )
        assert cfg.logging.wandb_project is None

    def test_check_mua_alpha_auto_fix(self):
        """muA init forces alpha = rank."""
        cfg = make_root_config(
            lora=LoraConfig(rank=32, alpha=16, use_mua_init=True),
        )
        assert cfg.lora.alpha == 32  # auto-fixed to match rank

    def test_check_mua_already_correct(self):
        """muA with alpha already matching rank is fine."""
        cfg = make_root_config(
            lora=LoraConfig(rank=16, alpha=16, use_mua_init=True),
        )
        assert cfg.lora.alpha == 16

    def test_check_fork_without_moe_error(self):
        """fork_enabled=True but moe.enabled=False raises."""
        with pytest.raises(Exception, match="fork_enabled is true but moe.enabled is false"):
            make_root_config(
                moe=MoeConfig(enabled=False, fork_enabled=True),
            )

    def test_check_fork_without_moe_ok(self):
        """Both disabled is fine."""
        cfg = make_root_config(
            model=ModelConfig(path="C:/m", is_moe=False),
            moe=MoeConfig(enabled=False, fork_enabled=False),
        )
        assert cfg.moe.fork_enabled is False

    def test_warn_aggressive_low_noise(self):
        """Low-noise LR > 2e-4 triggers a warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            make_root_config(
                moe=MoeConfig(
                    enabled=True,
                    low_noise=MoeExpertOverrides(learning_rate=5e-4),
                ),
            )
            assert len(w) == 1
            assert "Low-noise expert" in str(w[0].message)
            assert "2e-4" in str(w[0].message)

    def test_no_warning_normal_low_noise(self):
        """Low-noise LR <= 2e-4 does not warn."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            make_root_config(
                moe=MoeConfig(
                    enabled=True,
                    low_noise=MoeExpertOverrides(learning_rate=1e-4),
                ),
            )
            assert len(w) == 0

    def test_no_warning_when_moe_disabled(self):
        """No warning if MoE is disabled even with high LR."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            make_root_config(
                moe=MoeConfig(
                    enabled=False,
                    fork_enabled=False,
                    low_noise=MoeExpertOverrides(learning_rate=5e-4),
                ),
            )
            assert len(w) == 0

    def test_no_warning_when_low_noise_lr_none(self):
        """No warning if low-noise LR is None (inherits)."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            make_root_config(
                moe=MoeConfig(enabled=True),
            )
            assert len(w) == 0

    def test_all_subconfigs_present(self):
        """Root config has all expected sub-configs."""
        cfg = make_root_config()
        assert isinstance(cfg.model, ModelConfig)
        assert isinstance(cfg.lora, LoraConfig)
        assert isinstance(cfg.optimizer, OptimizerConfig)
        assert isinstance(cfg.scheduler, SchedulerConfig)
        assert isinstance(cfg.moe, MoeConfig)
        assert isinstance(cfg.training, TrainingLoopConfig)
        assert isinstance(cfg.save, SaveConfig)
        assert isinstance(cfg.logging, LoggingConfig)
        assert isinstance(cfg.sampling, SamplingConfig)
        assert isinstance(cfg.data_config, str)

    def test_data_config_required(self):
        """Root config without data_config raises."""
        with pytest.raises(Exception):
            FlimmerTrainingConfig(
                model=ModelConfig(path="C:/m"),
            )

    def test_model_required(self):
        """Root config without model raises."""
        with pytest.raises(Exception):
            FlimmerTrainingConfig(
                data_config="C:/fake.yaml",
            )


# ====================================================================
# 12. TestVariantDefaults
# ====================================================================


class TestVariantDefaults:
    """VARIANT_DEFAULTS keys and T2V vs I2V differences."""

    def test_known_variants(self):
        assert "2.2_t2v" in VARIANT_DEFAULTS
        assert "2.2_i2v" in VARIANT_DEFAULTS

    def test_only_two_variants(self):
        assert len(VARIANT_DEFAULTS) == 2

    def test_t2v_model_fields(self):
        t2v = VARIANT_DEFAULTS["2.2_t2v"]["model"]
        assert t2v["family"] == "wan"
        assert t2v["variant"] == "2.2_t2v"
        assert t2v["is_moe"] is True
        assert t2v["in_channels"] == 16
        assert t2v["num_layers"] == 40
        assert t2v["boundary_ratio"] == 0.875

    def test_i2v_model_fields(self):
        i2v = VARIANT_DEFAULTS["2.2_i2v"]["model"]
        assert i2v["variant"] == "2.2_i2v"
        assert i2v["in_channels"] == 36
        assert i2v["boundary_ratio"] == 0.900
        assert i2v["is_moe"] is True

    def test_i2v_training_overrides(self):
        i2v = VARIANT_DEFAULTS["2.2_i2v"]
        assert i2v["training"]["unified_epochs"] == 15
        assert i2v["training"]["caption_dropout_rate"] == 0.15

    def test_i2v_moe_low_noise_override(self):
        i2v = VARIANT_DEFAULTS["2.2_i2v"]
        assert i2v["moe"]["low_noise"]["max_epochs"] == 50

    def test_t2v_has_no_training_overrides(self):
        """T2V uses Pydantic defaults — no training block needed."""
        t2v = VARIANT_DEFAULTS["2.2_t2v"]
        assert "training" not in t2v

    def test_variants_are_dicts(self):
        for name, data in VARIANT_DEFAULTS.items():
            assert isinstance(data, dict), f"{name} is not a dict"
            assert "model" in data, f"{name} missing model block"


# ====================================================================
# 13. TestDeepMerge
# ====================================================================


class TestDeepMerge:
    """_deep_merge: scalar override, dict recursion, list replacement, no mutation."""

    def test_scalar_override(self):
        result = _deep_merge({"a": 1}, {"a": 2})
        assert result["a"] == 2

    def test_nested_dict_merge(self):
        base = {"model": {"family": "wan", "variant": "2.2_t2v"}}
        override = {"model": {"variant": "2.2_i2v"}}
        result = _deep_merge(base, override)
        assert result["model"]["family"] == "wan"
        assert result["model"]["variant"] == "2.2_i2v"

    def test_list_replacement(self):
        base = {"prompts": ["a", "b"]}
        override = {"prompts": ["c"]}
        result = _deep_merge(base, override)
        assert result["prompts"] == ["c"]

    def test_no_mutation_of_base(self):
        base = {"a": {"b": 1}}
        override = {"a": {"b": 2}}
        _deep_merge(base, override)
        assert base["a"]["b"] == 1

    def test_no_mutation_of_override(self):
        base = {"a": 1}
        override = {"a": {"nested": 2}}
        _deep_merge(base, override)
        assert override["a"] == {"nested": 2}

    def test_new_keys_added(self):
        result = _deep_merge({"a": 1}, {"b": 2})
        assert result == {"a": 1, "b": 2}

    def test_base_keys_preserved(self):
        result = _deep_merge({"a": 1, "b": 2}, {"b": 3})
        assert result == {"a": 1, "b": 3}

    def test_deeply_nested(self):
        base = {"l1": {"l2": {"l3": "old"}}}
        override = {"l1": {"l2": {"l3": "new"}}}
        result = _deep_merge(base, override)
        assert result["l1"]["l2"]["l3"] == "new"

    def test_empty_base(self):
        result = _deep_merge({}, {"a": 1})
        assert result == {"a": 1}

    def test_empty_override(self):
        result = _deep_merge({"a": 1}, {})
        assert result == {"a": 1}

    def test_both_empty(self):
        result = _deep_merge({}, {})
        assert result == {}


# ====================================================================
# 14. TestApplyVariantDefaults
# ====================================================================


class TestApplyVariantDefaults:
    """_apply_variant_defaults: variant lookup, unknown passthrough, user override wins."""

    def test_known_variant_merges(self):
        data = {"model": {"variant": "2.2_t2v", "path": "C:/m"}}
        result = _apply_variant_defaults(data)
        assert result["model"]["family"] == "wan"
        assert result["model"]["is_moe"] is True
        assert result["model"]["in_channels"] == 16

    def test_i2v_variant_merges(self):
        data = {"model": {"variant": "2.2_i2v", "path": "C:/m"}}
        result = _apply_variant_defaults(data)
        assert result["model"]["in_channels"] == 36
        assert result["training"]["unified_epochs"] == 15

    def test_unknown_variant_passthrough(self):
        """Unknown variant name is left unchanged (Pydantic will validate later)."""
        data = {"model": {"variant": "3.0_future", "path": "C:/m"}}
        result = _apply_variant_defaults(data)
        assert result["model"]["variant"] == "3.0_future"

    def test_no_variant_passthrough(self):
        """No variant field means no merging."""
        data = {"model": {"path": "C:/m"}}
        result = _apply_variant_defaults(data)
        assert "family" not in result.get("model", {})

    def test_user_override_wins(self):
        """User-specified values take precedence over variant defaults."""
        data = {
            "model": {"variant": "2.2_t2v", "path": "C:/m", "in_channels": 99},
        }
        result = _apply_variant_defaults(data)
        assert result["model"]["in_channels"] == 99

    def test_variant_preserved_in_output(self):
        data = {"model": {"variant": "2.2_t2v", "path": "C:/m"}}
        result = _apply_variant_defaults(data)
        assert result["model"]["variant"] == "2.2_t2v"

    def test_no_model_block_passthrough(self):
        data = {"data_config": "test.yaml"}
        result = _apply_variant_defaults(data)
        assert result == data

    def test_non_dict_model_passthrough(self):
        data = {"model": "not_a_dict"}
        result = _apply_variant_defaults(data)
        assert result["model"] == "not_a_dict"


# ====================================================================
# 15. TestAutoEnableMoe
# ====================================================================


class TestAutoEnableMoe:
    """_auto_enable_moe: auto-set when is_moe, respect explicit user setting."""

    def test_auto_enable_when_is_moe(self):
        data = {"model": {"is_moe": True}}
        result = _auto_enable_moe(data)
        assert result["moe"]["enabled"] is True

    def test_no_auto_enable_when_not_moe(self):
        data = {"model": {"is_moe": False}}
        result = _auto_enable_moe(data)
        assert "moe" not in result or "enabled" not in result.get("moe", {})

    def test_no_auto_enable_when_is_moe_none(self):
        data = {"model": {"is_moe": None}}
        result = _auto_enable_moe(data)
        assert "moe" not in result or "enabled" not in result.get("moe", {})

    def test_respect_explicit_false(self):
        """User explicitly sets moe.enabled=False — auto-enable does not override."""
        data = {"model": {"is_moe": True}, "moe": {"enabled": False}}
        result = _auto_enable_moe(data)
        assert result["moe"]["enabled"] is False

    def test_respect_explicit_true(self):
        data = {"model": {"is_moe": True}, "moe": {"enabled": True}}
        result = _auto_enable_moe(data)
        assert result["moe"]["enabled"] is True

    def test_no_model_block(self):
        data = {"data_config": "test.yaml"}
        result = _auto_enable_moe(data)
        assert result == data

    def test_creates_moe_block_if_missing(self):
        data = {"model": {"is_moe": True}}
        result = _auto_enable_moe(data)
        assert "moe" in result
        assert result["moe"]["enabled"] is True


# ====================================================================
# 16. TestHuggingFaceIdDetection
# ====================================================================


class TestHuggingFaceIdDetection:
    """_is_huggingface_id: Windows paths, drive letters, relative paths, HF IDs."""

    def test_hf_id_org_model(self):
        assert _is_huggingface_id("Wan-AI/Wan2.2-T2V-14B-Diffusers") is True

    def test_hf_id_user_model(self):
        assert _is_huggingface_id("myuser/my-model") is True

    def test_windows_backslash_path(self):
        assert _is_huggingface_id("C:\\models\\Wan") is False

    def test_windows_drive_letter(self):
        assert _is_huggingface_id("C:/models/Wan") is False

    def test_lowercase_drive_letter(self):
        assert _is_huggingface_id("d:/stuff") is False

    def test_relative_dot_path(self):
        assert _is_huggingface_id("./models/wan") is False

    def test_absolute_unix_path(self):
        assert _is_huggingface_id("/home/user/models") is False

    def test_three_part_path(self):
        """Three-part path (a/b/c) is not a HF ID."""
        assert _is_huggingface_id("a/b/c") is False

    def test_single_name(self):
        """Single name without slash is not a HF ID."""
        assert _is_huggingface_id("some-model") is False

    def test_empty_string(self):
        assert _is_huggingface_id("") is False

    def test_slash_only(self):
        assert _is_huggingface_id("/") is False

    def test_leading_slash_two_parts(self):
        """Starts with / => local path."""
        assert _is_huggingface_id("/org/model") is False


# ====================================================================
# 17. TestPathResolution
# ====================================================================


class TestPathResolution:
    """_resolve_paths: relative paths, absolute paths, HF passthrough,
    expert resume_from, sample_dir."""

    def test_relative_data_config(self, tmp_path: Path):
        data = {"data_config": "flimmer_data.yaml"}
        result = _resolve_paths(data, tmp_path)
        assert result["data_config"] == str((tmp_path / "flimmer_data.yaml").resolve())

    def test_absolute_data_config(self, tmp_path: Path):
        abs_path = str(tmp_path / "abs_config.yaml")
        data = {"data_config": abs_path}
        result = _resolve_paths(data, tmp_path)
        assert result["data_config"] == str(Path(abs_path).resolve())

    def test_relative_model_path(self, tmp_path: Path):
        data = {"model": {"path": "./models/Wan"}}
        result = _resolve_paths(data, tmp_path)
        assert result["model"]["path"] == str((tmp_path / "models/Wan").resolve())

    def test_hf_model_id_passthrough(self, tmp_path: Path):
        data = {"model": {"path": "Wan-AI/Wan2.2-T2V-14B-Diffusers"}}
        result = _resolve_paths(data, tmp_path)
        assert result["model"]["path"] == "Wan-AI/Wan2.2-T2V-14B-Diffusers"

    def test_relative_output_dir(self, tmp_path: Path):
        data = {"save": {"output_dir": "./output/annika"}}
        result = _resolve_paths(data, tmp_path)
        assert result["save"]["output_dir"] == str((tmp_path / "output/annika").resolve())

    def test_relative_resume_from(self, tmp_path: Path):
        data = {"training": {"resume_from": "checkpoints/epoch10.safetensors"}}
        result = _resolve_paths(data, tmp_path)
        expected = str((tmp_path / "checkpoints/epoch10.safetensors").resolve())
        assert result["training"]["resume_from"] == expected

    def test_expert_resume_from_resolved(self, tmp_path: Path):
        data = {
            "moe": {
                "high_noise": {"resume_from": "hn_lora.safetensors"},
                "low_noise": {"resume_from": "ln_lora.safetensors"},
            },
        }
        result = _resolve_paths(data, tmp_path)
        assert result["moe"]["high_noise"]["resume_from"] == str(
            (tmp_path / "hn_lora.safetensors").resolve()
        )
        assert result["moe"]["low_noise"]["resume_from"] == str(
            (tmp_path / "ln_lora.safetensors").resolve()
        )

    def test_sample_dir_resolved(self, tmp_path: Path):
        data = {"sampling": {"sample_dir": "my_samples"}}
        result = _resolve_paths(data, tmp_path)
        assert result["sampling"]["sample_dir"] == str(
            (tmp_path / "my_samples").resolve()
        )

    def test_none_resume_from_untouched(self, tmp_path: Path):
        data = {"training": {"resume_from": None}}
        result = _resolve_paths(data, tmp_path)
        assert result["training"]["resume_from"] is None

    def test_no_model_block_ok(self, tmp_path: Path):
        data = {"data_config": "test.yaml"}
        result = _resolve_paths(data, tmp_path)
        assert "data_config" in result

    def test_empty_moe_block_ok(self, tmp_path: Path):
        data = {"moe": {}}
        result = _resolve_paths(data, tmp_path)
        assert result["moe"] == {}

    def test_model_path_fallback_to_parent(self, tmp_path: Path):
        """Model weight in parent dir is found when config is in a subfolder."""
        # Create models/ at the root, config in a subfolder
        (tmp_path / "models").mkdir()
        (tmp_path / "models" / "vae.safetensors").touch()
        subfolder = tmp_path / "my_project"
        subfolder.mkdir()

        resolved = _resolve_model_path("./models/vae.safetensors", subfolder)
        assert resolved == (tmp_path / "models" / "vae.safetensors").resolve()

    def test_model_path_fallback_stops_at_git_root(self, tmp_path: Path):
        """Fallback stops at .git boundary and doesn't search above it."""
        # Create models/ above the git root — should NOT be found
        (tmp_path / "models").mkdir()
        (tmp_path / "models" / "vae.safetensors").touch()

        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / ".git").mkdir()
        subfolder = repo / "my_project"
        subfolder.mkdir()

        resolved = _resolve_model_path("./models/vae.safetensors", subfolder)
        # Should return the original (non-existent) resolution, not the one above .git
        assert resolved == (subfolder / "models" / "vae.safetensors").resolve()

    def test_model_path_no_fallback_when_found_locally(self, tmp_path: Path):
        """When the model exists relative to config dir, no fallback needed."""
        (tmp_path / "models").mkdir()
        (tmp_path / "models" / "vae.safetensors").touch()

        resolved = _resolve_model_path("./models/vae.safetensors", tmp_path)
        assert resolved == (tmp_path / "models" / "vae.safetensors").resolve()

    def test_model_path_not_found_returns_original(self, tmp_path: Path):
        """When model doesn't exist anywhere, returns original resolution."""
        subfolder = tmp_path / "my_project"
        subfolder.mkdir()

        resolved = _resolve_model_path("./models/vae.safetensors", subfolder)
        assert resolved == (subfolder / "models" / "vae.safetensors").resolve()

    def test_model_path_absolute_no_fallback(self, tmp_path: Path):
        """Absolute paths are returned as-is, no fallback."""
        abs_path = str(tmp_path / "models" / "vae.safetensors")
        resolved = _resolve_model_path(abs_path, tmp_path / "subfolder")
        assert resolved == Path(abs_path).resolve()

    def test_resolve_paths_uses_model_fallback(self, tmp_path: Path):
        """_resolve_paths uses fallback for model weights but not data_config."""
        # models/ at root, config in subfolder
        (tmp_path / "models").mkdir()
        (tmp_path / "models" / "vae.safetensors").touch()
        subfolder = tmp_path / "my_project"
        subfolder.mkdir()

        data = {
            "model": {"vae": "./models/vae.safetensors"},
            "data_config": "./flimmer_data.yaml",
        }
        result = _resolve_paths(data, subfolder)

        # Model path should resolve to parent's models/
        assert result["model"]["vae"] == str(
            (tmp_path / "models" / "vae.safetensors").resolve()
        )
        # data_config should stay relative to config dir (no fallback)
        assert result["data_config"] == str(
            (subfolder / "flimmer_data.yaml").resolve()
        )


# ====================================================================
# 18. TestFormatValidationError
# ====================================================================


class TestFormatValidationError:
    """_format_validation_error: human-readable error formatting."""

    def test_single_error(self):
        """Generate a real ValidationError and format it."""
        from pydantic import ValidationError
        try:
            FlimmerTrainingConfig.model_validate(
                {"data_config": "x.yaml", "model": {"path": "C:/m", "num_train_timesteps": -1}}
            )
            pytest.fail("Should have raised")
        except ValidationError as e:
            msg = _format_validation_error(e)
            assert "Training config validation failed" in msg
            assert "See config_templates/" in msg

    def test_strips_value_error_prefix(self):
        """Value error prefix is cleaned from custom validator messages."""
        from pydantic import ValidationError
        try:
            # Trigger the check_fork_without_moe validator
            FlimmerTrainingConfig.model_validate({
                "data_config": "x.yaml",
                "model": {"path": "C:/m"},
                "moe": {"enabled": False, "fork_enabled": True},
            })
            pytest.fail("Should have raised")
        except ValidationError as e:
            msg = _format_validation_error(e)
            # Should contain the message without "Value error, " prefix
            assert "fork_enabled is true" in msg

    def test_location_formatting(self):
        """Error locations are formatted with separators."""
        from pydantic import ValidationError
        try:
            FlimmerTrainingConfig.model_validate({
                "data_config": "x.yaml",
                "model": {"path": "C:/m"},
                "optimizer": {"type": "invalid_opt"},
            })
            pytest.fail("Should have raised")
        except ValidationError as e:
            msg = _format_validation_error(e)
            assert "optimizer" in msg.lower()
