"""Tests for flimmer.training.phase — phase resolution from config."""

import pytest

from flimmer.training.errors import PhaseConfigError
from flimmer.training.phase import (
    DEFAULT_EXPERT_ORDER,
    PhaseType,
    TrainingPhase,
    resolve_phases,
)


# ---------------------------------------------------------------------------
# Helpers — mock config objects that look like FlimmerTrainingConfig
# ---------------------------------------------------------------------------

class MockOptimizer:
    type = "adamw8bit"
    learning_rate = 5e-5
    weight_decay = 0.01

class MockScheduler:
    type = "cosine_with_min_lr"
    warmup_steps = 0
    min_lr_ratio = 0.01

class MockLora:
    dropout = 0.0

class MockExpertOverrides:
    def __init__(self, **kwargs):
        self.enabled = kwargs.get("enabled", True)
        self.learning_rate = kwargs.get("learning_rate", None)
        self.dropout = kwargs.get("dropout", None)
        self.max_epochs = kwargs.get("max_epochs", None)
        self.fork_targets = kwargs.get("fork_targets", None)
        self.block_targets = kwargs.get("block_targets", None)
        self.resume_from = kwargs.get("resume_from", None)
        self.batch_size = kwargs.get("batch_size", None)
        self.gradient_accumulation_steps = kwargs.get("gradient_accumulation_steps", None)
        self.caption_dropout_rate = kwargs.get("caption_dropout_rate", None)
        self.weight_decay = kwargs.get("weight_decay", None)
        self.min_lr_ratio = kwargs.get("min_lr_ratio", None)
        self.optimizer_type = kwargs.get("optimizer_type", None)
        self.scheduler_type = kwargs.get("scheduler_type", None)

class MockMoe:
    def __init__(self, **kwargs):
        self.enabled = kwargs.get("enabled", True)
        self.fork_enabled = kwargs.get("fork_enabled", True)
        self.expert_order = kwargs.get("expert_order", None) or DEFAULT_EXPERT_ORDER
        self.high_noise = kwargs.get("high_noise", MockExpertOverrides(max_epochs=30))
        self.low_noise = kwargs.get("low_noise", MockExpertOverrides(max_epochs=50))
        self.boundary_ratio = kwargs.get("boundary_ratio", None)

class MockTraining:
    def __init__(self, **kwargs):
        self.unified_epochs = kwargs.get("unified_epochs", 10)
        self.batch_size = kwargs.get("batch_size", 1)
        self.gradient_accumulation_steps = kwargs.get("gradient_accumulation_steps", 1)
        self.caption_dropout_rate = kwargs.get("caption_dropout_rate", 0.1)
        self.unified_targets = kwargs.get("unified_targets", None)
        self.unified_block_targets = kwargs.get("unified_block_targets", None)
        self.resume_from = kwargs.get("resume_from", None)

class MockModel:
    def __init__(self, **kwargs):
        self.boundary_ratio = kwargs.get("boundary_ratio", 0.875)

class MockConfig:
    def __init__(self, **kwargs):
        self.training = kwargs.get("training", MockTraining())
        self.optimizer = kwargs.get("optimizer", MockOptimizer())
        self.scheduler = kwargs.get("scheduler", MockScheduler())
        self.lora = kwargs.get("lora", MockLora())
        self.moe = kwargs.get("moe", MockMoe())
        self.model = kwargs.get("model", MockModel())


# ---------------------------------------------------------------------------
# PhaseType enum
# ---------------------------------------------------------------------------

class TestPhaseType:
    def test_values(self):
        assert PhaseType.UNIFIED.value == "full_noise"
        assert PhaseType.HIGH_NOISE.value == "high_noise"
        assert PhaseType.LOW_NOISE.value == "low_noise"

    def test_is_string_enum(self):
        assert isinstance(PhaseType.UNIFIED, str)
        assert PhaseType.UNIFIED == "full_noise"


# ---------------------------------------------------------------------------
# TrainingPhase dataclass
# ---------------------------------------------------------------------------

class TestTrainingPhase:
    def test_frozen(self):
        phase = TrainingPhase(
            phase_type=PhaseType.UNIFIED, max_epochs=10,
            learning_rate=5e-5, weight_decay=0.01,
            optimizer_type="adamw8bit", scheduler_type="cosine_with_min_lr",
            min_lr_ratio=0.01, warmup_steps=0, batch_size=1,
            gradient_accumulation_steps=1, caption_dropout_rate=0.1,
            first_frame_dropout_rate=0.0,
            lora_dropout=0.0, fork_targets=None, block_targets=None,
            resume_from=None, boundary_ratio=None, active_expert=None,
        )
        with pytest.raises(AttributeError):
            phase.max_epochs = 20  # type: ignore[misc]

    def test_all_fields_present(self):
        phase = TrainingPhase(
            phase_type=PhaseType.HIGH_NOISE, max_epochs=30,
            learning_rate=1e-4, weight_decay=0.02,
            optimizer_type="adamw", scheduler_type="constant",
            min_lr_ratio=0.0, warmup_steps=100, batch_size=2,
            gradient_accumulation_steps=4, caption_dropout_rate=0.15,
            first_frame_dropout_rate=0.05,
            lora_dropout=0.05, fork_targets=["ffn"], block_targets="0-11",
            resume_from="/path/to/lora.safetensors",
            boundary_ratio=0.875, active_expert="high_noise",
        )
        assert phase.phase_type == PhaseType.HIGH_NOISE
        assert phase.learning_rate == 1e-4
        assert phase.fork_targets == ["ffn"]
        assert phase.active_expert == "high_noise"


# ---------------------------------------------------------------------------
# Fork-and-specialize mode (PRIMARY)
# ---------------------------------------------------------------------------

class TestForkAndSpecialize:
    """fork_enabled=True, unified_epochs > 0."""

    def test_produces_three_phases(self):
        config = MockConfig()
        phases = resolve_phases(config)
        assert len(phases) == 3

    def test_phase_order(self):
        config = MockConfig()
        phases = resolve_phases(config)
        assert phases[0].phase_type == PhaseType.UNIFIED
        # Default order: low_noise first, then high_noise
        assert phases[1].phase_type == PhaseType.LOW_NOISE
        assert phases[2].phase_type == PhaseType.HIGH_NOISE

    def test_unified_has_no_expert_masking(self):
        config = MockConfig()
        phases = resolve_phases(config)
        unified = phases[0]
        assert unified.boundary_ratio is None
        assert unified.active_expert is None

    def test_expert_phases_have_masking(self):
        config = MockConfig()
        phases = resolve_phases(config)
        for phase in phases[1:]:
            assert phase.boundary_ratio == 0.875
            assert phase.active_expert is not None

    def test_unified_epochs(self):
        config = MockConfig()
        phases = resolve_phases(config)
        assert phases[0].max_epochs == 10

    def test_expert_epochs(self):
        config = MockConfig()
        phases = resolve_phases(config)
        # Low noise first (default order)
        assert phases[1].max_epochs == 50  # low_noise
        assert phases[2].max_epochs == 30  # high_noise


# ---------------------------------------------------------------------------
# Unified-only mode
# ---------------------------------------------------------------------------

class TestUnifiedOnly:
    """fork_enabled=False."""

    def test_produces_one_phase(self):
        config = MockConfig(moe=MockMoe(fork_enabled=False))
        phases = resolve_phases(config)
        assert len(phases) == 1
        assert phases[0].phase_type == PhaseType.UNIFIED

    def test_no_expert_masking(self):
        config = MockConfig(moe=MockMoe(fork_enabled=False))
        phases = resolve_phases(config)
        assert phases[0].boundary_ratio is None
        assert phases[0].active_expert is None


# ---------------------------------------------------------------------------
# Expert-from-scratch mode
# ---------------------------------------------------------------------------

class TestExpertFromScratch:
    """fork_enabled=True, unified_epochs=0."""

    def test_produces_two_phases(self):
        config = MockConfig(training=MockTraining(unified_epochs=0))
        phases = resolve_phases(config)
        assert len(phases) == 2

    def test_no_unified_phase(self):
        config = MockConfig(training=MockTraining(unified_epochs=0))
        phases = resolve_phases(config)
        for phase in phases:
            assert phase.phase_type != PhaseType.UNIFIED

    def test_expert_order_default(self):
        config = MockConfig(training=MockTraining(unified_epochs=0))
        phases = resolve_phases(config)
        assert phases[0].phase_type == PhaseType.LOW_NOISE
        assert phases[1].phase_type == PhaseType.HIGH_NOISE


# ---------------------------------------------------------------------------
# Non-MoE model (e.g., Wan 2.1)
# ---------------------------------------------------------------------------

class TestNonMoE:
    """moe.enabled=False (single transformer, no experts)."""

    def test_single_unified_phase(self):
        config = MockConfig(moe=MockMoe(enabled=False, fork_enabled=False))
        phases = resolve_phases(config)
        assert len(phases) == 1
        assert phases[0].phase_type == PhaseType.UNIFIED

    def test_no_masking(self):
        config = MockConfig(moe=MockMoe(enabled=False, fork_enabled=False))
        phases = resolve_phases(config)
        assert phases[0].boundary_ratio is None

    def test_zero_epochs_errors(self):
        config = MockConfig(
            moe=MockMoe(enabled=False, fork_enabled=False),
            training=MockTraining(unified_epochs=0),
        )
        with pytest.raises(PhaseConfigError, match="unified_epochs > 0"):
            resolve_phases(config)


# ---------------------------------------------------------------------------
# Expert order
# ---------------------------------------------------------------------------

class TestExpertOrder:
    """Expert training order is configurable."""

    def test_default_order_low_first(self):
        config = MockConfig()
        phases = resolve_phases(config)
        assert phases[1].phase_type == PhaseType.LOW_NOISE
        assert phases[2].phase_type == PhaseType.HIGH_NOISE

    def test_reversed_order(self):
        config = MockConfig(moe=MockMoe(
            expert_order=["high_noise", "low_noise"],
        ))
        phases = resolve_phases(config)
        assert phases[1].phase_type == PhaseType.HIGH_NOISE
        assert phases[2].phase_type == PhaseType.LOW_NOISE

    def test_invalid_expert_name(self):
        config = MockConfig(moe=MockMoe(
            expert_order=["invalid_expert"],
        ))
        with pytest.raises(PhaseConfigError, match="Invalid expert name"):
            resolve_phases(config)


# ---------------------------------------------------------------------------
# Skip single expert
# ---------------------------------------------------------------------------

class TestSkipExpert:
    """enabled: false skips an expert."""

    def test_skip_high_noise(self):
        config = MockConfig(moe=MockMoe(
            high_noise=MockExpertOverrides(enabled=False, max_epochs=30),
        ))
        phases = resolve_phases(config)
        assert len(phases) == 2  # unified + low_noise only
        assert phases[0].phase_type == PhaseType.UNIFIED
        assert phases[1].phase_type == PhaseType.LOW_NOISE

    def test_skip_low_noise(self):
        config = MockConfig(moe=MockMoe(
            low_noise=MockExpertOverrides(enabled=False, max_epochs=50),
        ))
        phases = resolve_phases(config)
        assert len(phases) == 2  # unified + high_noise only
        assert phases[0].phase_type == PhaseType.UNIFIED
        assert phases[1].phase_type == PhaseType.HIGH_NOISE

    def test_skip_both_experts_from_scratch_errors(self):
        config = MockConfig(
            training=MockTraining(unified_epochs=0),
            moe=MockMoe(
                high_noise=MockExpertOverrides(enabled=False, max_epochs=30),
                low_noise=MockExpertOverrides(enabled=False, max_epochs=50),
            ),
        )
        with pytest.raises(PhaseConfigError, match="No training phases"):
            resolve_phases(config)


# ---------------------------------------------------------------------------
# Override resolution
# ---------------------------------------------------------------------------

class TestOverrideResolution:
    """Expert overrides take precedence over base config."""

    def test_learning_rate_override(self):
        config = MockConfig(moe=MockMoe(
            high_noise=MockExpertOverrides(max_epochs=30, learning_rate=1e-4),
        ))
        phases = resolve_phases(config)
        high_phase = [p for p in phases if p.phase_type == PhaseType.HIGH_NOISE][0]
        assert high_phase.learning_rate == 1e-4

    def test_learning_rate_inherit(self):
        config = MockConfig()
        phases = resolve_phases(config)
        low_phase = [p for p in phases if p.phase_type == PhaseType.LOW_NOISE][0]
        assert low_phase.learning_rate == 5e-5  # from base optimizer

    def test_weight_decay_override(self):
        config = MockConfig(moe=MockMoe(
            low_noise=MockExpertOverrides(max_epochs=50, weight_decay=0.05),
        ))
        phases = resolve_phases(config)
        low_phase = [p for p in phases if p.phase_type == PhaseType.LOW_NOISE][0]
        assert low_phase.weight_decay == 0.05

    def test_batch_size_override(self):
        config = MockConfig(moe=MockMoe(
            high_noise=MockExpertOverrides(max_epochs=30, batch_size=2),
        ))
        phases = resolve_phases(config)
        high_phase = [p for p in phases if p.phase_type == PhaseType.HIGH_NOISE][0]
        assert high_phase.batch_size == 2

    def test_caption_dropout_override(self):
        config = MockConfig(moe=MockMoe(
            low_noise=MockExpertOverrides(max_epochs=50, caption_dropout_rate=0.2),
        ))
        phases = resolve_phases(config)
        low_phase = [p for p in phases if p.phase_type == PhaseType.LOW_NOISE][0]
        assert low_phase.caption_dropout_rate == 0.2

    def test_lora_dropout_override(self):
        config = MockConfig(moe=MockMoe(
            high_noise=MockExpertOverrides(max_epochs=30, dropout=0.1),
        ))
        phases = resolve_phases(config)
        high_phase = [p for p in phases if p.phase_type == PhaseType.HIGH_NOISE][0]
        assert high_phase.lora_dropout == 0.1

    def test_optimizer_type_override(self):
        config = MockConfig(moe=MockMoe(
            low_noise=MockExpertOverrides(max_epochs=50, optimizer_type="adafactor"),
        ))
        phases = resolve_phases(config)
        low_phase = [p for p in phases if p.phase_type == PhaseType.LOW_NOISE][0]
        assert low_phase.optimizer_type == "adafactor"

    def test_scheduler_type_override(self):
        config = MockConfig(moe=MockMoe(
            high_noise=MockExpertOverrides(max_epochs=30, scheduler_type="constant"),
        ))
        phases = resolve_phases(config)
        high_phase = [p for p in phases if p.phase_type == PhaseType.HIGH_NOISE][0]
        assert high_phase.scheduler_type == "constant"

    def test_min_lr_ratio_override(self):
        config = MockConfig(moe=MockMoe(
            low_noise=MockExpertOverrides(max_epochs=50, min_lr_ratio=0.05),
        ))
        phases = resolve_phases(config)
        low_phase = [p for p in phases if p.phase_type == PhaseType.LOW_NOISE][0]
        assert low_phase.min_lr_ratio == 0.05

    def test_gradient_accumulation_override(self):
        config = MockConfig(moe=MockMoe(
            high_noise=MockExpertOverrides(max_epochs=30, gradient_accumulation_steps=4),
        ))
        phases = resolve_phases(config)
        high_phase = [p for p in phases if p.phase_type == PhaseType.HIGH_NOISE][0]
        assert high_phase.gradient_accumulation_steps == 4

    def test_fork_targets_override(self):
        config = MockConfig(moe=MockMoe(
            high_noise=MockExpertOverrides(max_epochs=30, fork_targets=["ffn"]),
        ))
        phases = resolve_phases(config)
        high_phase = [p for p in phases if p.phase_type == PhaseType.HIGH_NOISE][0]
        assert high_phase.fork_targets == ["ffn"]

    def test_block_targets_override(self):
        config = MockConfig(moe=MockMoe(
            low_noise=MockExpertOverrides(max_epochs=50, block_targets="0-11"),
        ))
        phases = resolve_phases(config)
        low_phase = [p for p in phases if p.phase_type == PhaseType.LOW_NOISE][0]
        assert low_phase.block_targets == "0-11"

    def test_resume_from_expert(self):
        config = MockConfig(moe=MockMoe(
            high_noise=MockExpertOverrides(max_epochs=30, resume_from="/path/to/lora.safetensors"),
        ))
        phases = resolve_phases(config)
        high_phase = [p for p in phases if p.phase_type == PhaseType.HIGH_NOISE][0]
        assert high_phase.resume_from == "/path/to/lora.safetensors"


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------

class TestPhaseErrors:
    """Invalid configs produce clear errors."""

    def test_expert_missing_max_epochs(self):
        config = MockConfig(moe=MockMoe(
            high_noise=MockExpertOverrides(max_epochs=None),
        ))
        with pytest.raises(PhaseConfigError, match="max_epochs"):
            resolve_phases(config)

    def test_unified_only_zero_epochs(self):
        config = MockConfig(
            moe=MockMoe(fork_enabled=False),
            training=MockTraining(unified_epochs=0),
        )
        with pytest.raises(PhaseConfigError, match="unified_epochs > 0"):
            resolve_phases(config)


# ---------------------------------------------------------------------------
# Boundary ratio resolution
# ---------------------------------------------------------------------------

class TestBoundaryRatio:
    """boundary_ratio resolves from moe override or model default."""

    def test_model_default(self):
        config = MockConfig()
        phases = resolve_phases(config)
        expert_phases = [p for p in phases if p.active_expert]
        for phase in expert_phases:
            assert phase.boundary_ratio == 0.875

    def test_moe_override(self):
        config = MockConfig(moe=MockMoe(boundary_ratio=0.9))
        phases = resolve_phases(config)
        expert_phases = [p for p in phases if p.active_expert]
        for phase in expert_phases:
            assert phase.boundary_ratio == 0.9
