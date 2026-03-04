"""Tests for flimmer.training.logger — multi-backend logging."""

import pytest

from flimmer.training.logger import TrainingLogger, generate_run_name, save_resolved_config
from flimmer.training.phase import PhaseType, TrainingPhase


def _make_phase(**kwargs):
    """Create a minimal TrainingPhase for testing."""
    defaults = dict(
        phase_type=PhaseType.UNIFIED, max_epochs=10,
        learning_rate=5e-5, weight_decay=0.01,
        optimizer_type="adamw8bit", scheduler_type="cosine_with_min_lr",
        min_lr_ratio=0.01, warmup_steps=0, batch_size=1,
        gradient_accumulation_steps=1, caption_dropout_rate=0.1,
        first_frame_dropout_rate=0.0,
        lora_dropout=0.0, fork_targets=None, block_targets=None,
        resume_from=None, boundary_ratio=None, active_expert=None,
    )
    defaults.update(kwargs)
    return TrainingPhase(**defaults)


# ---------------------------------------------------------------------------
# Mock config objects for generate_run_name
# ---------------------------------------------------------------------------

class _MockModel:
    def __init__(self, variant="2.2_t2v", family="wan"):
        self.variant = variant
        self.family = family

class _MockSave:
    def __init__(self, name="test_lora"):
        self.name = name

class _MockLora:
    def __init__(self, rank=16):
        self.rank = rank

class _MockOptimizer:
    def __init__(self, learning_rate=5e-5):
        self.learning_rate = learning_rate

class _MockTraining:
    def __init__(self, unified_epochs=10):
        self.unified_epochs = unified_epochs

class _MockMoe:
    def __init__(self, enabled=True, fork_enabled=True):
        self.enabled = enabled
        self.fork_enabled = fork_enabled

class _MockConfig:
    def __init__(self, **kwargs):
        self.model = kwargs.get("model", _MockModel())
        self.save = kwargs.get("save", _MockSave())
        self.lora = kwargs.get("lora", _MockLora())
        self.optimizer = kwargs.get("optimizer", _MockOptimizer())
        self.training = kwargs.get("training", _MockTraining())
        self.moe = kwargs.get("moe", _MockMoe())


# ---------------------------------------------------------------------------
# Tests: TrainingLogger basics (console backend)
# ---------------------------------------------------------------------------

class TestTrainingLogger:
    """Console logger basics."""

    def test_create_console_only(self):
        logger = TrainingLogger(backends=["console"])
        assert logger is not None

    def test_print_training_plan(self, capsys):
        logger = TrainingLogger(backends=["console"])
        phases = [
            _make_phase(),
            _make_phase(phase_type=PhaseType.HIGH_NOISE, active_expert="high_noise"),
        ]
        logger.print_training_plan(phases)
        output = capsys.readouterr().out
        assert "TRAINING PLAN" in output
        assert "FULL_NOISE" in output
        assert "HIGH_NOISE" in output
        assert "Total phases: 2" in output

    def test_log_phase_start(self, capsys):
        logger = TrainingLogger(backends=["console"])
        phase = _make_phase()
        logger.log_phase_start(phase, phase_index=0)
        output = capsys.readouterr().out
        assert "FULL_NOISE" in output
        assert "10 epochs" in output

    def test_log_phase_end(self, capsys):
        logger = TrainingLogger(backends=["console"])
        phase = _make_phase()
        logger.log_phase_end(phase, phase_index=0)
        output = capsys.readouterr().out
        assert "complete" in output

    def test_log_fork(self, capsys):
        logger = TrainingLogger(backends=["console"])
        logger.log_fork()
        output = capsys.readouterr().out
        assert "FORK" in output

    def test_log_step_at_interval(self, capsys):
        logger = TrainingLogger(backends=["console"], log_every_n_steps=10)
        logger.log_step(
            metrics={"loss_ema": 0.05, "learning_rate": 5e-5, "epoch": 1},
            global_step=10,
        )
        output = capsys.readouterr().out
        assert "0.0500" in output

    def test_log_step_skipped(self, capsys):
        logger = TrainingLogger(backends=["console"], log_every_n_steps=10)
        logger.log_step(
            metrics={"loss_ema": 0.05},
            global_step=7,
        )
        output = capsys.readouterr().out
        assert output == ""

    def test_log_checkpoint_saved(self, capsys, tmp_path):
        logger = TrainingLogger(backends=["console"])
        path = tmp_path / "checkpoint.safetensors"
        logger.log_checkpoint_saved(path, PhaseType.UNIFIED, epoch=5)
        output = capsys.readouterr().out
        assert "checkpoint.safetensors" in output

    def test_log_sample_generated(self, capsys, tmp_path):
        logger = TrainingLogger(backends=["console"])
        path = tmp_path / "sample.mp4"
        logger.log_sample_generated(path, prompt_index=0)
        output = capsys.readouterr().out
        assert "sample.mp4" in output

    def test_close_is_safe(self):
        logger = TrainingLogger(backends=["console"])
        logger.close()  # Should not raise

    def test_plan_shows_fork_targets(self, capsys):
        logger = TrainingLogger(backends=["console"])
        phases = [_make_phase(fork_targets=["ffn", "self_attn"])]
        logger.print_training_plan(phases)
        output = capsys.readouterr().out
        assert "ffn" in output

    def test_plan_shows_block_targets(self, capsys):
        logger = TrainingLogger(backends=["console"])
        phases = [_make_phase(block_targets="0-11")]
        logger.print_training_plan(phases)
        output = capsys.readouterr().out
        assert "0-11" in output

    def test_plan_shows_boundary_ratio(self, capsys):
        logger = TrainingLogger(backends=["console"])
        phases = [_make_phase(boundary_ratio=0.875)]
        logger.print_training_plan(phases)
        output = capsys.readouterr().out
        assert "0.875" in output


# ---------------------------------------------------------------------------
# Tests: generate_run_name
# ---------------------------------------------------------------------------

class TestGenerateRunName:
    """Auto-descriptive W&B run naming from config."""

    def test_fork_mode(self):
        """MoE enabled + fork enabled + unified_epochs > 0 = fork mode."""
        config = _MockConfig(
            model=_MockModel(variant="2.2_t2v"),
            save=_MockSave(name="holly"),
            moe=_MockMoe(enabled=True, fork_enabled=True),
            training=_MockTraining(unified_epochs=10),
            lora=_MockLora(rank=16),
            optimizer=_MockOptimizer(learning_rate=1e-4),
        )
        name = generate_run_name(config)
        assert name == "wan22t2v-holly-fork-r16-lr1e-04"

    def test_unified_mode(self):
        """MoE disabled = unified mode."""
        config = _MockConfig(
            model=_MockModel(variant="2.2_t2v"),
            save=_MockSave(name="annika"),
            moe=_MockMoe(enabled=False, fork_enabled=False),
            lora=_MockLora(rank=16),
            optimizer=_MockOptimizer(learning_rate=5e-5),
        )
        name = generate_run_name(config)
        assert name == "wan22t2v-annika-full_noise-r16-lr5e-05"

    def test_expert_mode(self):
        """MoE enabled + fork enabled + unified_epochs == 0 = expert mode."""
        config = _MockConfig(
            model=_MockModel(variant="2.2_t2v"),
            save=_MockSave(name="test"),
            moe=_MockMoe(enabled=True, fork_enabled=True),
            training=_MockTraining(unified_epochs=0),
            lora=_MockLora(rank=24),
            optimizer=_MockOptimizer(learning_rate=8e-5),
        )
        name = generate_run_name(config)
        assert name == "wan22t2v-test-expert-r24-lr8e-05"

    def test_default_name_replaced(self):
        """'flimmer_lora' in save name is replaced with 'default'."""
        config = _MockConfig(
            save=_MockSave(name="flimmer_lora"),
        )
        name = generate_run_name(config)
        assert "default" in name
        assert "flimmer_lora" not in name

    def test_variant_dots_removed(self):
        """Dots in variant are removed for compact naming."""
        config = _MockConfig(
            model=_MockModel(variant="2.2_i2v"),
        )
        name = generate_run_name(config)
        assert name.startswith("wan22i2v-")


# ---------------------------------------------------------------------------
# Tests: log_vram
# ---------------------------------------------------------------------------

class TestLogVram:
    """VRAM metric logging."""

    def test_log_vram_console_only_no_crash(self, capsys):
        """log_vram() with console-only backend does not crash."""
        logger = TrainingLogger(backends=["console"])
        metrics = {"system/vram_allocated_gb": 4.5, "system/vram_reserved_gb": 6.0}
        logger.log_vram(metrics, global_step=50)
        # Console is intentionally skipped for VRAM, so no output
        output = capsys.readouterr().out
        assert output == ""

    def test_log_vram_no_wandb_no_crash(self):
        """log_vram() does not crash when W&B is not initialized."""
        logger = TrainingLogger(backends=["console"])
        assert logger._wandb_run is None
        logger.log_vram({"system/vram_allocated_gb": 4.0}, global_step=100)


# ---------------------------------------------------------------------------
# Tests: log_run_summary
# ---------------------------------------------------------------------------

class TestLogRunSummary:
    """End-of-run console summary."""

    def test_summary_includes_timing(self, capsys):
        """Summary shows total time and per-phase times."""
        logger = TrainingLogger(backends=["console"])
        logger.log_run_summary(
            total_time=120.0,
            phase_times={"full_noise": 60.0, "high_noise": 60.0},
            peak_vram_gb=12.5,
            phase_losses={"full_noise": 0.0123, "high_noise": 0.0089},
        )
        output = capsys.readouterr().out
        assert "TRAINING COMPLETE" in output
        assert "2.0 min" in output  # 120s = 2.0 min
        assert "full_noise" in output
        assert "high_noise" in output

    def test_summary_includes_loss(self, capsys):
        """Summary shows final EMA loss per phase."""
        logger = TrainingLogger(backends=["console"])
        logger.log_run_summary(
            total_time=60.0,
            phase_times={},
            peak_vram_gb=0.0,
            phase_losses={"full_noise": 0.012345},
        )
        output = capsys.readouterr().out
        assert "0.012345" in output

    def test_summary_includes_vram(self, capsys):
        """Summary shows peak VRAM when > 0."""
        logger = TrainingLogger(backends=["console"])
        logger.log_run_summary(
            total_time=60.0,
            phase_times={},
            peak_vram_gb=14.3,
            phase_losses={},
        )
        output = capsys.readouterr().out
        assert "14.30" in output

    def test_summary_no_vram_when_zero(self, capsys):
        """Summary skips VRAM line when peak is 0 (CPU-only)."""
        logger = TrainingLogger(backends=["console"])
        logger.log_run_summary(
            total_time=60.0,
            phase_times={},
            peak_vram_gb=0.0,
            phase_losses={},
        )
        output = capsys.readouterr().out
        assert "Peak VRAM" not in output


# ---------------------------------------------------------------------------
# Tests: save_resolved_config
# ---------------------------------------------------------------------------

class TestSaveResolvedConfig:
    """Resolved config YAML save to disk."""

    def test_saves_yaml_file(self, tmp_path):
        """save_resolved_config creates a YAML file in the output dir."""
        class FakeConfig:
            def model_dump(self):
                return {"model": {"variant": "2.2_t2v"}, "lora": {"rank": 16}}

        path = save_resolved_config(FakeConfig(), tmp_path)
        assert path.exists()
        assert path.name == "resolved_config.yaml"

    def test_yaml_contents(self, tmp_path):
        """Saved YAML contains the config dict."""
        import yaml

        class FakeConfig:
            def model_dump(self):
                return {"model": {"variant": "2.2_t2v"}}

        path = save_resolved_config(FakeConfig(), tmp_path)
        with open(path) as f:
            data = yaml.safe_load(f)
        assert data["model"]["variant"] == "2.2_t2v"

    def test_creates_parent_dirs(self, tmp_path):
        """save_resolved_config creates parent directories if needed."""
        deep_dir = tmp_path / "a" / "b" / "c"

        class FakeConfig:
            def model_dump(self):
                return {"key": "value"}

        path = save_resolved_config(FakeConfig(), deep_dir)
        assert path.exists()


# ---------------------------------------------------------------------------
# Tests: log_samples_to_wandb
# ---------------------------------------------------------------------------

class TestLogSamplesToWandb:
    """W&B media logging for sample videos and keyframe grids."""

    def test_no_crash_without_wandb(self, tmp_path):
        """log_samples_to_wandb does not crash when wandb_run is None."""
        logger = TrainingLogger(backends=["console"])
        assert logger._wandb_run is None
        # Should silently return without error
        video = tmp_path / "sample.mp4"
        video.write_bytes(b"fake video data")
        logger.log_samples_to_wandb(
            sample_paths=[video],
            phase_type="full_noise",
            epoch=5,
            global_step=100,
        )

    def test_no_crash_with_empty_paths(self):
        """log_samples_to_wandb handles empty path list gracefully."""
        logger = TrainingLogger(backends=["console"])
        logger.log_samples_to_wandb(
            sample_paths=[],
            phase_type="high_noise",
            epoch=1,
            global_step=50,
        )

    def test_skips_missing_files(self, tmp_path):
        """log_samples_to_wandb skips paths that don't exist on disk."""
        logger = TrainingLogger(backends=["console"])
        missing = tmp_path / "nonexistent.mp4"
        # Should not crash even with non-existent files
        logger.log_samples_to_wandb(
            sample_paths=[missing],
            phase_type="full_noise",
            epoch=1,
            global_step=10,
        )


# ---------------------------------------------------------------------------
# Tests: log_frozen_check
# ---------------------------------------------------------------------------

class TestLogFrozenCheck:
    """Frozen-expert verification console logging."""

    def test_prints_pass(self, capsys):
        """log_frozen_check prints PASS for a passing result."""
        logger = TrainingLogger(backends=["console"])

        class FakeResult:
            expert_name = "high_noise"
            passed = True
            details = "All good"

        logger.log_frozen_check(FakeResult())
        output = capsys.readouterr().out
        assert "PASS" in output
        assert "high_noise" in output

    def test_prints_fail(self, capsys):
        """log_frozen_check prints FAIL for a failing result."""
        logger = TrainingLogger(backends=["console"])

        class FakeResult:
            expert_name = "low_noise"
            passed = False
            details = "Changed"

        logger.log_frozen_check(FakeResult())
        # FAIL goes to stderr
        err = capsys.readouterr().err
        assert "FAIL" in err
        assert "low_noise" in err

    def test_no_crash_without_wandb(self):
        """log_frozen_check does not crash when wandb_run is None."""
        logger = TrainingLogger(backends=["console"])

        class FakeResult:
            expert_name = "high_noise"
            passed = True
            details = "OK"

        logger.log_frozen_check(FakeResult())  # Should not raise
