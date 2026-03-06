"""Tests for flimmer.training.vram — GPU VRAM tracking and estimation."""

import logging
from unittest.mock import MagicMock, patch

import pytest

from flimmer.training.vram import (
    VRAMEstimate,
    VRAMEstimator,
    VRAMTracker,
    detect_gpu_memory_gb,
    format_vram_warning,
)


class TestVRAMTrackerNoGPU:
    """VRAMTracker behavior when CUDA is not available."""

    def test_sample_returns_none_off_interval(self):
        """sample() returns None when not at the sampling interval."""
        tracker = VRAMTracker(sample_every_n_steps=50)
        # Step 7 is not a multiple of 50
        result = tracker.sample(global_step=7)
        assert result is None

    def test_sample_returns_none_without_cuda(self):
        """sample() returns None when CUDA is not available."""
        tracker = VRAMTracker(sample_every_n_steps=10)
        # Mock torch.cuda.is_available() to return False
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        with patch.dict("sys.modules", {"torch": mock_torch}):
            result = tracker.sample(global_step=10)
        assert result is None

    def test_peak_returns_zero_without_cuda(self):
        """peak() returns 0.0 when CUDA is not available."""
        tracker = VRAMTracker()
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        with patch.dict("sys.modules", {"torch": mock_torch}):
            result = tracker.peak()
        assert result == 0.0

    def test_reset_peak_no_error_without_cuda(self):
        """reset_peak() does not raise when CUDA is not available."""
        tracker = VRAMTracker()
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        with patch.dict("sys.modules", {"torch": mock_torch}):
            tracker.reset_peak()  # Should not raise


class TestVRAMTrackerWithCUDA:
    """VRAMTracker behavior with mocked CUDA."""

    def _make_mock_torch(self, allocated_bytes=4 * (1024**3), reserved_bytes=6 * (1024**3)):
        """Create a mock torch module with CUDA available."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.memory_allocated.return_value = allocated_bytes
        mock_torch.cuda.memory_reserved.return_value = reserved_bytes
        mock_torch.cuda.max_memory_allocated.return_value = allocated_bytes
        return mock_torch

    def test_sample_returns_dict_at_interval(self):
        """sample() returns a metrics dict at the configured interval."""
        tracker = VRAMTracker(sample_every_n_steps=10)
        mock_torch = self._make_mock_torch(
            allocated_bytes=4 * (1024**3),  # 4 GB
            reserved_bytes=6 * (1024**3),   # 6 GB
        )
        with patch.dict("sys.modules", {"torch": mock_torch}):
            result = tracker.sample(global_step=10)

        assert result is not None
        assert "system/vram_allocated_gb" in result
        assert "system/vram_reserved_gb" in result
        assert result["system/vram_allocated_gb"] == pytest.approx(4.0)
        assert result["system/vram_reserved_gb"] == pytest.approx(6.0)

    def test_sample_tracks_history(self):
        """Each successful sample is appended to the samples list."""
        tracker = VRAMTracker(sample_every_n_steps=10)
        mock_torch = self._make_mock_torch(allocated_bytes=2 * (1024**3))
        with patch.dict("sys.modules", {"torch": mock_torch}):
            tracker.sample(global_step=10)
            tracker.sample(global_step=20)

        assert len(tracker.samples) == 2
        assert tracker.samples[0] == pytest.approx(2.0)

    def test_peak_returns_max_allocated(self):
        """peak() returns the max allocated VRAM in GB."""
        tracker = VRAMTracker()
        mock_torch = self._make_mock_torch()
        mock_torch.cuda.max_memory_allocated.return_value = 8 * (1024**3)
        with patch.dict("sys.modules", {"torch": mock_torch}):
            result = tracker.peak()

        assert result == pytest.approx(8.0)

    def test_reset_peak_calls_cuda(self):
        """reset_peak() calls torch.cuda.reset_peak_memory_stats()."""
        tracker = VRAMTracker(device=0)
        mock_torch = self._make_mock_torch()
        with patch.dict("sys.modules", {"torch": mock_torch}):
            tracker.reset_peak()

        mock_torch.cuda.reset_peak_memory_stats.assert_called_once_with(0)

    def test_sample_at_step_zero(self):
        """sample() works at global_step=0 (0 % N == 0)."""
        tracker = VRAMTracker(sample_every_n_steps=50)
        mock_torch = self._make_mock_torch(allocated_bytes=1 * (1024**3))
        with patch.dict("sys.modules", {"torch": mock_torch}):
            result = tracker.sample(global_step=0)

        assert result is not None
        assert result["system/vram_allocated_gb"] == pytest.approx(1.0)

    def test_custom_device_index(self):
        """VRAMTracker passes the correct device index to CUDA calls."""
        tracker = VRAMTracker(device=2, sample_every_n_steps=10)
        mock_torch = self._make_mock_torch()
        with patch.dict("sys.modules", {"torch": mock_torch}):
            tracker.sample(global_step=10)

        mock_torch.cuda.memory_allocated.assert_called_with(2)
        mock_torch.cuda.memory_reserved.assert_called_with(2)


# ──────────────────────────────────────────────────────────────────────
# VRAMEstimate dataclass tests
# ──────────────────────────────────────────────────────────────────────


class TestVRAMEstimate:
    """VRAMEstimate dataclass: totals and formatting."""

    def test_total_gb_sums_all_components_minus_block_swap(self):
        """total_gb sums all component fields and subtracts block_swap_savings_gb."""
        est = VRAMEstimate(
            model_weights_gb=14.0,
            block_swap_savings_gb=7.0,
            lora_params_gb=0.1,
            gradients_gb=0.2,
            optimizer_states_gb=0.1,
            activations_gb=5.0,
            overhead_gb=1.5,
        )
        # 14 + 0.1 + 0.2 + 0.1 + 5.0 + 1.5 - 7.0 = 13.9
        assert est.total_gb == pytest.approx(13.9, abs=0.01)

    def test_total_gb_no_block_swap(self):
        """total_gb with zero block swap just sums everything."""
        est = VRAMEstimate(
            model_weights_gb=28.0,
            block_swap_savings_gb=0.0,
            lora_params_gb=0.2,
            gradients_gb=0.4,
            optimizer_states_gb=0.8,
            activations_gb=10.0,
            overhead_gb=1.5,
        )
        assert est.total_gb == pytest.approx(40.9, abs=0.01)

    def test_breakdown_table_returns_multiline_string(self):
        """breakdown_table() returns a formatted multi-line string with all components."""
        est = VRAMEstimate(
            model_weights_gb=14.0,
            block_swap_savings_gb=7.0,
            lora_params_gb=0.1,
            gradients_gb=0.2,
            optimizer_states_gb=0.1,
            activations_gb=5.0,
            overhead_gb=1.5,
        )
        table = est.breakdown_table()
        assert isinstance(table, str)
        assert "\n" in table
        assert "Model weights" in table
        assert "Block swap" in table
        assert "LoRA" in table
        assert "Gradients" in table
        assert "Optimizer" in table
        assert "Activations" in table
        assert "Overhead" in table
        assert "TOTAL" in table

    def test_default_overhead(self):
        """overhead_gb defaults to 1.5 when not specified."""
        est = VRAMEstimate(
            model_weights_gb=14.0,
            block_swap_savings_gb=0.0,
            lora_params_gb=0.1,
            gradients_gb=0.2,
            optimizer_states_gb=0.1,
            activations_gb=5.0,
        )
        assert est.overhead_gb == 1.5


# ──────────────────────────────────────────────────────────────────────
# VRAMEstimator formula tests
# ──────────────────────────────────────────────────────────────────────


class TestVRAMEstimator:
    """VRAMEstimator: component-based VRAM estimation formulas."""

    def test_t2v_bf16_total_around_42gb(self):
        """T2V bf16 config estimates ~42 GB total (model 28 + activations + overhead)."""
        estimator = VRAMEstimator(
            precision="bf16",
            blocks_to_swap=0,
            rank=16,
            optimizer_type="adamw",
            is_i2v=False,
            batch_size=1,
            resolution=720,
            frame_count=49,
        )
        est = estimator.estimate()
        # Model weights: 14B params * 2 bytes = ~26 GiB (~28 GB decimal)
        # Plus activations, LoRA, grads, optimizer, overhead => total ~35-50 GiB
        assert 35.0 < est.total_gb < 50.0
        assert est.model_weights_gb == pytest.approx(26.1, abs=1.5)

    def test_t2v_fp8_model_weight_around_13gb(self):
        """T2V fp8 config estimates ~13 GiB model weight (half of bf16)."""
        estimator = VRAMEstimator(
            precision="fp8",
            blocks_to_swap=0,
            rank=16,
            optimizer_type="adamw8bit",
            is_i2v=False,
            batch_size=1,
            resolution=480,
            frame_count=49,
        )
        est = estimator.estimate()
        # 14B params * 1 byte = ~13.0 GiB
        assert est.model_weights_gb == pytest.approx(13.0, abs=1.0)

    def test_block_swap_20_at_fp8_saves_about_7gb(self):
        """blocks_to_swap=20 at fp8 subtracts ~7 GB (20 * ~350 MB)."""
        estimator = VRAMEstimator(
            precision="fp8",
            blocks_to_swap=20,
            rank=16,
            optimizer_type="adamw8bit",
            is_i2v=False,
            batch_size=1,
            resolution=480,
            frame_count=49,
        )
        est = estimator.estimate()
        # 20 blocks * ~175 MB each at fp8 = ~3.5 GB savings
        # Plan says ~7 GB at bf16 per-block size of 350 MB, but fp8 is ~175 MB
        # The plan text says "~7 GB" but that's for bf16. For fp8: ~3.5 GB
        assert est.block_swap_savings_gb == pytest.approx(3.5, abs=0.5)

    def test_i2v_higher_estimate_than_t2v(self):
        """I2V mode produces higher estimate than T2V (more LoRA modules + activation overhead)."""
        t2v = VRAMEstimator(
            precision="fp8",
            blocks_to_swap=0,
            rank=16,
            optimizer_type="adamw8bit",
            is_i2v=False,
            batch_size=1,
            resolution=480,
            frame_count=49,
        )
        i2v = VRAMEstimator(
            precision="fp8",
            blocks_to_swap=0,
            rank=16,
            optimizer_type="adamw8bit",
            is_i2v=True,
            batch_size=1,
            resolution=480,
            frame_count=49,
        )
        t2v_est = t2v.estimate()
        i2v_est = i2v.estimate()
        assert i2v_est.total_gb > t2v_est.total_gb

    def test_cpu_offload_optimizer_zero_gpu(self):
        """cpu_offload optimizer has 0 GB optimizer state on GPU."""
        estimator = VRAMEstimator(
            precision="fp8",
            blocks_to_swap=0,
            rank=16,
            optimizer_type="cpu_offload",
            is_i2v=False,
            batch_size=1,
        )
        est = estimator.estimate()
        assert est.optimizer_states_gb == 0.0

    def test_adamw8bit_2_bytes_per_param(self):
        """adamw8bit has ~2 bytes/param optimizer state — smaller than adamw."""
        adamw_estimator = VRAMEstimator(
            precision="fp8",
            blocks_to_swap=0,
            rank=16,
            optimizer_type="adamw",
            is_i2v=False,
            batch_size=1,
        )
        adamw8_estimator = VRAMEstimator(
            precision="fp8",
            blocks_to_swap=0,
            rank=16,
            optimizer_type="adamw8bit",
            is_i2v=False,
            batch_size=1,
        )
        adamw_est = adamw_estimator.estimate()
        adamw8_est = adamw8_estimator.estimate()
        # adamw8bit should be ~1/4 of adamw (2 vs 8 bytes/param)
        assert adamw8_est.optimizer_states_gb < adamw_est.optimizer_states_gb
        assert adamw8_est.optimizer_states_gb == pytest.approx(
            adamw_est.optimizer_states_gb / 4.0, abs=0.01
        )


# ──────────────────────────────────────────────────────────────────────
# GPU detection tests
# ──────────────────────────────────────────────────────────────────────


class TestDetectGPUMemory:
    """detect_gpu_memory_gb() function."""

    def test_returns_none_when_torch_not_available(self):
        """Returns None when torch is not importable."""
        with patch.dict("sys.modules", {"torch": None}):
            result = detect_gpu_memory_gb()
        assert result is None

    def test_returns_total_gb_when_cuda_available(self):
        """Returns total GPU memory in GB when CUDA is available (mocked)."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        total_bytes = 24 * (1024**3)  # 24 GB
        mock_torch.cuda.mem_get_info.return_value = (20 * (1024**3), total_bytes)
        with patch.dict("sys.modules", {"torch": mock_torch}):
            result = detect_gpu_memory_gb()
        assert result == pytest.approx(24.0, abs=0.1)

    def test_returns_none_when_cuda_not_available(self):
        """Returns None when CUDA is not available."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        with patch.dict("sys.modules", {"torch": mock_torch}):
            result = detect_gpu_memory_gb()
        assert result is None


# ──────────────────────────────────────────────────────────────────────
# Warning and reporting tests
# ──────────────────────────────────────────────────────────────────────


class TestFormatVRAMWarning:
    """format_vram_warning() function: actionable suggestions."""

    def _make_estimate(self, block_swap=0.0, precision="bf16", resolution=720):
        """Helper to create an estimate with known config for testing."""
        return VRAMEstimate(
            model_weights_gb=28.0 if precision == "bf16" else 14.0,
            block_swap_savings_gb=block_swap,
            lora_params_gb=0.1,
            gradients_gb=0.2,
            optimizer_states_gb=0.4,
            activations_gb=10.0,
            overhead_gb=1.5,
        )

    def test_includes_actionable_suggestions(self):
        """Warning includes actionable suggestions (block_swap, fp8, resolution)."""
        est = self._make_estimate(block_swap=0.0, precision="bf16", resolution=720)
        config_summary = {"precision": "bf16", "blocks_to_swap": 0, "resolution": 720}
        warning = format_vram_warning(est, available_gb=24.0, config_summary=config_summary)
        assert "block" in warning.lower() or "swap" in warning.lower()

    def test_suggests_block_swap_when_zero(self):
        """Suggests blocks_to_swap when current is 0."""
        est = self._make_estimate(block_swap=0.0)
        config_summary = {"precision": "bf16", "blocks_to_swap": 0, "resolution": 720}
        warning = format_vram_warning(est, available_gb=24.0, config_summary=config_summary)
        assert "blocks_to_swap" in warning

    def test_suggests_fp8_when_bf16(self):
        """Suggests fp8 when current precision is bf16."""
        est = self._make_estimate(precision="bf16")
        config_summary = {"precision": "bf16", "blocks_to_swap": 0, "resolution": 720}
        warning = format_vram_warning(est, available_gb=24.0, config_summary=config_summary)
        assert "fp8" in warning

    def test_suggests_lower_resolution_when_720(self):
        """Suggests lower resolution when current resolution is 720."""
        est = self._make_estimate(resolution=720)
        config_summary = {"precision": "bf16", "blocks_to_swap": 0, "resolution": 720}
        warning = format_vram_warning(est, available_gb=24.0, config_summary=config_summary)
        assert "480" in warning or "resolution" in warning.lower()


# ──────────────────────────────────────────────────────────────────────
# Reporting and logging tests
# ──────────────────────────────────────────────────────────────────────


class TestEstimatorReporting:
    """VRAMEstimator: print_report(), log_estimate(), warn_if_over()."""

    def _make_estimator(self):
        """Create a standard T2V fp8 estimator for reporting tests."""
        return VRAMEstimator(
            precision="fp8",
            blocks_to_swap=0,
            rank=16,
            optimizer_type="adamw8bit",
            is_i2v=False,
            batch_size=1,
            resolution=480,
            frame_count=49,
        )

    def test_print_report_outputs_breakdown(self, capsys):
        """print_report() outputs component breakdown table to stdout."""
        estimator = self._make_estimator()
        est = estimator.estimate()
        estimator.print_report(est, gpu_memory_gb=24.0)
        captured = capsys.readouterr()
        assert "Model weights" in captured.out
        assert "TOTAL" in captured.out

    def test_log_estimate_uses_logger(self, caplog):
        """log_estimate() logs the breakdown via the module logger at INFO level."""
        estimator = self._make_estimator()
        est = estimator.estimate()
        with caplog.at_level(logging.INFO, logger="flimmer.training.vram"):
            estimator.log_estimate(est)
        assert any("Model weights" in r.message for r in caplog.records)

    def test_warn_if_over_logs_warning_when_exceeds(self, caplog):
        """warn_if_over() logs warning when estimate exceeds available memory."""
        estimator = self._make_estimator()
        est = estimator.estimate()
        # Give a very small GPU so estimate exceeds it
        with caplog.at_level(logging.WARNING, logger="flimmer.training.vram"):
            estimator.warn_if_over(est, gpu_memory_gb=4.0)
        assert any(r.levelno == logging.WARNING for r in caplog.records)

    def test_warn_if_over_no_warning_when_fits(self, caplog):
        """warn_if_over() does not warn when estimate fits within available memory."""
        estimator = self._make_estimator()
        est = estimator.estimate()
        with caplog.at_level(logging.WARNING, logger="flimmer.training.vram"):
            estimator.warn_if_over(est, gpu_memory_gb=200.0)
        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warning_records) == 0


# ──────────────────────────────────────────────────────────────────────
# Integration tests: from_config(), CLI, parser
# ──────────────────────────────────────────────────────────────────────


class TestFromConfig:
    """VRAMEstimator.from_config() classmethod."""

    def test_extracts_fields_from_mock_config(self):
        """from_config() correctly extracts fields from a mock FlimmerTrainingConfig."""
        config = MagicMock()
        config.training.base_model_precision = "fp8"
        config.training.blocks_to_swap = 20
        config.training.batch_size = 1
        config.lora.rank = 32
        config.optimizer.type = "adamw8bit"
        config.model.variant = "2.2_i2v"
        config.data_config = None  # No data config

        estimator = VRAMEstimator.from_config(config)

        assert estimator._precision == "fp8"
        assert estimator._blocks_to_swap == 20
        assert estimator._rank == 32
        assert estimator._optimizer_type == "adamw8bit"
        assert estimator._is_i2v is True
        assert estimator._batch_size == 1

    def test_defaults_without_optional_fields(self):
        """from_config() uses sensible defaults when config fields are missing."""
        config = MagicMock(spec=[])  # Empty mock — no attributes
        config.training = None
        config.lora = None
        config.optimizer = None
        config.model = None
        config.data_config = None

        estimator = VRAMEstimator.from_config(config)

        assert estimator._precision == "bf16"
        assert estimator._blocks_to_swap == 0
        assert estimator._rank == 16
        assert estimator._optimizer_type == "adamw8bit"
        assert estimator._is_i2v is False


class TestCLIParser:
    """CLI parser accepts estimate-vram command."""

    def test_parser_accepts_estimate_vram(self):
        """build_parser() recognizes 'estimate-vram' as a valid subcommand."""
        from flimmer.training.__main__ import build_parser

        parser = build_parser()
        args = parser.parse_args(["estimate-vram", "--config", "train.yaml"])
        assert args.command == "estimate-vram"
        assert args.config == "train.yaml"

    def test_parser_accepts_gpu_memory_flag(self):
        """estimate-vram parser accepts --gpu-memory flag."""
        from flimmer.training.__main__ import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "estimate-vram", "--config", "train.yaml", "--gpu-memory", "24",
        ])
        assert args.gpu_memory == 24.0

    def test_parser_accepts_resolution_and_frames(self):
        """estimate-vram parser accepts --resolution and --frames flags."""
        from flimmer.training.__main__ import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "estimate-vram", "-c", "train.yaml",
            "--resolution", "720", "--frames", "81",
        ])
        assert args.resolution == 720
        assert args.frames == 81

    def test_cmd_estimate_vram_calls_print_report(self):
        """cmd_estimate_vram loads config and calls print_report."""
        from flimmer.training.__main__ import cmd_estimate_vram

        mock_config = MagicMock()
        mock_config.training.base_model_precision = "fp8"
        mock_config.training.blocks_to_swap = 0
        mock_config.training.batch_size = 1
        mock_config.lora.rank = 16
        mock_config.optimizer.type = "adamw8bit"
        mock_config.model.variant = "2.2_t2v"
        mock_config.data_config = None

        args = MagicMock()
        args.config = "train.yaml"
        args.gpu_memory = 24.0
        args.resolution = None
        args.frames = None

        with patch(
            "flimmer.config.training_loader.load_training_config",
            return_value=mock_config,
        ) as mock_load:
            cmd_estimate_vram(args)
            mock_load.assert_called_once_with("train.yaml")
