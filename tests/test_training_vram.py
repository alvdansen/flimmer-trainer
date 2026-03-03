"""Tests for flimmer.training.vram — GPU VRAM tracking utility."""

from unittest.mock import MagicMock, patch

import pytest

from flimmer.training.vram import VRAMTracker


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
