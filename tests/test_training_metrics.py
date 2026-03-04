"""Tests for flimmer.training.metrics — per-phase metric tracking and timing."""

import time

import pytest

from flimmer.training.metrics import MetricsTracker, PhaseMetrics, RunTimer
from flimmer.training.phase import PhaseType


class TestPhaseMetrics:
    """PhaseMetrics accumulation."""

    def test_initial_state(self):
        m = PhaseMetrics(phase_type=PhaseType.UNIFIED)
        assert m.loss_ema == 0.0
        assert m.step_count == 0

    def test_first_update_sets_ema(self):
        m = PhaseMetrics(phase_type=PhaseType.UNIFIED)
        m.update(loss=1.0)
        assert m.loss_ema == 1.0
        assert m.loss_raw == 1.0
        assert m.step_count == 1

    def test_ema_smoothing(self):
        m = PhaseMetrics(phase_type=PhaseType.UNIFIED)
        m.update(loss=1.0, ema_decay=0.9)
        m.update(loss=0.0, ema_decay=0.9)
        # EMA = 0.9 * 1.0 + 0.1 * 0.0 = 0.9
        assert m.loss_ema == pytest.approx(0.9)
        assert m.loss_raw == 0.0

    def test_grad_norm_tracked(self):
        m = PhaseMetrics(phase_type=PhaseType.UNIFIED)
        m.update(loss=0.5, grad_norm=1.5)
        assert m.grad_norm == 1.5

    def test_learning_rate_tracked(self):
        m = PhaseMetrics(phase_type=PhaseType.UNIFIED)
        m.update(loss=0.5, learning_rate=5e-5)
        assert m.learning_rate == 5e-5

    def test_to_dict(self):
        m = PhaseMetrics(phase_type=PhaseType.HIGH_NOISE)
        m.update(loss=0.1, grad_norm=0.5, learning_rate=1e-4)
        d = m.to_dict(prefix="train/")
        assert "train/loss_ema" in d
        assert "train/grad_norm" in d
        assert d["train/step"] == 1.0

    def test_reset(self):
        m = PhaseMetrics(phase_type=PhaseType.UNIFIED)
        m.update(loss=1.0)
        m.reset()
        assert m.loss_ema == 0.0
        assert m.step_count == 0


class TestMetricsTracker:
    """Multi-phase metric tracking."""

    def test_start_phase(self):
        tracker = MetricsTracker()
        metrics = tracker.start_phase(PhaseType.UNIFIED)
        assert metrics.phase_type == PhaseType.UNIFIED
        assert tracker.current_phase == PhaseType.UNIFIED

    def test_update_current(self):
        tracker = MetricsTracker()
        tracker.start_phase(PhaseType.UNIFIED)
        tracker.update(loss=0.5)
        current = tracker.get_current()
        assert current is not None
        assert current.loss_raw == 0.5

    def test_update_without_phase_raises(self):
        tracker = MetricsTracker()
        with pytest.raises(RuntimeError, match="No active phase"):
            tracker.update(loss=0.5)

    def test_set_epoch(self):
        tracker = MetricsTracker()
        tracker.start_phase(PhaseType.HIGH_NOISE)
        tracker.set_epoch(5)
        current = tracker.get_current()
        assert current is not None
        assert current.epoch == 5

    def test_get_phase(self):
        tracker = MetricsTracker()
        tracker.start_phase(PhaseType.UNIFIED)
        tracker.update(loss=0.1)
        tracker.start_phase(PhaseType.HIGH_NOISE)
        tracker.update(loss=0.2)
        unified = tracker.get_phase(PhaseType.UNIFIED)
        assert unified is not None
        assert unified.loss_raw == 0.1

    def test_get_all_metrics(self):
        tracker = MetricsTracker()
        tracker.start_phase(PhaseType.UNIFIED)
        tracker.update(loss=0.1)
        tracker.start_phase(PhaseType.HIGH_NOISE)
        tracker.update(loss=0.2)
        all_metrics = tracker.get_all_metrics()
        assert "full_noise/loss_raw" in all_metrics
        assert "high_noise/loss_raw" in all_metrics

    def test_tracked_phases(self):
        tracker = MetricsTracker()
        tracker.start_phase(PhaseType.UNIFIED)
        tracker.start_phase(PhaseType.LOW_NOISE)
        assert PhaseType.UNIFIED in tracker.tracked_phases
        assert PhaseType.LOW_NOISE in tracker.tracked_phases


class TestRunTimer:
    """Wall-clock timer for training phases and total run."""

    def test_start_run_total_elapsed(self):
        """start_run + total_elapsed gives a positive value."""
        timer = RunTimer()
        timer.start_run()
        # Even a tiny sleep ensures elapsed > 0
        time.sleep(0.01)
        elapsed = timer.total_elapsed()
        assert elapsed > 0.0

    def test_start_phase_end_phase(self):
        """start_phase + end_phase records timing for the named phase."""
        timer = RunTimer()
        timer.start_run()
        timer.start_phase("full_noise")
        time.sleep(0.01)
        elapsed = timer.end_phase("full_noise")
        assert elapsed > 0.0

    def test_phase_times_property(self):
        """phase_times returns a dict of recorded phase durations."""
        timer = RunTimer()
        timer.start_run()
        timer.start_phase("high_noise")
        time.sleep(0.01)
        timer.end_phase("high_noise")
        timer.start_phase("low_noise")
        time.sleep(0.01)
        timer.end_phase("low_noise")

        pt = timer.phase_times
        assert "high_noise" in pt
        assert "low_noise" in pt
        assert pt["high_noise"] > 0.0
        assert pt["low_noise"] > 0.0

    def test_phase_times_returns_copy(self):
        """phase_times returns a copy, not the internal dict."""
        timer = RunTimer()
        pt1 = timer.phase_times
        pt1["injected"] = 999.0
        pt2 = timer.phase_times
        assert "injected" not in pt2

    def test_initial_state_zero(self):
        """Before start_run, total_elapsed returns a value near perf_counter."""
        timer = RunTimer()
        # total_elapsed = perf_counter() - 0.0, which is a large number
        # but should not raise
        _ = timer.total_elapsed()
