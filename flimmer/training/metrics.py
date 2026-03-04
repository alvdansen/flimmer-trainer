"""Per-phase metric accumulation and tracking.

Tracks training metrics (loss, gradient norm, learning rate) per phase.
Uses exponential moving average (EMA) for smoothed loss reporting and
exports flat dicts for logger consumption.

Also provides RunTimer for wall-clock timing of training phases and
the total run.

GPU-free — pure math on Python floats.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from flimmer.training.phase import PhaseType


@dataclass
class PhaseMetrics:
    """Accumulated metrics for a single training phase.

    Tracks both raw and smoothed loss (EMA), gradient norm, and learning
    rate. Step count is tracked for averaging and logging intervals.
    """
    phase_type: PhaseType
    """Which phase these metrics belong to."""

    loss_ema: float = 0.0
    """Exponential moving average of loss. Updated every step."""

    loss_raw: float = 0.0
    """Last raw (unsmoothed) loss value."""

    grad_norm: float = 0.0
    """Last gradient norm value (after clipping)."""

    learning_rate: float = 0.0
    """Current learning rate (from scheduler)."""

    step_count: int = 0
    """Total steps recorded for this phase."""

    epoch: int = 0
    """Current epoch within this phase."""

    _ema_initialized: bool = field(default=False, repr=False)
    """Whether the EMA has been initialized with a first value."""

    def update(
        self,
        loss: float,
        grad_norm: float = 0.0,
        learning_rate: float = 0.0,
        ema_decay: float = 0.99,
    ) -> None:
        """Record one training step's metrics.

        Updates the EMA loss with exponential smoothing. On the first
        call, initializes EMA to the raw loss value (no smoothing).

        Args:
            loss: Raw loss value for this step.
            grad_norm: Gradient norm after clipping.
            learning_rate: Current learning rate from scheduler.
            ema_decay: EMA decay factor (0.99 = slow smoothing, 0.9 = fast).
        """
        self.loss_raw = loss
        self.grad_norm = grad_norm
        self.learning_rate = learning_rate
        self.step_count += 1

        if not self._ema_initialized:
            self.loss_ema = loss
            self._ema_initialized = True
        else:
            self.loss_ema = ema_decay * self.loss_ema + (1 - ema_decay) * loss

    def to_dict(self, prefix: str = "") -> dict[str, float]:
        """Export metrics as a flat dict for logger consumption.

        Args:
            prefix: Optional prefix for all keys (e.g. 'train/').

        Returns:
            Dict with float values for each metric.
        """
        return {
            f"{prefix}loss_ema": self.loss_ema,
            f"{prefix}loss_raw": self.loss_raw,
            f"{prefix}grad_norm": self.grad_norm,
            f"{prefix}learning_rate": self.learning_rate,
            f"{prefix}step": float(self.step_count),
            f"{prefix}epoch": float(self.epoch),
        }

    def reset(self) -> None:
        """Reset all metrics for a new phase or run."""
        self.loss_ema = 0.0
        self.loss_raw = 0.0
        self.grad_norm = 0.0
        self.learning_rate = 0.0
        self.step_count = 0
        self.epoch = 0
        self._ema_initialized = False


class MetricsTracker:
    """Manages per-phase metrics for the training loop.

    Creates and tracks PhaseMetrics for each phase. The training loop
    calls update() on the current phase; loggers call get_metrics()
    to read the latest values.
    """

    def __init__(self) -> None:
        self._phases: dict[PhaseType, PhaseMetrics] = {}
        self._current_phase: PhaseType | None = None

    def start_phase(self, phase_type: PhaseType) -> PhaseMetrics:
        """Start tracking a new phase.

        Creates a fresh PhaseMetrics for this phase type. If a previous
        PhaseMetrics existed for this type, it is replaced.

        Args:
            phase_type: The phase to start tracking.

        Returns:
            The new PhaseMetrics instance.
        """
        metrics = PhaseMetrics(phase_type=phase_type)
        self._phases[phase_type] = metrics
        self._current_phase = phase_type
        return metrics

    def update(
        self,
        loss: float,
        grad_norm: float = 0.0,
        learning_rate: float = 0.0,
        ema_decay: float = 0.99,
    ) -> None:
        """Record metrics for the current phase.

        Args:
            loss: Raw loss value.
            grad_norm: Gradient norm after clipping.
            learning_rate: Current learning rate.
            ema_decay: EMA smoothing factor.

        Raises:
            RuntimeError: If no phase has been started.
        """
        if self._current_phase is None:
            raise RuntimeError(
                "No active phase. Call start_phase() before update()."
            )
        self._phases[self._current_phase].update(
            loss=loss,
            grad_norm=grad_norm,
            learning_rate=learning_rate,
            ema_decay=ema_decay,
        )

    def set_epoch(self, epoch: int) -> None:
        """Update the epoch counter for the current phase.

        Args:
            epoch: Current epoch number (1-based).
        """
        if self._current_phase is not None and self._current_phase in self._phases:
            self._phases[self._current_phase].epoch = epoch

    def get_current(self) -> PhaseMetrics | None:
        """Get the current phase's metrics.

        Returns:
            PhaseMetrics for the current phase, or None if no phase is active.
        """
        if self._current_phase is None:
            return None
        return self._phases.get(self._current_phase)

    def get_phase(self, phase_type: PhaseType) -> PhaseMetrics | None:
        """Get metrics for a specific phase.

        Args:
            phase_type: The phase to query.

        Returns:
            PhaseMetrics if the phase has been tracked, None otherwise.
        """
        return self._phases.get(phase_type)

    def get_all_metrics(self) -> dict[str, float]:
        """Export all phase metrics as a flat dict.

        Keys are prefixed with the phase type name (e.g. 'unified/loss_ema',
        'high_noise/grad_norm').

        Returns:
            Dict with all metrics from all tracked phases.
        """
        result: dict[str, float] = {}
        for phase_type, metrics in self._phases.items():
            prefix = f"{phase_type.value}/"
            result.update(metrics.to_dict(prefix=prefix))
        return result

    @property
    def current_phase(self) -> PhaseType | None:
        """The currently active phase type."""
        return self._current_phase

    @property
    def tracked_phases(self) -> list[PhaseType]:
        """List of phase types that have been tracked."""
        return list(self._phases.keys())


class RunTimer:
    """Wall-clock timer for training phases and total run.

    Uses time.perf_counter() for highest-resolution monotonic clock on
    Windows. Tracks per-phase durations and total run elapsed time.

    Usage:
        timer = RunTimer()
        timer.start_run()

        timer.start_phase("full_noise")
        # ... training ...
        elapsed = timer.end_phase("full_noise")  # returns seconds

        total = timer.total_elapsed()
    """

    def __init__(self) -> None:
        self._run_start: float = 0.0
        self._phase_start: float = 0.0
        self._phase_times: dict[str, float] = {}

    def start_run(self) -> None:
        """Mark the start of the training run."""
        self._run_start = time.perf_counter()

    def start_phase(self, phase_name: str) -> None:
        """Mark the start of a training phase.

        Args:
            phase_name: Name of the phase (e.g. 'unified', 'high_noise').
        """
        self._phase_start = time.perf_counter()

    def end_phase(self, phase_name: str) -> float:
        """Mark the end of a training phase and record elapsed time.

        Args:
            phase_name: Name of the phase (must match start_phase call).

        Returns:
            Elapsed wall-clock seconds for this phase.
        """
        elapsed = time.perf_counter() - self._phase_start
        self._phase_times[phase_name] = elapsed
        return elapsed

    def total_elapsed(self) -> float:
        """Return total wall-clock seconds since start_run().

        Returns:
            Elapsed seconds since start_run() was called.
        """
        return time.perf_counter() - self._run_start

    @property
    def phase_times(self) -> dict[str, float]:
        """Recorded phase durations in seconds (name -> elapsed)."""
        return dict(self._phase_times)
