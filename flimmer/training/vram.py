"""GPU VRAM tracking utility for training observability.

Samples GPU memory usage at configurable step intervals and tracks peak
allocation. All values are reported in GB for human readability.

GPU-safe: returns empty/zero metrics when CUDA is not available, so the
utility can be instantiated and called on CPU-only machines without error.

Why periodic sampling instead of every-step: W&B logging has serialization
and network overhead that compounds at high frequency. Sampling every 50
steps (~1 sample per minute at typical training speed) gives a useful
memory curve without impacting training throughput.
"""

from __future__ import annotations


class VRAMTracker:
    """Tracks GPU memory usage during training.

    Samples VRAM at configurable step intervals and records peak usage.
    All values are in GB. Returns metrics dicts ready for W&B logging.

    GPU-safe: returns empty/zero metrics when CUDA is not available.

    Args:
        device: CUDA device index to monitor (default 0).
        sample_every_n_steps: How often to sample VRAM. Only samples when
            global_step is a multiple of this value.
    """

    def __init__(self, device: int = 0, sample_every_n_steps: int = 50) -> None:
        self._device = device
        self._sample_interval = sample_every_n_steps
        self._samples: list[float] = []

    def sample(self, global_step: int) -> dict[str, float] | None:
        """Sample VRAM if at the configured interval.

        Only performs a sample when global_step is a multiple of the
        configured interval. This keeps the overhead minimal during
        the inner training loop.

        Args:
            global_step: Current global training step counter.

        Returns:
            Dict with keys 'system/vram_allocated_gb' and
            'system/vram_reserved_gb' if at interval and CUDA is
            available, otherwise None.
        """
        if global_step % self._sample_interval != 0:
            return None

        try:
            import torch
            if not torch.cuda.is_available():
                return None

            allocated = torch.cuda.memory_allocated(self._device) / (1024**3)
            reserved = torch.cuda.memory_reserved(self._device) / (1024**3)
            self._samples.append(allocated)

            return {
                "system/vram_allocated_gb": allocated,
                "system/vram_reserved_gb": reserved,
            }
        except (ImportError, RuntimeError):
            # torch not installed or CUDA error — graceful fallback
            return None

    def peak(self) -> float:
        """Return peak allocated VRAM in GB via torch.cuda.max_memory_allocated().

        Returns 0.0 when CUDA is not available or torch is not installed.
        """
        try:
            import torch
            if not torch.cuda.is_available():
                return 0.0
            return torch.cuda.max_memory_allocated(self._device) / (1024**3)
        except (ImportError, RuntimeError):
            return 0.0

    def reset_peak(self) -> None:
        """Reset PyTorch's peak memory tracking.

        Calls torch.cuda.reset_peak_memory_stats() so subsequent peak()
        calls only reflect memory usage after this reset. Safe to call
        when CUDA is not available (no-op).
        """
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(self._device)
        except (ImportError, RuntimeError):
            pass

    @property
    def samples(self) -> list[float]:
        """All recorded allocated VRAM samples in GB."""
        return list(self._samples)
