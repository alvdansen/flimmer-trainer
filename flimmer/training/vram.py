"""GPU VRAM tracking and pre-flight estimation.

Two complementary tools for VRAM management:

1. **VRAMTracker** (runtime): Samples GPU memory at configurable step intervals
   during training. Returns metrics dicts ready for W&B logging.

2. **VRAMEstimator** (pre-flight): Estimates VRAM usage from training config
   parameters WITHOUT requiring a GPU. Uses component-based formulas derived
   from Wan 14B architecture constants and empirical per-block weight data.

Why pre-flight estimation: Users should never waste 10+ minutes loading a
14B model only to OOM. The estimator runs instantly from config YAML, showing
a component breakdown and warning when estimated usage exceeds GPU capacity.

GPU-safe: Both tools work on CPU-only machines without error. VRAMTracker
returns empty/zero metrics; VRAMEstimator returns estimates based on formulas.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


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


# ──────────────────────────────────────────────────────────────────────
# Pre-flight VRAM Estimation
# ──────────────────────────────────────────────────────────────────────

# Total model parameter count for Wan 14B (one expert).
_TOTAL_PARAMS: int = 14_000_000_000

# Bytes per parameter for each precision mode.
_BYTES_PER_PARAM: dict[str, float] = {
    "bf16": 2.0,
    "fp16": 2.0,
    "fp32": 4.0,
    "fp8": 1.0,
    "fp8_scaled": 0.5,
}
"""Bytes per model weight parameter at each precision."""

# Empirical per-block SWAP weight sizes from block_swap.py measurements.
# Block swap only moves Linear weights (not LayerNorm, not buffers), so these
# are smaller than the full per-block parameter weight. These are what actually
# gets offloaded to CPU.
_PER_BLOCK_SWAP_MB: dict[str, float] = {
    "bf16": 350.0,
    "fp16": 350.0,
    "fp32": 700.0,
    "fp8": 175.0,
    "fp8_scaled": 88.0,
}
"""Per-block swappable weight in MB. Source: block_swap.py docstring."""

# Optimizer state sizes in bytes per trainable parameter.
_OPTIMIZER_BYTES_PER_PARAM: dict[str, int] = {
    "adamw": 8,          # 2 states (m, v) x fp32 = 8 bytes
    "adamw8bit": 2,      # 2 states x 8-bit = 2 bytes
    "adafactor": 4,      # ~1 state row/col factors
    "came": 4,           # ~1 state (momentum)
    "prodigy": 8,        # similar to adamw
    "ademamix": 12,      # 3 EMA states x fp32
    "schedule_free_adamw": 8,
    "cpu_offload": 0,    # states on CPU, 0 on GPU
    "adam_mini": 4,       # ~50% of adamw
}
"""GPU-side optimizer state bytes per trainable parameter."""

# Average sum of (in_features + out_features) across all LoRA target modules
# in one transformer block.  Derived from Wan architecture:
#   T2V (10 modules): attn1 QKV(5120+5120)*3 + out(5120+5120) +
#                     attn2 QKV(4096+5120)*3 + out(5120+5120) +
#                     ffn gate(5120+13824) + out(13824+5120) = 86,784
#   I2V (12 modules): T2V total + add_k_proj(4096+5120) + add_v_proj(4096+5120) = 105,216
_T2V_AVG_IO_SUM: int = 86_784
_I2V_AVG_IO_SUM: int = 105_216

# Architecture constants (imported values for reference, kept local to avoid
# circular import risk with constants module).
_NUM_BLOCKS: int = 40
_HIDDEN_DIM: int = 5120
_NUM_HEADS: int = 40
_VAE_SPATIAL_COMPRESS: int = 8
_VAE_TEMPORAL_COMPRESS: int = 4


@dataclass
class VRAMEstimate:
    """Component-wise VRAM estimate for Wan 14B LoRA training.

    Each field represents one category of GPU memory usage in GB.
    The total_gb property sums all components and subtracts block swap
    savings to produce the final estimate.

    Why component breakdown: Users need to understand WHAT is eating VRAM
    and WHICH lever to pull. A single number is not actionable.
    """

    model_weights_gb: float
    block_swap_savings_gb: float
    lora_params_gb: float
    gradients_gb: float
    optimizer_states_gb: float
    activations_gb: float
    overhead_gb: float = 1.5

    @property
    def total_gb(self) -> float:
        """Total estimated VRAM in GB (all components minus block swap savings)."""
        return (
            self.model_weights_gb
            + self.lora_params_gb
            + self.gradients_gb
            + self.optimizer_states_gb
            + self.activations_gb
            + self.overhead_gb
            - self.block_swap_savings_gb
        )

    def breakdown_table(self) -> str:
        """Format the estimate as a human-readable breakdown table.

        Returns a multi-line string with each component on its own line,
        suitable for logging or CLI output.
        """
        lines = [
            f"  Model weights:       {self.model_weights_gb:6.1f} GB",
            f"  Block swap:         -{self.block_swap_savings_gb:5.1f} GB",
            f"  LoRA parameters:     {self.lora_params_gb:6.1f} GB",
            f"  Gradients:           {self.gradients_gb:6.1f} GB",
            f"  Optimizer states:    {self.optimizer_states_gb:6.1f} GB",
            f"  Activations:         {self.activations_gb:6.1f} GB",
            f"  Overhead:            {self.overhead_gb:6.1f} GB",
            f"  ─────────────────────────────",
            f"  TOTAL:               {self.total_gb:6.1f} GB",
        ]
        return "\n".join(lines)


class VRAMEstimator:
    """Pre-flight VRAM estimator for Wan 14B LoRA training.

    Estimates GPU memory usage from training config parameters using
    component-based formulas. Runs on CPU — no GPU required.

    The estimate is approximate but conservative. It accounts for:
    - Model weights at the configured precision
    - Block swap savings (blocks offloaded to CPU)
    - LoRA adapter parameters (always bf16)
    - Gradients for trainable parameters (fp32)
    - Optimizer states (varies by optimizer type)
    - Activation memory (depends on resolution, frame count, batch size)
    - PyTorch overhead (CUDA context, allocator fragmentation)

    Args:
        precision: Base model weight precision (bf16, fp8, fp8_scaled).
        blocks_to_swap: Number of transformer blocks offloaded to CPU.
        rank: LoRA rank.
        optimizer_type: Optimizer name (adamw, adamw8bit, cpu_offload, etc.).
        is_i2v: Whether this is an I2V model (more LoRA targets, higher activations).
        batch_size: Training batch size.
        resolution: Video resolution height in pixels (default 480).
        frame_count: Number of video frames (default 49).
        aspect_ratio: Width/height ratio for computing latent dimensions (default 16/9).
    """

    def __init__(
        self,
        precision: str = "bf16",
        blocks_to_swap: int = 0,
        rank: int = 16,
        optimizer_type: str = "adamw8bit",
        is_i2v: bool = False,
        batch_size: int = 1,
        resolution: int = 480,
        frame_count: int = 49,
        aspect_ratio: float = 16 / 9,
    ) -> None:
        self._precision = precision
        self._blocks_to_swap = blocks_to_swap
        self._rank = rank
        self._optimizer_type = optimizer_type
        self._is_i2v = is_i2v
        self._batch_size = batch_size
        self._resolution = resolution
        self._frame_count = frame_count
        self._aspect_ratio = aspect_ratio

    @classmethod
    def from_config(cls, config: object) -> VRAMEstimator:
        """Create a VRAMEstimator from a FlimmerTrainingConfig instance.

        Extracts all needed fields from the config. If the data config path
        exists, attempts to load resolution and frame_count from it.

        Args:
            config: FlimmerTrainingConfig instance.

        Returns:
            Configured VRAMEstimator.
        """
        training = getattr(config, "training", None)
        lora = getattr(config, "lora", None)
        optimizer = getattr(config, "optimizer", None)
        model = getattr(config, "model", None)

        precision = getattr(training, "base_model_precision", "bf16") if training else "bf16"
        blocks_to_swap = getattr(training, "blocks_to_swap", 0) if training else 0
        batch_size = getattr(training, "batch_size", 1) if training else 1
        rank = getattr(lora, "rank", 16) if lora else 16
        optimizer_type = getattr(optimizer, "type", "adamw8bit") if optimizer else "adamw8bit"

        variant = getattr(model, "variant", "2.2_t2v") if model else "2.2_t2v"
        is_i2v = "i2v" in variant.lower() if variant else False

        # Try to load resolution/frame_count from data config
        resolution = 480
        frame_count = 49
        data_config_path = getattr(config, "data_config", None)
        if data_config_path:
            try:
                from pathlib import Path
                import yaml

                data_path = Path(data_config_path)
                if data_path.exists():
                    with open(data_path) as f:
                        data = yaml.safe_load(f)
                    if data:
                        video = data.get("video", {})
                        resolution = video.get("resolution", resolution)
                        frame_count = video.get("max_frames", video.get("frame_count", frame_count))
            except Exception:
                pass  # Fall back to defaults

        return cls(
            precision=precision,
            blocks_to_swap=blocks_to_swap,
            rank=rank,
            optimizer_type=optimizer_type,
            is_i2v=is_i2v,
            batch_size=batch_size,
            resolution=resolution,
            frame_count=frame_count,
        )

    def estimate(self) -> VRAMEstimate:
        """Compute the component-wise VRAM estimate.

        Uses empirical per-block weight sizes and architecture constants
        to estimate each VRAM component independently.

        Returns:
            VRAMEstimate with all components populated.
        """
        # 1. Model weights — total params * bytes per param
        bpp = _BYTES_PER_PARAM.get(self._precision, 2.0)
        model_weights_gb = _TOTAL_PARAMS * bpp / (1024**3)

        # 2. Block swap savings — use empirical per-block swap sizes
        # Block swap only offloads Linear weights (not norms/buffers),
        # so the savings use the measured per-block swap size, not total/40.
        per_block_swap_mb = _PER_BLOCK_SWAP_MB.get(self._precision, 350.0)
        block_swap_savings_gb = self._blocks_to_swap * per_block_swap_mb / 1024

        # 3. LoRA parameters (always bf16 = 2 bytes per param)
        avg_io_sum = _I2V_AVG_IO_SUM if self._is_i2v else _T2V_AVG_IO_SUM
        lora_params = self._rank * avg_io_sum * _NUM_BLOCKS
        lora_params_gb = lora_params * 2 / (1024**3)  # bf16

        # 4. Gradients (fp32 = 4 bytes per param, same count as LoRA)
        gradients_gb = lora_params * 4 / (1024**3)

        # 5. Optimizer states
        opt_bytes = _OPTIMIZER_BYTES_PER_PARAM.get(self._optimizer_type, 8)
        optimizer_states_gb = lora_params * opt_bytes / (1024**3)

        # 6. Activations
        activations_gb = self._estimate_activations()

        return VRAMEstimate(
            model_weights_gb=model_weights_gb,
            block_swap_savings_gb=block_swap_savings_gb,
            lora_params_gb=lora_params_gb,
            gradients_gb=gradients_gb,
            optimizer_states_gb=optimizer_states_gb,
            activations_gb=activations_gb,
        )

    def _estimate_activations(self) -> float:
        """Estimate activation memory in GB.

        Uses latent dimensions derived from resolution/frame_count via
        VAE compression ratios, combined with the transformer hidden dim.

        With gradient checkpointing (always enabled for Wan 14B training),
        only ~2 blocks worth of activations are held at any time. With
        flash attention (standard for Wan training), the O(n^2) attention
        maps are NOT stored -- only the hidden states, intermediate FFN
        tensors, and recomputation buffers consume VRAM.

        The formula uses a practical per-token activation budget derived
        from empirical observations: each token in the sequence needs
        approximately hidden_dim * 20 bytes of activation storage per
        checkpointed block (hidden states + FFN intermediates + attention
        output buffers in bf16/fp32 mix).

        I2V adds a 1.3x multiplier for the 36-channel input processing
        overhead in the initial patch embedding and first few layers.
        """
        # Compute latent spatial dimensions
        width = int(self._resolution * self._aspect_ratio)
        latent_h = math.ceil(self._resolution / _VAE_SPATIAL_COMPRESS)
        latent_w = math.ceil(width / _VAE_SPATIAL_COMPRESS)
        latent_t = (self._frame_count - 1) // _VAE_TEMPORAL_COMPRESS + 1

        # Sequence length for attention
        seq_len = latent_h * latent_w * latent_t

        # Activation per block (with flash attention + gradient checkpointing):
        # With flash attention, O(n^2) attention maps are NOT stored.
        # Remaining activations per block:
        # - Hidden states: seq_len * hidden_dim * 2 bytes (bf16)
        # - Attention Q/K/V projections: seq_len * hidden_dim * 2 * 3
        # - Cross-attention buffers: smaller due to shorter text seq
        # Calibrated against empirical ~42 GB for bf16 720p/49f T2V.
        _BYTES_PER_TOKEN_PER_BLOCK = _HIDDEN_DIM * 8
        activation_per_block_gb = seq_len * _BYTES_PER_TOKEN_PER_BLOCK / (1024**3)

        # With gradient checkpointing: ~2 blocks worth at any time
        activation_gb = activation_per_block_gb * 2 * self._batch_size

        # I2V has ~30% more activation overhead from 36-channel input processing
        if self._is_i2v:
            activation_gb *= 1.3

        return activation_gb

    def log_estimate(self, estimate: VRAMEstimate) -> None:
        """Log the VRAM breakdown via the module logger at INFO level.

        Used by training-start integration to always show the estimate
        in the training logs.

        Args:
            estimate: The computed VRAMEstimate.
        """
        logger.info("VRAM Estimate:\n%s", estimate.breakdown_table())

    def warn_if_over(
        self,
        estimate: VRAMEstimate,
        gpu_memory_gb: float | None = None,
    ) -> None:
        """Warn if the estimate exceeds available GPU memory.

        Detects GPU memory automatically if not provided. If no GPU is
        available and no explicit value is given, skips the warning.

        Args:
            estimate: The computed VRAMEstimate.
            gpu_memory_gb: Explicit GPU memory in GB (overrides auto-detect).
        """
        available = gpu_memory_gb if gpu_memory_gb is not None else detect_gpu_memory_gb()
        if available is None:
            return

        if estimate.total_gb > available:
            config_summary = {
                "precision": self._precision,
                "blocks_to_swap": self._blocks_to_swap,
                "resolution": self._resolution,
            }
            warning = format_vram_warning(estimate, available, config_summary)
            logger.warning(warning)

    def print_report(
        self,
        estimate: VRAMEstimate,
        gpu_memory_gb: float | None = None,
    ) -> None:
        """Print a full VRAM report to stdout (for CLI use).

        Shows the config summary, component breakdown, and GPU status.

        Args:
            estimate: The computed VRAMEstimate.
            gpu_memory_gb: Explicit GPU memory in GB (overrides auto-detect).
        """
        mode = "I2V" if self._is_i2v else "T2V"
        header = (
            f"VRAM Estimate for Wan 2.2 {mode} "
            f"({self._precision}, rank {self._rank}, "
            f"{self._resolution}p, {self._frame_count} frames)"
        )
        print(header)
        print("=" * len(header))
        print(estimate.breakdown_table())

        # GPU status
        available = gpu_memory_gb if gpu_memory_gb is not None else detect_gpu_memory_gb()
        if available is not None:
            if estimate.total_gb <= available:
                headroom = available - estimate.total_gb
                print(f"\n  GPU: {available:.0f} GB -- OK ({headroom:.1f} GB headroom)")
            else:
                excess = estimate.total_gb - available
                print(f"\n  GPU: {available:.0f} GB -- WARNING: exceeds by {excess:.1f} GB")
                config_summary = {
                    "precision": self._precision,
                    "blocks_to_swap": self._blocks_to_swap,
                    "resolution": self._resolution,
                }
                suggestions = _build_suggestions(config_summary)
                if suggestions:
                    print("\n  Suggestions:")
                    for s in suggestions:
                        print(f"    * {s}")
        else:
            print("\n  GPU: not detected (use --gpu-memory to specify)")


def detect_gpu_memory_gb() -> float | None:
    """Detect total GPU memory in GB.

    Uses torch.cuda.mem_get_info() to get the total memory of GPU 0.
    Returns None if torch is not installed or CUDA is not available.

    Why total and not free: The estimate compares against total GPU capacity.
    Other processes using GPU memory is a runtime concern, not a config
    planning concern.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        _free, total = torch.cuda.mem_get_info(0)
        return total / (1024**3)
    except (ImportError, RuntimeError, ModuleNotFoundError):
        return None


def _build_suggestions(config_summary: dict) -> list[str]:
    """Build a list of actionable suggestions based on current config.

    Checks which optimizations are NOT yet applied and suggests them.
    Only suggests things that would actually help.

    Args:
        config_summary: Dict with keys 'precision', 'blocks_to_swap', 'resolution'.

    Returns:
        List of suggestion strings.
    """
    suggestions: list[str] = []

    if config_summary.get("blocks_to_swap", 0) == 0:
        suggestions.append("Add blocks_to_swap: 20 (saves ~7 GB at bf16, ~3.5 GB at fp8)")

    precision = config_summary.get("precision", "bf16")
    if precision in ("bf16", "fp16", "fp32"):
        suggestions.append("Use base_model_precision: fp8 (saves ~14 GB)")

    resolution = config_summary.get("resolution", 480)
    if resolution >= 720:
        suggestions.append("Reduce resolution to 480p (saves ~40% activation VRAM)")

    return suggestions


def format_vram_warning(
    estimate: VRAMEstimate,
    available_gb: float,
    config_summary: dict,
) -> str:
    """Format a VRAM warning with actionable suggestions.

    Called when estimated VRAM exceeds available GPU memory.
    Includes specific config changes the user can make to reduce usage.

    Args:
        estimate: The computed VRAMEstimate.
        available_gb: Available GPU memory in GB.
        config_summary: Dict with current config values for suggestion logic.

    Returns:
        Formatted warning string.
    """
    excess = estimate.total_gb - available_gb
    msg = (
        f"Estimated VRAM: {estimate.total_gb:.1f} GB exceeds "
        f"GPU capacity of {available_gb:.0f} GB (by {excess:.1f} GB)."
    )

    suggestions = _build_suggestions(config_summary)
    if suggestions:
        msg += "\n  Try: " + " | ".join(suggestions)

    return msg
