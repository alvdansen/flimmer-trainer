"""LtxModelBackend — implements the ModelBackend protocol for LTX-2.3.

Handles:
    - Loading the LTX-2.3 transformer via ltx-core's SingleGPUModelBuilder
    - LoRA adapter placement via PEFT target modules
    - Optional int8/fp8 quantization via optimum-quanto
    - Preparing Modality-based inputs for the LTX forward pass
    - Running the forward pass
    - Gradient checkpointing setup

LTX-2.3 uses a Modality-based forward API (not standard diffusers),
with separate video, text, and audio modality objects. The audio
modality is set to None for the video-only MVP.

Requires: torch, ltx-core, peft (the 'ltx' dependency group).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from flimmer.training.ltx.constants import (
    LTX_VAE_SPATIAL_COMPRESSION,
    LTX_VAE_TEMPORAL_COMPRESSION,
)


# ---------------------------------------------------------------------------
# LTX Noise Schedule
# ---------------------------------------------------------------------------

class LtxNoiseSchedule:
    """NoiseSchedule for LTX-2.3 flow matching with shifted logit-normal sampling.

    LTX-2.3 uses a sequence-length-dependent shift for timestep sampling:
        shift = lerp(0.95, 2.05, clamp((seq_len - 1024) / 3072, 0, 1))

    This is fundamentally different from Wan's uniform/shifted sampling,
    so it gets its own class rather than extending FlowMatchingSchedule.

    The schedule still uses flow matching interpolation:
        noisy = (1 - t) * clean + t * noise
        target (velocity) = noise - clean

    Args:
        num_timesteps: Total discrete timesteps (1000).
        seq_len: Latent sequence length, used to compute the shift parameter.
            If None, uses a default shift of 1.0 (no shift).
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        seq_len: int | None = None,
    ) -> None:
        self._num_timesteps = num_timesteps
        self._seq_len = seq_len

    @property
    def num_timesteps(self) -> int:
        """Total number of discrete timesteps."""
        return self._num_timesteps

    def _compute_shift(self, seq_len: int | None = None) -> float:
        """Compute the logit-normal shift based on sequence length.

        shift = lerp(0.95, 2.05, clamp((seq_len - 1024) / 3072, 0, 1))

        Args:
            seq_len: Override sequence length. Uses constructor value if None.

        Returns:
            Shift parameter in [0.95, 2.05].
        """
        sl = seq_len if seq_len is not None else self._seq_len
        if sl is None:
            return 1.0  # neutral shift
        t = max(0.0, min(1.0, (sl - 1024) / 3072))
        return 0.95 + t * (2.05 - 0.95)

    def sample_timesteps(
        self,
        batch_size: int,
        strategy: str = "logit_normal_shifted",
        flow_shift: float = 1.0,
        generator: Any = None,
    ) -> np.ndarray:
        """Sample timesteps using shifted logit-normal distribution.

        10% of samples use uniform fallback to prevent mode collapse.
        The remaining 90% use logit-normal with sequence-length-dependent shift.

        Args:
            batch_size: Number of timesteps to sample.
            strategy: Ignored (always uses shifted logit-normal). Accepted
                for protocol compatibility.
            flow_shift: Ignored (shift is computed from sequence length).
                Accepted for protocol compatibility.
            generator: Numpy random Generator for reproducibility.

        Returns:
            Float64 array of shape [batch_size] with values in (0, 1).
        """
        rng = generator if generator is not None else np.random.default_rng()

        shift = self._compute_shift()
        result = np.empty(batch_size, dtype=np.float64)

        # 10% uniform fallback to prevent mode collapse
        uniform_count = max(1, batch_size // 10)
        normal_count = batch_size - uniform_count

        # Uniform samples
        result[:uniform_count] = rng.uniform(
            low=1e-5, high=1.0 - 1e-5, size=uniform_count
        )

        # Shifted logit-normal samples
        if normal_count > 0:
            z = rng.normal(loc=shift, scale=1.0, size=normal_count)
            t = 1.0 / (1.0 + np.exp(-z))  # sigmoid
            result[uniform_count:] = np.clip(t, 1e-5, 1.0 - 1e-5)

        return result

    def compute_noisy_latent(
        self,
        clean: np.ndarray,
        noise: np.ndarray,
        timesteps: np.ndarray,
    ) -> np.ndarray:
        """Compute noisy latents: (1-t)*clean + t*noise."""
        t = self._broadcast_timesteps(timesteps, clean.ndim)
        return (1.0 - t) * clean + t * noise

    def compute_target(
        self,
        clean: np.ndarray,
        noise: np.ndarray,
        timesteps: np.ndarray,
    ) -> np.ndarray:
        """Compute velocity target: noise - clean."""
        return noise - clean

    def get_signal_to_noise_ratio(self, timesteps: np.ndarray) -> np.ndarray:
        """Compute SNR: (1-t)/t."""
        return (1.0 - timesteps) / timesteps

    @staticmethod
    def _broadcast_timesteps(timesteps: np.ndarray, ndim: int) -> np.ndarray:
        """Reshape timesteps [B] to [B, 1, 1, ...] for broadcasting."""
        if timesteps.ndim == 0:
            return timesteps
        shape = (-1,) + (1,) * (ndim - 1)
        return timesteps.reshape(shape)


# ---------------------------------------------------------------------------
# LTX Model Backend
# ---------------------------------------------------------------------------

class LtxModelBackend:
    """ModelBackend implementation for LTX-2.3.

    Uses ltx-core's SingleGPUModelBuilder for model loading and PEFT for
    LoRA adapter placement. Supports optional quantization via optimum-quanto.

    Args:
        model_id: Human-readable identifier (e.g. 'ltx-2.3-t2v').
        model_path: Path to model directory or HuggingFace ID.
        is_moe: Whether this is an MoE model (always False for LTX).
        is_i2v: Whether this is an I2V model (always False for LTX MVP).
        in_channels: Input channel count (128 for LTX).
        num_blocks: Number of transformer blocks (48).
        lora_targets: List of module suffixes for LoRA placement.
        hf_repo: HuggingFace repository ID for model config.
        quantization: Quantization mode ('int8', 'fp8', or None).
    """

    def __init__(
        self,
        model_id: str,
        model_path: str | None = None,
        is_moe: bool = False,
        is_i2v: bool = False,
        in_channels: int = 128,
        num_blocks: int = 48,
        lora_targets: list[str] | None = None,
        hf_repo: str | None = None,
        quantization: str | None = None,
    ) -> None:
        self._model_id = model_id
        self._model_path = model_path or ""
        self._is_moe = is_moe
        self._is_i2v = is_i2v
        self._in_channels = in_channels
        self._num_blocks = num_blocks
        self._lora_targets = lora_targets or []
        self._hf_repo = hf_repo
        self._quantization = quantization
        self._noise_schedule: LtxNoiseSchedule | None = None

    # ── Properties ──────────────────────────────────────────────────────

    @property
    def model_id(self) -> str:
        """Human-readable model identifier."""
        return self._model_id

    @property
    def supports_moe(self) -> bool:
        """LTX-2.3 is a single transformer — no MoE."""
        return False

    @property
    def supports_reference_image(self) -> bool:
        """LTX-2.3 MVP is text-to-video only."""
        return False

    @property
    def is_quantized(self) -> bool:
        """Whether quantization is enabled."""
        return self._quantization is not None

    # ── Model loading ───────────────────────────────────────────────────

    def load_model(self, config: Any, expert: str | None = None) -> Any:
        """Load the LTX-2.3 transformer via ltx-core.

        Uses SingleGPUModelBuilder from ltx-core for model construction.
        Optionally applies LoRA and quantization.

        Args:
            config: FlimmerTrainingConfig instance.
            expert: Ignored (LTX has no expert switching).

        Returns:
            The LTX transformer model, ready for training.

        Raises:
            ImportError: If ltx-core is not installed.
        """
        from ltx_core import SingleGPUModelBuilder  # type: ignore[import-untyped]

        builder = SingleGPUModelBuilder(self._model_path or self._hf_repo)

        # Apply LoRA if targets are configured
        lora_rank = 32
        lora_alpha = 32
        if hasattr(config, "lora"):
            lora_rank = getattr(config.lora, "rank", 32)
            lora_alpha = getattr(config.lora, "alpha", 32)

        if self._lora_targets:
            builder = builder.lora(
                rank=lora_rank,
                alpha=lora_alpha,
                target_modules=self._lora_targets,
            )

        # Apply quantization if configured
        if self._quantization:
            builder = builder.quantize(self._quantization)

        model = builder.build()
        return model

    # ── Training interface ──────────────────────────────────────────────

    def prepare_model_inputs(
        self,
        batch: dict[str, Any],
        timesteps: Any,
        noisy_latents: Any,
    ) -> dict[str, Any]:
        """Prepare inputs for the LTX forward pass.

        Constructs Modality objects for video and text, wrapping them in
        a dict with underscore-prefixed keys. The training loop passes
        this dict to forward() without knowing about Modality internals.

        Audio is set to None for the video-only MVP.

        Args:
            batch: Training batch dict with 'text_embeddings' or
                'encoder_hidden_states'.
            timesteps: Timestep tensor for the current batch.
            noisy_latents: Noisy video latent tensor.

        Returns:
            Dict with keys: '_video', '_text', '_audio',
            'hidden_states', 'timesteps'.
        """
        from ltx_core import Modality  # type: ignore[import-untyped]

        # Build video modality
        video_modality = Modality(data=noisy_latents, timesteps=timesteps)

        # Build text modality — support both key names
        text_data = batch.get("text_embeddings") or batch.get(
            "encoder_hidden_states"
        )
        text_modality = Modality(data=text_data) if text_data is not None else None

        return {
            "_video": video_modality,
            "_text": text_modality,
            "_audio": None,  # Audio skipped in video-only MVP
            "hidden_states": noisy_latents,
            "timesteps": timesteps,
        }

    def forward(self, model: Any, **inputs: Any) -> Any:
        """Run the LTX forward pass.

        Unpacks Modality objects from the inputs dict and calls the
        model's forward method with video, text, and audio arguments.

        Args:
            model: The LTX transformer model.
            **inputs: Dict from prepare_model_inputs().

        Returns:
            Model prediction tensor (velocity).
        """
        video_mod = inputs.get("_video")
        text_mod = inputs.get("_text")
        audio_mod = inputs.get("_audio")

        output = model(
            video=video_mod,
            audio=audio_mod,
            text=text_mod,
            perturbations=None,
        )

        # Extract the prediction tensor from the output
        if hasattr(output, "sample"):
            return output.sample
        return output

    def setup_gradient_checkpointing(self, model: Any) -> None:
        """Enable gradient checkpointing on the LTX model.

        Reduces GPU memory usage during training at the cost of
        recomputing activations during the backward pass.

        Args:
            model: The LTX transformer model.
        """
        if hasattr(model, "enable_gradient_checkpointing"):
            model.enable_gradient_checkpointing()
        elif hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()

    def get_noise_schedule(self) -> LtxNoiseSchedule:
        """Return the LTX noise schedule.

        Creates a single LtxNoiseSchedule instance on first call
        and caches it for subsequent calls.

        Returns:
            LtxNoiseSchedule configured for this model.
        """
        if self._noise_schedule is None:
            self._noise_schedule = LtxNoiseSchedule(num_timesteps=1000)
        return self._noise_schedule

    def get_lora_target_modules(self) -> list[str]:
        """Return the list of LoRA target module suffixes."""
        return list(self._lora_targets)

    def get_expert_mask(
        self,
        timesteps: Any,
        boundary_ratio: float | None = None,
    ) -> tuple[Any, Any]:
        """LTX has no expert routing — returns (None, None)."""
        return (None, None)

    def switch_expert(self, expert: str) -> None:
        """LTX has no expert switching — no-op."""
        pass
