"""Training protocols — the contracts Phase 8 must implement.

Three protocols define what a model backend must provide:

    NoiseSchedule       — pure math: timestep sampling, noisy latent computation
    ModelBackend        — training-time: model loading, forward pass, LoRA targets
    InferencePipeline   — sampling-time: full denoising loop for preview generation

Why three separate protocols:
    - NoiseSchedule is pure math — testable with numpy, no model needed
    - ModelBackend is training-time — single forward step, gradient computation
    - InferencePipeline is for sampling — iterative denoising, different code path

All protocols are @runtime_checkable so isinstance() works for validation.
Phase 7 tests use mock implementations; Phase 8 fills them with real Wan code.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class NoiseSchedule(Protocol):
    """Protocol for noise scheduling in flow matching / diffusion training.

    Handles timestep sampling, noisy latent computation, and target
    computation. The math is model-family-specific (flow matching for Wan,
    could be DDPM for other models), but the interface is universal.

    Phase 7 provides FlowMatchingSchedule as the default implementation.
    Other model families can provide their own.
    """

    @property
    def num_timesteps(self) -> int:
        """Total number of discrete timesteps in the noise schedule.

        For Wan models this is 1000. Used to validate timestep ranges
        and normalize timestep values.
        """
        ...

    def sample_timesteps(
        self,
        batch_size: int,
        strategy: str = "uniform",
        flow_shift: float = 1.0,
        generator: Any = None,
    ) -> Any:
        """Sample random timesteps for a training batch.

        Args:
            batch_size: Number of timesteps to sample.
            strategy: Sampling strategy name ('uniform', 'shift',
                'logit_normal', 'sigmoid').
            flow_shift: Shift parameter for the 'shift' strategy.
                Higher values bias toward higher noise levels.
            generator: Optional random generator for reproducibility.

        Returns:
            Tensor of shape [batch_size] with timestep values in [0, 1].
            The exact tensor type depends on the implementation (numpy
            for testing, torch for real training).
        """
        ...

    def compute_noisy_latent(
        self,
        clean: Any,
        noise: Any,
        timesteps: Any,
    ) -> Any:
        """Compute noisy latents by interpolating between clean and noise.

        For flow matching: noisy = (1 - t) * clean + t * noise
        For DDPM: noisy = sqrt(alpha_bar) * clean + sqrt(1-alpha_bar) * noise

        Args:
            clean: Clean latent tensor [B, C, F, H, W].
            noise: Pure noise tensor, same shape as clean.
            timesteps: Timestep values [B], in [0, 1].

        Returns:
            Noisy latent tensor, same shape as clean.
        """
        ...

    def compute_target(
        self,
        clean: Any,
        noise: Any,
        timesteps: Any,
    ) -> Any:
        """Compute the training target (what the model should predict).

        For flow matching velocity: target = noise - clean
        For DDPM epsilon: target = noise

        Args:
            clean: Clean latent tensor [B, C, F, H, W].
            noise: Pure noise tensor, same shape as clean.
            timesteps: Timestep values [B], in [0, 1].

        Returns:
            Target tensor, same shape as clean.
        """
        ...

    def get_signal_to_noise_ratio(self, timesteps: Any) -> Any:
        """Compute signal-to-noise ratio for given timesteps.

        Used for expert routing — timesteps with SNR above the boundary
        go to the high-noise expert, below go to low-noise.

        For flow matching: SNR = (1 - t) / t

        Args:
            timesteps: Timestep values [B], in (0, 1).

        Returns:
            SNR values [B]. Higher = less noise (cleaner signal).
        """
        ...


@runtime_checkable
class ModelBackend(Protocol):
    """Protocol for model-specific training operations.

    Encapsulates everything that changes between model architectures:
    how to load the model, which layers get LoRA, how to compute expert
    masks, how to run the forward pass. The training loop calls these
    methods without knowing whether it's Wan 2.1, 2.2, or a future model.

    Phase 8 implements WanModelBackend for all Wan variants.
    """

    @property
    def model_id(self) -> str:
        """Human-readable identifier for this model backend.

        Used in logging and checkpoint metadata. Should include the
        model family and variant (e.g. 'wan-2.2-t2v-14b').
        """
        ...

    @property
    def supports_moe(self) -> bool:
        """Whether this model has Mixture of Experts architecture.

        Controls whether expert masking and fork-and-specialize are
        available. Wan 2.2 = True, Wan 2.1 = False.
        """
        ...

    @property
    def supports_reference_image(self) -> bool:
        """Whether this model accepts a reference image as conditioning.

        Controls whether reference image tensors are passed to the model.
        I2V = True, T2V = False.
        """
        ...

    def load_model(self, config: Any) -> Any:
        """Load the base model from disk or HuggingFace Hub.

        Args:
            config: Model configuration (ModelConfig from training schema).

        Returns:
            The loaded model object (e.g. a torch.nn.Module).
            The training loop treats this as opaque — it's passed back
            to forward() and other methods.
        """
        ...

    def get_lora_target_modules(self) -> list[str]:
        """Return the list of module names that should get LoRA adapters.

        These are the actual nn.Module names in the model (not the
        abstract fork target names). The training loop uses this to
        create LoRA layers at the right locations.

        Returns:
            List of module name patterns (e.g. ['attn1.to_q', 'attn1.to_k']).
        """
        ...

    def get_expert_mask(
        self,
        timesteps: Any,
        boundary_ratio: float,
    ) -> tuple[Any, Any]:
        """Compute expert masks for a batch of timesteps.

        Determines which samples in a batch belong to which expert
        based on the noise level and the boundary ratio.

        Args:
            timesteps: Timestep values [B], in [0, 1].
            boundary_ratio: SNR boundary between experts (e.g. 0.875).

        Returns:
            Tuple of (high_noise_mask, low_noise_mask), each [B].
            Masks are floats (0.0 or 1.0) for loss weighting.
        """
        ...

    def prepare_model_inputs(
        self,
        batch: dict[str, Any],
        timesteps: Any,
        noisy_latents: Any,
    ) -> dict[str, Any]:
        """Prepare inputs for the model's forward pass.

        Converts the generic batch dict into model-specific kwargs.
        Handles text embedding injection, reference image concatenation,
        timestep embedding, etc.

        Args:
            batch: Collated batch from CachedLatentDataset.
            timesteps: Sampled timesteps [B].
            noisy_latents: Noise-corrupted latents [B, C, F, H, W].

        Returns:
            Dict of kwargs to pass to forward().
        """
        ...

    def forward(self, model: Any, **inputs: Any) -> Any:
        """Run one forward pass through the model.

        Args:
            model: The loaded model object (from load_model).
            **inputs: Model-specific inputs (from prepare_model_inputs).

        Returns:
            Model prediction tensor, same shape as the training target.
        """
        ...

    def setup_gradient_checkpointing(self, model: Any) -> None:
        """Enable gradient checkpointing on the model.

        Trades compute for VRAM by recomputing activations during
        backward pass instead of storing them. Essential for video
        training where activation memory is enormous.

        Args:
            model: The loaded model object.
        """
        ...

    def get_noise_schedule(self) -> NoiseSchedule:
        """Return the noise schedule for this model.

        The model backend owns its noise schedule because the math
        is model-family-specific (flow matching for Wan, potentially
        DDPM for other models).

        Returns:
            A NoiseSchedule implementation appropriate for this model.
        """
        ...


@runtime_checkable
class InferencePipeline(Protocol):
    """Protocol for sample generation during training.

    Separate from ModelBackend because inference follows a completely
    different code path: iterative denoising with a scheduler, classifier-free
    guidance, and video decoding. The training loop only does single-step
    forward passes.

    Phase 8 implements WanInferencePipeline using diffusers.
    """

    def generate(
        self,
        model: Any,
        lora_state_dict: dict[str, Any] | None,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 30,
        guidance_scale: float = 5.0,
        seed: int = 42,
        reference_image: Any = None,
    ) -> Any:
        """Generate a sample video or image.

        Runs the full denoising loop: encode prompt → create noise →
        iteratively denoise → decode latents → return output.

        Args:
            model: The loaded model object.
            lora_state_dict: LoRA weights to apply during generation.
                None = generate without LoRA (baseline comparison).
            prompt: Positive text prompt.
            negative_prompt: Negative text prompt for CFG.
            num_inference_steps: Number of denoising steps.
            guidance_scale: Classifier-free guidance scale.
            seed: Random seed for reproducibility.
            reference_image: Optional reference image for I2V models.

        Returns:
            Generated output (video frames, image, or file path).
            The exact type depends on the implementation.
        """
        ...
