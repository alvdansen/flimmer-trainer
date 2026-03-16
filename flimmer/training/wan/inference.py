"""WanInferencePipeline — memory-efficient sample generation during training.

Generates preview videos during training using the already-loaded training
model. Avoids loading a second copy of the 14B transformer by reusing the
training model directly.

Memory strategy:
    1. Load T5 text encoder temporarily → encode all prompts → free T5
    2. Build diffusers WanPipeline from components:
       - transformer = HIGH-noise expert (runs first, large timesteps)
       - transformer_2 = LOW-noise expert (runs second, small timesteps)
       - boundary_ratio = 0.6 for T2V, 0.9 for I2V
       - VAE = loaded from disk (small: ~243MB), always float32
       - scheduler = UniPCMultistepScheduler with flow matching
    3. Generate using prompt_embeds (pre-encoded) → skip T5 entirely
    4. Free VAE + pipeline after generation

Dual-expert support:
    When a partner_model is passed to generate(), the pipeline uses both
    experts for coherent output. When only the training model is available,
    falls back to single-expert mode (acceptable for training previews).

    Diffusers convention:
        transformer   = HIGH-noise expert (handles timesteps >= boundary)
        transformer_2 = LOW-noise expert  (handles timesteps < boundary)

Supports both individual safetensors files and Diffusers directories.

Requires: torch, diffusers, transformers (the 'wan' dependency group).
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

from flimmer.training.errors import SamplingError


def _clean_gpu_memory() -> None:
    """Force Python GC and clear CUDA cache."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


class WanInferencePipeline:
    """InferencePipeline for Wan model sample generation.

    Two usage modes:

    1. Training preview mode (model already has LoRA applied via PEFT):
       Pass the training model directly — LoRA is already injected.

    2. Standalone inference mode (load LoRA from file):
       Set lora_path to a saved checkpoint. The pipeline loads the LoRA
       via pipeline.load_lora_weights() after building the diffusers pipeline.
       The checkpoint must have diffusers-compatible keys (transformer.blocks.0...).

    Memory strategy:
        - Load T5 temporarily → encode all prompts → free T5
        - Build diffusers WanPipeline from components
        - Load LoRA from file if lora_path is set
        - Generate using prompt_embeds → skip T5 during denoising
        - Free VAE + pipeline after generation

    Supports two model loading modes:
        - Individual safetensors: provide vae_path and t5_path
        - Diffusers directory: provide diffusers_path (fallback)

    Args:
        vae_path: Path to VAE safetensors file (individual file mode).
        t5_path: Path to T5 encoder weights (.pth or .safetensors).
        diffusers_path: Path to Diffusers model directory (fallback).
            Used when individual file paths are not provided.
        is_i2v: Whether this is an I2V pipeline.
        dtype: Tensor dtype for inference ('bf16', 'fp16', 'fp32').
        device: Target device ('cuda', 'cpu').
        flow_shift: Flow matching shift parameter for the scheduler.
            Must match the value used during training to avoid noise
            level mismatch. Defaults to 5.0 (Wan 2.2 T2V default).
            Wan 2.1 T2V uses 3.0.
        lora_path: Path to a saved LoRA checkpoint (.safetensors).
            If set, the LoRA is loaded via pipeline.load_lora_weights()
            after building the pipeline. The file must have diffusers-
            compatible keys (with transformer./transformer_2. prefix).
    """

    def __init__(
        self,
        vae_path: str | None = None,
        t5_path: str | None = None,
        diffusers_path: str | None = None,
        is_i2v: bool = False,
        dtype: str = "bf16",
        device: str = "cuda",
        flow_shift: float = 5.0,
        lora_path: str | None = None,
    ) -> None:
        self._vae_path = vae_path
        self._t5_path = t5_path
        self._diffusers_path = diffusers_path
        self._is_i2v = is_i2v
        self._dtype_str = dtype
        self._device = device
        self._flow_shift = flow_shift
        self._lora_path = lora_path

        # Cached prompt embeddings (populated on first generate call)
        self._cached_prompt_embeds: dict[str, Any] = {}

    def _get_torch_dtype(self) -> Any:
        """Map dtype string to torch dtype."""
        import torch
        return {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }.get(self._dtype_str, torch.bfloat16)

    def _encode_prompt(self, prompt: str) -> Any:
        """Encode a single prompt using T5, with caching.

        Loads the T5 encoder on first call, encodes the prompt, and
        caches the result. The encoder is freed after all prompts in
        a generation batch are encoded.

        Returns:
            Tensor of shape [1, seq_len, hidden_dim].
        """
        if prompt in self._cached_prompt_embeds:
            return self._cached_prompt_embeds[prompt]

        raise SamplingError(
            "Prompt not pre-encoded. Call _precompute_embeddings() first."
        )

    def _precompute_embeddings(
        self, prompts: list[str], negative_prompt: str = ""
    ) -> None:
        """Pre-encode all prompts using T5, then free the encoder.

        This is the memory-efficient approach: load T5 once, encode
        everything, then completely free the ~10GB encoder before
        starting the diffusion loop.

        Args:
            prompts: All positive prompts to encode.
            negative_prompt: The negative prompt (shared across all samples).
        """
        # Skip if all prompts already cached
        all_texts = set(prompts)
        if negative_prompt:
            all_texts.add(negative_prompt)

        uncached = [t for t in all_texts if t not in self._cached_prompt_embeds]
        if not uncached:
            return

        import torch

        dtype = self._get_torch_dtype()

        # Free CUDA cache before loading T5 — during training, the caching
        # allocator holds reserved blocks from the last training step. Without
        # this, T5 may OOM even though there's enough total VRAM.
        _clean_gpu_memory()

        tokenizer = None
        text_encoder = None
        try:
            # Load T5 encoder and tokenizer
            tokenizer, text_encoder = self._load_t5()
            text_encoder = text_encoder.to(self._device, dtype=dtype)
            text_encoder.eval()

            # Replicate diffusers WanPipeline._get_t5_prompt_embeds() behavior:
            # 1. Tokenize with max_length=512, padding="max_length"
            # 2. Encode through T5
            # 3. Truncate to actual token length (using attention mask)
            # 4. Re-pad with ZEROS to max_sequence_length
            #
            # Step 3-4 is critical: T5 produces non-zero embeddings for padding
            # tokens. If these are left in, the model receives ~490 garbage tokens
            # that drown out the real prompt, producing black/empty output.
            max_seq_len = 512

            print("  Encoding sampling prompts...")
            with torch.no_grad():
                for text in uncached:
                    tokens = tokenizer(
                        text,
                        max_length=max_seq_len,
                        padding="max_length",
                        truncation=True,
                        add_special_tokens=True,
                        return_attention_mask=True,
                        return_tensors="pt",
                    )
                    input_ids = tokens.input_ids.to(self._device)
                    attention_mask = tokens.attention_mask.to(self._device)

                    output = text_encoder(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )
                    # Get the last hidden state
                    if hasattr(output, "last_hidden_state"):
                        embeds = output.last_hidden_state
                    else:
                        embeds = output[0]

                    # Truncate to actual token length, then zero-pad.
                    # This matches WanPipeline._get_t5_prompt_embeds() exactly.
                    seq_len = attention_mask.gt(0).sum(dim=1).long()
                    for i in range(embeds.shape[0]):
                        actual_len = seq_len[i].item()
                        real_tokens = embeds[i, :actual_len]
                        padding = embeds.new_zeros(
                            max_seq_len - actual_len, embeds.shape[-1]
                        )
                        embeds[i] = torch.cat([real_tokens, padding])

                    # Store on CPU to save GPU memory
                    self._cached_prompt_embeds[text] = embeds.cpu()

            print(f"  Encoded {len(uncached)} prompts")

        finally:
            # Free T5 regardless of success/failure.
            # Variables are pre-initialized to None so del is always safe
            # (previously, a _load_t5() failure caused UnboundLocalError here,
            # masking the real error).
            del text_encoder, tokenizer
            _clean_gpu_memory()

    def _load_t5(self) -> tuple[Any, Any]:
        """Load T5 tokenizer and encoder model.

        Supports:
            - Individual .pth file (torch.load)
            - Individual .safetensors file
            - Diffusers directory (subfolder 'text_encoder')

        Returns:
            (tokenizer, text_encoder) tuple.
        """
        from transformers import AutoTokenizer

        if self._t5_path and Path(self._t5_path).exists():
            # Individual file mode — use the same loading as text_encoder.py
            return self._load_t5_from_file(self._t5_path)
        elif self._diffusers_path:
            # Diffusers directory mode — load directly in target dtype to halve
            # CPU memory (bf16 ~11GB vs fp32 ~23GB). During training, system RAM
            # may be constrained by optimizer states and cached data.
            import torch
            from transformers import UMT5EncoderModel
            tokenizer = AutoTokenizer.from_pretrained(
                self._diffusers_path, subfolder="tokenizer"
            )
            text_encoder = UMT5EncoderModel.from_pretrained(
                self._diffusers_path,
                subfolder="text_encoder",
                torch_dtype=self._get_torch_dtype(),
                low_cpu_mem_usage=True,
            )
            # Fix embed_tokens weight tying (same bug affects from_pretrained)
            self._fix_t5_embed_tokens(text_encoder)
            return tokenizer, text_encoder
        else:
            raise SamplingError(
                "No T5 encoder path available for sampling. "
                "Set model.t5 (individual file) or model.path (Diffusers directory) "
                "in your training config."
            )

    def _load_t5_from_file(self, t5_path: str) -> tuple[Any, Any]:
        """Load T5 from an individual weight file (.pth or .safetensors).

        Uses the same approach as flimmer.encoding.text_encoder:
        UMT5Config → empty model → load_state_dict.

        Returns:
            (tokenizer, text_encoder) tuple.
        """
        import torch
        from transformers import AutoTokenizer, UMT5EncoderModel, UMT5Config

        # Load tokenizer and config from HuggingFace cache (pre-cached during setup).
        # local_files_only=True prevents network access — if not cached, falls back
        # to downloading (only happens if setup.sh cache step was skipped).
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                "google/umt5-xxl", local_files_only=True
            )
            config = UMT5Config.from_pretrained(
                "google/umt5-xxl", local_files_only=True
            )
        except OSError:
            # Not cached locally — fall back to download (first run without setup.sh)
            tokenizer = AutoTokenizer.from_pretrained("google/umt5-xxl")
            config = UMT5Config.from_pretrained("google/umt5-xxl")
        text_encoder = UMT5EncoderModel(config)

        path = Path(t5_path)
        if path.suffix == ".pth":
            state_dict = torch.load(str(path), map_location="cpu", weights_only=True)
        else:
            from safetensors.torch import load_file
            state_dict = load_file(str(path))

        text_encoder.load_state_dict(state_dict, strict=False)

        # Fix T5 embed_tokens weight tying: the Wan-AI checkpoint stores the
        # token embedding as "shared.weight" but UMT5EncoderModel expects it
        # at "encoder.embed_tokens.weight". The weight tying doesn't happen
        # automatically during load_state_dict, leaving embed_tokens as zeros.
        # Without this fix, the T5 produces all-zero embeddings and the model
        # generates unconditionally (ignoring the prompt entirely).
        self._fix_t5_embed_tokens(text_encoder)

        return tokenizer, text_encoder

    @staticmethod
    def _fix_t5_embed_tokens(text_encoder: Any) -> None:
        """Fix T5 embed_tokens weight tying bug.

        The Wan-AI T5 checkpoint stores the token embedding as
        "shared.weight", but UMT5EncoderModel keeps a separate
        "encoder.embed_tokens.weight" that should be tied to it.
        Neither load_state_dict(strict=False) nor from_pretrained()
        performs this tying automatically.

        When loaded via load_state_dict (from .pth), embed_tokens is
        left as all zeros. When loaded via from_pretrained, embed_tokens
        is randomly initialized. In both cases, the weight is WRONG --
        it should be identical to shared.weight.

        Fix: always copy shared.weight to encoder.embed_tokens if they
        differ. The shared.weight always has the correct trained values.
        """
        if (
            hasattr(text_encoder, "shared")
            and hasattr(text_encoder, "encoder")
            and hasattr(text_encoder.encoder, "embed_tokens")
        ):
            import torch
            shared = text_encoder.shared.weight
            embed = text_encoder.encoder.embed_tokens.weight
            # Fix if embed_tokens differs from shared (wrong initialization)
            # shared.weight always has the correct trained values
            if not torch.equal(shared, embed):
                text_encoder.encoder.embed_tokens.weight = shared

    def _load_vae(self) -> Any:
        """Load the Wan VAE for decoding latents to video frames.

        Always loads in float32 regardless of training precision — the Wan VAE
        produces artifacts and gridded noise patterns when run in bf16/fp16.
        This matches the official recommendation and our validated inference.

        Returns:
            AutoencoderKLWan instance.
        """
        import torch
        from diffusers.models import AutoencoderKLWan

        if self._vae_path and Path(self._vae_path).exists():
            vae = AutoencoderKLWan.from_single_file(
                self._vae_path, torch_dtype=torch.float32
            )
        elif self._diffusers_path:
            vae = AutoencoderKLWan.from_pretrained(
                self._diffusers_path, subfolder="vae", torch_dtype=torch.float32
            )
        else:
            raise SamplingError(
                "No VAE path available for sampling. "
                "Set model.vae (individual file) or model.path (Diffusers directory) "
                "in your training config."
            )

        return vae.to(self._device)

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
        partner_model: Any = None,
        active_expert: str | None = None,
    ) -> Any:
        """Generate a sample video using the training model.

        Memory-efficient approach:
        1. Pre-encode prompts with T5 (loaded temporarily, then freed)
        2. Build a minimal pipeline: training model + VAE + scheduler
        3. Generate using prompt_embeds (no T5 needed during denoising)
        4. Export frames to video and clean up

        Dual-expert support:
        When partner_model is provided, both experts are used for coherent
        output. The active_expert parameter tells us which role the training
        model plays so we can assign transformer/transformer_2 correctly.

        Diffusers convention:
            transformer   = HIGH-noise expert (large timesteps, runs first)
            transformer_2 = LOW-noise expert  (small timesteps, runs second)

        Args:
            model: The training transformer (already on GPU with LoRA).
            lora_state_dict: LoRA state dict (unused — LoRA is already
                applied to the model during training).
            prompt: Positive text prompt.
            negative_prompt: Negative text prompt for CFG.
            num_inference_steps: Number of denoising steps.
            guidance_scale: Classifier-free guidance scale.
            seed: Random seed for reproducibility.
            reference_image: Reference image for I2V (PIL Image or path).
            partner_model: Optional second expert transformer for dual-expert
                inference. If None, uses single-expert mode.
            active_expert: Which expert the training model is: 'high_noise'
                or 'low_noise'. Required when partner_model is provided.
                None = unified phase (training model used as sole transformer).

        Returns:
            Generated video frames (list of lists of PIL Images).

        Raises:
            SamplingError: If generation fails.
        """
        try:
            import torch
        except ImportError as e:
            raise SamplingError(f"torch required for inference: {e}")

        try:
            # Step 1: Pre-encode prompts (loads T5 temporarily, then frees it)
            # Always include the negative prompt (even empty string) so the
            # pipeline receives negative_prompt_embeds. Without this, diffusers
            # tries to encode the negative prompt via self.text_encoder which
            # is None (we only load T5 temporarily), causing:
            #   'NoneType' object has no attribute 'dtype'
            neg_prompt = negative_prompt if negative_prompt else ""
            all_prompts = [prompt, neg_prompt]
            self._precompute_embeddings(all_prompts, neg_prompt)

            # Move cached embeddings to GPU
            prompt_embeds = self._cached_prompt_embeds[prompt].to(
                self._device, dtype=self._get_torch_dtype()
            )
            neg_embeds = self._cached_prompt_embeds[neg_prompt].to(
                self._device, dtype=self._get_torch_dtype()
            )

            # Step 2: Merge LoRA weights into base model for inference.
            # PeftModel wrappers can confuse diffusers pipelines (dtype/device
            # detection, internal attribute access). Merging bakes LoRA deltas
            # into the base weights so the pipeline sees clean tensors.
            _peft_merged = False
            if hasattr(model, "merge_adapter"):
                model.merge_adapter()
                _peft_merged = True

            # Step 2b: Build pipeline — pass the PeftModel directly.
            # After merge_adapter(), the base weights contain LoRA deltas
            # and PeftModel forwards cleanly. Unwrapping via get_base_model()
            # can break device placement when block swap is active.
            pipeline = self._build_pipeline(
                model,
                partner_model=partner_model,
                active_expert=active_expert,
            )

            # Step 2c: Load LoRA from file if configured (standalone inference)
            if self._lora_path is not None:
                self._apply_lora_from_file(pipeline, self._lora_path)

            # Step 3: Switch model(s) to eval mode for inference
            was_training = model.training
            model.eval()
            partner_was_training = False
            if partner_model is not None:
                partner_was_training = partner_model.training
                partner_model.eval()

            generator = torch.Generator(device=self._device).manual_seed(seed)

            # Build generation kwargs
            # Explicit height/width/num_frames avoid diffusers defaults (81 frames)
            # which would OOM during training. 17 frames is the minimum Wan supports.
            kwargs: dict[str, Any] = {
                "prompt_embeds": prompt_embeds,
                "negative_prompt_embeds": neg_embeds,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "height": 480,
                "width": 832,
                "num_frames": 17,
                "generator": generator,
            }

            # Dual-expert parameters
            if partner_model is not None:
                # Official recommendation: 4.0 high-noise, 3.0 low-noise
                kwargs["guidance_scale_2"] = 3.0

            # Add reference image for I2V
            if self._is_i2v and reference_image is not None:
                kwargs["image"] = reference_image

            with torch.no_grad():
                output = pipeline(**kwargs)

            # NOTE: LoRA unmerge is deferred to after block swap resumes
            # (in loop.py _generate_samples). Unmerging while blocks are on
            # GPU but LoRA adapter weights are on CPU causes device mismatch.

            # Restore training mode
            if was_training:
                model.train()
            if partner_model is not None and partner_was_training:
                partner_model.train()

            # Extract frames
            frames = output.frames if hasattr(output, "frames") else output

            # Clean up pipeline (keeps training model, frees VAE + scheduler)
            # Don't delete models — they're the training models
            pipeline.transformer = None
            if hasattr(pipeline, "transformer_2"):
                pipeline.transformer_2 = None
            del pipeline
            _clean_gpu_memory()

            return frames

        except SamplingError:
            raise
        except torch.cuda.OutOfMemoryError:
            _clean_gpu_memory()
            raise SamplingError(
                "Out of GPU memory during sampling. "
                "This is expected when training uses most of the VRAM. "
                "Set sampling.enabled: false in your config, or reduce "
                "training batch size to free memory for sampling."
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            _clean_gpu_memory()
            raise SamplingError(
                f"Video generation failed: {e}\n"
                f"This may indicate insufficient VRAM or model incompatibility."
            ) from e

    def _apply_lora_from_file(self, pipeline: Any, lora_path: str) -> None:
        """Load a LoRA checkpoint and apply it to the pipeline.

        The LoRA file must have diffusers-compatible keys with component
        prefixes (transformer.blocks.0... and/or transformer_2.blocks.0...).
        Files saved by flimmer with diffusers_prefix are directly compatible.

        If the file has clean keys (no prefix), they are auto-prefixed with
        'transformer.' so they apply to the first/only transformer.

        Args:
            pipeline: Built WanPipeline or WanImageToVideoPipeline.
            lora_path: Path to the .safetensors LoRA file.

        Raises:
            SamplingError: If the LoRA file cannot be loaded.
        """
        from pathlib import Path

        lora_file = Path(lora_path)
        if not lora_file.is_file():
            raise SamplingError(
                f"LoRA file not found: {lora_path}\n"
                f"Check that the path is correct."
            )

        try:
            from safetensors.torch import load_file

            state_dict = load_file(str(lora_file))

            # Auto-detect prefix: if keys don't have 'transformer.' prefix,
            # add it so load_lora_weights() routes them correctly.
            has_prefix = any(
                k.startswith("transformer.") or k.startswith("transformer_2.")
                for k in state_dict
            )
            if not has_prefix:
                # Clean keys → add transformer. prefix for the primary model
                state_dict = {
                    f"transformer.{k}": v for k, v in state_dict.items()
                }

            print(f"  Loading LoRA: {lora_file.name} ({len(state_dict)} keys)")
            pipeline.load_lora_weights(state_dict, adapter_name="flimmer")
            print(f"  LoRA applied successfully")

        except SamplingError:
            raise
        except Exception as e:
            raise SamplingError(
                f"Failed to load LoRA from '{lora_path}': {e}\n"
                f"Ensure the file is a valid safetensors LoRA checkpoint."
            ) from e

    def _build_pipeline(
        self,
        model: Any,
        partner_model: Any = None,
        active_expert: str | None = None,
    ) -> Any:
        """Build a diffusers pipeline using the training model and fresh VAE.

        Constructs the pipeline from individual components rather than
        from_pretrained(), so it works with both individual safetensors
        files and Diffusers directories.

        Dual-expert wiring:
        When partner_model is provided, we assign transformer/transformer_2
        based on which expert the training model is:
            - active_expert='high_noise' → model=transformer, partner=transformer_2
            - active_expert='low_noise'  → partner=transformer, model=transformer_2
            - active_expert=None (unified) → model=transformer only

        Args:
            model: Training transformer (already on GPU).
            partner_model: Optional second expert for dual-expert mode.
            active_expert: Role of the training model ('high_noise', 'low_noise', None).

        Returns:
            WanPipeline or WanImageToVideoPipeline instance.
        """
        from diffusers import UniPCMultistepScheduler

        # Load VAE (small: ~243MB)
        vae = self._load_vae()

        # UniPCMultistepScheduler with flow matching — matches musubi-tuner
        # and ai-toolkit sampling. FlowMatchEulerDiscreteScheduler produced
        # noise; UniPC's higher-order solver handles the denoising trajectory
        # better for LoRA-trained models with block swap.
        scheduler = UniPCMultistepScheduler(
            prediction_type="flow_prediction",
            num_train_timesteps=1000,
            flow_shift=self._flow_shift,
            use_flow_sigmas=True,
        )

        # Determine expert assignment for the pipeline
        # Diffusers convention: transformer = HIGH-noise, transformer_2 = LOW-noise
        if partner_model is not None and active_expert == "high_noise":
            # Training model is high-noise expert, partner is low-noise
            transformer_high = model
            transformer_low = partner_model
        elif partner_model is not None and active_expert == "low_noise":
            # Training model is low-noise expert, partner is high-noise
            transformer_high = partner_model
            transformer_low = model
        else:
            # Single-expert mode (unified phase or no partner available)
            transformer_high = model
            transformer_low = None

        # Boundary ratio: controls handoff between high-noise and low-noise
        # experts. Only meaningful when both transformers are present.
        # T2V: 0.6 (validated against ComfyUI), I2V: 0.9.
        # Single-expert: must be None — the pipeline checks
        #   `if boundary_timestep is None: use self.transformer`
        # Any numeric value (even 1.0) causes boundary_timestep comparison
        # against descending timesteps, eventually routing to transformer_2
        # (None), causing: 'NoneType' object has no attribute 'cache_context'
        if transformer_low is None:
            boundary = None
        elif self._is_i2v:
            boundary = 0.9
        else:
            boundary = 0.6



        if self._is_i2v:
            from diffusers import WanImageToVideoPipeline
            kwargs: dict[str, Any] = {
                "transformer": transformer_high,
                "vae": vae,
                "text_encoder": None,
                "tokenizer": None,
                "scheduler": scheduler,
                "boundary_ratio": boundary,
            }
            if transformer_low is not None:
                kwargs["transformer_2"] = transformer_low
            pipeline = WanImageToVideoPipeline(**kwargs)
        else:
            from diffusers import WanPipeline
            kwargs = {
                "transformer": transformer_high,
                "vae": vae,
                "text_encoder": None,
                "tokenizer": None,
                "scheduler": scheduler,
                "boundary_ratio": boundary,
            }
            if transformer_low is not None:
                kwargs["transformer_2"] = transformer_low
            pipeline = WanPipeline(**kwargs)

        return pipeline

    def cleanup(self) -> None:
        """Release GPU memory and clear cached embeddings."""
        self._cached_prompt_embeds.clear()
        _clean_gpu_memory()
