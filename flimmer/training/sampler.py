"""Preview generation framework for training monitoring.

Generates sample videos/images during training so you can visually
track training progress without waiting for the full run to complete.

Key behaviors:
    - Per-phase sampling control via skip_phases list
    - Partner LoRA resolution: during expert phases, samples use BOTH
      experts — the active one's current state + the partner's latest
      checkpoint (or unified base if partner hasn't trained yet)
    - Seed walking: seed + prompt_index for variety across prompts
    - Output organized by phase and epoch

Uses InferencePipeline protocol — actual generation is model-specific
(Phase 8). This module handles WHEN and WHAT to sample.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from flimmer.training.errors import SamplingError
from flimmer.training.phase import PhaseType


def _prepare_frames(frames: Any) -> list[Any]:
    """Normalize diffusers pipeline output into a flat list of (H, W, C) frames.

    Diffusers WanPipeline output varies by version and backend:
      - list[list[PIL.Image]]  — standard with imageio
      - list[list[ndarray]]    — OpenCV fallback
      - ndarray (F, H, W, C)  — single video tensor
      - ndarray (1, F, H, W, C) — batched video tensor

    This normalizes all formats to a list of (H, W, C) arrays or PIL Images.
    """
    import numpy as np

    if frames is None:
        return []

    # Case 1: Single numpy array — a video tensor (F, H, W, C) or (1, F, H, W, C)
    if isinstance(frames, np.ndarray):
        # Squeeze leading batch dimensions
        while frames.ndim > 4:
            frames = frames[0]
        # Now (F, H, W, C) — split into list of (H, W, C) frames
        if frames.ndim == 4:
            return [frames[i] for i in range(frames.shape[0])]
        # (H, W, C) single frame
        if frames.ndim == 3:
            return [frames]
        return []

    # Case 2: list[list[...]] — batch of frame lists
    if isinstance(frames, (list, tuple)) and len(frames) > 0:
        inner = frames[0]
        if isinstance(inner, (list, tuple)):
            # Take first batch item
            frame_list = list(inner)
        elif isinstance(inner, np.ndarray) and inner.ndim >= 3:
            # list of ndarray frames — already flat
            frame_list = list(frames)
        else:
            frame_list = list(frames)

        # Squeeze extra dims on individual numpy frames
        result = []
        for f in frame_list:
            if isinstance(f, np.ndarray):
                while f.ndim > 3:
                    f = f[0]
                result.append(f)
            else:
                result.append(f)
        return result

    return []


def _save_keyframe_grid(
    frames: Any,
    output_path: Path,
    frame_indices: tuple[int, ...] = (0, 4, 8, 12, 16),
) -> None:
    """Save a horizontal grid of keyframes as a PNG for quick visual review.

    Extracts specific frames from the video and arranges them side-by-side
    in a single image. This lets you evaluate training quality at a glance
    without watching the full video -- especially useful when reviewing
    samples from RunPod or other remote environments.

    Default frame_indices (0, 4, 8, 12, 16) correspond to frames 1, 5, 9,
    13, 17 of a 17-frame Wan output -- evenly spaced across the clip.

    Args:
        frames: Pipeline output in any supported format (same as
            _save_frames_to_video). Gets normalized via _prepare_frames().
        output_path: Path to save the .png grid image.
        frame_indices: 0-based indices of frames to include in the grid.
            Indices beyond the frame count are silently skipped.
    """
    import numpy as np

    frame_list = _prepare_frames(frames)
    if not frame_list:
        print("  Warning: No frames for keyframe grid")
        return

    # Extract requested frames, skipping any out-of-range indices
    selected = []
    for idx in frame_indices:
        if idx < len(frame_list):
            selected.append(frame_list[idx])

    if not selected:
        print("  Warning: No valid frame indices for keyframe grid")
        return

    # Convert all frames to PIL Images for consistent grid assembly
    from PIL import Image

    pil_frames: list[Image.Image] = []
    for frame in selected:
        if hasattr(frame, "save"):
            # Already a PIL Image
            pil_frames.append(frame)
        elif isinstance(frame, np.ndarray):
            if frame.dtype != np.uint8:
                frame = (frame * 255).clip(0, 255).astype(np.uint8)
            pil_frames.append(Image.fromarray(frame))
        else:
            print(f"  Warning: Unknown frame type {type(frame)}, skipping in grid")
            continue

    if not pil_frames:
        print("  Warning: Could not convert any frames for keyframe grid")
        return

    # Create horizontal grid: all frames side by side
    widths = [img.width for img in pil_frames]
    heights = [img.height for img in pil_frames]
    grid_width = sum(widths)
    grid_height = max(heights)

    grid = Image.new("RGB", (grid_width, grid_height))
    x_offset = 0
    for img in pil_frames:
        grid.paste(img, (x_offset, 0))
        x_offset += img.width

    # Save the grid
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output_path)
    size_kb = output_path.stat().st_size / 1024
    print(f"  Keyframe grid saved: {output_path} ({size_kb:.0f} KB, {grid.width}x{grid.height}, {len(pil_frames)} frames)")


def _save_frames_to_video(frames: Any, output_path: Path, fps: int = 16) -> None:
    """Save diffusers pipeline output frames to an MP4 file.

    Handles all WanPipeline output formats: PIL Images, numpy arrays,
    and raw video tensors. Uses export_to_video with imageio when
    available, falls back to PIL PNG saving.

    Args:
        frames: Pipeline output in any supported format.
        output_path: Path to save the .mp4 file.
        fps: Frames per second for the output video.
    """
    frame_list = _prepare_frames(frames)
    if not frame_list:
        print("  Warning: No frames to save")
        return

    print(f"  Prepared {len(frame_list)} frames for saving")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Try export_to_video (handles both PIL and numpy arrays)
    try:
        from diffusers.utils import export_to_video
        export_to_video(frame_list, str(output_path), fps=fps)
        size_mb = output_path.stat().st_size / 1024 / 1024
        print(f"  Sample saved: {output_path} ({size_mb:.1f} MB)")

        # Save keyframe grid alongside the video for quick visual review.
        # This lets you evaluate training quality at a glance without
        # watching the full video — invaluable for remote pod workflows.
        grid_path = output_path.with_suffix(".grid.png")
        _save_keyframe_grid(frames, grid_path)

        return
    except Exception as e:
        print(f"  Warning: export_to_video failed ({e}), trying fallback...")

    # Fallback: save individual frames as PNG
    import numpy as np

    png_dir = output_path.with_suffix("")
    png_dir.mkdir(parents=True, exist_ok=True)

    for idx, frame in enumerate(frame_list):
        png_path = png_dir / f"frame_{idx:04d}.png"
        try:
            # PIL Image
            if hasattr(frame, "save"):
                frame.save(png_path)
            # Numpy array
            elif isinstance(frame, np.ndarray):
                from PIL import Image
                if frame.dtype != np.uint8:
                    frame = (frame * 255).clip(0, 255).astype(np.uint8)
                Image.fromarray(frame).save(png_path)
            else:
                print(f"  Warning: Unknown frame type {type(frame)} at index {idx}")
        except Exception as e:
            print(f"  Warning: Failed to save frame {idx}: {e}")

    print(f"  Sample frames saved to: {png_dir} ({len(frame_list)} frames)")

    # Save keyframe grid even when falling back to individual PNGs
    grid_path = output_path.with_suffix(".grid.png")
    _save_keyframe_grid(frames, grid_path)


class SamplingEngine:
    """Orchestrates when and how to generate training samples.

    Manages sampling schedule, prompt iteration, seed handling, and
    partner LoRA resolution. The actual generation is delegated to
    an InferencePipeline implementation.

    Args:
        enabled: Whether sampling is active at all.
        every_n_epochs: Generate samples every N epochs.
        prompts: List of positive prompts to generate.
        negative_prompt: Negative prompt applied to all generations.
        seed: Base seed for reproducibility.
        walk_seed: If True, increment seed by 1 per prompt.
        num_inference_steps: Denoising steps per sample.
        guidance_scale: Classifier-free guidance scale.
        sample_dir: Root directory for sample output.
        skip_phases: List of phase type value strings to skip.
    """

    def __init__(
        self,
        enabled: bool = False,
        every_n_epochs: int = 5,
        prompts: list[str] | None = None,
        negative_prompt: str = "",
        seed: int = 42,
        walk_seed: bool = True,
        num_inference_steps: int = 30,
        guidance_scale: float = 4.0,
        sample_dir: str | Path | None = None,
        skip_phases: list[str] | None = None,
    ) -> None:
        self._enabled = enabled
        self._every_n_epochs = every_n_epochs
        self._prompts = prompts or []
        self._negative_prompt = negative_prompt
        self._seed = seed
        self._walk_seed = walk_seed
        self._num_inference_steps = num_inference_steps
        self._guidance_scale = guidance_scale
        self._sample_dir = Path(sample_dir) if sample_dir else None
        self._skip_phases: set[str] = set(skip_phases or [])

    @property
    def enabled(self) -> bool:
        """Whether sampling is enabled."""
        return self._enabled

    @property
    def prompts(self) -> list[str]:
        """List of generation prompts."""
        return self._prompts

    def should_sample(self, epoch: int, phase_type: PhaseType) -> bool:
        """Determine whether to generate samples at this epoch.

        Checks:
        1. Is sampling enabled?
        2. Are there any prompts?
        3. Is this epoch on the sampling interval?
        4. Is this phase NOT in the skip list?

        Args:
            epoch: Current epoch number (1-based).
            phase_type: Current training phase type.

        Returns:
            True if samples should be generated.
        """
        if not self._enabled:
            return False
        if not self._prompts:
            return False
        if epoch <= 0:
            return False
        if epoch % self._every_n_epochs != 0:
            return False
        if phase_type.value in self._skip_phases:
            return False
        return True

    def resolve_partner_lora(
        self,
        active_expert: str | None,
        high_noise_path: str | Path | None,
        low_noise_path: str | Path | None,
        unified_path: str | Path | None,
    ) -> str | Path | None:
        """Resolve which partner LoRA to use during expert sampling.

        During an expert phase, samples should show BOTH experts working
        together. The active expert's current state is used directly;
        this method resolves the PARTNER's state.

        Resolution order:
        1. If the partner has a trained checkpoint → use it
        2. If the partner hasn't trained yet → use the unified base
        3. If no unified base exists → return None (sample with active only)

        Args:
            active_expert: Currently training expert ('high_noise' or 'low_noise').
                None = unified phase (no partner needed).
            high_noise_path: Path to latest high-noise checkpoint, or None.
            low_noise_path: Path to latest low-noise checkpoint, or None.
            unified_path: Path to unified checkpoint (fork point), or None.

        Returns:
            Path to the partner LoRA, or None if no partner is available.
        """
        if active_expert is None:
            # Unified phase — no partner needed
            return None

        # Determine the partner expert
        if active_expert == "high_noise":
            partner_path = low_noise_path
        elif active_expert == "low_noise":
            partner_path = high_noise_path
        else:
            return None

        # If partner has a trained checkpoint, use it
        if partner_path is not None:
            return partner_path

        # Fall back to unified base
        return unified_path

    def get_seed_for_prompt(self, prompt_index: int) -> int:
        """Get the seed for a specific prompt.

        If walk_seed is True, each prompt gets a different seed
        (base_seed + prompt_index). Otherwise all prompts share
        the same seed.

        Args:
            prompt_index: Index of the prompt in the prompts list.

        Returns:
            Seed value for this prompt.
        """
        if self._walk_seed:
            return self._seed + prompt_index
        return self._seed

    def get_output_dir(
        self,
        phase_type: PhaseType,
        epoch: int,
        base_dir: str | Path | None = None,
    ) -> Path:
        """Get the output directory for samples at a specific checkpoint.

        Directory format: {sample_dir}/{phase_abbrev}_epoch{N:03d}/

        Args:
            phase_type: Current training phase.
            epoch: Current epoch number.
            base_dir: Override base directory (default: self._sample_dir).

        Returns:
            Path to the sample output directory (created if needed).
        """
        root = Path(base_dir) if base_dir else self._sample_dir
        if root is None:
            raise SamplingError(
                "No sample directory configured. "
                "Set sampling.sample_dir in your config or provide base_dir."
            )

        # Use the same abbreviations as checkpoint manager
        abbrevs = {
            PhaseType.UNIFIED: "unified",
            PhaseType.HIGH_NOISE: "high",
            PhaseType.LOW_NOISE: "low",
        }
        abbrev = abbrevs.get(phase_type, phase_type.value)
        out_dir = root / f"{abbrev}_epoch{epoch:03d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    def generate_samples(
        self,
        pipeline: Any,
        model: Any,
        lora_state_dict: dict[str, Any] | None,
        phase_type: PhaseType,
        epoch: int,
        base_dir: str | Path | None = None,
        reference_image: Any = None,
        partner_model: Any = None,
        active_expert: str | None = None,
    ) -> list[Path]:
        """Generate sample videos for all prompts.

        Iterates through prompts, generates one sample per prompt,
        and saves them to the output directory.

        For expert phases, pass partner_model and active_expert to enable
        dual-expert inference — the trained expert uses its live LoRA while
        the partner expert runs with its own LoRA (or base weights).

        Args:
            pipeline: InferencePipeline implementation.
            model: The loaded model object.
            lora_state_dict: LoRA weights to apply (may include partner).
            phase_type: Current training phase.
            epoch: Current epoch number.
            base_dir: Override base directory for output.
            reference_image: Optional reference image for I2V.
            partner_model: The other expert's transformer for dual-expert
                inference. None = single-expert mode.
            active_expert: Which expert the training model is ('high_noise'
                or 'low_noise'). Required when partner_model is provided.

        Returns:
            List of paths to generated sample files.
        """
        out_dir = self.get_output_dir(phase_type, epoch, base_dir)
        generated: list[Path] = []

        for i, prompt in enumerate(self._prompts):
            seed = self.get_seed_for_prompt(i)
            output_path = out_dir / f"prompt_{i}_seed{seed}.mp4"

            try:
                result = pipeline.generate(
                    model=model,
                    lora_state_dict=lora_state_dict,
                    prompt=prompt,
                    negative_prompt=self._negative_prompt,
                    num_inference_steps=self._num_inference_steps,
                    guidance_scale=self._guidance_scale,
                    seed=seed,
                    reference_image=reference_image,
                    partner_model=partner_model,
                    active_expert=active_expert,
                )
                # If result is a path, use it; otherwise save frames to video
                if isinstance(result, (str, Path)):
                    generated.append(Path(result))
                else:
                    # Pipeline returned frames — export to video file
                    _save_frames_to_video(result, output_path)
                    generated.append(output_path)

            except Exception as e:
                raise SamplingError(
                    f"Failed to generate sample for prompt {i} "
                    f"('{prompt[:50]}...'): {e}"
                ) from e

        return generated
