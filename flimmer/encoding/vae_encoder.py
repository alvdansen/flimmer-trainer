"""Wan VAE encoder for video/image latent encoding.

Encodes video frames and images through the Wan 3D causal VAE to produce
latent representations for training. This is the heaviest encoding step —
each video clip is loaded, resized, normalized, and passed through the VAE.

The VAE encoder is model-specific (Wan family). Other model families
would implement their own VAE encoder following the ControlEncoder protocol.

Wan VAE specifics:
    - 3D causal VAE with ~4x temporal compression
    - 81 frames → ~21 temporal tokens in latent space
    - Spatial compression: 8x in each dimension
    - Input: pixel values in [-1, 1], shape [B, C, F, H, W]
    - Output: latent in model dtype, shape [B, latent_ch, F//4, H//8, W//8]

This module requires torch and diffusers. It is NOT imported by default —
only loaded when cache-latents is actually run.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from flimmer.dataset.discover import IMAGE_EXTENSIONS
from flimmer.encoding.errors import EncoderError


class WanVaeEncoder:
    """Wan VAE encoder implementing the ControlEncoder protocol.

    Loads the VAE from a Wan model directory or HuggingFace ID.
    Manages GPU memory — the VAE is loaded on first encode() call
    and released on cleanup().

    Usage::

        encoder = WanVaeEncoder(
            model_path="Wan-AI/Wan2.2-T2V-14B-Diffusers",
            dtype="bf16",
        )
        result = encoder.encode(
            "path/to/clip.mp4",
            target_width=848,
            target_height=480,
            target_frames=81,
        )
        latent = result["latent"]  # torch.Tensor
        encoder.cleanup()
    """

    def __init__(
        self,
        model_path: str,
        dtype: str = "bf16",
        device: str = "cuda",
        vae_path: str | None = None,
    ) -> None:
        """Initialize the VAE encoder (model loaded lazily on first encode).

        Args:
            model_path: Path to Wan model directory or HuggingFace model ID.
                The VAE is loaded from the 'vae' subdirectory. Used as
                fallback when vae_path is not set.
            dtype: Tensor dtype ('bf16', 'fp16', 'fp32').
            device: Target device ('cuda', 'cpu').
            vae_path: Optional path to a single VAE .safetensors file.
                When set, loads via from_single_file() instead of from
                a Diffusers directory.
        """
        self._model_path = model_path
        self._dtype_str = dtype
        self._device = device
        self._vae_path = vae_path
        self._vae = None  # Lazy loaded

    @property
    def encoder_id(self) -> str:
        """Unique identifier for this encoder configuration."""
        return f"{self._model_path}/vae/{self._dtype_str}"

    @property
    def signal_type(self) -> str:
        """Signal type: 'latent' for VAE encodings."""
        return "latent"

    def _ensure_vae(self) -> None:
        """Load the VAE model if not already loaded.

        Raises:
            EncoderError: If loading fails.
        """
        if self._vae is not None:
            return

        try:
            import torch
            from diffusers import AutoencoderKLWan
        except ImportError as e:
            raise EncoderError(
                "wan_vae",
                f"Required packages not installed: {e}\n"
                f"Install with: pip install 'flimmer[wan]'"
            )

        dtype_map = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }
        dtype = dtype_map.get(self._dtype_str, torch.bfloat16)

        try:
            # Priority 1: Direct path to a single .safetensors file
            if self._vae_path is not None:
                vae_file = Path(self._vae_path)
                if not vae_file.is_file():
                    raise EncoderError(
                        "wan_vae",
                        f"VAE file not found: '{self._vae_path}'"
                    )
                self._vae = AutoencoderKLWan.from_single_file(
                    str(vae_file),
                    torch_dtype=dtype,
                )
            # Priority 2: Diffusers directory or HuggingFace ID
            else:
                model_path = Path(self._model_path)
                if model_path.is_dir():
                    self._vae = AutoencoderKLWan.from_pretrained(
                        str(model_path),
                        subfolder="vae",
                        torch_dtype=dtype,
                    )
                else:
                    self._vae = AutoencoderKLWan.from_pretrained(
                        self._model_path,
                        subfolder="vae",
                        torch_dtype=dtype,
                    )
            self._vae.to(self._device)
            self._vae.eval()
        except EncoderError:
            raise
        except Exception as e:
            source = self._vae_path if self._vae_path else self._model_path
            raise EncoderError(
                "wan_vae",
                f"Failed to load Wan VAE from '{source}': {e}\n"
                f"Check that the path is correct."
            ) from e

    def _encode_image_as_latent(
        self,
        image_path: Path,
        target_width: int,
        target_height: int,
    ) -> dict[str, Any]:
        """Encode a single image through VAE as a 1-frame target latent.

        Same PIL-based loading and normalization as encode_reference_image(),
        but returns {"latent": tensor} with the batch dimension kept — matching
        the shape convention of video latents [1, C, 1, H//8, W//8].

        This avoids invoking ffmpeg for single images, which is faster and
        more reliable (ffmpeg can fail on unusual image formats).

        Args:
            image_path: Path to the image file (PNG, JPG, etc.).
            target_width: Target width in pixels (must be multiple of 8).
            target_height: Target height in pixels (must be multiple of 8).

        Returns:
            Dict with 'latent' key containing tensor of shape
            [1, C, 1, H//8, W//8].

        Raises:
            EncoderError: If loading or encoding fails.
        """
        self._ensure_vae()

        try:
            import torch
            from PIL import Image
            import numpy as np
        except ImportError as e:
            raise EncoderError("wan_vae", f"Required packages not installed: {e}")

        if not image_path.is_file():
            raise EncoderError(
                "wan_vae",
                f"Image file not found: '{image_path}'"
            )

        try:
            # Load and resize image
            img = Image.open(image_path).convert("RGB")
            if img.size != (target_width, target_height):
                img = img.resize((target_width, target_height), Image.LANCZOS)

            # Convert to tensor and normalize to [-1, 1]
            pixels = np.array(img, dtype=np.float32)
            pixels = (pixels / 127.5) - 1.0

            # Reshape: [H, W, C] -> [1, C, 1, H, W] (batch=1, single frame)
            pixels = np.transpose(pixels, (2, 0, 1))  # [C, H, W]
            tensor = torch.from_numpy(pixels).unsqueeze(0).unsqueeze(2)  # [1, C, 1, H, W]

            # Convert to model dtype and device
            dtype_map = {
                "bf16": torch.bfloat16,
                "fp16": torch.float16,
                "fp32": torch.float32,
            }
            dtype = dtype_map.get(self._dtype_str, torch.bfloat16)
            tensor = tensor.to(dtype=dtype, device=self._device)

            # Encode through VAE
            with torch.no_grad():
                latent_dist = self._vae.encode(tensor)
                if hasattr(latent_dist, "latent_dist"):
                    latent = latent_dist.latent_dist.sample()
                elif hasattr(latent_dist, "sample"):
                    latent = latent_dist.sample()
                else:
                    latent = latent_dist

            # Apply scaling factor if the VAE uses one
            if hasattr(self._vae.config, "scaling_factor"):
                latent = latent * self._vae.config.scaling_factor

            # Keep batch dimension: [1, C, 1, H//8, W//8]
            return {"latent": latent.cpu()}

        except EncoderError:
            raise
        except Exception as e:
            raise EncoderError(
                "wan_vae",
                f"Image encoding failed for '{image_path}': {e}"
            ) from e

    def encode(self, input_path: str, **kwargs: Any) -> dict[str, Any]:
        """Encode a video or image through the VAE.

        Images (.jpg, .png, etc.) are routed through PIL for efficiency.
        Videos are routed through ffmpeg for frame extraction.

        Args:
            input_path: Path to the video or image file.
            target_width: Target width in pixels (must be multiple of 8).
            target_height: Target height in pixels (must be multiple of 8).
            target_frames: Number of frames to extract and encode.
            frame_extraction: 'head' or 'uniform'.

        Returns:
            Dict with 'latent' key containing the encoded tensor.

        Raises:
            EncoderError: If encoding fails (VRAM, model loading, etc.).
        """
        self._ensure_vae()

        try:
            import torch
        except ImportError as e:
            raise EncoderError("wan_vae", f"torch required: {e}")

        target_width = kwargs.get("target_width", 848)
        target_height = kwargs.get("target_height", 480)
        target_frames = kwargs.get("target_frames", 81)

        # VAE requires dimensions to be multiples of 8
        if target_width % 8 != 0 or target_height % 8 != 0:
            raise EncoderError(
                "wan_vae",
                f"Dimensions must be multiples of 8, got {target_width}x{target_height}. "
                f"Try {target_width - target_width % 8}x{target_height - target_height % 8} instead."
            )

        # Route images through PIL (faster, no subprocess overhead)
        input_p = Path(input_path)
        if target_frames == 1 or input_p.suffix.lower() in IMAGE_EXTENSIONS:
            return self._encode_image_as_latent(input_p, target_width, target_height)

        frame_extraction = kwargs.get("frame_extraction", "head")

        try:
            # Step 1: Extract frames from video using ffmpeg
            pixels = self._load_frames(
                input_path,
                target_width=target_width,
                target_height=target_height,
                target_frames=target_frames,
                frame_extraction=frame_extraction,
            )

            # Step 2: Encode through VAE
            # pixels shape: [1, C, F, H, W] in [-1, 1]
            with torch.no_grad():
                latent_dist = self._vae.encode(pixels.to(self._device))
                if hasattr(latent_dist, "latent_dist"):
                    latent = latent_dist.latent_dist.sample()
                elif hasattr(latent_dist, "sample"):
                    latent = latent_dist.sample()
                else:
                    latent = latent_dist

            # Step 3: Apply scaling if the VAE has config for it
            if hasattr(self._vae.config, "scaling_factor"):
                latent = latent * self._vae.config.scaling_factor

            return {"latent": latent.cpu()}

        except EncoderError:
            raise
        except Exception as e:
            raise EncoderError(
                "wan_vae",
                f"VAE encoding failed for '{input_path}': {e}"
            ) from e

    def _load_frames(
        self,
        input_path: str,
        target_width: int,
        target_height: int,
        target_frames: int,
        frame_extraction: str = "head",
    ) -> Any:
        """Load and preprocess video frames for VAE encoding.

        Uses ffmpeg to extract, resize, and normalize frames.

        Args:
            input_path: Path to video file.
            target_width: Output width.
            target_height: Output height.
            target_frames: Number of frames to extract.
            frame_extraction: 'head' (first N frames) or 'uniform'.

        Returns:
            torch.Tensor of shape [1, 3, F, H, W] in [-1, 1].
        """
        import torch
        import numpy as np

        input_p = Path(input_path)
        if not input_p.is_file():
            raise EncoderError(
                "wan_vae",
                f"Input file not found: '{input_path}'"
            )

        # Use ffmpeg to extract raw frames as RGB bytes
        cmd = [
            "ffmpeg", "-i", str(input_p),
            "-vf", f"scale={target_width}:{target_height}",
            "-frames:v", str(target_frames),
            "-pix_fmt", "rgb24",
            "-f", "rawvideo",
            "-v", "error",
            "pipe:1",
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                check=True,
                timeout=120,
            )
        except FileNotFoundError:
            raise EncoderError(
                "wan_vae",
                "ffmpeg not found. Install ffmpeg and ensure it's on PATH."
            )
        except subprocess.CalledProcessError as e:
            raise EncoderError(
                "wan_vae",
                f"ffmpeg failed for '{input_path}': {e.stderr.decode()}"
            )
        except subprocess.TimeoutExpired:
            raise EncoderError(
                "wan_vae",
                f"ffmpeg timed out processing '{input_path}'"
            )

        # Parse raw bytes into numpy array
        raw = np.frombuffer(result.stdout, dtype=np.uint8)
        actual_frames = len(raw) // (target_height * target_width * 3)

        if actual_frames == 0:
            raise EncoderError(
                "wan_vae",
                f"No frames extracted from '{input_path}'. "
                f"Check that the file is a valid video."
            )

        # Reshape: [F, H, W, C] → normalize → [1, C, F, H, W]
        frames_to_use = min(actual_frames, target_frames)
        raw = raw[: frames_to_use * target_height * target_width * 3]
        frames = raw.reshape(frames_to_use, target_height, target_width, 3)

        # Convert to float and normalize to [-1, 1]
        frames_float = frames.astype(np.float32) / 127.5 - 1.0

        # Rearrange: [F, H, W, C] → [C, F, H, W]
        frames_float = np.transpose(frames_float, (3, 0, 1, 2))

        # Add batch dimension: [1, C, F, H, W]
        tensor = torch.from_numpy(frames_float).unsqueeze(0)

        # Convert to model dtype
        dtype_map = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }
        tensor = tensor.to(dtype_map.get(self._dtype_str, torch.bfloat16))

        return tensor

    def encode_reference_image(
        self,
        image_path: Path,
        target_width: int,
        target_height: int,
    ) -> dict[str, "torch.Tensor"]:
        """Encode a single reference image through the VAE for I2V conditioning.

        Loads a PNG/JPG, resizes to the target resolution (matching the video
        bucket), normalizes to [-1, 1], and encodes through the VAE as a
        single frame. This produces the first-frame latent that I2V models
        use as conditioning — without it, I2V trains as T2V (16 channels
        instead of 36).

        Uses PIL instead of ffmpeg because it's a single image — no process
        overhead, simpler error handling.

        Args:
            image_path: Path to the reference image (PNG or JPG).
            target_width: Target width in pixels (must be multiple of 8).
            target_height: Target height in pixels (must be multiple of 8).

        Returns:
            Dict with 'reference' key containing tensor of shape
            [C, 1, H//8, W//8] (latent channels, 1 temporal frame,
            spatially compressed).

        Raises:
            EncoderError: If loading or encoding fails.
        """
        self._ensure_vae()

        try:
            import torch
            from PIL import Image
        except ImportError as e:
            raise EncoderError("wan_vae", f"Required packages not installed: {e}")

        if not image_path.is_file():
            raise EncoderError(
                "wan_vae",
                f"Reference image not found: '{image_path}'"
            )

        try:
            # Load and resize image
            img = Image.open(image_path).convert("RGB")
            if img.size != (target_width, target_height):
                img = img.resize((target_width, target_height), Image.LANCZOS)

            # Convert to tensor and normalize to [-1, 1]
            import numpy as np
            pixels = np.array(img, dtype=np.float32)
            pixels = (pixels / 127.5) - 1.0

            # Reshape: [H, W, C] -> [1, C, 1, H, W] (batch=1, single frame)
            pixels = np.transpose(pixels, (2, 0, 1))  # [C, H, W]
            tensor = torch.from_numpy(pixels).unsqueeze(0).unsqueeze(2)  # [1, C, 1, H, W]

            # Convert to model dtype and device
            dtype_map = {
                "bf16": torch.bfloat16,
                "fp16": torch.float16,
                "fp32": torch.float32,
            }
            dtype = dtype_map.get(self._dtype_str, torch.bfloat16)
            tensor = tensor.to(dtype=dtype, device=self._device)

            # Encode through VAE
            with torch.no_grad():
                latent_dist = self._vae.encode(tensor)
                if hasattr(latent_dist, "latent_dist"):
                    latent = latent_dist.latent_dist.sample()
                elif hasattr(latent_dist, "sample"):
                    latent = latent_dist.sample()
                else:
                    latent = latent_dist

            # Apply scaling factor if the VAE uses one
            if hasattr(self._vae.config, "scaling_factor"):
                latent = latent * self._vae.config.scaling_factor

            # Squeeze batch dim: [1, C, 1, H//8, W//8] -> [C, 1, H//8, W//8]
            return {"reference": latent.squeeze(0).cpu()}

        except EncoderError:
            raise
        except Exception as e:
            raise EncoderError(
                "wan_vae",
                f"Reference image encoding failed for '{image_path}': {e}"
            ) from e

    def cleanup(self) -> None:
        """Release GPU memory.

        Uses gc.collect() before torch.cuda.empty_cache() to ensure
        Python objects holding CUDA tensor references are freed first.
        Without gc.collect(), empty_cache() cannot reclaim memory held
        by unreachable but not-yet-collected Python objects.
        """
        if self._vae is not None:
            del self._vae
            self._vae = None
            try:
                import gc
                import torch
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
