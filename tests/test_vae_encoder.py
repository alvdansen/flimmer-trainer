"""Tests for flimmer.encoding.vae_encoder — VAE encoding with mocked models.

Tests cover:
    - encode_reference_image(): single image encoding for I2V first-frame
    - Error handling for missing images
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from flimmer.encoding.errors import EncoderError
from flimmer.encoding.vae_encoder import WanVaeEncoder


# ---------------------------------------------------------------------------
# encode_reference_image
# ---------------------------------------------------------------------------

class TestEncodeReferenceImage:
    """Tests for WanVaeEncoder.encode_reference_image with mocked VAE."""

    def _make_encoder_with_mock_vae(self) -> WanVaeEncoder:
        """Create a WanVaeEncoder with a mocked VAE that returns fake latents."""
        import torch

        encoder = WanVaeEncoder(model_path="dummy", dtype="fp32", device="cpu")

        # Mock the VAE object
        mock_vae = MagicMock()

        # Mock config with scaling_factor
        mock_config = MagicMock()
        mock_config.scaling_factor = 1.0
        mock_vae.config = mock_config

        # Mock encode() to return a fake latent distribution
        # Input shape: [1, 3, 1, H, W] -> output shape: [1, C_latent, 1, H//8, W//8]
        def fake_encode(pixel_values):
            b, c, f, h, w = pixel_values.shape
            latent = torch.randn(b, 16, f, h // 8, w // 8, dtype=pixel_values.dtype)
            result = MagicMock()
            result.latent_dist = MagicMock()
            result.latent_dist.sample.return_value = latent
            return result

        mock_vae.encode = fake_encode

        # Inject the mock VAE (bypass _ensure_vae)
        encoder._vae = mock_vae

        return encoder

    def test_returns_reference_key(self, tmp_path: Path) -> None:
        """encode_reference_image returns dict with 'reference' key."""
        from PIL import Image

        img = Image.new("RGB", (64, 64), color=(128, 128, 128))
        img_path = tmp_path / "test_ref.png"
        img.save(img_path)

        encoder = self._make_encoder_with_mock_vae()
        result = encoder.encode_reference_image(img_path, target_width=64, target_height=64)

        assert "reference" in result
        assert result["reference"].ndim == 4  # [C, 1, H//8, W//8]

    def test_output_shape(self, tmp_path: Path) -> None:
        """Output tensor has correct shape [C, 1, H//8, W//8]."""
        from PIL import Image

        img = Image.new("RGB", (128, 64), color=(200, 100, 50))
        img_path = tmp_path / "ref.png"
        img.save(img_path)

        encoder = self._make_encoder_with_mock_vae()
        result = encoder.encode_reference_image(img_path, target_width=128, target_height=64)

        ref_tensor = result["reference"]
        # Shape should be [C_latent, 1, H//8, W//8]
        assert ref_tensor.shape[1] == 1  # Single frame
        assert ref_tensor.shape[2] == 64 // 8  # H // 8
        assert ref_tensor.shape[3] == 128 // 8  # W // 8

    def test_resizes_to_target(self, tmp_path: Path) -> None:
        """Image is resized to target dimensions before encoding."""
        from PIL import Image

        # Create image with different size than target
        img = Image.new("RGB", (200, 300), color=(128, 128, 128))
        img_path = tmp_path / "ref_big.png"
        img.save(img_path)

        encoder = self._make_encoder_with_mock_vae()
        result = encoder.encode_reference_image(img_path, target_width=64, target_height=64)

        ref_tensor = result["reference"]
        # Should be based on target dimensions, not original image dimensions
        assert ref_tensor.shape[2] == 64 // 8
        assert ref_tensor.shape[3] == 64 // 8

    def test_missing_image_raises_error(self, tmp_path: Path) -> None:
        """Raises EncoderError when image file doesn't exist."""
        encoder = self._make_encoder_with_mock_vae()
        missing_path = tmp_path / "nonexistent.png"

        with pytest.raises(EncoderError, match="not found"):
            encoder.encode_reference_image(missing_path, target_width=64, target_height=64)

    def test_cpu_output(self, tmp_path: Path) -> None:
        """Output tensor is on CPU (for safetensors serialization)."""
        from PIL import Image

        img = Image.new("RGB", (64, 64), color=(0, 0, 0))
        img_path = tmp_path / "ref.png"
        img.save(img_path)

        encoder = self._make_encoder_with_mock_vae()
        result = encoder.encode_reference_image(img_path, target_width=64, target_height=64)

        assert result["reference"].device.type == "cpu"
