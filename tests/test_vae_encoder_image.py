"""Tests for image encoding in WanVaeEncoder.

Tests cover:
    - _encode_image_as_latent(): PIL-based single image encoding
    - encode() routing: images go through PIL, videos through ffmpeg
    - No regression for existing video encoding path
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from flimmer.encoding.errors import EncoderError
from flimmer.encoding.vae_encoder import WanVaeEncoder

_has_torch = False
try:
    import torch  # noqa: F401
    _has_torch = True
except ImportError:
    pass


def _make_encoder_with_mock_vae() -> WanVaeEncoder:
    """Create a WanVaeEncoder with a mocked VAE that returns fake latents."""
    import torch

    encoder = WanVaeEncoder(model_path="dummy", dtype="fp32", device="cpu")

    mock_vae = MagicMock()
    mock_config = MagicMock()
    mock_config.scaling_factor = 1.0
    mock_vae.config = mock_config

    def fake_encode(pixel_values):
        b, c, f, h, w = pixel_values.shape
        latent = torch.randn(b, 16, f, h // 8, w // 8, dtype=pixel_values.dtype)
        result = MagicMock()
        result.latent_dist = MagicMock()
        result.latent_dist.sample.return_value = latent
        return result

    mock_vae.encode = fake_encode
    encoder._vae = mock_vae

    return encoder


# ---------------------------------------------------------------------------
# _encode_image_as_latent
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _has_torch, reason="PyTorch not installed")
class TestEncodeImageAsLatent:
    """Tests for _encode_image_as_latent — PIL-based single image encoding."""

    def test_returns_latent_key(self, tmp_path: Path) -> None:
        """_encode_image_as_latent returns dict with 'latent' key."""
        from PIL import Image

        img = Image.new("RGB", (64, 64), color=(128, 128, 128))
        img_path = tmp_path / "test.jpg"
        img.save(img_path)

        encoder = _make_encoder_with_mock_vae()
        result = encoder._encode_image_as_latent(img_path, target_width=64, target_height=64)

        assert "latent" in result

    def test_output_shape(self, tmp_path: Path) -> None:
        """Output tensor has shape [1, C, 1, H//8, W//8]."""
        from PIL import Image

        img = Image.new("RGB", (128, 64), color=(200, 100, 50))
        img_path = tmp_path / "test.png"
        img.save(img_path)

        encoder = _make_encoder_with_mock_vae()
        result = encoder._encode_image_as_latent(img_path, target_width=128, target_height=64)

        latent = result["latent"]
        assert latent.ndim == 5  # [1, C, 1, H//8, W//8]
        assert latent.shape[0] == 1  # batch
        assert latent.shape[2] == 1  # single frame
        assert latent.shape[3] == 64 // 8  # H // 8
        assert latent.shape[4] == 128 // 8  # W // 8

    def test_resizes_to_target(self, tmp_path: Path) -> None:
        """Image is resized to target dimensions before encoding."""
        from PIL import Image

        img = Image.new("RGB", (200, 300), color=(128, 128, 128))
        img_path = tmp_path / "big.png"
        img.save(img_path)

        encoder = _make_encoder_with_mock_vae()
        result = encoder._encode_image_as_latent(img_path, target_width=64, target_height=64)

        latent = result["latent"]
        assert latent.shape[3] == 64 // 8
        assert latent.shape[4] == 64 // 8

    def test_cpu_output(self, tmp_path: Path) -> None:
        """Output tensor is on CPU (for safetensors serialization)."""
        from PIL import Image

        img = Image.new("RGB", (64, 64), color=(0, 0, 0))
        img_path = tmp_path / "test.png"
        img.save(img_path)

        encoder = _make_encoder_with_mock_vae()
        result = encoder._encode_image_as_latent(img_path, target_width=64, target_height=64)

        assert result["latent"].device.type == "cpu"


# ---------------------------------------------------------------------------
# encode() routing
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _has_torch, reason="PyTorch not installed")
class TestEncodeRouting:
    """Tests for encode() routing images vs videos."""

    def test_routes_image_extension_through_pil(self, tmp_path: Path) -> None:
        """encode() with .jpg file routes to _encode_image_as_latent, not ffmpeg."""
        from PIL import Image

        img = Image.new("RGB", (64, 64), color=(128, 128, 128))
        img_path = tmp_path / "test.jpg"
        img.save(img_path)

        encoder = _make_encoder_with_mock_vae()
        # Patch _load_frames to detect if ffmpeg path is called
        with patch.object(encoder, "_load_frames") as mock_load_frames:
            result = encoder.encode(
                str(img_path),
                target_width=64,
                target_height=64,
                target_frames=1,
            )

            # ffmpeg path (_load_frames) should NOT have been called
            mock_load_frames.assert_not_called()
            # Should have "latent" key from PIL path
            assert "latent" in result

    def test_routes_png_through_pil(self, tmp_path: Path) -> None:
        """encode() with .png file routes through PIL."""
        from PIL import Image

        img = Image.new("RGB", (64, 64), color=(128, 128, 128))
        img_path = tmp_path / "test.png"
        img.save(img_path)

        encoder = _make_encoder_with_mock_vae()
        with patch.object(encoder, "_load_frames") as mock_load_frames:
            result = encoder.encode(
                str(img_path),
                target_width=64,
                target_height=64,
                target_frames=1,
            )
            mock_load_frames.assert_not_called()
            assert "latent" in result

    def test_routes_target_frames_1_through_pil(self, tmp_path: Path) -> None:
        """encode() with target_frames=1 routes through PIL regardless of extension."""
        from PIL import Image

        img = Image.new("RGB", (64, 64), color=(128, 128, 128))
        # Even with a .mp4 extension but target_frames=1 it could be routed
        # But we use a .png to be sure
        img_path = tmp_path / "test.png"
        img.save(img_path)

        encoder = _make_encoder_with_mock_vae()
        with patch.object(encoder, "_load_frames") as mock_load_frames:
            result = encoder.encode(
                str(img_path),
                target_width=64,
                target_height=64,
                target_frames=1,
            )
            mock_load_frames.assert_not_called()

    def test_video_still_uses_ffmpeg(self, tmp_path: Path) -> None:
        """encode() with .mp4 and target_frames > 1 still uses ffmpeg path."""
        import torch

        encoder = _make_encoder_with_mock_vae()

        # Mock _load_frames to return a valid tensor (simulating ffmpeg path)
        fake_tensor = torch.randn(1, 3, 17, 64, 64, dtype=torch.float32)
        with patch.object(encoder, "_load_frames", return_value=fake_tensor) as mock_load:
            result = encoder.encode(
                str(tmp_path / "clip.mp4"),
                target_width=64,
                target_height=64,
                target_frames=17,
            )
            # ffmpeg path SHOULD be called for video files
            mock_load.assert_called_once()
            assert "latent" in result
