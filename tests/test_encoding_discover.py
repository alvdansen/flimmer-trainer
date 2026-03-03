"""Tests for flimmer.encoding.discover — sample discovery.

Tests cover:
    - _classify_target_role(): video vs image classification
    - discover_from_directory(): flat and flimmer layouts
    - discover_from_manifest(): JSON manifest reading
    - discover_samples(): unified entry point auto-detection
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from flimmer.encoding.discover import (
    _classify_target_role,
    discover_from_directory,
    discover_from_manifest,
    discover_samples,
)
from flimmer.encoding.errors import FlimmerEncodingError
from flimmer.encoding.models import SampleRole


# ---------------------------------------------------------------------------
# Role classification
# ---------------------------------------------------------------------------

class TestClassifyTargetRole:
    """Tests for _classify_target_role."""

    @pytest.mark.parametrize("ext", [".mp4", ".mov", ".avi", ".mkv", ".webm"])
    def test_video_extensions(self, ext: str) -> None:
        assert _classify_target_role(Path(f"/test{ext}")) == SampleRole.TARGET_VIDEO

    @pytest.mark.parametrize("ext", [".png", ".jpg", ".jpeg", ".webp"])
    def test_image_extensions(self, ext: str) -> None:
        assert _classify_target_role(Path(f"/test{ext}")) == SampleRole.TARGET_IMAGE

    def test_unknown_defaults_to_video(self) -> None:
        assert _classify_target_role(Path("/test.xyz")) == SampleRole.TARGET_VIDEO


# ---------------------------------------------------------------------------
# Directory discovery
# ---------------------------------------------------------------------------

class TestDiscoverFromDirectory:
    """Tests for discover_from_directory with tmp_path fixtures."""

    def test_flat_layout_video(self, tmp_path: Path) -> None:
        """Discovers video samples in flat layout."""
        (tmp_path / "clip_001.mp4").write_bytes(b"\x00" * 100)
        (tmp_path / "clip_001.txt").write_text("A scene", encoding="utf-8")

        samples = discover_from_directory(tmp_path, probe=False)
        assert len(samples) == 1
        assert samples[0].stem == "clip_001"
        assert samples[0].target_role == SampleRole.TARGET_VIDEO
        assert samples[0].caption is not None

    def test_flat_layout_image_only_is_reference(self, tmp_path: Path) -> None:
        """In flat layout, standalone images are classified as references, not targets.

        The existing dataset discovery treats images as 'references'. An image
        without a matching video target creates no SamplePair. Image targets
        must be specified via manifest or mixed with videos in the dataset.
        """
        (tmp_path / "photo_001.png").write_bytes(b"\x89PNG" + b"\x00" * 100)
        (tmp_path / "photo_001.txt").write_text("A photo", encoding="utf-8")

        # No video target → no sample pair → no discovered sample
        samples = discover_from_directory(tmp_path, probe=False)
        assert len(samples) == 0

    def test_flat_layout_image_via_manifest(self, tmp_path: Path) -> None:
        """Image targets CAN be discovered via manifest (explicit role assignment)."""
        (tmp_path / "photo_001.png").write_bytes(b"\x89PNG" + b"\x00" * 100)
        manifest = {
            "samples": [{
                "stem": "photo_001",
                "target": "photo_001.png",
                "width": 1024,
                "height": 768,
                "frame_count": 1,
            }],
        }
        manifest_path = tmp_path / "manifest.json"
        import json
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        samples = discover_from_manifest(manifest_path)
        assert len(samples) == 1
        assert samples[0].target_role == SampleRole.TARGET_IMAGE

    def test_flat_layout_with_reference(self, tmp_path: Path) -> None:
        """Discovers samples with video + caption + reference image."""
        (tmp_path / "clip_001.mp4").write_bytes(b"\x00" * 100)
        (tmp_path / "clip_001.txt").write_text("caption", encoding="utf-8")
        (tmp_path / "clip_001.png").write_bytes(b"\x89PNG" + b"\x00" * 100)

        samples = discover_from_directory(tmp_path, probe=False)
        assert len(samples) == 1
        assert samples[0].caption is not None
        assert samples[0].reference is not None

    def test_flimmer_layout(self, tmp_path: Path) -> None:
        """Discovers samples in flimmer hierarchical layout."""
        targets = tmp_path / "training" / "targets"
        captions = tmp_path / "training" / "signals" / "captions"
        targets.mkdir(parents=True)
        captions.mkdir(parents=True)

        (targets / "clip_001.mp4").write_bytes(b"\x00" * 100)
        (captions / "clip_001.txt").write_text("caption", encoding="utf-8")

        samples = discover_from_directory(tmp_path, probe=False)
        assert len(samples) == 1
        assert samples[0].stem == "clip_001"

    def test_multiple_samples(self, tmp_path: Path) -> None:
        """Discovers multiple samples."""
        for i in range(5):
            (tmp_path / f"clip_{i:03d}.mp4").write_bytes(b"\x00" * 100)
            (tmp_path / f"clip_{i:03d}.txt").write_text(f"Scene {i}", encoding="utf-8")

        samples = discover_from_directory(tmp_path, probe=False)
        assert len(samples) == 5

    def test_no_caption_without_requirement(self, tmp_path: Path) -> None:
        """Samples without captions are included when not required."""
        (tmp_path / "clip_001.mp4").write_bytes(b"\x00" * 100)

        samples = discover_from_directory(tmp_path, caption_required=False, probe=False)
        assert len(samples) == 1
        assert samples[0].caption is None

    def test_caption_required_filters(self, tmp_path: Path) -> None:
        """Samples without captions are skipped when required."""
        (tmp_path / "clip_001.mp4").write_bytes(b"\x00" * 100)  # no caption

        samples = discover_from_directory(tmp_path, caption_required=True, probe=False)
        assert len(samples) == 0

    def test_empty_directory(self, tmp_path: Path) -> None:
        """Empty directory returns empty list."""
        samples = discover_from_directory(tmp_path, probe=False)
        assert len(samples) == 0

    def test_nonexistent_directory(self) -> None:
        """Raises error for missing directory."""
        with pytest.raises(FlimmerEncodingError, match="not found"):
            discover_from_directory("/nonexistent/path", probe=False)


# ---------------------------------------------------------------------------
# Manifest discovery
# ---------------------------------------------------------------------------

class TestDiscoverFromManifest:
    """Tests for discover_from_manifest."""

    def test_basic_manifest(self, tmp_path: Path) -> None:
        """Reads samples from a manifest file."""
        # Create a target file so path resolution works
        (tmp_path / "clip_001.mp4").write_bytes(b"\x00" * 100)
        (tmp_path / "clip_001.txt").write_text("caption", encoding="utf-8")

        manifest = {
            "samples": [
                {
                    "stem": "clip_001",
                    "target": "clip_001.mp4",
                    "caption": "clip_001.txt",
                    "width": 848,
                    "height": 480,
                    "frame_count": 81,
                    "fps": 16.0,
                    "duration": 5.0625,
                },
            ],
        }
        manifest_path = tmp_path / "flimmer_manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        samples = discover_from_manifest(manifest_path)
        assert len(samples) == 1
        assert samples[0].stem == "clip_001"
        assert samples[0].width == 848
        assert samples[0].frame_count == 81

    def test_manifest_with_repeats(self, tmp_path: Path) -> None:
        """Reads per-sample repeats from manifest."""
        manifest = {
            "samples": [
                {
                    "stem": "clip_001",
                    "target": "clip_001.mp4",
                    "repeats": 5,
                    "loss_multiplier": 2.0,
                },
            ],
        }
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        (tmp_path / "clip_001.mp4").write_bytes(b"\x00")

        samples = discover_from_manifest(manifest_path)
        assert samples[0].repeats == 5
        assert samples[0].loss_multiplier == 2.0

    def test_missing_manifest(self) -> None:
        with pytest.raises(FlimmerEncodingError, match="not found"):
            discover_from_manifest("/nonexistent/manifest.json")

    def test_invalid_json(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text("{invalid", encoding="utf-8")
        with pytest.raises(FlimmerEncodingError, match="parse"):
            discover_from_manifest(bad)

    def test_empty_samples_list(self, tmp_path: Path) -> None:
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text('{"samples": []}', encoding="utf-8")
        samples = discover_from_manifest(manifest_path)
        assert len(samples) == 0

    def test_skips_invalid_entries(self, tmp_path: Path) -> None:
        """Entries missing required fields are skipped."""
        manifest = {
            "samples": [
                {"stem": "clip_001"},  # no target → skipped
                {"target": "clip_002.mp4"},  # no stem → skipped
                42,  # not a dict → skipped
            ],
        }
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        samples = discover_from_manifest(manifest_path)
        assert len(samples) == 0


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

class TestDiscoverSamples:
    """Tests for the discover_samples unified entry point."""

    def test_directory_mode(self, tmp_path: Path) -> None:
        """Auto-detects directory and uses directory discovery."""
        (tmp_path / "clip_001.mp4").write_bytes(b"\x00" * 100)
        samples = discover_samples(tmp_path, probe=False)
        assert len(samples) == 1

    def test_manifest_mode(self, tmp_path: Path) -> None:
        """Auto-detects .json file and uses manifest discovery."""
        manifest = {"samples": []}
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        samples = discover_samples(manifest_path)
        assert len(samples) == 0
