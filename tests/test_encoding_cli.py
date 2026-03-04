"""Tests for flimmer.encoding.__main__ — CLI argument parsing and helpers.

Tests cover:
    - build_parser(): argument structure for all commands
    - Command dispatching
    - _auto_extract_first_frames(): first-frame reference auto-extraction
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from flimmer.encoding.__main__ import build_parser, _auto_extract_first_frames


class TestBuildParser:
    """Tests for CLI argument parsing."""

    def test_info_command(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["info", "--config", "train.yaml"])
        assert args.command == "info"
        assert args.config == "train.yaml"

    def test_info_short_flag(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["info", "-c", "train.yaml"])
        assert args.config == "train.yaml"

    def test_cache_latents_command(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["cache-latents", "--config", "train.yaml"])
        assert args.command == "cache-latents"
        assert args.config == "train.yaml"
        assert not args.dry_run
        assert not args.force

    def test_cache_latents_dry_run(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["cache-latents", "-c", "t.yaml", "--dry-run"])
        assert args.dry_run is True

    def test_cache_latents_force(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["cache-latents", "-c", "t.yaml", "--force"])
        assert args.force is True

    def test_cache_text_command(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["cache-text", "--config", "train.yaml"])
        assert args.command == "cache-text"
        assert args.config == "train.yaml"

    def test_cache_text_dry_run(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["cache-text", "-c", "t.yaml", "--dry-run"])
        assert args.dry_run is True

    def test_no_command_raises(self) -> None:
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_missing_config_raises(self) -> None:
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["info"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _touch(path: Path, content: bytes = b"") -> Path:
    """Create a file with optional content, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


# ---------------------------------------------------------------------------
# _auto_extract_first_frames tests
# ---------------------------------------------------------------------------

class TestAutoExtractFirstFrames:
    """Tests for the _auto_extract_first_frames helper."""

    @patch("flimmer.video.extract.extract_first_frame")
    def test_extracts_for_flimmer_layout(self, mock_extract, tmp_path: Path) -> None:
        """First frames are extracted to training/signals/first_frame/ for flimmer layout."""
        ds = tmp_path / "ds"
        _touch(ds / "training" / "targets" / "clip_001.mp4")
        _touch(ds / "training" / "targets" / "clip_002.mp4")
        _touch(ds / "training" / "signals" / "captions" / "clip_001.txt", b"cap")
        _touch(ds / "training" / "signals" / "captions" / "clip_002.txt", b"cap")

        count = _auto_extract_first_frames("unused", [ds])

        assert count == 2
        assert mock_extract.call_count == 2
        # Check output paths point to references dir
        for call in mock_extract.call_args_list:
            out_path = call[0][1]  # second positional arg
            assert "first_frame" in str(out_path)
            assert str(out_path).endswith(".png")

    @patch("flimmer.video.extract.extract_first_frame")
    def test_extracts_for_flat_layout(self, mock_extract, tmp_path: Path) -> None:
        """First frames are extracted to same dir for flat layout."""
        _touch(tmp_path / "clip_a.mp4")
        _touch(tmp_path / "clip_a.txt", b"caption")

        count = _auto_extract_first_frames("unused", [tmp_path])

        assert count == 1
        out_path = mock_extract.call_args[0][1]
        assert out_path == tmp_path / "clip_a.png"

    @patch("flimmer.video.extract.extract_first_frame")
    def test_skips_existing_references(self, mock_extract, tmp_path: Path) -> None:
        """Already-existing reference PNGs are not re-extracted."""
        ds = tmp_path / "ds"
        _touch(ds / "training" / "targets" / "clip_001.mp4")
        # Reference already exists
        _touch(ds / "training" / "signals" / "first_frame" / "clip_001.png")

        count = _auto_extract_first_frames("unused", [ds])

        assert count == 0
        mock_extract.assert_not_called()

    @patch("flimmer.video.extract.extract_first_frame")
    def test_skips_image_targets(self, mock_extract, tmp_path: Path) -> None:
        """Image targets (not videos) are skipped — they ARE their own reference."""
        _touch(tmp_path / "still_001.png")
        _touch(tmp_path / "still_001.txt", b"caption")

        count = _auto_extract_first_frames("unused", [tmp_path])

        assert count == 0
        mock_extract.assert_not_called()

    @patch("flimmer.video.extract.extract_first_frame")
    def test_handles_multiple_dataset_dirs(self, mock_extract, tmp_path: Path) -> None:
        """Works across multiple dataset directories."""
        ds1 = tmp_path / "ds1"
        ds2 = tmp_path / "ds2"
        _touch(ds1 / "clip_a.mp4")
        _touch(ds2 / "clip_b.mp4")

        count = _auto_extract_first_frames("unused", [ds1, ds2])

        assert count == 2
        assert mock_extract.call_count == 2
