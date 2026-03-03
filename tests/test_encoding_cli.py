"""Tests for flimmer.encoding.__main__ — CLI argument parsing.

Tests cover:
    - build_parser(): argument structure for all commands
    - Command dispatching
"""

from __future__ import annotations

import pytest

from flimmer.encoding.__main__ import build_parser


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
