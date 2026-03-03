"""Tests for flimmer.training.__main__ — CLI entry point."""

import pytest

from flimmer.training.__main__ import build_parser, _StubBackend


class TestParser:
    """CLI argument parser."""

    def test_train_command(self):
        parser = build_parser()
        args = parser.parse_args(["train", "--config", "path/to/config.yaml"])
        assert args.command == "train"
        assert args.config == "path/to/config.yaml"

    def test_plan_command(self):
        parser = build_parser()
        args = parser.parse_args(["plan", "--config", "config.yaml"])
        assert args.command == "plan"

    def test_dry_run_flag(self):
        parser = build_parser()
        args = parser.parse_args(["train", "-c", "config.yaml", "--dry-run"])
        assert args.dry_run is True

    def test_short_flag(self):
        parser = build_parser()
        args = parser.parse_args(["train", "-c", "config.yaml"])
        assert args.config == "config.yaml"

    def test_no_command_errors(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])


class TestStubBackend:
    """Stub backend for dry-run."""

    def test_model_id(self):
        stub = _StubBackend()
        assert stub.model_id == "stub"

    def test_supports_moe(self):
        stub = _StubBackend()
        assert stub.supports_moe is True

    def test_load_model(self):
        stub = _StubBackend()
        assert stub.load_model(None) is None

    def test_get_noise_schedule(self):
        stub = _StubBackend()
        schedule = stub.get_noise_schedule()
        assert schedule.num_timesteps == 1000

    def test_forward(self):
        stub = _StubBackend()
        assert stub.forward(None) is None
