"""Tests for flimmer.training.__main__ — CLI entry point."""

import logging
import os

import pytest

from flimmer.training.__main__ import build_parser, _StubBackend, _configure_cuda_allocator


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


class TestConfigureCudaAllocator:
    """FIX-02: _configure_cuda_allocator sets PYTORCH_CUDA_ALLOC_CONF."""

    ENV_KEY = "PYTORCH_CUDA_ALLOC_CONF"

    def test_sets_expandable_segments_when_unset(self, monkeypatch):
        """When env var is unset, set it to expandable_segments:True."""
        monkeypatch.delenv(self.ENV_KEY, raising=False)
        _configure_cuda_allocator()
        assert os.environ[self.ENV_KEY] == "expandable_segments:True"

    def test_appends_to_existing_value(self, monkeypatch):
        """When env var has other settings, append expandable_segments:True."""
        monkeypatch.setenv(self.ENV_KEY, "max_split_size_mb:512")
        _configure_cuda_allocator()
        assert os.environ[self.ENV_KEY] == "max_split_size_mb:512,expandable_segments:True"

    def test_does_not_modify_when_already_present(self, monkeypatch):
        """When expandable_segments is already in the env var, leave it alone."""
        original = "expandable_segments:True,max_split_size_mb:512"
        monkeypatch.setenv(self.ENV_KEY, original)
        _configure_cuda_allocator()
        assert os.environ[self.ENV_KEY] == original

    def test_logs_info_when_setting(self, monkeypatch, caplog):
        """Info log when setting the env var."""
        monkeypatch.delenv(self.ENV_KEY, raising=False)
        with caplog.at_level(logging.INFO, logger="flimmer.training.__main__"):
            _configure_cuda_allocator()
        assert any("expandable_segments" in msg for msg in caplog.messages)

    def test_logs_info_when_already_set(self, monkeypatch, caplog):
        """Info log (different message) when env var already has expandable_segments."""
        monkeypatch.setenv(self.ENV_KEY, "expandable_segments:True")
        with caplog.at_level(logging.INFO, logger="flimmer.training.__main__"):
            _configure_cuda_allocator()
        assert any("already set" in msg.lower() for msg in caplog.messages)
