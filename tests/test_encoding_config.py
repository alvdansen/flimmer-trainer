"""Tests for CacheConfig in wan22_training_master.py.

Tests cover:
    - CacheConfig defaults
    - target_frames 4n+1 validation
    - dtype validation
    - frame_extraction validation
    - CacheConfig in FlimmerTrainingConfig
    - Path resolution for cache.cache_dir
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from flimmer.config.wan22_training_master import CacheConfig, FlimmerTrainingConfig


# ---------------------------------------------------------------------------
# CacheConfig standalone
# ---------------------------------------------------------------------------

class TestCacheConfig:
    """Tests for the CacheConfig model."""

    def test_defaults(self) -> None:
        """Default values are sensible."""
        cfg = CacheConfig()
        assert cfg.cache_dir == "./cache"
        assert cfg.dtype == "bf16"
        assert cfg.target_frames == [17, 33, 49, 81]
        assert cfg.frame_extraction == "head"
        assert cfg.include_head_frame is False
        assert cfg.reso_step == 16

    def test_valid_target_frames(self) -> None:
        """Custom target_frames that satisfy 4n+1."""
        cfg = CacheConfig(target_frames=[17, 33])
        assert cfg.target_frames == [17, 33]

    def test_invalid_target_frames_not_4n1(self) -> None:
        """Target frames violating 4n+1 are rejected."""
        with pytest.raises(Exception, match="4n\\+1"):
            CacheConfig(target_frames=[16])

    def test_invalid_target_frames_zero(self) -> None:
        with pytest.raises(Exception, match="positive"):
            CacheConfig(target_frames=[0])

    def test_empty_target_frames(self) -> None:
        with pytest.raises(Exception, match="empty"):
            CacheConfig(target_frames=[])

    def test_invalid_dtype(self) -> None:
        with pytest.raises(Exception, match="dtype"):
            CacheConfig(dtype="int8")

    def test_valid_dtypes(self) -> None:
        for dtype in ("bf16", "fp16", "fp32"):
            cfg = CacheConfig(dtype=dtype)
            assert cfg.dtype == dtype

    def test_invalid_frame_extraction(self) -> None:
        with pytest.raises(Exception, match="frame_extraction"):
            CacheConfig(frame_extraction="random")

    def test_valid_frame_extractions(self) -> None:
        for mode in ("head", "uniform"):
            cfg = CacheConfig(frame_extraction=mode)
            assert cfg.frame_extraction == mode

    def test_reso_step_positive(self) -> None:
        with pytest.raises(Exception):
            CacheConfig(reso_step=0)


# ---------------------------------------------------------------------------
# CacheConfig in root config
# ---------------------------------------------------------------------------

class TestCacheConfigInTraining:
    """Tests for CacheConfig integrated into FlimmerTrainingConfig."""

    def test_default_cache_in_training(self) -> None:
        """Training config includes cache with defaults."""
        config = FlimmerTrainingConfig(
            data_config="./data.yaml",
            model={"variant": "2.2_t2v", "path": "Wan-AI/Wan2.2-T2V-14B"},
        )
        assert config.cache.cache_dir == "./cache"
        assert config.cache.dtype == "bf16"

    def test_custom_cache_in_training(self) -> None:
        """Training config accepts custom cache settings."""
        config = FlimmerTrainingConfig(
            data_config="./data.yaml",
            model={"variant": "2.2_t2v", "path": "Wan-AI/Wan2.2-T2V-14B"},
            cache={"cache_dir": "./my_cache", "dtype": "fp16", "target_frames": [17, 33]},
        )
        assert config.cache.cache_dir == "./my_cache"
        assert config.cache.dtype == "fp16"
        assert config.cache.target_frames == [17, 33]


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

class TestCachePathResolution:
    """Tests for cache.cache_dir path resolution in training_loader."""

    def test_cache_dir_resolved(self, tmp_path: Path) -> None:
        """cache_dir is resolved relative to config file location."""
        from flimmer.config.training_loader import load_training_config

        # Create a minimal training config
        config_path = tmp_path / "train.yaml"
        data_config_path = tmp_path / "data.yaml"
        data_config_path.write_text(
            "datasets:\n  - path: .\n",
            encoding="utf-8",
        )

        config_data = {
            "model": {
                "variant": "2.2_t2v",
                "path": "Wan-AI/Wan2.2-T2V-14B",
            },
            "data_config": "data.yaml",
            "cache": {
                "cache_dir": "./my_cache",
            },
        }
        config_path.write_text(yaml.dump(config_data), encoding="utf-8")

        config = load_training_config(str(config_path))
        # cache_dir should be resolved to absolute path
        resolved = Path(config.cache.cache_dir)
        assert resolved.is_absolute()
        assert "my_cache" in str(resolved)
