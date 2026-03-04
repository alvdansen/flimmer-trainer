"""Tests for per-phase dataset configuration on PhaseConfig and ResolvedPhase."""

import pytest

from flimmer.phases.phase_model import PhaseConfig, ResolvedPhase


class TestPhaseConfigDataset:
    """Test PhaseConfig.dataset field."""

    def test_default_dataset_is_none(self) -> None:
        """PhaseConfig with no dataset argument defaults to None."""
        config = PhaseConfig(phase_type="unified")
        assert config.dataset is None

    def test_dataset_stores_path_string(self) -> None:
        """PhaseConfig with dataset="./data/config.yaml" stores the string."""
        config = PhaseConfig(
            phase_type="unified",
            dataset="./data/config.yaml",
        )
        assert config.dataset == "./data/config.yaml"

    def test_dataset_with_absolute_path(self) -> None:
        """PhaseConfig accepts absolute path strings."""
        config = PhaseConfig(
            phase_type="unified",
            dataset="/home/user/data/phase1.yaml",
        )
        assert config.dataset == "/home/user/data/phase1.yaml"

    def test_dataset_none_explicit(self) -> None:
        """PhaseConfig with dataset=None is the same as default."""
        config = PhaseConfig(phase_type="unified", dataset=None)
        assert config.dataset is None


class TestResolvedPhaseDataset:
    """Test ResolvedPhase.dataset field."""

    def test_resolved_phase_dataset_none(self) -> None:
        """ResolvedPhase with dataset=None (inherits from run-level)."""
        resolved = ResolvedPhase(
            phase_type="unified",
            display_name="",
            enabled=True,
            params={"learning_rate": 5e-5},
            extras={},
            dataset=None,
            signals={"text": True},
        )
        assert resolved.dataset is None

    def test_resolved_phase_dataset_explicit(self) -> None:
        """ResolvedPhase with explicit dataset path."""
        resolved = ResolvedPhase(
            phase_type="unified",
            display_name="",
            enabled=True,
            params={"learning_rate": 5e-5},
            extras={},
            dataset="./path.yaml",
            signals={"text": True},
        )
        assert resolved.dataset == "./path.yaml"

    def test_resolved_phase_dataset_accessible_on_frozen(self) -> None:
        """Dataset field is accessible on frozen dataclass instance."""
        resolved = ResolvedPhase(
            phase_type="unified",
            display_name="",
            enabled=True,
            params={},
            extras={},
            dataset="./data.yaml",
            signals={"text": True},
        )
        # Should not raise -- just reading is fine
        _ = resolved.dataset
        assert resolved.dataset == "./data.yaml"
