"""Tests for per-phase signal configuration on PhaseConfig and ResolvedPhase."""

import pytest

from flimmer.phases import ModelDefinition, PhaseConfigError
from flimmer.phases.phase_model import PhaseConfig, ResolvedPhase


class TestPhaseConfigSignals:
    """Test PhaseConfig.signals field."""

    def test_default_signals_is_none(self) -> None:
        """PhaseConfig with no signals argument defaults to None."""
        config = PhaseConfig(phase_type="unified")
        assert config.signals is None

    def test_signals_stores_dict(self) -> None:
        """PhaseConfig with signals dict stores the selective overrides."""
        config = PhaseConfig(
            phase_type="unified",
            signals={"text": True, "reference_image": False},
        )
        assert config.signals == {"text": True, "reference_image": False}

    def test_signals_none_explicit(self) -> None:
        """PhaseConfig with signals=None is the same as default (inherit all)."""
        config = PhaseConfig(phase_type="unified", signals=None)
        assert config.signals is None

    def test_signals_empty_dict(self) -> None:
        """PhaseConfig with signals={} means no overrides (all inherited)."""
        config = PhaseConfig(phase_type="unified", signals={})
        assert config.signals == {}


class TestPhaseConfigSignalValidation:
    """Test validate_against() signal validation."""

    def test_rejects_unsupported_modality(
        self, sample_model_definition: ModelDefinition
    ) -> None:
        """Signal referencing modality not in model's supported_signals raises."""
        config = PhaseConfig(
            phase_type="unified",
            signals={"audio": True},  # "audio" not in sample model
        )
        with pytest.raises(PhaseConfigError, match="audio"):
            config.validate_against(sample_model_definition)

    def test_accepts_supported_modalities(
        self, sample_model_definition: ModelDefinition
    ) -> None:
        """Signals referencing supported modalities passes validation."""
        config = PhaseConfig(
            phase_type="unified",
            signals={"text": True, "reference_image": False},
        )
        # Should not raise
        config.validate_against(sample_model_definition)

    def test_accepts_signals_none(
        self, sample_model_definition: ModelDefinition
    ) -> None:
        """signals=None means inherit all -- passes validation."""
        config = PhaseConfig(
            phase_type="unified",
            signals=None,
        )
        config.validate_against(sample_model_definition)

    def test_signal_validation_after_phase_type_check(
        self, sample_model_definition: ModelDefinition
    ) -> None:
        """Invalid phase_type raises BEFORE signal check (ordering per Pitfall 4).

        A config with both invalid phase_type and invalid signals should raise
        about the phase_type, not about signals.
        """
        config = PhaseConfig(
            phase_type="nonexistent",
            signals={"audio": True},  # Also invalid, but phase_type checked first
        )
        with pytest.raises(PhaseConfigError, match="Unknown phase type"):
            config.validate_against(sample_model_definition)

    def test_rejects_multiple_unsupported_modalities(
        self, sample_model_definition: ModelDefinition
    ) -> None:
        """Multiple unsupported modalities still caught."""
        config = PhaseConfig(
            phase_type="unified",
            signals={"audio": True, "lidar": False},
        )
        with pytest.raises(PhaseConfigError):
            config.validate_against(sample_model_definition)

    def test_accepts_all_model_signals(
        self, sample_model_definition: ModelDefinition
    ) -> None:
        """Specifying every supported modality passes."""
        config = PhaseConfig(
            phase_type="unified",
            signals={
                "text": True,
                "video": True,
                "reference_image": False,
                "custom": True,
            },
        )
        config.validate_against(sample_model_definition)


class TestResolvedPhaseSignals:
    """Test ResolvedPhase.signals field."""

    def test_resolved_phase_has_signals(self) -> None:
        """ResolvedPhase carries a fully-specified signals dict."""
        resolved = ResolvedPhase(
            phase_type="unified",
            display_name="",
            enabled=True,
            params={},
            extras={},
            dataset=None,
            signals={"text": True, "video": True, "reference_image": False},
        )
        assert resolved.signals == {
            "text": True,
            "video": True,
            "reference_image": False,
        }

    def test_resolved_phase_signals_accessible_on_frozen(self) -> None:
        """Signals field is accessible on frozen dataclass instance."""
        resolved = ResolvedPhase(
            phase_type="unified",
            display_name="",
            enabled=True,
            params={},
            extras={},
            dataset=None,
            signals={"text": True},
        )
        _ = resolved.signals
        assert resolved.signals["text"] is True
