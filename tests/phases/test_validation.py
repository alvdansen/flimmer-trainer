"""Tests for batch validation system: ValidationIssue, ValidationResult, validate_project.

Tests use the register_all_models fixture so real model definitions are available.
"""

from __future__ import annotations

import re
from dataclasses import FrozenInstanceError

import pytest

from flimmer.phases.phase_model import PhaseConfig
from flimmer.phases.validation import ValidationIssue, ValidationResult, validate_project


# ---------------------------------------------------------------------------
# We need Project for validate_project() -- import from project module.
# For TDD RED, we use a minimal stand-in since project.py may not exist yet.
# However, validate_project() takes a Project-like object with model_id and phases.
# We'll import the real Project once it exists; for now use a SimpleNamespace stub.
# ---------------------------------------------------------------------------

class _StubPhaseEntry:
    """Minimal stand-in for PhaseEntry (status not relevant for validation)."""

    def __init__(self, config: PhaseConfig):
        self.config = config
        self.status = "pending"


class _StubProject:
    """Minimal stand-in for Project to test validation independently."""

    def __init__(self, model_id: str, phases: list[_StubPhaseEntry]):
        self.model_id = model_id
        self.phases = phases


class TestValidationIssue:
    """ValidationIssue is a frozen (immutable) dataclass."""

    def test_create_with_all_fields(self):
        issue = ValidationIssue(
            phase_index=0,
            phase_type="full_noise",
            param="learning_rate",
            message="Value out of range",
            severity="error",
        )
        assert issue.phase_index == 0
        assert issue.phase_type == "full_noise"
        assert issue.param == "learning_rate"
        assert issue.message == "Value out of range"
        assert issue.severity == "error"

    def test_create_with_param_none(self):
        issue = ValidationIssue(
            phase_index=1,
            phase_type="high_noise",
            param=None,
            message="Unknown phase type",
            severity="error",
        )
        assert issue.param is None

    def test_is_frozen_immutable(self):
        issue = ValidationIssue(
            phase_index=0,
            phase_type="full_noise",
            param=None,
            message="test",
            severity="error",
        )
        with pytest.raises(FrozenInstanceError):
            issue.phase_index = 99  # type: ignore[misc]


class TestValidationResult:
    """ValidationResult aggregates issues with valid property and format()."""

    def test_valid_when_no_issues(self):
        result = ValidationResult(issues=[])
        assert result.valid is True

    def test_valid_when_only_warnings(self):
        result = ValidationResult(issues=[
            ValidationIssue(0, "full_noise", "learning_rate", "Suspiciously low", "warning"),
        ])
        assert result.valid is True

    def test_invalid_when_has_error(self):
        result = ValidationResult(issues=[
            ValidationIssue(0, "full_noise", "learning_rate", "Out of range", "error"),
        ])
        assert result.valid is False

    def test_invalid_when_mixed_errors_and_warnings(self):
        result = ValidationResult(issues=[
            ValidationIssue(0, "full_noise", "learning_rate", "Suspiciously low", "warning"),
            ValidationIssue(1, "full_noise", None, "Bad phase type", "error"),
        ])
        assert result.valid is False

    def test_format_no_issues(self):
        result = ValidationResult(issues=[])
        assert result.format() == "Validation passed: no issues found."

    def test_format_with_error_and_param(self):
        result = ValidationResult(issues=[
            ValidationIssue(0, "full_noise", "learning_rate", "Out of range", "error"),
        ])
        text = result.format()
        assert "ERROR" in text
        assert "Phase 0" in text
        assert "(full_noise)" in text
        assert "[learning_rate]" in text
        assert "Out of range" in text

    def test_format_with_warning_and_no_param(self):
        result = ValidationResult(issues=[
            ValidationIssue(2, "high_noise", None, "Suspicious config", "warning"),
        ])
        text = result.format()
        assert "WARNING" in text
        assert "Phase 2" in text
        assert "(high_noise)" in text
        # No [param] bracket when param is None
        assert "[" not in text or "[None]" not in text

    def test_format_omits_param_bracket_when_none(self):
        issue = ValidationIssue(0, "full_noise", None, "msg", "error")
        result = ValidationResult(issues=[issue])
        text = result.format()
        # Should NOT have [] brackets for param
        # Pattern: "Phase 0 (full_noise): msg" not "Phase 0 (full_noise)[]: msg"
        assert "[]" not in text
        assert "[None]" not in text


class TestValidateProject:
    """Integration tests for validate_project() batch collection."""

    def test_valid_project_returns_valid_result(self, register_all_models):
        project = _StubProject(
            model_id="wan-2.1-t2v-14b",
            phases=[
                _StubPhaseEntry(PhaseConfig(phase_type="full_noise")),
            ],
        )
        result = validate_project(project)
        assert result.valid is True
        assert len(result.issues) == 0

    def test_invalid_phase_type_collects_error(self, register_all_models):
        project = _StubProject(
            model_id="wan-2.1-t2v-14b",
            phases=[
                _StubPhaseEntry(PhaseConfig(phase_type="nonexistent")),
            ],
        )
        result = validate_project(project)
        assert result.valid is False
        assert len(result.issues) == 1
        issue = result.issues[0]
        assert issue.severity == "error"
        assert issue.phase_index == 0
        assert issue.phase_type == "nonexistent"
        assert "nonexistent" in issue.message

    def test_out_of_range_override_collects_error(self, register_all_models):
        project = _StubProject(
            model_id="wan-2.1-t2v-14b",
            phases=[
                _StubPhaseEntry(PhaseConfig(
                    phase_type="full_noise",
                    overrides={"learning_rate": 999.0},  # way above max 1e-2
                )),
            ],
        )
        result = validate_project(project)
        assert result.valid is False
        assert len(result.issues) >= 1
        issue = result.issues[0]
        assert issue.severity == "error"
        assert issue.param == "learning_rate"

    def test_batch_collects_all_errors_not_just_first(self, register_all_models):
        """Multiple invalid phases -> ALL errors collected."""
        project = _StubProject(
            model_id="wan-2.1-t2v-14b",
            phases=[
                _StubPhaseEntry(PhaseConfig(phase_type="nonexistent_1")),
                _StubPhaseEntry(PhaseConfig(phase_type="nonexistent_2")),
            ],
        )
        result = validate_project(project)
        assert result.valid is False
        assert len(result.issues) >= 2
        # Both phases should be reported
        indices = {issue.phase_index for issue in result.issues}
        assert 0 in indices
        assert 1 in indices

    def test_warning_for_learning_rate_at_minimum(self, register_all_models):
        """learning_rate at its min_value (1e-6) triggers a warning."""
        project = _StubProject(
            model_id="wan-2.1-t2v-14b",
            phases=[
                _StubPhaseEntry(PhaseConfig(
                    phase_type="full_noise",
                    overrides={"learning_rate": 1e-6},  # exactly at min
                )),
            ],
        )
        result = validate_project(project)
        # Should be valid (warning only, no error)
        assert result.valid is True
        warnings = [i for i in result.issues if i.severity == "warning"]
        assert len(warnings) >= 1
        assert warnings[0].param == "learning_rate"

    def test_validation_uses_model_from_registry(self, register_all_models):
        """validate_project looks up model_id from the registry."""
        # Using a model that supports high_noise (MoE model)
        project = _StubProject(
            model_id="wan-2.2-i2v-14b",
            phases=[
                _StubPhaseEntry(PhaseConfig(
                    phase_type="high_noise",
                    extras={"boundary_ratio": 0.875},
                )),
            ],
        )
        result = validate_project(project)
        assert result.valid is True
