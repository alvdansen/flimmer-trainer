"""Tests for template factory functions and public API completeness.

Covers: template_moe_standard, template_wan21_finetune, project validation,
correct model IDs, boundary_ratio in expert extras, and public API exports.
"""

import pytest

from flimmer.phases.phase_model import PhaseConfig


class TestMoeStandardTemplate:
    """template_moe_standard creates a 3-phase MoE project."""

    def test_returns_project_with_3_phases(self, register_all_models):
        from flimmer.phases.templates import template_moe_standard

        project = template_moe_standard("test_moe")
        assert len(project.phases) == 3

    def test_phase_types(self, register_all_models):
        from flimmer.phases.templates import template_moe_standard

        project = template_moe_standard("test_moe")
        types = [entry.config.phase_type for entry in project.phases]
        assert types == ["unified", "high_noise", "low_noise"]

    def test_model_id(self, register_all_models):
        from flimmer.phases.templates import template_moe_standard

        project = template_moe_standard("test_moe")
        assert project.model_id == "wan-2.2-t2v-14b"

    def test_validates_successfully(self, register_all_models):
        from flimmer.phases.templates import template_moe_standard

        project = template_moe_standard("test_moe")
        result = project.validate()
        # Should have no errors (warnings are ok)
        assert result.valid, f"Validation failed: {result.format()}"

    def test_warmup_phase_has_lower_lr(self, register_all_models):
        from flimmer.phases.templates import template_moe_standard

        project = template_moe_standard("test_moe")
        warmup = project.phases[0].config
        high_noise = project.phases[1].config
        # Warmup LR should be <= expert LR
        warmup_lr = warmup.overrides.get("learning_rate", 5e-5)
        expert_lr = high_noise.overrides.get("learning_rate", 5e-5)
        assert warmup_lr <= expert_lr

    def test_expert_phases_have_boundary_ratio(self, register_all_models):
        from flimmer.phases.templates import template_moe_standard

        project = template_moe_standard("test_moe")
        for entry in project.phases[1:]:  # high_noise, low_noise
            assert "boundary_ratio" in entry.config.extras
            assert entry.config.extras["boundary_ratio"] == pytest.approx(0.875)

    def test_experts_have_more_epochs(self, register_all_models):
        from flimmer.phases.templates import template_moe_standard

        project = template_moe_standard("test_moe")
        warmup_epochs = project.phases[0].config.overrides.get("max_epochs", 10)
        hn_epochs = project.phases[1].config.overrides.get("max_epochs", 10)
        ln_epochs = project.phases[2].config.overrides.get("max_epochs", 10)
        assert hn_epochs > warmup_epochs
        assert ln_epochs > warmup_epochs

    def test_custom_name(self, register_all_models):
        from flimmer.phases.templates import template_moe_standard

        project = template_moe_standard("holly_training")
        assert project.name == "holly_training"


class TestWan21FinetuneTemplate:
    """template_wan21_finetune creates a 1-phase non-MoE project."""

    def test_returns_project_with_1_phase(self, register_all_models):
        from flimmer.phases.templates import template_wan21_finetune

        project = template_wan21_finetune("test_finetune")
        assert len(project.phases) == 1

    def test_phase_type_is_unified(self, register_all_models):
        from flimmer.phases.templates import template_wan21_finetune

        project = template_wan21_finetune("test_finetune")
        assert project.phases[0].config.phase_type == "unified"

    def test_model_id(self, register_all_models):
        from flimmer.phases.templates import template_wan21_finetune

        project = template_wan21_finetune("test_finetune")
        assert project.model_id == "wan-2.1-t2v-14b"

    def test_validates_successfully(self, register_all_models):
        from flimmer.phases.templates import template_wan21_finetune

        project = template_wan21_finetune("test_finetune")
        result = project.validate()
        assert result.valid, f"Validation failed: {result.format()}"

    def test_has_training_params(self, register_all_models):
        from flimmer.phases.templates import template_wan21_finetune

        project = template_wan21_finetune("test_finetune")
        overrides = project.phases[0].config.overrides
        assert "learning_rate" in overrides
        assert "max_epochs" in overrides


class TestPublicApiExports:
    """flimmer.phases top-level exports all Phase 3 types."""

    def test_import_all_phase3_types(self, register_all_models):
        from flimmer.phases import (
            Project,
            PhaseEntry,
            PhaseStatus,
            ValidationIssue,
            ValidationResult,
            validate_project,
            export_yaml,
            template_moe_standard,
            template_wan21_finetune,
        )
        # Verify they're all callable/classes (not None)
        assert Project is not None
        assert PhaseEntry is not None
        assert PhaseStatus is not None
        assert ValidationIssue is not None
        assert ValidationResult is not None
        assert callable(validate_project)
        assert callable(export_yaml)
        assert callable(template_moe_standard)
        assert callable(template_wan21_finetune)
