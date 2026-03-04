"""Tests for auto-registration of model definitions on package import.

These tests verify that `import flimmer.phases` populates MODEL_REGISTRY
with all four model definitions without explicit model imports or test
fixture workarounds. This closes the AUTO-REG-GAP and TEMPLATE-E2E gaps
identified in the v1.0 milestone audit.
"""

from flimmer.phases import (
    MODEL_REGISTRY,
    template_moe_standard,
    template_wan21_finetune,
)


class TestAutoRegistration:
    """Verify that bare `import flimmer.phases` populates MODEL_REGISTRY."""

    def test_package_import_populates_registry(self):
        """AUTO-REG-GAP: MODEL_REGISTRY has >= 4 entries after bare import.

        Uses >= 4 (not == 4) to tolerate any test ordering effects where
        additional models might be registered by other test modules.
        """
        assert len(MODEL_REGISTRY) >= 4

    def test_expected_model_ids_registered(self):
        """AUTO-REG-GAP: All four expected model IDs are present in the registry."""
        expected_ids = {
            "wan-2.1-t2v-14b",
            "wan-2.1-i2v-14b",
            "wan-2.2-t2v-14b",
            "wan-2.2-i2v-14b",
        }
        assert expected_ids.issubset(MODEL_REGISTRY.keys()), (
            f"Missing model IDs: {expected_ids - MODEL_REGISTRY.keys()}"
        )

    def test_template_moe_standard_works_without_explicit_model_import(self):
        """TEMPLATE-E2E: template_moe_standard() works via auto-registration.

        No imports from flimmer.phases.models.* -- only the top-level package.
        """
        project = template_moe_standard("auto_reg_test")
        assert project.model_id == "wan-2.2-t2v-14b"
        assert len(project.phases) == 3

    def test_template_wan21_finetune_works_without_explicit_model_import(self):
        """TEMPLATE-E2E: template_wan21_finetune() works via auto-registration.

        No imports from flimmer.phases.models.* -- only the top-level package.
        """
        project = template_wan21_finetune("auto_reg_wan21_test")
        assert project.model_id == "wan-2.1-t2v-14b"
        assert len(project.phases) == 1
