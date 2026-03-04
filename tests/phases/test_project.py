"""Tests for Project lifecycle model: create, save, load, modify, status locking.

Tests use the register_all_models fixture so real model definitions are available.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from flimmer.phases.errors import ModelNotFoundError, PhaseConfigError
from flimmer.phases.phase_model import PhaseConfig
from flimmer.phases.project import PhaseEntry, PhaseStatus, Project
from flimmer.phases.validation import ValidationResult


class TestPhaseStatus:
    """PhaseStatus enum has pending, running, completed values."""

    def test_pending_value(self):
        assert PhaseStatus.PENDING == "pending"
        assert PhaseStatus.PENDING.value == "pending"

    def test_running_value(self):
        assert PhaseStatus.RUNNING == "running"
        assert PhaseStatus.RUNNING.value == "running"

    def test_completed_value(self):
        assert PhaseStatus.COMPLETED == "completed"
        assert PhaseStatus.COMPLETED.value == "completed"


class TestPhaseEntry:
    """PhaseEntry holds a config and a status."""

    def test_default_status_is_pending(self):
        config = PhaseConfig(phase_type="full_noise")
        entry = PhaseEntry(config=config)
        assert entry.status == PhaseStatus.PENDING

    def test_status_can_be_set(self):
        config = PhaseConfig(phase_type="full_noise")
        entry = PhaseEntry(config=config, status=PhaseStatus.RUNNING)
        assert entry.status == PhaseStatus.RUNNING


class TestProjectCreate:
    """Project.create() validates model and sets up run-level params."""

    def test_create_with_valid_model(self, register_all_models):
        project = Project.create("my-project", "wan-2.1-t2v-14b")
        assert project.name == "my-project"
        assert project.model_id == "wan-2.1-t2v-14b"
        assert project.phases == []
        assert project.created_at != ""

    def test_create_with_invalid_model_raises(self, register_all_models):
        with pytest.raises(ModelNotFoundError):
            Project.create("bad-project", "nonexistent-model")

    def test_create_sets_default_run_level_params(self, register_all_models):
        project = Project.create("my-project", "wan-2.1-t2v-14b")
        # wan-2.1-t2v-14b has run-level params: lora_rank, lora_alpha,
        # mixed_precision, base_model_precision
        assert "lora_rank" in project.run_level_params
        assert "lora_alpha" in project.run_level_params
        assert "mixed_precision" in project.run_level_params
        assert "base_model_precision" in project.run_level_params
        # Defaults should match the model definition
        assert project.run_level_params["lora_rank"] == 16
        assert project.run_level_params["lora_alpha"] == 16

    def test_create_with_custom_run_level_params(self, register_all_models):
        project = Project.create(
            "my-project",
            "wan-2.1-t2v-14b",
            run_level_params={"lora_rank": 32, "lora_alpha": 32},
        )
        assert project.run_level_params["lora_rank"] == 32
        assert project.run_level_params["lora_alpha"] == 32
        # Others should still be defaults
        assert project.run_level_params["mixed_precision"] == "bf16"


class TestProjectAddPhase:
    """project.add_phase() appends or inserts a phase with PENDING status."""

    def test_add_phase_appends(self, register_all_models):
        project = Project.create("test", "wan-2.1-t2v-14b")
        idx = project.add_phase(PhaseConfig(phase_type="full_noise"))
        assert idx == 0
        assert len(project.phases) == 1
        assert project.phases[0].status == PhaseStatus.PENDING
        assert project.phases[0].config.phase_type == "full_noise"

    def test_add_phase_returns_index(self, register_all_models):
        project = Project.create("test", "wan-2.1-t2v-14b")
        project.add_phase(PhaseConfig(phase_type="full_noise"))
        idx = project.add_phase(PhaseConfig(phase_type="full_noise", display_name="Phase 2"))
        assert idx == 1

    def test_add_phase_insert_at_index(self, register_all_models):
        project = Project.create("test", "wan-2.1-t2v-14b")
        project.add_phase(PhaseConfig(phase_type="full_noise", display_name="First"))
        project.add_phase(PhaseConfig(phase_type="full_noise", display_name="Third"))
        idx = project.add_phase(
            PhaseConfig(phase_type="full_noise", display_name="Second"),
            index=1,
        )
        assert idx == 1
        assert project.phases[0].config.display_name == "First"
        assert project.phases[1].config.display_name == "Second"
        assert project.phases[2].config.display_name == "Third"


class TestProjectModifyPhase:
    """project.modify_phase() replaces config for PENDING phases only."""

    def test_modify_pending_phase(self, register_all_models):
        project = Project.create("test", "wan-2.1-t2v-14b")
        project.add_phase(PhaseConfig(phase_type="full_noise"))
        new_config = PhaseConfig(phase_type="full_noise", display_name="Modified")
        project.modify_phase(0, new_config)
        assert project.phases[0].config.display_name == "Modified"

    def test_modify_running_raises(self, register_all_models):
        project = Project.create("test", "wan-2.1-t2v-14b")
        project.add_phase(PhaseConfig(phase_type="full_noise"))
        project.mark_phase_running(0)
        with pytest.raises(PhaseConfigError):
            project.modify_phase(0, PhaseConfig(phase_type="full_noise"))

    def test_modify_completed_raises(self, register_all_models):
        project = Project.create("test", "wan-2.1-t2v-14b")
        project.add_phase(PhaseConfig(phase_type="full_noise"))
        project.mark_phase_running(0)
        project.mark_phase_completed(0)
        with pytest.raises(PhaseConfigError):
            project.modify_phase(0, PhaseConfig(phase_type="full_noise"))


class TestProjectRemovePhase:
    """project.remove_phase() removes PENDING phases and returns config."""

    def test_remove_pending_phase(self, register_all_models):
        project = Project.create("test", "wan-2.1-t2v-14b")
        project.add_phase(PhaseConfig(phase_type="full_noise", display_name="ToRemove"))
        removed = project.remove_phase(0)
        assert removed.display_name == "ToRemove"
        assert len(project.phases) == 0

    def test_remove_running_raises(self, register_all_models):
        project = Project.create("test", "wan-2.1-t2v-14b")
        project.add_phase(PhaseConfig(phase_type="full_noise"))
        project.mark_phase_running(0)
        with pytest.raises(PhaseConfigError):
            project.remove_phase(0)

    def test_remove_completed_raises(self, register_all_models):
        project = Project.create("test", "wan-2.1-t2v-14b")
        project.add_phase(PhaseConfig(phase_type="full_noise"))
        project.mark_phase_running(0)
        project.mark_phase_completed(0)
        with pytest.raises(PhaseConfigError):
            project.remove_phase(0)


class TestProjectReorderPhases:
    """project.reorder_phases() reorders PENDING phases only."""

    def test_reorder_pending_phases(self, register_all_models):
        project = Project.create("test", "wan-2.1-t2v-14b")
        project.add_phase(PhaseConfig(phase_type="full_noise", display_name="A"))
        project.add_phase(PhaseConfig(phase_type="full_noise", display_name="B"))
        project.add_phase(PhaseConfig(phase_type="full_noise", display_name="C"))
        project.reorder_phases([2, 0, 1])
        assert project.phases[0].config.display_name == "C"
        assert project.phases[1].config.display_name == "A"
        assert project.phases[2].config.display_name == "B"

    def test_reorder_raises_if_non_pending(self, register_all_models):
        project = Project.create("test", "wan-2.1-t2v-14b")
        project.add_phase(PhaseConfig(phase_type="full_noise", display_name="A"))
        project.add_phase(PhaseConfig(phase_type="full_noise", display_name="B"))
        project.mark_phase_running(0)
        with pytest.raises(PhaseConfigError):
            project.reorder_phases([1, 0])


class TestProjectStatusTransitions:
    """Phase status transitions: PENDING -> RUNNING -> COMPLETED."""

    def test_mark_running(self, register_all_models):
        project = Project.create("test", "wan-2.1-t2v-14b")
        project.add_phase(PhaseConfig(phase_type="full_noise"))
        project.mark_phase_running(0)
        assert project.phases[0].status == PhaseStatus.RUNNING

    def test_mark_completed(self, register_all_models):
        project = Project.create("test", "wan-2.1-t2v-14b")
        project.add_phase(PhaseConfig(phase_type="full_noise"))
        project.mark_phase_running(0)
        project.mark_phase_completed(0)
        assert project.phases[0].status == PhaseStatus.COMPLETED

    def test_mark_running_not_pending_raises(self, register_all_models):
        project = Project.create("test", "wan-2.1-t2v-14b")
        project.add_phase(PhaseConfig(phase_type="full_noise"))
        project.mark_phase_running(0)
        project.mark_phase_completed(0)
        with pytest.raises(PhaseConfigError):
            project.mark_phase_running(0)

    def test_mark_completed_not_running_raises(self, register_all_models):
        project = Project.create("test", "wan-2.1-t2v-14b")
        project.add_phase(PhaseConfig(phase_type="full_noise"))
        with pytest.raises(PhaseConfigError):
            project.mark_phase_completed(0)


class TestProjectSaveLoad:
    """Save/load round-trip preserves all project state."""

    def test_save_creates_file(self, register_all_models, tmp_path):
        project = Project.create("test", "wan-2.1-t2v-14b")
        project.add_phase(PhaseConfig(phase_type="full_noise"))
        project.save(tmp_path)
        assert (tmp_path / "flimmer_project.json").exists()

    def test_save_load_round_trip(self, register_all_models, tmp_path):
        project = Project.create(
            "round-trip",
            "wan-2.1-t2v-14b",
            run_level_params={"lora_rank": 64},
        )
        project.add_phase(PhaseConfig(phase_type="full_noise", display_name="P1"))
        project.add_phase(PhaseConfig(phase_type="full_noise", display_name="P2"))
        project.mark_phase_running(0)
        project.mark_phase_completed(0)

        project.save(tmp_path)
        loaded = Project.load(tmp_path)

        assert loaded.name == "round-trip"
        assert loaded.model_id == "wan-2.1-t2v-14b"
        assert loaded.run_level_params["lora_rank"] == 64
        assert loaded.created_at == project.created_at
        assert len(loaded.phases) == 2
        assert loaded.phases[0].status == PhaseStatus.COMPLETED
        assert loaded.phases[0].config.display_name == "P1"
        assert loaded.phases[1].status == PhaseStatus.PENDING
        assert loaded.phases[1].config.display_name == "P2"

    def test_save_writes_valid_json(self, register_all_models, tmp_path):
        project = Project.create("json-test", "wan-2.1-t2v-14b")
        project.save(tmp_path)
        with open(tmp_path / "flimmer_project.json") as f:
            data = json.load(f)
        assert data["name"] == "json-test"
        assert data["model_id"] == "wan-2.1-t2v-14b"


class TestProjectValidate:
    """project.validate() delegates to validate_project()."""

    def test_validate_returns_validation_result(self, register_all_models):
        project = Project.create("test", "wan-2.1-t2v-14b")
        project.add_phase(PhaseConfig(phase_type="full_noise"))
        result = project.validate()
        assert isinstance(result, ValidationResult)
        assert result.valid is True

    def test_validate_catches_errors(self, register_all_models):
        project = Project.create("test", "wan-2.1-t2v-14b")
        project.add_phase(PhaseConfig(phase_type="nonexistent"))
        result = project.validate()
        assert result.valid is False
        assert len(result.issues) >= 1
