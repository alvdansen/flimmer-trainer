"""Tests for flimmer.training.checkpoint — checkpoint management."""

import json
import pytest

from flimmer.training.checkpoint import (
    PHASE_ABBREV,
    PHASE_DIRS,
    TRAINING_STATE_FILENAME,
    CheckpointManager,
    CheckpointMetadata,
    TrainingState,
)
from flimmer.training.errors import CheckpointError, ResumptionError
from flimmer.training.phase import PhaseType


class TestTrainingState:
    """TrainingState serialization."""

    def test_to_dict(self):
        state = TrainingState(
            phase_index=1, phase_type="high_noise",
            epoch=7, global_step=3500,
            unified_lora_path="unified/lora.safetensors",
        )
        d = state.to_dict()
        assert d["phase_index"] == 1
        assert d["phase_type"] == "high_noise"
        assert d["epoch"] == 7
        assert d["unified_lora_path"] == "unified/lora.safetensors"

    def test_from_dict(self):
        d = {"phase_index": 2, "phase_type": "low_noise", "epoch": 10, "global_step": 5000}
        state = TrainingState.from_dict(d)
        assert state.phase_index == 2
        assert state.epoch == 10

    def test_round_trip(self):
        original = TrainingState(phase_index=1, epoch=5, global_step=100)
        restored = TrainingState.from_dict(original.to_dict())
        assert restored.phase_index == original.phase_index
        assert restored.epoch == original.epoch

    def test_defaults(self):
        state = TrainingState()
        assert state.phase_index == 0
        assert state.unified_lora_path is None

    def test_wandb_run_id_serialization(self):
        """wandb_run_id roundtrips through to_dict/from_dict."""
        state = TrainingState(
            phase_index=1, epoch=3, global_step=150,
            wandb_run_id="abc123",
        )
        d = state.to_dict()
        assert d["wandb_run_id"] == "abc123"

        restored = TrainingState.from_dict(d)
        assert restored.wandb_run_id == "abc123"

    def test_wandb_run_id_default_none(self):
        """from_dict({}) gives wandb_run_id=None (backward compat)."""
        state = TrainingState.from_dict({})
        assert state.wandb_run_id is None

    def test_wandb_run_id_backward_compat(self):
        """Old-format JSON (no wandb_run_id key) loads without error."""
        old_format = {
            "phase_index": 2,
            "phase_type": "high_noise",
            "epoch": 10,
            "global_step": 500,
            "unified_lora_path": "unified/lora.safetensors",
            "high_noise_lora_path": None,
            "low_noise_lora_path": None,
            "optimizer_path": None,
            # NOTE: no wandb_run_id key — simulates pre-04-01 state files
        }
        state = TrainingState.from_dict(old_format)
        assert state.phase_index == 2
        assert state.epoch == 10
        assert state.wandb_run_id is None  # Missing key → defaults to None


class TestCheckpointMetadata:
    """CheckpointMetadata frozen dataclass."""

    def test_create(self):
        meta = CheckpointMetadata(
            phase="unified", epoch=5, global_step=500, loss=0.05
        )
        assert meta.phase == "unified"
        assert meta.loss == 0.05

    def test_frozen(self):
        meta = CheckpointMetadata(phase="unified", epoch=5, global_step=500, loss=0.05)
        with pytest.raises(AttributeError):
            meta.epoch = 10  # type: ignore[misc]


class TestCheckpointManager:
    """CheckpointManager directory and file operations."""

    def test_ensure_dirs(self, tmp_path):
        mgr = CheckpointManager(tmp_path / "output", name="test")
        mgr.ensure_dirs()
        assert (tmp_path / "output" / "unified").is_dir()
        assert (tmp_path / "output" / "high_noise").is_dir()
        assert (tmp_path / "output" / "low_noise").is_dir()
        assert (tmp_path / "output" / "final").is_dir()
        assert (tmp_path / "output" / "samples").is_dir()

    def test_checkpoint_path(self, tmp_path):
        mgr = CheckpointManager(tmp_path, name="annika")
        path = mgr.checkpoint_path(PhaseType.UNIFIED, epoch=5)
        assert path.name == "annika_unified_epoch005.safetensors"
        assert path.parent.name == "unified"

    def test_checkpoint_path_expert(self, tmp_path):
        mgr = CheckpointManager(tmp_path, name="lora")
        path = mgr.checkpoint_path(PhaseType.HIGH_NOISE, epoch=15)
        assert path.name == "lora_high_epoch015.safetensors"
        assert path.parent.name == "high_noise"

    def test_final_path(self, tmp_path):
        mgr = CheckpointManager(tmp_path, name="test")
        path = mgr.final_path()
        assert path.name == "test_merged.safetensors"
        assert path.parent.name == "final"

    def test_sample_dir(self, tmp_path):
        mgr = CheckpointManager(tmp_path, name="test")
        d = mgr.sample_dir(PhaseType.LOW_NOISE, epoch=25)
        assert "low_epoch025" in d.name

    def test_save_and_load_state(self, tmp_path):
        mgr = CheckpointManager(tmp_path, name="test")
        state = TrainingState(phase_index=1, epoch=7, global_step=350)
        mgr.save_training_state(state)

        loaded = mgr.load_training_state()
        assert loaded is not None
        assert loaded.phase_index == 1
        assert loaded.epoch == 7

    def test_load_state_missing(self, tmp_path):
        mgr = CheckpointManager(tmp_path, name="test")
        result = mgr.load_training_state()
        assert result is None

    def test_load_state_corrupt(self, tmp_path):
        mgr = CheckpointManager(tmp_path, name="test")
        tmp_path.mkdir(parents=True, exist_ok=True)
        state_path = tmp_path / TRAINING_STATE_FILENAME
        state_path.write_text("{invalid json", encoding="utf-8")
        with pytest.raises(ResumptionError, match="Corrupt"):
            mgr.load_training_state()

    def test_list_checkpoints_empty(self, tmp_path):
        mgr = CheckpointManager(tmp_path, name="test")
        mgr.ensure_dirs()
        assert mgr.list_checkpoints(PhaseType.UNIFIED) == []

    def test_list_checkpoints_sorted(self, tmp_path):
        mgr = CheckpointManager(tmp_path, name="test")
        mgr.ensure_dirs()
        # Create some dummy checkpoint files
        for epoch in [5, 15, 10]:
            path = mgr.checkpoint_path(PhaseType.UNIFIED, epoch)
            path.write_text("dummy", encoding="utf-8")
        checkpoints = mgr.list_checkpoints(PhaseType.UNIFIED)
        assert len(checkpoints) == 3
        assert "epoch005" in checkpoints[0].name
        assert "epoch010" in checkpoints[1].name
        assert "epoch015" in checkpoints[2].name

    def test_find_latest_checkpoint(self, tmp_path):
        mgr = CheckpointManager(tmp_path, name="test")
        mgr.ensure_dirs()
        for epoch in [5, 10, 15]:
            path = mgr.checkpoint_path(PhaseType.HIGH_NOISE, epoch)
            path.write_text("dummy", encoding="utf-8")
        latest = mgr.find_latest_checkpoint(PhaseType.HIGH_NOISE)
        assert latest is not None
        assert "epoch015" in latest.name

    def test_find_latest_checkpoint_none(self, tmp_path):
        mgr = CheckpointManager(tmp_path, name="test")
        mgr.ensure_dirs()
        assert mgr.find_latest_checkpoint(PhaseType.LOW_NOISE) is None

    def test_prune_checkpoints(self, tmp_path):
        mgr = CheckpointManager(tmp_path, name="test", max_checkpoints=2)
        mgr.ensure_dirs()
        for epoch in [5, 10, 15, 20]:
            path = mgr.checkpoint_path(PhaseType.UNIFIED, epoch)
            path.write_text("dummy", encoding="utf-8")
        deleted = mgr.prune_checkpoints(PhaseType.UNIFIED)
        assert len(deleted) == 2
        remaining = mgr.list_checkpoints(PhaseType.UNIFIED)
        assert len(remaining) == 2
        assert "epoch015" in remaining[0].name
        assert "epoch020" in remaining[1].name

    def test_prune_no_limit(self, tmp_path):
        mgr = CheckpointManager(tmp_path, name="test", max_checkpoints=None)
        mgr.ensure_dirs()
        for epoch in [5, 10]:
            path = mgr.checkpoint_path(PhaseType.UNIFIED, epoch)
            path.write_text("dummy", encoding="utf-8")
        deleted = mgr.prune_checkpoints(PhaseType.UNIFIED)
        assert len(deleted) == 0

    def test_find_resume_point(self, tmp_path):
        mgr = CheckpointManager(tmp_path, name="test")
        state = TrainingState(phase_index=1, epoch=7, global_step=350)
        mgr.save_training_state(state)
        result = mgr.find_resume_point(phases=["phase0", "phase1", "phase2"])
        assert result is not None
        idx, epoch, loaded_state = result
        assert idx == 1
        assert epoch == 7

    def test_find_resume_point_fresh(self, tmp_path):
        mgr = CheckpointManager(tmp_path, name="test")
        result = mgr.find_resume_point(phases=["phase0"])
        assert result is None

    def test_find_resume_point_out_of_range(self, tmp_path):
        mgr = CheckpointManager(tmp_path, name="test")
        state = TrainingState(phase_index=5, epoch=1, global_step=10)
        mgr.save_training_state(state)
        result = mgr.find_resume_point(phases=["only_one_phase"])
        assert result is None
