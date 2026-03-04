"""Tests for flimmer.project — loader and CLI.

Tests the project YAML loading, Project creation, config merging,
and CLI subcommands (status, plan). Uses tmp_path for all file I/O
and mocks subprocess.run for the CLI run command.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml


# ── Sample data ──────────────────────────────────────────────────────

SAMPLE_PROJECT_YAML = {
    "name": "test_project",
    "model_id": "wan-2.2-t2v-14b",
    "base_config": "./base_train.yaml",
    "run_level_params": {
        "lora_rank": 32,
        "lora_alpha": 32,
        "mixed_precision": "bf16",
        "base_model_precision": "bf16",
    },
    "phases": [
        {
            "type": "unified",
            "name": "Unified Warmup",
            "overrides": {
                "learning_rate": 5e-5,
                "max_epochs": 15,
            },
        },
        {
            "type": "high_noise",
            "name": "High Noise Expert",
            "overrides": {
                "learning_rate": 1e-4,
                "max_epochs": 30,
            },
            "extras": {
                "boundary_ratio": 0.9,
            },
        },
        {
            "type": "low_noise",
            "name": "Low Noise Expert",
            "overrides": {
                "learning_rate": 8e-5,
                "max_epochs": 50,
            },
            "extras": {
                "boundary_ratio": 0.9,
            },
        },
    ],
}


SAMPLE_BASE_CONFIG = {
    "model": {
        "variant": "2.2_t2v",
        "dit_high": "/workspace/models/high.safetensors",
        "dit_low": "/workspace/models/low.safetensors",
        "vae": "/workspace/models/vae.safetensors",
        "t5": "/workspace/models/t5.safetensors",
    },
    "data_config": "./holly/flimmer_data.yaml",
    "lora": {
        "rank": 16,
        "alpha": 16,
        "loraplus_lr_ratio": 4.0,
        "dropout": 0.05,
    },
    "optimizer": {
        "type": "adamw8bit",
        "learning_rate": 5e-5,
        "weight_decay": 0.01,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
    },
    "scheduler": {
        "type": "cosine_with_min_lr",
        "warmup_steps": 0,
        "min_lr_ratio": 0.01,
    },
    "training": {
        "mixed_precision": "bf16",
        "base_model_precision": "bf16",
        "unified_epochs": 10,
        "batch_size": 1,
        "gradient_accumulation_steps": 1,
        "caption_dropout_rate": 0.15,
        "gradient_checkpointing": True,
        "seed": 42,
    },
    "moe": {
        "enabled": True,
        "fork_enabled": False,
        "boundary_ratio": 0.5,
    },
    "save": {
        "save_every_n_epochs": 5,
        "output_dir": "./output/test",
        "name": "test_lora",
    },
    "sampling": {
        "enabled": False,
    },
    "logging": {
        "backends": ["console"],
    },
}


def _write_yaml(path: Path, data: dict) -> Path:
    """Write a dict as YAML to the given path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f)
    return path


# ── Tests: load_project_yaml ────────────────────────────────────────

class TestLoadProjectYaml:
    """Test YAML parsing of project files."""

    def test_loads_valid_yaml(self, tmp_path):
        """load_project_yaml() parses a valid project YAML file."""
        yaml_path = _write_yaml(tmp_path / "project.yaml", SAMPLE_PROJECT_YAML)

        from flimmer.project.loader import load_project_yaml

        data = load_project_yaml(yaml_path)

        assert data["name"] == "test_project"
        assert data["model_id"] == "wan-2.2-t2v-14b"
        assert data["base_config"] == "./base_train.yaml"
        assert len(data["phases"]) == 3
        assert data["phases"][0]["type"] == "unified"
        assert data["run_level_params"]["lora_rank"] == 32

    def test_loads_phases_with_extras(self, tmp_path):
        """load_project_yaml() preserves extras on phases."""
        yaml_path = _write_yaml(tmp_path / "project.yaml", SAMPLE_PROJECT_YAML)

        from flimmer.project.loader import load_project_yaml

        data = load_project_yaml(yaml_path)

        assert data["phases"][1]["extras"]["boundary_ratio"] == 0.9
        assert data["phases"][2]["extras"]["boundary_ratio"] == 0.9


# ── Tests: project_from_yaml ────────────────────────────────────────

class TestProjectFromYaml:
    """Test YAML -> Project conversion and persistence."""

    def test_creates_project_with_correct_phases(self, tmp_path):
        """project_from_yaml() creates a Project with correct phases from YAML."""
        yaml_path = _write_yaml(tmp_path / "project.yaml", SAMPLE_PROJECT_YAML)

        from flimmer.project.loader import project_from_yaml

        project = project_from_yaml(yaml_path)

        assert project.name == "test_project"
        assert project.model_id == "wan-2.2-t2v-14b"
        assert len(project.phases) == 3
        assert project.phases[0].config.phase_type == "unified"
        assert project.phases[0].config.display_name == "Unified Warmup"
        assert project.phases[1].config.phase_type == "high_noise"
        assert project.phases[2].config.phase_type == "low_noise"

    def test_saves_flimmer_project_json(self, tmp_path):
        """project_from_yaml() saves flimmer_project.json alongside the YAML."""
        yaml_path = _write_yaml(tmp_path / "project.yaml", SAMPLE_PROJECT_YAML)

        from flimmer.project.loader import project_from_yaml

        project_from_yaml(yaml_path)

        json_path = tmp_path / "flimmer_project.json"
        assert json_path.exists()
        data = json.loads(json_path.read_text())
        assert data["name"] == "test_project"

    def test_loads_existing_project_json(self, tmp_path):
        """project_from_yaml() loads existing flimmer_project.json if present."""
        yaml_path = _write_yaml(tmp_path / "project.yaml", SAMPLE_PROJECT_YAML)

        from flimmer.project.loader import project_from_yaml

        # First call creates the project
        project = project_from_yaml(yaml_path)
        project.mark_phase_running(0)
        project.mark_phase_completed(0)
        project.save(tmp_path)

        # Second call should load existing state (phase 0 completed)
        reloaded = project_from_yaml(yaml_path)
        assert reloaded.phases[0].status.value == "completed"

    def test_applies_run_level_params(self, tmp_path):
        """project_from_yaml() passes run_level_params from YAML to Project."""
        yaml_path = _write_yaml(tmp_path / "project.yaml", SAMPLE_PROJECT_YAML)

        from flimmer.project.loader import project_from_yaml

        project = project_from_yaml(yaml_path)

        assert project.run_level_params["lora_rank"] == 32
        assert project.run_level_params["lora_alpha"] == 32
        assert project.run_level_params["mixed_precision"] == "bf16"

    def test_phase_overrides_preserved(self, tmp_path):
        """project_from_yaml() preserves phase overrides in PhaseConfig."""
        yaml_path = _write_yaml(tmp_path / "project.yaml", SAMPLE_PROJECT_YAML)

        from flimmer.project.loader import project_from_yaml

        project = project_from_yaml(yaml_path)

        assert project.phases[0].config.overrides["learning_rate"] == 5e-5
        assert project.phases[0].config.overrides["max_epochs"] == 15
        assert project.phases[1].config.overrides["learning_rate"] == 1e-4

    def test_phase_extras_preserved(self, tmp_path):
        """project_from_yaml() preserves extras (e.g. boundary_ratio) in PhaseConfig."""
        yaml_path = _write_yaml(tmp_path / "project.yaml", SAMPLE_PROJECT_YAML)

        from flimmer.project.loader import project_from_yaml

        project = project_from_yaml(yaml_path)

        assert project.phases[1].config.extras["boundary_ratio"] == 0.9


# ── Tests: merge_phase_config ───────────────────────────────────────

class TestMergePhaseConfig:
    """Test base config + phase override merging."""

    def _setup_merge(self, tmp_path):
        """Create base config, project YAML, and Project for merge tests."""
        base_path = _write_yaml(tmp_path / "base_train.yaml", SAMPLE_BASE_CONFIG)
        yaml_path = _write_yaml(tmp_path / "project.yaml", SAMPLE_PROJECT_YAML)

        from flimmer.project.loader import project_from_yaml

        project = project_from_yaml(yaml_path)
        return base_path, project

    def test_applies_learning_rate_override(self, tmp_path):
        """merge_phase_config() applies learning_rate to optimizer section."""
        base_path, project = self._setup_merge(tmp_path)
        output_path = tmp_path / "merged.yaml"

        from flimmer.project.loader import merge_phase_config

        merge_phase_config(base_path, project, 0, output_path)

        merged = yaml.safe_load(output_path.read_text())
        assert merged["optimizer"]["learning_rate"] == 5e-5

    def test_applies_max_epochs_to_unified(self, tmp_path):
        """merge_phase_config() sets training.unified_epochs for unified phases."""
        base_path, project = self._setup_merge(tmp_path)
        output_path = tmp_path / "merged.yaml"

        from flimmer.project.loader import merge_phase_config

        merge_phase_config(base_path, project, 0, output_path)

        merged = yaml.safe_load(output_path.read_text())
        assert merged["training"]["unified_epochs"] == 15

    def test_applies_run_level_params(self, tmp_path):
        """merge_phase_config() applies run_level_params from project."""
        base_path, project = self._setup_merge(tmp_path)
        output_path = tmp_path / "merged.yaml"

        from flimmer.project.loader import merge_phase_config

        merge_phase_config(base_path, project, 0, output_path)

        merged = yaml.safe_load(output_path.read_text())
        assert merged["training"]["mixed_precision"] == "bf16"
        assert merged["training"]["base_model_precision"] == "bf16"
        assert merged["lora"]["rank"] == 32
        assert merged["lora"]["alpha"] == 32

    def test_moe_expert_phase(self, tmp_path):
        """merge_phase_config() sets MoE block correctly for expert phases."""
        base_path, project = self._setup_merge(tmp_path)
        output_path = tmp_path / "merged.yaml"

        from flimmer.project.loader import merge_phase_config

        # Phase 1 is high_noise expert
        merge_phase_config(base_path, project, 1, output_path)

        merged = yaml.safe_load(output_path.read_text())
        assert merged["moe"]["fork_enabled"] is True
        assert merged["moe"]["boundary_ratio"] == 0.9
        # Expert-specific overrides should be in the expert section
        assert merged["moe"]["high_noise"]["learning_rate"] == 1e-4
        assert merged["moe"]["high_noise"]["max_epochs"] == 30

    def test_preserves_base_config_keys(self, tmp_path):
        """merge_phase_config() preserves all base config keys not overridden."""
        base_path, project = self._setup_merge(tmp_path)
        output_path = tmp_path / "merged.yaml"

        from flimmer.project.loader import merge_phase_config

        merge_phase_config(base_path, project, 0, output_path)

        merged = yaml.safe_load(output_path.read_text())
        # Model paths should be preserved
        assert merged["model"]["dit_high"] == "/workspace/models/high.safetensors"
        assert merged["model"]["dit_low"] == "/workspace/models/low.safetensors"
        assert merged["model"]["vae"] == "/workspace/models/vae.safetensors"
        # Data config preserved
        assert merged["data_config"] == "./holly/flimmer_data.yaml"
        # Save config preserved
        assert merged["save"]["output_dir"] == "./output/test"
        # Scheduler preserved
        assert merged["scheduler"]["type"] == "cosine_with_min_lr"
        # Other training params preserved
        assert merged["training"]["gradient_checkpointing"] is True
        assert merged["training"]["seed"] == 42

    def test_applies_weight_decay_override(self, tmp_path):
        """merge_phase_config() applies weight_decay override."""
        # Add weight_decay override to the first phase
        project_data = dict(SAMPLE_PROJECT_YAML)
        project_data = json.loads(json.dumps(SAMPLE_PROJECT_YAML))
        project_data["phases"][0]["overrides"]["weight_decay"] = 0.05

        yaml_path = _write_yaml(tmp_path / "project.yaml", project_data)
        base_path = _write_yaml(tmp_path / "base_train.yaml", SAMPLE_BASE_CONFIG)

        from flimmer.project.loader import merge_phase_config, project_from_yaml

        project = project_from_yaml(yaml_path)
        output_path = tmp_path / "merged.yaml"
        merge_phase_config(base_path, project, 0, output_path)

        merged = yaml.safe_load(output_path.read_text())
        assert merged["optimizer"]["weight_decay"] == 0.05

    def test_applies_batch_size_override(self, tmp_path):
        """merge_phase_config() applies batch_size override."""
        project_data = json.loads(json.dumps(SAMPLE_PROJECT_YAML))
        project_data["phases"][0]["overrides"]["batch_size"] = 2

        yaml_path = _write_yaml(tmp_path / "project.yaml", project_data)
        base_path = _write_yaml(tmp_path / "base_train.yaml", SAMPLE_BASE_CONFIG)

        from flimmer.project.loader import merge_phase_config, project_from_yaml

        project = project_from_yaml(yaml_path)
        output_path = tmp_path / "merged.yaml"
        merge_phase_config(base_path, project, 0, output_path)

        merged = yaml.safe_load(output_path.read_text())
        assert merged["training"]["batch_size"] == 2

    def test_applies_gradient_accumulation_override(self, tmp_path):
        """merge_phase_config() applies gradient_accumulation_steps override."""
        project_data = json.loads(json.dumps(SAMPLE_PROJECT_YAML))
        project_data["phases"][0]["overrides"]["gradient_accumulation_steps"] = 4

        yaml_path = _write_yaml(tmp_path / "project.yaml", project_data)
        base_path = _write_yaml(tmp_path / "base_train.yaml", SAMPLE_BASE_CONFIG)

        from flimmer.project.loader import merge_phase_config, project_from_yaml

        project = project_from_yaml(yaml_path)
        output_path = tmp_path / "merged.yaml"
        merge_phase_config(base_path, project, 0, output_path)

        merged = yaml.safe_load(output_path.read_text())
        assert merged["training"]["gradient_accumulation_steps"] == 4

    def test_applies_caption_dropout_override(self, tmp_path):
        """merge_phase_config() applies caption_dropout_rate override."""
        project_data = json.loads(json.dumps(SAMPLE_PROJECT_YAML))
        project_data["phases"][0]["overrides"]["caption_dropout_rate"] = 0.2

        yaml_path = _write_yaml(tmp_path / "project.yaml", project_data)
        base_path = _write_yaml(tmp_path / "base_train.yaml", SAMPLE_BASE_CONFIG)

        from flimmer.project.loader import merge_phase_config, project_from_yaml

        project = project_from_yaml(yaml_path)
        output_path = tmp_path / "merged.yaml"
        merge_phase_config(base_path, project, 0, output_path)

        merged = yaml.safe_load(output_path.read_text())
        assert merged["training"]["caption_dropout_rate"] == 0.2

    def test_low_noise_expert_phase(self, tmp_path):
        """merge_phase_config() handles low_noise expert phases correctly."""
        base_path, project = self._setup_merge(tmp_path)
        output_path = tmp_path / "merged.yaml"

        from flimmer.project.loader import merge_phase_config

        # Phase 2 is low_noise expert
        merge_phase_config(base_path, project, 2, output_path)

        merged = yaml.safe_load(output_path.read_text())
        assert merged["moe"]["fork_enabled"] is True
        assert merged["moe"]["boundary_ratio"] == 0.9
        assert merged["moe"]["low_noise"]["learning_rate"] == 8e-5
        assert merged["moe"]["low_noise"]["max_epochs"] == 50

    def test_returns_output_path(self, tmp_path):
        """merge_phase_config() returns the output path."""
        base_path, project = self._setup_merge(tmp_path)
        output_path = tmp_path / "merged.yaml"

        from flimmer.project.loader import merge_phase_config

        result = merge_phase_config(base_path, project, 0, output_path)

        assert result == output_path


# ── Tests: CLI ──────────────────────────────────────────────────────

class TestCLIStatus:
    """Test the status subcommand."""

    def test_prints_phase_statuses(self, tmp_path, capsys):
        """cmd_status() prints each phase with its status."""
        yaml_path = _write_yaml(tmp_path / "project.yaml", SAMPLE_PROJECT_YAML)

        from flimmer.project.__main__ import cmd_status

        args = MagicMock()
        args.project = str(yaml_path)
        cmd_status(args)

        output = capsys.readouterr().out
        assert "PENDING" in output
        assert "Unified Warmup" in output
        assert "High Noise Expert" in output
        assert "Low Noise Expert" in output

    def test_shows_completed_status(self, tmp_path, capsys):
        """cmd_status() shows COMPLETED for completed phases."""
        yaml_path = _write_yaml(tmp_path / "project.yaml", SAMPLE_PROJECT_YAML)

        from flimmer.project.loader import project_from_yaml

        project = project_from_yaml(yaml_path)
        project.mark_phase_running(0)
        project.mark_phase_completed(0)
        project.save(tmp_path)

        from flimmer.project.__main__ import cmd_status

        args = MagicMock()
        args.project = str(yaml_path)
        cmd_status(args)

        output = capsys.readouterr().out
        assert "COMPLETED" in output


class TestCLIPlan:
    """Test the plan subcommand."""

    def test_shows_pending_phases(self, tmp_path, capsys):
        """cmd_plan() shows what phases would run."""
        yaml_path = _write_yaml(tmp_path / "project.yaml", SAMPLE_PROJECT_YAML)

        from flimmer.project.__main__ import cmd_plan

        args = MagicMock()
        args.project = str(yaml_path)
        cmd_plan(args)

        output = capsys.readouterr().out
        assert "Unified Warmup" in output
        assert "PENDING" in output or "pending" in output.lower()

    def test_shows_overrides(self, tmp_path, capsys):
        """cmd_plan() shows what overrides would be applied."""
        yaml_path = _write_yaml(tmp_path / "project.yaml", SAMPLE_PROJECT_YAML)

        from flimmer.project.__main__ import cmd_plan

        args = MagicMock()
        args.project = str(yaml_path)
        cmd_plan(args)

        output = capsys.readouterr().out
        assert "learning_rate" in output


class TestCLIParser:
    """Test the argument parser."""

    def test_run_command(self):
        """Parser accepts run subcommand."""
        from flimmer.project.__main__ import build_parser

        parser = build_parser()
        args = parser.parse_args(["run", "--project", "project.yaml"])
        assert args.command == "run"
        assert args.project == "project.yaml"

    def test_status_command(self):
        """Parser accepts status subcommand."""
        from flimmer.project.__main__ import build_parser

        parser = build_parser()
        args = parser.parse_args(["status", "--project", "project.yaml"])
        assert args.command == "status"

    def test_plan_command(self):
        """Parser accepts plan subcommand."""
        from flimmer.project.__main__ import build_parser

        parser = build_parser()
        args = parser.parse_args(["plan", "--project", "project.yaml"])
        assert args.command == "plan"

    def test_run_all_flag(self):
        """Parser accepts --all flag for run command."""
        from flimmer.project.__main__ import build_parser

        parser = build_parser()
        args = parser.parse_args(["run", "--project", "p.yaml", "--all"])
        assert args.all is True

    def test_run_output_dir(self):
        """Parser accepts --output-dir for run command."""
        from flimmer.project.__main__ import build_parser

        parser = build_parser()
        args = parser.parse_args(["run", "--project", "p.yaml", "--output-dir", "/out"])
        assert args.output_dir == "/out"

    def test_no_command_errors(self):
        """Parser errors when no subcommand given."""
        from flimmer.project.__main__ import build_parser

        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])


# ── Tests: Public API ───────────────────────────────────────────────

class TestPublicAPI:
    """Test that flimmer.project exposes the public API."""

    def test_imports(self):
        """flimmer.project exports the expected names."""
        from flimmer.project import (
            load_project_yaml,
            merge_phase_config,
            project_from_yaml,
        )

        assert callable(load_project_yaml)
        assert callable(merge_phase_config)
        assert callable(project_from_yaml)
