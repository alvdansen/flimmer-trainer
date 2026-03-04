"""Tests for YAML export: ResolvedPhase -> flimmer config dict -> YAML file.

Covers: non-MoE T2V, MoE T2V, I2V signals, float formatting,
disabled phase skipping, and round-trip YAML structure verification.
"""

from pathlib import Path

import yaml
import pytest

from flimmer.phases.phase_model import PhaseConfig
from flimmer.phases.project import Project


@pytest.fixture
def non_moe_project(register_all_models):
    """Wan 2.1 T2V project with one unified phase (non-MoE)."""
    project = Project.create(name="test_t2v", model_id="wan-2.1-t2v-14b")
    project.add_phase(PhaseConfig(
        phase_type="unified",
        display_name="Standard Training",
        overrides={"learning_rate": 5e-5, "max_epochs": 100},
    ))
    return project


@pytest.fixture
def moe_project(register_all_models):
    """Wan 2.2 T2V project with unified + high_noise + low_noise phases (MoE)."""
    project = Project.create(name="test_moe", model_id="wan-2.2-t2v-14b")
    project.add_phase(PhaseConfig(
        phase_type="unified",
        display_name="Warmup",
        overrides={"learning_rate": 5e-5, "max_epochs": 10},
    ))
    project.add_phase(PhaseConfig(
        phase_type="high_noise",
        display_name="High Noise Expert",
        overrides={"learning_rate": 1e-4, "max_epochs": 30},
        extras={"boundary_ratio": 0.875},
    ))
    project.add_phase(PhaseConfig(
        phase_type="low_noise",
        display_name="Low Noise Expert",
        overrides={"learning_rate": 8e-5, "max_epochs": 50},
        extras={"boundary_ratio": 0.875},
    ))
    return project


@pytest.fixture
def i2v_project(register_all_models):
    """Wan 2.1 I2V project with one unified phase (non-MoE, has reference_image signal)."""
    project = Project.create(name="test_i2v", model_id="wan-2.1-i2v-14b")
    project.add_phase(PhaseConfig(
        phase_type="unified",
        display_name="I2V Training",
        overrides={"learning_rate": 5e-5, "max_epochs": 50},
    ))
    return project


class TestNonMoeExport:
    """Non-MoE T2V model YAML export."""

    def test_model_variant(self, non_moe_project, tmp_path):
        from flimmer.phases.yaml_export import export_yaml

        out = export_yaml(non_moe_project, tmp_path / "config.yaml")
        data = yaml.safe_load(out.read_text())
        assert data["model"]["variant"] == "2.1_t2v"

    def test_optimizer_block(self, non_moe_project, tmp_path):
        from flimmer.phases.yaml_export import export_yaml

        out = export_yaml(non_moe_project, tmp_path / "config.yaml")
        data = yaml.safe_load(out.read_text())
        assert "optimizer" in data
        assert data["optimizer"]["learning_rate"] == pytest.approx(5e-5)
        # Key name mapping: optimizer_type -> type
        assert data["optimizer"]["type"] == "adamw8bit"
        assert "optimizer_type" not in data["optimizer"]

    def test_scheduler_block(self, non_moe_project, tmp_path):
        from flimmer.phases.yaml_export import export_yaml

        out = export_yaml(non_moe_project, tmp_path / "config.yaml")
        data = yaml.safe_load(out.read_text())
        assert data["scheduler"]["type"] == "cosine_with_min_lr"
        assert "scheduler_type" not in data["scheduler"]

    def test_training_block(self, non_moe_project, tmp_path):
        from flimmer.phases.yaml_export import export_yaml

        out = export_yaml(non_moe_project, tmp_path / "config.yaml")
        data = yaml.safe_load(out.read_text())
        assert data["training"]["unified_epochs"] == 100
        assert data["training"]["batch_size"] == 1
        assert data["training"]["gradient_accumulation_steps"] == 1
        assert data["training"]["caption_dropout_rate"] == pytest.approx(0.10)
        assert data["training"]["mixed_precision"] == "bf16"
        assert data["training"]["base_model_precision"] == "bf16"

    def test_lora_block(self, non_moe_project, tmp_path):
        from flimmer.phases.yaml_export import export_yaml

        out = export_yaml(non_moe_project, tmp_path / "config.yaml")
        data = yaml.safe_load(out.read_text())
        assert data["lora"]["rank"] == 16
        assert data["lora"]["alpha"] == 16
        # Key name mapping: lora_dropout -> dropout
        assert data["lora"]["dropout"] == pytest.approx(0.0)
        assert "lora_dropout" not in data["lora"]

    def test_save_block(self, non_moe_project, tmp_path):
        from flimmer.phases.yaml_export import export_yaml

        out = export_yaml(non_moe_project, tmp_path / "config.yaml")
        data = yaml.safe_load(out.read_text())
        assert data["save"]["output_dir"] == "./output/test_t2v"
        assert data["save"]["name"] == "test_t2v"

    def test_no_moe_block(self, non_moe_project, tmp_path):
        from flimmer.phases.yaml_export import export_yaml

        out = export_yaml(non_moe_project, tmp_path / "config.yaml")
        data = yaml.safe_load(out.read_text())
        assert "moe" not in data

    def test_no_signals_block_for_t2v(self, non_moe_project, tmp_path):
        """T2V models only have text+video (both assumed), no signals block."""
        from flimmer.phases.yaml_export import export_yaml

        out = export_yaml(non_moe_project, tmp_path / "config.yaml")
        data = yaml.safe_load(out.read_text())
        assert "signals" not in data

    def test_returns_output_path(self, non_moe_project, tmp_path):
        from flimmer.phases.yaml_export import export_yaml

        out_path = tmp_path / "config.yaml"
        result = export_yaml(non_moe_project, out_path)
        assert result == out_path
        assert out_path.exists()


class TestMoeExport:
    """MoE T2V model YAML export."""

    def test_moe_block_present(self, moe_project, tmp_path):
        from flimmer.phases.yaml_export import export_yaml

        out = export_yaml(moe_project, tmp_path / "config.yaml")
        data = yaml.safe_load(out.read_text())
        assert "moe" in data
        assert data["moe"]["enabled"] is True
        assert data["moe"]["fork_enabled"] is True

    def test_moe_boundary_ratio(self, moe_project, tmp_path):
        from flimmer.phases.yaml_export import export_yaml

        out = export_yaml(moe_project, tmp_path / "config.yaml")
        data = yaml.safe_load(out.read_text())
        assert data["moe"]["boundary_ratio"] == pytest.approx(0.875)

    def test_moe_expert_overrides(self, moe_project, tmp_path):
        """Expert phases include only params that differ from the base/unified phase."""
        from flimmer.phases.yaml_export import export_yaml

        out = export_yaml(moe_project, tmp_path / "config.yaml")
        data = yaml.safe_load(out.read_text())

        # high_noise expert: learning_rate and max_epochs differ from unified
        hn = data["moe"]["high_noise"]
        assert hn["learning_rate"] == pytest.approx(1e-4)
        assert hn["max_epochs"] == 30

        # low_noise expert: learning_rate and max_epochs differ from unified
        ln = data["moe"]["low_noise"]
        assert ln["learning_rate"] == pytest.approx(8e-5)
        assert ln["max_epochs"] == 50

    def test_moe_expert_minimal_overrides(self, moe_project, tmp_path):
        """Expert phases only include differing values (not everything)."""
        from flimmer.phases.yaml_export import export_yaml

        out = export_yaml(moe_project, tmp_path / "config.yaml")
        data = yaml.safe_load(out.read_text())

        # Params that are the same as unified should NOT be in expert blocks
        hn = data["moe"]["high_noise"]
        assert "batch_size" not in hn
        assert "weight_decay" not in hn

    def test_moe_no_signals_for_t2v(self, moe_project, tmp_path):
        """T2V MoE model also has no signals block."""
        from flimmer.phases.yaml_export import export_yaml

        out = export_yaml(moe_project, tmp_path / "config.yaml")
        data = yaml.safe_load(out.read_text())
        assert "signals" not in data

    def test_moe_base_config_from_unified(self, moe_project, tmp_path):
        """The base optimizer/training/scheduler values come from the unified phase."""
        from flimmer.phases.yaml_export import export_yaml

        out = export_yaml(moe_project, tmp_path / "config.yaml")
        data = yaml.safe_load(out.read_text())

        # Unified phase has lr=5e-5, max_epochs=10
        assert data["optimizer"]["learning_rate"] == pytest.approx(5e-5)
        assert data["training"]["unified_epochs"] == 10


class TestI2vExport:
    """I2V model YAML export (has reference_image signal)."""

    def test_signals_block_present(self, i2v_project, tmp_path):
        """I2V model has reference_image signal -> signals block should appear."""
        from flimmer.phases.yaml_export import export_yaml

        out = export_yaml(i2v_project, tmp_path / "config.yaml")
        data = yaml.safe_load(out.read_text())
        assert "signals" in data
        assert data["signals"]["reference_image"] is True


class TestFloatFormatting:
    """YAML float representation is human-readable."""

    def test_scientific_notation_readable(self, non_moe_project, tmp_path):
        """5e-5 should appear as human-readable notation, not 4.999...e-05."""
        from flimmer.phases.yaml_export import export_yaml

        out = export_yaml(non_moe_project, tmp_path / "config.yaml")
        content = out.read_text()
        # Should contain a clean representation, not precision artifacts
        assert "4.9999" not in content
        assert "4.999" not in content


class TestDisabledPhaseSkipping:
    """Disabled phases are skipped during export."""

    def test_disabled_phase_omitted(self, register_all_models, tmp_path):
        from flimmer.phases.yaml_export import export_yaml

        project = Project.create(name="test_disabled", model_id="wan-2.1-t2v-14b")
        project.add_phase(PhaseConfig(
            phase_type="unified",
            display_name="Active Phase",
            overrides={"learning_rate": 5e-5, "max_epochs": 50},
        ))
        project.add_phase(PhaseConfig(
            phase_type="unified",
            display_name="Disabled Phase",
            enabled=False,
            overrides={"learning_rate": 1e-4, "max_epochs": 200},
        ))

        out = export_yaml(project, tmp_path / "config.yaml")
        data = yaml.safe_load(out.read_text())
        # Should use the active phase's max_epochs, not the disabled one
        assert data["training"]["unified_epochs"] == 50


class TestDatasetPath:
    """Dataset path appears as data_config in YAML."""

    def test_dataset_in_yaml(self, register_all_models, tmp_path):
        from flimmer.phases.yaml_export import export_yaml

        project = Project.create(name="test_ds", model_id="wan-2.1-t2v-14b")
        project.add_phase(PhaseConfig(
            phase_type="unified",
            display_name="With Dataset",
            overrides={"learning_rate": 5e-5, "max_epochs": 10},
            dataset="./data/train.yaml",
        ))

        out = export_yaml(project, tmp_path / "config.yaml")
        data = yaml.safe_load(out.read_text())
        assert data["data_config"] == "./data/train.yaml"


class TestParentDirectoryCreation:
    """export_yaml creates parent directories if needed."""

    def test_creates_parent_dirs(self, non_moe_project, tmp_path):
        from flimmer.phases.yaml_export import export_yaml

        nested = tmp_path / "a" / "b" / "c" / "config.yaml"
        result = export_yaml(non_moe_project, nested)
        assert result.exists()


class TestRoundTrip:
    """Export -> load YAML -> verify key structure."""

    def test_round_trip_structure(self, moe_project, tmp_path):
        from flimmer.phases.yaml_export import export_yaml

        out = export_yaml(moe_project, tmp_path / "config.yaml")
        data = yaml.safe_load(out.read_text())

        # All expected top-level keys present
        expected_keys = {"model", "lora", "optimizer", "scheduler", "training", "moe", "save"}
        assert expected_keys.issubset(set(data.keys()))

        # Nested structure is correct
        assert isinstance(data["model"], dict)
        assert isinstance(data["moe"]["high_noise"], dict)
        assert isinstance(data["moe"]["low_noise"], dict)
