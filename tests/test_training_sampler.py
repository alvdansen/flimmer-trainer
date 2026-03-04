"""Tests for flimmer.training.sampler — preview generation framework."""

import pytest
from pathlib import Path

from flimmer.training.errors import SamplingError
from flimmer.training.phase import PhaseType
from flimmer.training.sampler import SamplingEngine


class TestShouldSample:
    """Sampling schedule logic."""

    def test_disabled(self):
        engine = SamplingEngine(enabled=False)
        assert not engine.should_sample(5, PhaseType.UNIFIED)

    def test_no_prompts(self):
        engine = SamplingEngine(enabled=True, prompts=[])
        assert not engine.should_sample(5, PhaseType.UNIFIED)

    def test_on_interval(self):
        engine = SamplingEngine(enabled=True, prompts=["test"], every_n_epochs=5)
        assert engine.should_sample(5, PhaseType.UNIFIED)
        assert engine.should_sample(10, PhaseType.UNIFIED)

    def test_off_interval(self):
        engine = SamplingEngine(enabled=True, prompts=["test"], every_n_epochs=5)
        assert not engine.should_sample(3, PhaseType.UNIFIED)
        assert not engine.should_sample(7, PhaseType.UNIFIED)

    def test_epoch_zero(self):
        engine = SamplingEngine(enabled=True, prompts=["test"], every_n_epochs=1)
        assert not engine.should_sample(0, PhaseType.UNIFIED)

    def test_skip_phase(self):
        engine = SamplingEngine(
            enabled=True, prompts=["test"], every_n_epochs=5,
            skip_phases=["full_noise"],
        )
        assert not engine.should_sample(5, PhaseType.UNIFIED)
        assert engine.should_sample(5, PhaseType.HIGH_NOISE)

    def test_skip_multiple_phases(self):
        engine = SamplingEngine(
            enabled=True, prompts=["test"], every_n_epochs=5,
            skip_phases=["high_noise", "low_noise"],
        )
        assert engine.should_sample(5, PhaseType.UNIFIED)
        assert not engine.should_sample(5, PhaseType.HIGH_NOISE)
        assert not engine.should_sample(5, PhaseType.LOW_NOISE)


class TestPartnerResolution:
    """Partner LoRA resolution during expert phases."""

    def test_unified_no_partner(self):
        engine = SamplingEngine()
        result = engine.resolve_partner_lora(
            active_expert=None,
            high_noise_path=None, low_noise_path=None, unified_path=None,
        )
        assert result is None

    def test_high_noise_uses_low_partner(self):
        engine = SamplingEngine()
        result = engine.resolve_partner_lora(
            active_expert="high_noise",
            high_noise_path=None,
            low_noise_path="/path/to/low.safetensors",
            unified_path=None,
        )
        assert result == "/path/to/low.safetensors"

    def test_low_noise_uses_high_partner(self):
        engine = SamplingEngine()
        result = engine.resolve_partner_lora(
            active_expert="low_noise",
            high_noise_path="/path/to/high.safetensors",
            low_noise_path=None,
            unified_path=None,
        )
        assert result == "/path/to/high.safetensors"

    def test_fallback_to_unified(self):
        engine = SamplingEngine()
        result = engine.resolve_partner_lora(
            active_expert="high_noise",
            high_noise_path=None,
            low_noise_path=None,
            unified_path="/path/to/unified.safetensors",
        )
        assert result == "/path/to/unified.safetensors"

    def test_no_partner_available(self):
        engine = SamplingEngine()
        result = engine.resolve_partner_lora(
            active_expert="high_noise",
            high_noise_path=None, low_noise_path=None, unified_path=None,
        )
        assert result is None


class TestSeedWalking:
    """Seed walking across prompts."""

    def test_walk_seed_true(self):
        engine = SamplingEngine(seed=42, walk_seed=True)
        assert engine.get_seed_for_prompt(0) == 42
        assert engine.get_seed_for_prompt(1) == 43
        assert engine.get_seed_for_prompt(2) == 44

    def test_walk_seed_false(self):
        engine = SamplingEngine(seed=42, walk_seed=False)
        assert engine.get_seed_for_prompt(0) == 42
        assert engine.get_seed_for_prompt(1) == 42


class TestOutputDir:
    """Output directory generation."""

    def test_creates_dir(self, tmp_path):
        engine = SamplingEngine(sample_dir=tmp_path)
        out = engine.get_output_dir(PhaseType.UNIFIED, epoch=5)
        assert out.is_dir()
        assert "full_noise_epoch005" in out.name

    def test_expert_dir(self, tmp_path):
        engine = SamplingEngine(sample_dir=tmp_path)
        out = engine.get_output_dir(PhaseType.HIGH_NOISE, epoch=15)
        assert "high_epoch015" in out.name

    def test_no_sample_dir_raises(self):
        engine = SamplingEngine(sample_dir=None)
        with pytest.raises(SamplingError, match="No sample directory"):
            engine.get_output_dir(PhaseType.UNIFIED, epoch=5)

    def test_base_dir_override(self, tmp_path):
        engine = SamplingEngine(sample_dir="/should/not/use")
        out = engine.get_output_dir(PhaseType.LOW_NOISE, epoch=10, base_dir=tmp_path)
        assert str(tmp_path) in str(out)


class TestGenerateSamples:
    """Sample generation with mock pipeline."""

    def test_basic_generation(self, tmp_path):
        class MockPipeline:
            def generate(self, **kwargs):
                return str(tmp_path / f"sample_{kwargs['seed']}.mp4")

        engine = SamplingEngine(
            enabled=True, prompts=["test prompt"], seed=42,
            walk_seed=True, sample_dir=tmp_path,
        )
        results = engine.generate_samples(
            pipeline=MockPipeline(),
            model=None,
            lora_state_dict=None,
            phase_type=PhaseType.UNIFIED,
            epoch=5,
        )
        assert len(results) == 1

    def test_multiple_prompts(self, tmp_path):
        class MockPipeline:
            def generate(self, **kwargs):
                return str(tmp_path / f"s_{kwargs['seed']}.mp4")

        engine = SamplingEngine(
            enabled=True, prompts=["p1", "p2", "p3"], seed=100,
            walk_seed=True, sample_dir=tmp_path,
        )
        results = engine.generate_samples(
            pipeline=MockPipeline(), model=None, lora_state_dict=None,
            phase_type=PhaseType.HIGH_NOISE, epoch=10,
        )
        assert len(results) == 3
