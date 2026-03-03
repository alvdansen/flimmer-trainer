"""Tests for flimmer.training.protocols — protocol compliance."""

import numpy as np
import pytest

from flimmer.training.protocols import (
    InferencePipeline,
    ModelBackend,
    NoiseSchedule,
)


class MockNoiseSchedule:
    """Mock implementation of the NoiseSchedule protocol."""

    @property
    def num_timesteps(self) -> int:
        return 1000

    def sample_timesteps(self, batch_size, strategy="uniform", flow_shift=1.0, generator=None):
        return np.random.uniform(0, 1, size=batch_size)

    def compute_noisy_latent(self, clean, noise, timesteps):
        return clean + noise

    def compute_target(self, clean, noise, timesteps):
        return noise - clean

    def get_signal_to_noise_ratio(self, timesteps):
        return (1.0 - timesteps) / timesteps


class MockModelBackend:
    """Mock implementation of the ModelBackend protocol."""

    @property
    def model_id(self) -> str:
        return "mock-model-v1"

    @property
    def supports_moe(self) -> bool:
        return True

    @property
    def supports_reference_image(self) -> bool:
        return False

    def load_model(self, config):
        return "mock_model"

    def get_lora_target_modules(self):
        return ["attn.to_q", "attn.to_k"]

    def get_expert_mask(self, timesteps, boundary_ratio):
        high = (timesteps >= boundary_ratio).astype(float)
        low = (timesteps < boundary_ratio).astype(float)
        return high, low

    def prepare_model_inputs(self, batch, timesteps, noisy_latents):
        return {"input": batch}

    def forward(self, model, **inputs):
        return 0.5

    def setup_gradient_checkpointing(self, model):
        pass

    def get_noise_schedule(self):
        return MockNoiseSchedule()


class MockInferencePipeline:
    """Mock implementation of the InferencePipeline protocol."""

    def generate(self, model, lora_state_dict, prompt, negative_prompt="",
                 num_inference_steps=30, guidance_scale=5.0, seed=42,
                 reference_image=None):
        return f"sample_{seed}.mp4"


class TestNoiseScheduleProtocol:
    """NoiseSchedule protocol compliance."""

    def test_isinstance_check(self):
        schedule = MockNoiseSchedule()
        assert isinstance(schedule, NoiseSchedule)

    def test_num_timesteps(self):
        schedule = MockNoiseSchedule()
        assert schedule.num_timesteps == 1000

    def test_sample_timesteps(self):
        schedule = MockNoiseSchedule()
        t = schedule.sample_timesteps(batch_size=4)
        assert len(t) == 4

    def test_compute_noisy_latent(self):
        schedule = MockNoiseSchedule()
        clean = np.zeros(4)
        noise = np.ones(4)
        result = schedule.compute_noisy_latent(clean, noise, np.array([0.5]))
        assert result is not None

    def test_compute_target(self):
        schedule = MockNoiseSchedule()
        result = schedule.compute_target(np.zeros(4), np.ones(4), np.array([0.5]))
        assert result is not None

    def test_get_snr(self):
        schedule = MockNoiseSchedule()
        snr = schedule.get_signal_to_noise_ratio(np.array([0.5]))
        assert snr is not None


class TestModelBackendProtocol:
    """ModelBackend protocol compliance."""

    def test_isinstance_check(self):
        backend = MockModelBackend()
        assert isinstance(backend, ModelBackend)

    def test_model_id(self):
        assert MockModelBackend().model_id == "mock-model-v1"

    def test_supports_moe(self):
        assert MockModelBackend().supports_moe is True

    def test_load_model(self):
        model = MockModelBackend().load_model(None)
        assert model == "mock_model"

    def test_get_lora_targets(self):
        targets = MockModelBackend().get_lora_target_modules()
        assert "attn.to_q" in targets

    def test_forward(self):
        result = MockModelBackend().forward("model", input="data")
        assert result == 0.5


class TestInferencePipelineProtocol:
    """InferencePipeline protocol compliance."""

    def test_isinstance_check(self):
        pipeline = MockInferencePipeline()
        assert isinstance(pipeline, InferencePipeline)

    def test_generate(self):
        pipeline = MockInferencePipeline()
        result = pipeline.generate(
            model=None,
            lora_state_dict=None,
            prompt="test prompt",
        )
        assert "sample_42.mp4" in result
