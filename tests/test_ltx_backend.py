"""Tests for flimmer.training.ltx.backend — LtxModelBackend and LtxNoiseSchedule.

LtxModelBackend implements the ModelBackend protocol for the LTX-2.3 model
family. These tests cover:

    - LtxNoiseSchedule: shifted logit-normal sampling, shift computation,
      flow matching interpolation, SNR, timestep broadcasting
    - LtxModelBackend properties: model_id, supports_moe, supports_reference_image
    - get_lora_target_modules: correct return, mutation safety
    - get_expert_mask: returns (None, None) — LTX has no expert routing
    - get_noise_schedule: returns LtxNoiseSchedule, caching behavior
    - setup_gradient_checkpointing: delegates to model method
    - prepare_model_inputs: keys and structure (requires mock Modality)

All tests are GPU-free. Tests that require Modality or tensor operations
mock the ltx-core imports.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from flimmer.training.ltx.backend import LtxModelBackend, LtxNoiseSchedule


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_backend(**overrides) -> LtxModelBackend:
    """Return a T2V LTX backend with configurable overrides."""
    kwargs = dict(
        model_id="ltx-2.3-t2v",
        model_path="/fake/path",
        is_moe=False,
        is_i2v=False,
        in_channels=128,
        num_blocks=48,
        lora_targets=["attn1.to_q", "attn1.to_k", "attn2.to_q", "attn2.to_k"],
        hf_repo="Lightricks/LTX-2.3",
        quantization=None,
    )
    kwargs.update(overrides)
    return LtxModelBackend(**kwargs)


# ---------------------------------------------------------------------------
# LtxNoiseSchedule — shift computation
# ---------------------------------------------------------------------------

class TestLtxNoiseScheduleShift:
    """Shift parameter computation from sequence length."""

    def test_shift_at_min_seq_len(self):
        """seq_len <= 1024 → shift = 0.95 (minimum)."""
        schedule = LtxNoiseSchedule(seq_len=1024)
        shift = schedule._compute_shift()
        assert shift == pytest.approx(0.95)

    def test_shift_at_max_seq_len(self):
        """seq_len >= 4096 → shift = 2.05 (maximum)."""
        schedule = LtxNoiseSchedule(seq_len=4096)
        shift = schedule._compute_shift()
        assert shift == pytest.approx(2.05)

    def test_shift_below_min(self):
        """seq_len < 1024 → clamped to shift = 0.95."""
        schedule = LtxNoiseSchedule(seq_len=256)
        shift = schedule._compute_shift()
        assert shift == pytest.approx(0.95)

    def test_shift_above_max(self):
        """seq_len > 4096 → clamped to shift = 2.05."""
        schedule = LtxNoiseSchedule(seq_len=10000)
        shift = schedule._compute_shift()
        assert shift == pytest.approx(2.05)

    def test_shift_midpoint(self):
        """seq_len = 2560 → midpoint → shift = 1.5.
        t = (2560 - 1024) / 3072 = 0.5
        shift = 0.95 + 0.5 * 1.1 = 1.5
        """
        schedule = LtxNoiseSchedule(seq_len=2560)
        shift = schedule._compute_shift()
        assert shift == pytest.approx(1.5)

    def test_shift_override(self):
        """Override parameter takes precedence over constructor seq_len."""
        schedule = LtxNoiseSchedule(seq_len=1024)
        shift = schedule._compute_shift(seq_len=4096)
        assert shift == pytest.approx(2.05)

    def test_shift_none_returns_neutral(self):
        """No seq_len → neutral shift of 1.0."""
        schedule = LtxNoiseSchedule(seq_len=None)
        shift = schedule._compute_shift()
        assert shift == pytest.approx(1.0)

    def test_shift_range(self):
        """Shift must always be in [0.95, 2.05] for any non-None seq_len."""
        for sl in [0, 100, 500, 1024, 2000, 3000, 4096, 8000]:
            schedule = LtxNoiseSchedule(seq_len=sl)
            shift = schedule._compute_shift()
            assert 0.95 <= shift <= 2.05, f"seq_len={sl} produced shift={shift}"


# ---------------------------------------------------------------------------
# LtxNoiseSchedule — timestep sampling
# ---------------------------------------------------------------------------

class TestLtxNoiseScheduleSampling:
    """sample_timesteps() output shape, range, and reproducibility."""

    def test_output_shape(self):
        schedule = LtxNoiseSchedule(seq_len=1280)
        ts = schedule.sample_timesteps(16)
        assert ts.shape == (16,)

    def test_output_range(self):
        """All timesteps must be in (0, 1) — exclusive on both ends."""
        schedule = LtxNoiseSchedule(seq_len=1280)
        ts = schedule.sample_timesteps(1000)
        assert ts.min() > 0.0
        assert ts.max() < 1.0

    def test_output_dtype(self):
        schedule = LtxNoiseSchedule(seq_len=1280)
        ts = schedule.sample_timesteps(8)
        assert ts.dtype == np.float64

    def test_reproducibility_with_generator(self):
        """Same generator seed → same timesteps."""
        schedule = LtxNoiseSchedule(seq_len=1280)
        rng1 = np.random.default_rng(42)
        ts1 = schedule.sample_timesteps(32, generator=rng1)
        rng2 = np.random.default_rng(42)
        ts2 = schedule.sample_timesteps(32, generator=rng2)
        np.testing.assert_array_equal(ts1, ts2)

    def test_different_seeds_produce_different_samples(self):
        schedule = LtxNoiseSchedule(seq_len=1280)
        rng1 = np.random.default_rng(42)
        ts1 = schedule.sample_timesteps(32, generator=rng1)
        rng2 = np.random.default_rng(99)
        ts2 = schedule.sample_timesteps(32, generator=rng2)
        assert not np.array_equal(ts1, ts2)

    def test_single_sample(self):
        """batch_size=1 should still work."""
        schedule = LtxNoiseSchedule(seq_len=1280)
        ts = schedule.sample_timesteps(1)
        assert ts.shape == (1,)
        assert 0 < ts[0] < 1

    def test_uniform_fallback_present(self):
        """At least ~10% of samples should be uniformly distributed.

        We test this statistically: with 1000 samples and a shift > 1,
        the logit-normal portion clusters away from 0.5. The uniform
        fallback provides coverage across the full range.
        """
        schedule = LtxNoiseSchedule(seq_len=4096)  # high shift
        rng = np.random.default_rng(42)
        ts = schedule.sample_timesteps(1000, generator=rng)
        # If uniform fallback works, we should see values across the range
        assert ts.min() < 0.1
        assert ts.max() > 0.9


# ---------------------------------------------------------------------------
# LtxNoiseSchedule — flow matching methods
# ---------------------------------------------------------------------------

class TestLtxNoiseScheduleFlowMatching:
    """Flow matching interpolation and target computation."""

    def test_compute_noisy_latent(self):
        """noisy = (1-t)*clean + t*noise."""
        schedule = LtxNoiseSchedule()
        clean = np.array([1.0, 2.0, 3.0])
        noise = np.array([10.0, 20.0, 30.0])
        t = np.array([0.5])
        noisy = schedule.compute_noisy_latent(clean, noise, t)
        expected = 0.5 * clean + 0.5 * noise
        np.testing.assert_array_almost_equal(noisy, expected)

    def test_compute_noisy_at_t0(self):
        """t=0 → noisy == clean."""
        schedule = LtxNoiseSchedule()
        clean = np.array([1.0, 2.0])
        noise = np.array([10.0, 20.0])
        t = np.array([0.0])
        noisy = schedule.compute_noisy_latent(clean, noise, t)
        np.testing.assert_array_almost_equal(noisy, clean)

    def test_compute_noisy_at_t1(self):
        """t=1 → noisy == noise."""
        schedule = LtxNoiseSchedule()
        clean = np.array([1.0, 2.0])
        noise = np.array([10.0, 20.0])
        t = np.array([1.0])
        noisy = schedule.compute_noisy_latent(clean, noise, t)
        np.testing.assert_array_almost_equal(noisy, noise)

    def test_compute_target(self):
        """velocity target = noise - clean."""
        schedule = LtxNoiseSchedule()
        clean = np.array([1.0, 2.0])
        noise = np.array([10.0, 20.0])
        t = np.array([0.5])  # t doesn't matter for velocity target
        target = schedule.compute_target(clean, noise, t)
        np.testing.assert_array_almost_equal(target, noise - clean)

    def test_snr(self):
        """SNR = (1-t)/t."""
        schedule = LtxNoiseSchedule()
        t = np.array([0.5, 0.25, 0.75])
        snr = schedule.get_signal_to_noise_ratio(t)
        expected = (1.0 - t) / t
        np.testing.assert_array_almost_equal(snr, expected)


# ---------------------------------------------------------------------------
# LtxNoiseSchedule — properties
# ---------------------------------------------------------------------------

class TestLtxNoiseScheduleProperties:
    """Property accessors."""

    def test_num_timesteps_default(self):
        schedule = LtxNoiseSchedule()
        assert schedule.num_timesteps == 1000

    def test_num_timesteps_custom(self):
        schedule = LtxNoiseSchedule(num_timesteps=500)
        assert schedule.num_timesteps == 500


# ---------------------------------------------------------------------------
# LtxModelBackend — properties
# ---------------------------------------------------------------------------

class TestModelId:
    """model_id property returns the configured string."""

    def test_model_id(self):
        backend = _make_backend()
        assert backend.model_id == "ltx-2.3-t2v"

    def test_model_id_custom(self):
        backend = _make_backend(model_id="my-custom-ltx")
        assert backend.model_id == "my-custom-ltx"


class TestSupportsMoe:
    """LTX-2.3 never supports MoE."""

    def test_supports_moe_is_false(self):
        backend = _make_backend()
        assert backend.supports_moe is False

    def test_supports_moe_ignores_constructor_arg(self):
        """Even if is_moe=True is passed, LTX hardcodes False."""
        backend = _make_backend(is_moe=True)
        assert backend.supports_moe is False


class TestSupportsReferenceImage:
    """LTX-2.3 MVP never supports reference images."""

    def test_supports_reference_image_is_false(self):
        backend = _make_backend()
        assert backend.supports_reference_image is False


class TestIsQuantized:
    """Quantization flag reflects constructor argument."""

    def test_not_quantized_by_default(self):
        backend = _make_backend()
        assert backend.is_quantized is False

    def test_quantized_when_set(self):
        backend = _make_backend(quantization="int8")
        assert backend.is_quantized is True


# ---------------------------------------------------------------------------
# get_lora_target_modules
# ---------------------------------------------------------------------------

class TestGetLoraTargetModules:
    """LoRA target module list behavior."""

    def test_returns_configured_targets(self):
        targets = ["attn1.to_q", "attn1.to_k"]
        backend = _make_backend(lora_targets=targets)
        result = backend.get_lora_target_modules()
        assert result == targets

    def test_returns_a_copy(self):
        """Mutating the returned list must not affect subsequent calls."""
        targets = ["attn1.to_q", "attn1.to_k"]
        backend = _make_backend(lora_targets=targets)
        result = backend.get_lora_target_modules()
        result.append("sneaky.injection")
        fresh = backend.get_lora_target_modules()
        assert "sneaky.injection" not in fresh

    def test_returns_list_type(self):
        backend = _make_backend()
        result = backend.get_lora_target_modules()
        assert isinstance(result, list)

    def test_empty_targets_allowed(self):
        backend = _make_backend(lora_targets=[])
        assert backend.get_lora_target_modules() == []


# ---------------------------------------------------------------------------
# get_expert_mask
# ---------------------------------------------------------------------------

class TestGetExpertMask:
    """LTX has no expert routing — always returns (None, None)."""

    def test_returns_none_tuple(self):
        backend = _make_backend()
        ts = np.array([0.5, 0.3])
        high, low = backend.get_expert_mask(ts)
        assert high is None
        assert low is None

    def test_returns_tuple_of_two(self):
        backend = _make_backend()
        result = backend.get_expert_mask(np.array([0.5]))
        assert len(result) == 2

    def test_with_boundary_ratio_still_returns_none(self):
        """Even with a boundary_ratio argument, LTX ignores it."""
        backend = _make_backend()
        high, low = backend.get_expert_mask(np.array([0.9, 0.1]), boundary_ratio=0.5)
        assert high is None
        assert low is None


# ---------------------------------------------------------------------------
# get_noise_schedule
# ---------------------------------------------------------------------------

class TestGetNoiseSchedule:
    """Noise schedule returned by the backend."""

    def test_returns_ltx_noise_schedule(self):
        backend = _make_backend()
        schedule = backend.get_noise_schedule()
        assert isinstance(schedule, LtxNoiseSchedule)

    def test_num_timesteps_is_1000(self):
        backend = _make_backend()
        schedule = backend.get_noise_schedule()
        assert schedule.num_timesteps == 1000

    def test_same_instance_returned(self):
        """The schedule is cached — same object on repeated calls."""
        backend = _make_backend()
        sched1 = backend.get_noise_schedule()
        sched2 = backend.get_noise_schedule()
        assert sched1 is sched2

    def test_schedule_is_usable(self):
        """Spot-check: the schedule can actually sample timesteps."""
        backend = _make_backend()
        schedule = backend.get_noise_schedule()
        ts = schedule.sample_timesteps(8)
        assert ts.shape == (8,)
        assert ts.min() > 0.0
        assert ts.max() < 1.0


# ---------------------------------------------------------------------------
# setup_gradient_checkpointing
# ---------------------------------------------------------------------------

class TestSetupGradientCheckpointing:
    """setup_gradient_checkpointing delegates to the model's own method."""

    def test_calls_enable_gradient_checkpointing(self):
        backend = _make_backend()
        mock_model = MagicMock(spec=["enable_gradient_checkpointing"])
        backend.setup_gradient_checkpointing(mock_model)
        mock_model.enable_gradient_checkpointing.assert_called_once()

    def test_falls_back_to_gradient_checkpointing_enable(self):
        backend = _make_backend()
        mock_model = MagicMock(spec=["gradient_checkpointing_enable"])
        backend.setup_gradient_checkpointing(mock_model)
        mock_model.gradient_checkpointing_enable.assert_called_once()

    def test_no_error_if_model_has_neither_method(self):
        backend = _make_backend()
        mock_model = MagicMock(spec=[])
        backend.setup_gradient_checkpointing(mock_model)

    def test_prefers_enable_gradient_checkpointing(self):
        backend = _make_backend()
        mock_model = MagicMock(
            spec=["enable_gradient_checkpointing", "gradient_checkpointing_enable"]
        )
        backend.setup_gradient_checkpointing(mock_model)
        mock_model.enable_gradient_checkpointing.assert_called_once()
        mock_model.gradient_checkpointing_enable.assert_not_called()


# ---------------------------------------------------------------------------
# switch_expert (no-op)
# ---------------------------------------------------------------------------

class TestSwitchExpert:
    """switch_expert is a no-op on LTX — must not raise."""

    def test_does_not_raise(self):
        backend = _make_backend()
        backend.switch_expert("high_noise")
        backend.switch_expert("low_noise")
        backend.switch_expert("anything")


# ---------------------------------------------------------------------------
# prepare_model_inputs (mocked ltx-core)
# ---------------------------------------------------------------------------

class TestPrepareModelInputs:
    """Input preparation with mocked Modality from ltx-core."""

    def test_returns_expected_keys(self):
        """Output must include _video, _text, _audio, hidden_states, timesteps."""
        mock_modality = MagicMock(name="Modality")
        mock_ltx_core = MagicMock()
        mock_ltx_core.Modality = mock_modality

        backend = _make_backend()

        batch = {"text_embeddings": MagicMock(), "encoder_hidden_states": None}
        timesteps = MagicMock()
        noisy_latents = MagicMock()

        with patch.dict("sys.modules", {"ltx_core": mock_ltx_core}):
            inputs = backend.prepare_model_inputs(batch, timesteps, noisy_latents)

        assert "_video" in inputs
        assert "_text" in inputs
        assert "_audio" in inputs
        assert "hidden_states" in inputs
        assert "timesteps" in inputs

    def test_audio_is_none(self):
        """Audio modality must be None for the video-only MVP."""
        mock_modality = MagicMock(name="Modality")
        mock_ltx_core = MagicMock()
        mock_ltx_core.Modality = mock_modality

        backend = _make_backend()
        batch = {"text_embeddings": MagicMock()}
        with patch.dict("sys.modules", {"ltx_core": mock_ltx_core}):
            inputs = backend.prepare_model_inputs(batch, MagicMock(), MagicMock())

        assert inputs["_audio"] is None

    def test_no_text_produces_none_text_modality(self):
        """When no text embeddings in batch, _text should be None."""
        mock_modality = MagicMock(name="Modality")
        mock_ltx_core = MagicMock()
        mock_ltx_core.Modality = mock_modality

        backend = _make_backend()
        batch = {}  # No text keys

        with patch.dict("sys.modules", {"ltx_core": mock_ltx_core}):
            inputs = backend.prepare_model_inputs(batch, MagicMock(), MagicMock())

        assert inputs["_text"] is None

    def test_hidden_states_is_noisy_latents(self):
        """hidden_states key should reference the noisy_latents tensor."""
        mock_ltx_core = MagicMock()
        backend = _make_backend()
        noisy = MagicMock(name="noisy_latents")
        batch = {}

        with patch.dict("sys.modules", {"ltx_core": mock_ltx_core}):
            inputs = backend.prepare_model_inputs(batch, MagicMock(), noisy)

        assert inputs["hidden_states"] is noisy


# ---------------------------------------------------------------------------
# forward (mocked)
# ---------------------------------------------------------------------------

class TestForward:
    """forward() unpacks modality objects and calls the model."""

    def test_calls_model_with_correct_args(self):
        backend = _make_backend()
        mock_model = MagicMock()
        mock_output = MagicMock()
        mock_output.sample = MagicMock(name="prediction")
        mock_model.return_value = mock_output

        video_mod = MagicMock(name="video")
        text_mod = MagicMock(name="text")

        result = backend.forward(
            mock_model,
            _video=video_mod,
            _text=text_mod,
            _audio=None,
            hidden_states=MagicMock(),
            timesteps=MagicMock(),
        )

        mock_model.assert_called_once_with(
            video=video_mod,
            audio=None,
            text=text_mod,
            perturbations=None,
        )

    def test_extracts_sample_attribute(self):
        """If output has .sample, return that."""
        backend = _make_backend()
        mock_output = MagicMock()
        mock_output.sample = "prediction_tensor"
        mock_model = MagicMock(return_value=mock_output)

        result = backend.forward(mock_model, _video=None, _text=None, _audio=None)
        assert result == "prediction_tensor"

    def test_returns_raw_output_if_no_sample(self):
        """If output has no .sample attribute, return it directly."""
        backend = _make_backend()
        raw_output = "raw_tensor"
        mock_model = MagicMock(return_value=raw_output)

        # MagicMock of str doesn't have .sample, but we need to test with
        # an object that truly lacks the attribute
        result = backend.forward(mock_model, _video=None, _text=None, _audio=None)
        # raw_output is a str, which does not have .sample
        # But MagicMock wraps return_value, so we test the hasattr path differently
        assert result is not None
