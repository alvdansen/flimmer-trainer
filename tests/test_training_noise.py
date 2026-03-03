"""Tests for flimmer.training.noise — flow matching math."""

import numpy as np
import pytest

from flimmer.training.noise import (
    SAMPLING_STRATEGIES,
    FlowMatchingSchedule,
    compute_snr,
    flow_matching_interpolate,
    flow_matching_velocity,
    get_expert_masks,
    logit_normal_timesteps,
    shift_timesteps,
    sigmoid_timesteps,
    uniform_timesteps,
)


class TestUniformTimesteps:
    """Uniform timestep sampling."""

    def test_shape(self):
        t = uniform_timesteps(10)
        assert t.shape == (10,)

    def test_range(self):
        t = uniform_timesteps(1000)
        assert t.min() > 0.0
        assert t.max() < 1.0

    def test_reproducible(self):
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        t1 = uniform_timesteps(10, rng=rng1)
        t2 = uniform_timesteps(10, rng=rng2)
        np.testing.assert_array_equal(t1, t2)

    def test_different_seeds_differ(self):
        t1 = uniform_timesteps(10, rng=np.random.default_rng(1))
        t2 = uniform_timesteps(10, rng=np.random.default_rng(2))
        assert not np.allclose(t1, t2)


class TestShiftTimesteps:
    """Shifted timestep sampling (Wan default)."""

    def test_shape(self):
        t = shift_timesteps(10, flow_shift=3.0)
        assert t.shape == (10,)

    def test_range(self):
        t = shift_timesteps(1000, flow_shift=3.0)
        assert t.min() > 0.0
        assert t.max() < 1.0

    def test_higher_shift_biases_up(self):
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        t_low = shift_timesteps(10000, flow_shift=1.0, rng=rng1)
        t_high = shift_timesteps(10000, flow_shift=5.0, rng=rng2)
        # Higher shift should push the mean toward higher noise
        assert t_high.mean() > t_low.mean()

    def test_shift_1_close_to_uniform(self):
        rng = np.random.default_rng(42)
        t = shift_timesteps(10000, flow_shift=1.0, rng=rng)
        # With shift=1.0, the formula simplifies to identity
        assert abs(t.mean() - 0.5) < 0.05


class TestLogitNormalTimesteps:
    """Logit-normal timestep sampling."""

    def test_shape(self):
        t = logit_normal_timesteps(10)
        assert t.shape == (10,)

    def test_range(self):
        t = logit_normal_timesteps(1000)
        assert t.min() > 0.0
        assert t.max() < 1.0

    def test_centered(self):
        t = logit_normal_timesteps(10000, mean=0.0, std=1.0,
                                    rng=np.random.default_rng(42))
        assert abs(t.mean() - 0.5) < 0.05


class TestSigmoidTimesteps:
    """Sigmoid timestep sampling."""

    def test_shape(self):
        t = sigmoid_timesteps(10)
        assert t.shape == (10,)

    def test_range(self):
        t = sigmoid_timesteps(1000)
        assert t.min() > 0.0
        assert t.max() < 1.0


class TestSamplingStrategies:
    """Strategy name → function mapping."""

    def test_all_strategies_present(self):
        assert "uniform" in SAMPLING_STRATEGIES
        assert "shift" in SAMPLING_STRATEGIES
        assert "logit_normal" in SAMPLING_STRATEGIES
        assert "sigmoid" in SAMPLING_STRATEGIES

    def test_strategies_callable(self):
        for name, fn in SAMPLING_STRATEGIES.items():
            assert callable(fn), f"Strategy {name} is not callable"


class TestFlowMatchingInterpolate:
    """noisy = (1-t)*clean + t*noise"""

    def test_t0_returns_clean(self):
        clean = np.array([1.0, 2.0, 3.0])
        noise = np.array([10.0, 20.0, 30.0])
        t = np.array([0.0])
        result = flow_matching_interpolate(clean, noise, t)
        np.testing.assert_allclose(result, clean)

    def test_t1_returns_noise(self):
        clean = np.array([1.0, 2.0, 3.0])
        noise = np.array([10.0, 20.0, 30.0])
        t = np.array([1.0])
        result = flow_matching_interpolate(clean, noise, t)
        np.testing.assert_allclose(result, noise)

    def test_t05_returns_midpoint(self):
        clean = np.array([0.0, 0.0])
        noise = np.array([2.0, 4.0])
        t = np.array([0.5])
        result = flow_matching_interpolate(clean, noise, t)
        np.testing.assert_allclose(result, [1.0, 2.0])

    def test_batched(self):
        clean = np.array([[1.0], [2.0]])
        noise = np.array([[10.0], [20.0]])
        t = np.array([0.0, 1.0])
        result = flow_matching_interpolate(clean, noise, t)
        np.testing.assert_allclose(result[0], [1.0])
        np.testing.assert_allclose(result[1], [20.0])


class TestFlowMatchingVelocity:
    """velocity = noise - clean"""

    def test_basic(self):
        clean = np.array([1.0, 2.0])
        noise = np.array([3.0, 5.0])
        result = flow_matching_velocity(clean, noise, np.array([0.5]))
        np.testing.assert_allclose(result, [2.0, 3.0])

    def test_timestep_independent(self):
        clean = np.array([1.0])
        noise = np.array([5.0])
        r1 = flow_matching_velocity(clean, noise, np.array([0.1]))
        r2 = flow_matching_velocity(clean, noise, np.array([0.9]))
        np.testing.assert_allclose(r1, r2)


class TestComputeSNR:
    """SNR = (1-t)/t"""

    def test_low_noise(self):
        snr = compute_snr(np.array([0.1]))
        assert snr[0] == pytest.approx(9.0)

    def test_high_noise(self):
        snr = compute_snr(np.array([0.9]))
        assert snr[0] == pytest.approx(1.0 / 9.0)

    def test_midpoint(self):
        snr = compute_snr(np.array([0.5]))
        assert snr[0] == pytest.approx(1.0)


class TestExpertMasks:
    """Expert masking based on boundary ratio."""

    def test_basic_split(self):
        t = np.array([0.1, 0.5, 0.8, 0.9, 0.95])
        high, low = get_expert_masks(t, boundary_ratio=0.875)
        # t >= 0.875 → high-noise expert
        np.testing.assert_allclose(high, [0, 0, 0, 1, 1])
        np.testing.assert_allclose(low, [1, 1, 1, 0, 0])

    def test_masks_sum_to_one(self):
        t = np.random.uniform(0, 1, size=100)
        high, low = get_expert_masks(t, 0.5)
        np.testing.assert_allclose(high + low, np.ones(100))

    def test_boundary_exact(self):
        t = np.array([0.875])
        high, low = get_expert_masks(t, 0.875)
        assert high[0] == 1.0
        assert low[0] == 0.0


class TestFlowMatchingSchedule:
    """FlowMatchingSchedule protocol implementation."""

    def test_num_timesteps(self):
        schedule = FlowMatchingSchedule(1000)
        assert schedule.num_timesteps == 1000

    def test_sample_uniform(self):
        schedule = FlowMatchingSchedule()
        t = schedule.sample_timesteps(10, strategy="uniform")
        assert t.shape == (10,)

    def test_sample_shift(self):
        schedule = FlowMatchingSchedule()
        t = schedule.sample_timesteps(10, strategy="shift", flow_shift=3.0)
        assert t.shape == (10,)

    def test_sample_invalid_strategy(self):
        schedule = FlowMatchingSchedule()
        with pytest.raises(ValueError, match="Unknown timestep"):
            schedule.sample_timesteps(10, strategy="invalid")

    def test_compute_noisy_latent(self):
        schedule = FlowMatchingSchedule()
        clean = np.zeros((2, 3))
        noise = np.ones((2, 3))
        t = np.array([0.0, 1.0])
        result = schedule.compute_noisy_latent(clean, noise, t)
        np.testing.assert_allclose(result[0], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(result[1], [1.0, 1.0, 1.0])

    def test_compute_target(self):
        schedule = FlowMatchingSchedule()
        result = schedule.compute_target(
            np.array([1.0]), np.array([3.0]), np.array([0.5])
        )
        np.testing.assert_allclose(result, [2.0])

    def test_get_snr(self):
        schedule = FlowMatchingSchedule()
        snr = schedule.get_signal_to_noise_ratio(np.array([0.5]))
        assert snr[0] == pytest.approx(1.0)
