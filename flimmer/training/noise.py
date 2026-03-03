"""Flow matching noise schedule — pure math, GPU-free.

All functions work with numpy arrays for testing. The FlowMatchingSchedule
class wraps them into the NoiseSchedule protocol for use in the training loop.

Flow matching interpolation:
    noisy = (1 - t) * clean + t * noise
    target (velocity) = noise - clean
    SNR = (1 - t) / t

Timestep sampling strategies:
    uniform      — flat distribution across [0, 1]
    shift        — shifted distribution favoring mid-to-high noise (Wan default)
    logit_normal — logit-normal distribution, concentrates around center
    sigmoid      — sigmoid-based, similar to logit_normal with different tails

Expert masking:
    Timesteps are split into high-noise and low-noise groups by the
    boundary_ratio. The mask is binary (0 or 1) — loss for non-matching
    timesteps is zeroed out rather than sampling only matching timesteps.
    This gives better gradient estimates with ~12% waste at boundary_ratio=0.875.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Timestep sampling strategies (pure functions, numpy)
# ---------------------------------------------------------------------------

def uniform_timesteps(
    batch_size: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample timesteps uniformly from (0, 1).

    Excludes exact 0.0 and 1.0 to avoid division-by-zero in SNR computation
    and degenerate noise levels.

    Args:
        batch_size: Number of timesteps to sample.
        rng: Optional numpy random generator for reproducibility.

    Returns:
        Float64 array of shape [batch_size] with values in (0, 1).
    """
    if rng is None:
        rng = np.random.default_rng()
    # Sample from (0, 1) — open interval
    return rng.uniform(low=1e-5, high=1.0 - 1e-5, size=batch_size)


def shift_timesteps(
    batch_size: int,
    flow_shift: float = 3.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample timesteps with shifted distribution (Wan default).

    Applies a shift that biases sampling toward higher noise levels.
    The shift formula: t_shifted = t * flow_shift / (1 + (flow_shift - 1) * t)

    This matches Wan's pretraining distribution — using the same distribution
    during fine-tuning maintains consistency with the model's expectations.

    Args:
        batch_size: Number of timesteps to sample.
        flow_shift: Shift parameter. Higher = more bias toward high noise.
            Wan uses 3.0 for 480p, 5.0 for 720p.
        rng: Optional numpy random generator.

    Returns:
        Float64 array of shape [batch_size] with shifted values in (0, 1).
    """
    u = uniform_timesteps(batch_size, rng)
    # Apply shift: t_shifted = u * s / (1 + (s - 1) * u)
    shifted = u * flow_shift / (1.0 + (flow_shift - 1.0) * u)
    return np.clip(shifted, 1e-5, 1.0 - 1e-5)


def logit_normal_timesteps(
    batch_size: int,
    mean: float = 0.0,
    std: float = 1.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample timesteps from logit-normal distribution.

    Samples z ~ N(mean, std²) then applies sigmoid: t = 1 / (1 + exp(-z)).
    Concentrates samples around the center of [0, 1].

    Args:
        batch_size: Number of timesteps to sample.
        mean: Mean of the underlying normal distribution.
        std: Standard deviation of the underlying normal distribution.
        rng: Optional numpy random generator.

    Returns:
        Float64 array of shape [batch_size] with values in (0, 1).
    """
    if rng is None:
        rng = np.random.default_rng()
    z = rng.normal(loc=mean, scale=std, size=batch_size)
    t = 1.0 / (1.0 + np.exp(-z))  # sigmoid
    return np.clip(t, 1e-5, 1.0 - 1e-5)


def sigmoid_timesteps(
    batch_size: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample timesteps using sigmoid-based sampling.

    Similar to logit-normal but with a fixed mapping that creates
    different tail behavior. Uses the inverse CDF of the sigmoid.

    Args:
        batch_size: Number of timesteps to sample.
        rng: Optional numpy random generator.

    Returns:
        Float64 array of shape [batch_size] with values in (0, 1).
    """
    u = uniform_timesteps(batch_size, rng)
    # Apply sigmoid transformation
    t = 1.0 / (1.0 + np.exp(-(np.log(u / (1.0 - u)))))
    return np.clip(t, 1e-5, 1.0 - 1e-5)


# Strategy name → function mapping
SAMPLING_STRATEGIES: dict[str, Any] = {
    "uniform": uniform_timesteps,
    "shift": shift_timesteps,
    "logit_normal": logit_normal_timesteps,
    "sigmoid": sigmoid_timesteps,
}


# ---------------------------------------------------------------------------
# Flow matching math (pure functions, numpy)
# ---------------------------------------------------------------------------

def flow_matching_interpolate(
    clean: np.ndarray,
    noise: np.ndarray,
    timesteps: np.ndarray,
) -> np.ndarray:
    """Compute noisy latents via flow matching interpolation.

    noisy = (1 - t) * clean + t * noise

    Args:
        clean: Clean data array of any shape.
        noise: Noise array, same shape as clean.
        timesteps: Timestep values in [0, 1]. Shape must be broadcastable
            to clean/noise (typically [B] or [B, 1, 1, 1, 1]).

    Returns:
        Noisy array, same shape as clean.
    """
    # Reshape timesteps for broadcasting: [B] → [B, 1, 1, ...]
    t = _broadcast_timesteps(timesteps, clean.ndim)
    return (1.0 - t) * clean + t * noise


def flow_matching_velocity(
    clean: np.ndarray,
    noise: np.ndarray,
    timesteps: np.ndarray,
) -> np.ndarray:
    """Compute flow matching velocity target.

    velocity = noise - clean

    The velocity is constant along the flow path — it doesn't depend
    on the timestep. The timesteps parameter is accepted for protocol
    compatibility but unused.

    Args:
        clean: Clean data array.
        noise: Noise array, same shape as clean.
        timesteps: Unused (accepted for protocol compatibility).

    Returns:
        Velocity target, same shape as clean.
    """
    return noise - clean


def compute_snr(timesteps: np.ndarray) -> np.ndarray:
    """Compute signal-to-noise ratio for flow matching.

    SNR = (1 - t) / t

    Higher SNR = less noise (cleaner signal). At t=0 (pure signal), SNR→∞.
    At t=1 (pure noise), SNR→0.

    Args:
        timesteps: Timestep values in (0, 1). Must not include exact 0 or 1.

    Returns:
        SNR values, same shape as timesteps.
    """
    return (1.0 - timesteps) / timesteps


def get_expert_masks(
    timesteps: np.ndarray,
    boundary_ratio: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute expert masks based on timestep noise levels.

    High-noise expert handles timesteps where t >= boundary_ratio
    (high noise, low signal). Low-noise expert handles t < boundary_ratio
    (low noise, high signal).

    The masks are binary (0.0 or 1.0) — loss for non-matching timesteps
    is zeroed out. Every timestep belongs to exactly one expert.

    Args:
        timesteps: Timestep values in [0, 1], shape [B].
        boundary_ratio: Boundary between experts (e.g. 0.875).
            Timesteps >= this go to high-noise expert.

    Returns:
        Tuple of (high_noise_mask, low_noise_mask), each shape [B].
        Both are float64 arrays with values 0.0 or 1.0.
    """
    high_mask = (timesteps >= boundary_ratio).astype(np.float64)
    low_mask = (timesteps < boundary_ratio).astype(np.float64)
    return high_mask, low_mask


# ---------------------------------------------------------------------------
# FlowMatchingSchedule — protocol implementation
# ---------------------------------------------------------------------------

class FlowMatchingSchedule:
    """NoiseSchedule implementation for flow matching models.

    Wraps the pure functions above into the NoiseSchedule protocol.
    Uses numpy for computation — the training loop converts to torch
    tensors as needed.

    This is the default schedule for all Wan models.

    Args:
        num_timesteps: Total discrete timesteps (1000 for Wan).
    """

    def __init__(self, num_timesteps: int = 1000) -> None:
        self._num_timesteps = num_timesteps

    @property
    def num_timesteps(self) -> int:
        """Total number of discrete timesteps."""
        return self._num_timesteps

    def sample_timesteps(
        self,
        batch_size: int,
        strategy: str = "uniform",
        flow_shift: float = 1.0,
        generator: Any = None,
    ) -> np.ndarray:
        """Sample random timesteps using the specified strategy.

        Args:
            batch_size: Number of timesteps to sample.
            strategy: Strategy name from SAMPLING_STRATEGIES.
            flow_shift: Shift parameter (only used by 'shift' strategy).
            generator: Numpy random Generator for reproducibility.

        Returns:
            Float64 array of shape [batch_size] with values in (0, 1).

        Raises:
            ValueError: If the strategy name is unknown.
        """
        if strategy not in SAMPLING_STRATEGIES:
            valid = ", ".join(sorted(SAMPLING_STRATEGIES.keys()))
            raise ValueError(
                f"Unknown timestep sampling strategy '{strategy}'. "
                f"Valid strategies: {valid}."
            )

        fn = SAMPLING_STRATEGIES[strategy]

        # Build kwargs based on what the function accepts
        kwargs: dict[str, Any] = {"batch_size": batch_size, "rng": generator}
        if strategy == "shift":
            kwargs["flow_shift"] = flow_shift

        return fn(**kwargs)

    def compute_noisy_latent(
        self,
        clean: np.ndarray,
        noise: np.ndarray,
        timesteps: np.ndarray,
    ) -> np.ndarray:
        """Compute noisy latents: (1-t)*clean + t*noise."""
        return flow_matching_interpolate(clean, noise, timesteps)

    def compute_target(
        self,
        clean: np.ndarray,
        noise: np.ndarray,
        timesteps: np.ndarray,
    ) -> np.ndarray:
        """Compute velocity target: noise - clean."""
        return flow_matching_velocity(clean, noise, timesteps)

    def get_signal_to_noise_ratio(self, timesteps: np.ndarray) -> np.ndarray:
        """Compute SNR: (1-t)/t."""
        return compute_snr(timesteps)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _broadcast_timesteps(timesteps: np.ndarray, ndim: int) -> np.ndarray:
    """Reshape timesteps [B] to [B, 1, 1, ...] for broadcasting.

    Args:
        timesteps: 1D array of shape [B].
        ndim: Number of dimensions in the target array.

    Returns:
        Reshaped timesteps with (ndim - 1) trailing dimensions of size 1.
    """
    if timesteps.ndim == 0:
        return timesteps
    shape = (-1,) + (1,) * (ndim - 1)
    return timesteps.reshape(shape)
