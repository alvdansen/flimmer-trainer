"""Tests for flimmer.training.optimizer — optimizer and scheduler construction."""

import math
import pytest

from flimmer.training.optimizer import (
    _cosine_with_min_lr_lambda,
    _polynomial_lambda,
    _rex_lambda,
    _warmup_lambda,
    compute_total_steps,
)


class TestWarmupLambda:
    """Linear warmup from 0 to 1."""

    def test_no_warmup(self):
        fn = _warmup_lambda(warmup_steps=0)
        assert fn(0) == 1.0
        assert fn(100) == 1.0

    def test_warmup_midpoint(self):
        fn = _warmup_lambda(warmup_steps=100)
        assert fn(50) == pytest.approx(0.5)

    def test_warmup_complete(self):
        fn = _warmup_lambda(warmup_steps=100)
        assert fn(100) == 1.0
        assert fn(200) == 1.0

    def test_warmup_start(self):
        fn = _warmup_lambda(warmup_steps=10)
        assert fn(0) == pytest.approx(0.0)
        assert fn(1) == pytest.approx(0.1)


class TestCosineWithMinLR:
    """Cosine decay with warmup and floor."""

    def test_starts_at_one(self):
        fn = _cosine_with_min_lr_lambda(total_steps=100, warmup_steps=0, min_lr_ratio=0.01)
        assert fn(0) == pytest.approx(1.0)

    def test_ends_at_min_lr(self):
        fn = _cosine_with_min_lr_lambda(total_steps=100, warmup_steps=0, min_lr_ratio=0.01)
        assert fn(100) == pytest.approx(0.01)

    def test_midpoint(self):
        fn = _cosine_with_min_lr_lambda(total_steps=100, warmup_steps=0, min_lr_ratio=0.0)
        val = fn(50)
        # At midpoint of cosine, should be ~0.5
        assert 0.4 < val < 0.6

    def test_with_warmup(self):
        fn = _cosine_with_min_lr_lambda(total_steps=100, warmup_steps=10, min_lr_ratio=0.01)
        assert fn(5) == pytest.approx(0.5)
        assert fn(10) == pytest.approx(1.0)

    def test_never_below_min(self):
        fn = _cosine_with_min_lr_lambda(total_steps=100, warmup_steps=0, min_lr_ratio=0.05)
        for step in range(101):
            assert fn(step) >= 0.05 - 1e-6


class TestPolynomialLambda:
    """Polynomial decay."""

    def test_starts_at_one(self):
        fn = _polynomial_lambda(total_steps=100, warmup_steps=0, min_lr_ratio=0.0)
        assert fn(0) == pytest.approx(1.0)

    def test_ends_at_min(self):
        fn = _polynomial_lambda(total_steps=100, warmup_steps=0, min_lr_ratio=0.01)
        assert fn(100) == pytest.approx(0.01)

    def test_monotonically_decreasing(self):
        fn = _polynomial_lambda(total_steps=100, warmup_steps=0, min_lr_ratio=0.01)
        values = [fn(s) for s in range(101)]
        for i in range(1, len(values)):
            assert values[i] <= values[i - 1] + 1e-6


class TestRexLambda:
    """Rex scheduler."""

    def test_starts_at_one(self):
        fn = _rex_lambda(total_steps=100, warmup_steps=0)
        assert fn(0) == pytest.approx(1.0)

    def test_decreases(self):
        fn = _rex_lambda(total_steps=100, warmup_steps=0)
        assert fn(50) < fn(0)
        assert fn(100) < fn(50)

    def test_with_warmup(self):
        fn = _rex_lambda(total_steps=100, warmup_steps=10)
        assert fn(5) == pytest.approx(0.5)


class TestComputeTotalSteps:
    """Total optimizer step computation."""

    def test_basic(self):
        steps = compute_total_steps(
            num_samples=100, batch_size=1,
            gradient_accumulation_steps=1, max_epochs=10,
        )
        assert steps == 1000

    def test_with_grad_accum(self):
        steps = compute_total_steps(
            num_samples=100, batch_size=1,
            gradient_accumulation_steps=4, max_epochs=10,
        )
        assert steps == 250

    def test_with_batch_size(self):
        steps = compute_total_steps(
            num_samples=100, batch_size=2,
            gradient_accumulation_steps=1, max_epochs=10,
        )
        assert steps == 500

    def test_minimum_one_step_per_epoch(self):
        steps = compute_total_steps(
            num_samples=1, batch_size=10,
            gradient_accumulation_steps=10, max_epochs=5,
        )
        assert steps == 5  # At least 1 step per epoch
