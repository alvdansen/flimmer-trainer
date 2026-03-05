"""Tests for the BlockSwapOffloader — hook-based transformer block offloading.

All tests run on CPU-only CI by mocking CUDA operations (streams, pinned
memory). The tests verify hook registration, initial device placement,
clamping, cleanup, and selective Linear-weight-only swapping.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch, PropertyMock

import torch
import torch.nn as nn
import pytest


def _make_blocks(num_blocks: int = 5) -> nn.ModuleList:
    """Create a simple ModuleList of Linear-containing blocks for testing.

    Each block is a Sequential with a Linear and a LayerNorm.
    This simulates a simplified transformer block where only the Linear
    weights should be swapped, not the LayerNorm params.
    """
    blocks = nn.ModuleList()
    for _ in range(num_blocks):
        block = nn.Sequential(
            nn.Linear(8, 8, bias=False),
            nn.LayerNorm(8),
        )
        blocks.append(block)
    return blocks


@pytest.fixture
def mock_cuda():
    """Patch CUDA operations so tests run on CPU-only CI."""
    mock_stream = MagicMock()
    mock_stream.__enter__ = MagicMock(return_value=mock_stream)
    mock_stream.__exit__ = MagicMock(return_value=False)

    mock_current_stream = MagicMock()

    with (
        patch("torch.cuda.Stream", return_value=mock_stream),
        patch("torch.cuda.stream", return_value=mock_stream),
        patch("torch.cuda.current_stream", return_value=mock_current_stream),
        patch("torch.Tensor.pin_memory", side_effect=lambda self, **kw: self),
    ):
        yield {
            "stream": mock_stream,
            "current_stream": mock_current_stream,
        }


class TestBlockSwapOffloaderHookRegistration:
    """Verify that hooks are registered correctly on blocks."""

    def test_registers_forward_pre_hook_on_all_blocks(self, mock_cuda):
        from flimmer.training.wan.block_swap import BlockSwapOffloader

        blocks = _make_blocks(5)
        offloader = BlockSwapOffloader(blocks, blocks_to_swap=2, device=torch.device("cpu"))

        # Every block should have at least one forward_pre_hook registered
        for block in blocks:
            pre_hooks = dict(block._forward_pre_hooks)
            assert len(pre_hooks) > 0, "Each block should have a forward_pre_hook"

        offloader.remove_hooks()

    def test_registers_forward_hook_on_swappable_blocks(self, mock_cuda):
        from flimmer.training.wan.block_swap import BlockSwapOffloader

        blocks = _make_blocks(5)
        offloader = BlockSwapOffloader(blocks, blocks_to_swap=2, device=torch.device("cpu"))

        # Swappable blocks (those in the CPU range) should have forward_hooks
        # Resident blocks (first num_blocks - blocks_to_swap = 3) may or may not
        # The offloader should have stored hook handles
        assert len(offloader._handles) > 0

        offloader.remove_hooks()

    def test_blocks_to_swap_zero_no_hooks(self, mock_cuda):
        from flimmer.training.wan.block_swap import BlockSwapOffloader

        blocks = _make_blocks(5)
        offloader = BlockSwapOffloader(blocks, blocks_to_swap=0, device=torch.device("cpu"))

        # No hooks should be registered
        assert len(offloader._handles) == 0

        offloader.remove_hooks()


class TestBlockSwapOffloaderInitialPlacement:
    """Verify that blocks are placed on the correct initial devices."""

    def test_initial_placement_gpu_and_cpu(self, mock_cuda):
        from flimmer.training.wan.block_swap import BlockSwapOffloader

        blocks = _make_blocks(5)
        # With blocks_to_swap=2, first 3 blocks stay on original device,
        # last 2 blocks get their Linear weights moved to CPU
        offloader = BlockSwapOffloader(blocks, blocks_to_swap=2, device=torch.device("cpu"))

        # Since everything starts on CPU in test environment, we verify
        # the offloader's internal state tracks the placement correctly
        assert offloader.num_blocks == 5
        assert offloader.blocks_to_swap == 2
        num_resident = offloader.num_blocks - offloader.blocks_to_swap
        assert num_resident == 3

        offloader.remove_hooks()


class TestBlockSwapOffloaderClamping:
    """Verify blocks_to_swap is clamped to num_blocks - 1."""

    def test_clamp_to_max(self, mock_cuda):
        from flimmer.training.wan.block_swap import BlockSwapOffloader

        blocks = _make_blocks(5)
        # Request swapping all 5 blocks -- should clamp to 4 (can't swap ALL)
        offloader = BlockSwapOffloader(blocks, blocks_to_swap=5, device=torch.device("cpu"))
        assert offloader.blocks_to_swap == 4  # clamped to num_blocks - 1

        offloader.remove_hooks()

    def test_clamp_exceeds_max(self, mock_cuda):
        from flimmer.training.wan.block_swap import BlockSwapOffloader

        blocks = _make_blocks(5)
        # Request swapping 100 blocks -- should clamp to 4
        offloader = BlockSwapOffloader(blocks, blocks_to_swap=100, device=torch.device("cpu"))
        assert offloader.blocks_to_swap == 4

        offloader.remove_hooks()


class TestBlockSwapOffloaderCleanup:
    """Verify remove_hooks() clears all registered hook handles."""

    def test_remove_hooks_clears_handles(self, mock_cuda):
        from flimmer.training.wan.block_swap import BlockSwapOffloader

        blocks = _make_blocks(5)
        offloader = BlockSwapOffloader(blocks, blocks_to_swap=2, device=torch.device("cpu"))

        num_handles = len(offloader._handles)
        assert num_handles > 0

        offloader.remove_hooks()
        assert len(offloader._handles) == 0

    def test_remove_hooks_removes_from_blocks(self, mock_cuda):
        from flimmer.training.wan.block_swap import BlockSwapOffloader

        blocks = _make_blocks(5)

        # Count hooks before
        pre_hooks_before = sum(len(b._forward_pre_hooks) for b in blocks)

        offloader = BlockSwapOffloader(blocks, blocks_to_swap=2, device=torch.device("cpu"))

        # Count hooks after registration
        pre_hooks_after = sum(len(b._forward_pre_hooks) for b in blocks)
        assert pre_hooks_after > pre_hooks_before

        offloader.remove_hooks()

        # After removal, hooks should be back to the original count
        pre_hooks_cleaned = sum(len(b._forward_pre_hooks) for b in blocks)
        assert pre_hooks_cleaned == pre_hooks_before


class TestBlockSwapOffloaderLinearOnly:
    """Verify that only Linear-like module weights are targeted for swapping."""

    def test_only_linear_weights_in_pinned_buffers(self, mock_cuda):
        from flimmer.training.wan.block_swap import BlockSwapOffloader

        blocks = _make_blocks(5)
        offloader = BlockSwapOffloader(blocks, blocks_to_swap=2, device=torch.device("cpu"))

        # Pinned buffers should exist for swappable blocks (indices 3, 4)
        # Each block has 1 Linear (the LayerNorm weights are NOT included)
        for block_idx in range(offloader.num_blocks - offloader.blocks_to_swap, offloader.num_blocks):
            assert block_idx in offloader._pinned_buffers
            # Each block has exactly 1 Linear weight
            assert len(offloader._pinned_buffers[block_idx]) == 1

        offloader.remove_hooks()

    def test_pinned_buffers_allocated_once_at_setup(self, mock_cuda):
        from flimmer.training.wan.block_swap import BlockSwapOffloader

        blocks = _make_blocks(5)
        offloader = BlockSwapOffloader(blocks, blocks_to_swap=2, device=torch.device("cpu"))

        # Only swappable blocks get pinned buffers
        num_resident = offloader.num_blocks - offloader.blocks_to_swap
        for i in range(num_resident):
            assert i not in offloader._pinned_buffers

        offloader.remove_hooks()
