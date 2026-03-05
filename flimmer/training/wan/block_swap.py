"""Hook-based transformer block CPU<->GPU weight swapping for VRAM reduction.

This module provides block-level CPU<->GPU weight swapping for training VRAM
reduction. Based on the musubi-tuner Offloader algorithm but implemented via
PyTorch hooks for compatibility with standard diffusers models.

How it works:
    Wan 2.2 has 40 transformer blocks, each ~175MB at fp8 (~350MB at bf16).
    Block swapping offloads inactive blocks to CPU during forward/backward
    passes, keeping only the currently-executing block and a configurable
    number of "resident" blocks on GPU at all times.

    A secondary CUDA stream handles async weight transfers so that the GPU
    can compute on one block while transferring the next. Pinned (page-locked)
    CPU memory enables direct DMA access by the GPU, roughly 2-3x faster
    than pageable memory transfers.

Algorithm (forward pass, blocks 0..N-1):
    1. forward_pre_hook(block[i]): Synchronize -- wait for any pending async
       transfer of block[i] to complete on the secondary stream.
    2. Block[i] executes forward pass on GPU.
    3. forward_hook(block[i]): If block[i] is in the swappable range, submit
       async transfer: move block[i] Linear weights to CPU pinned buffers,
       and prefetch the next block from CPU to GPU.

Algorithm (backward pass, blocks N-1..0):
    1. full_backward_hook(block[i]): After backward through block[i], submit
       async offload of block[i] to CPU. Prefetch block[i-1] if it needs
       to come from CPU.

Only Linear-like module weights are swapped (nn.Linear, bitsandbytes
Linear8bitLt/Linear4bit). Buffers (e.g., LayerNorm parameters) stay on GPU.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Threshold for logging a warning about extreme swap counts.
# Above this, training speed is significantly impacted.
_EXTREME_SWAP_THRESHOLD = 35


def _get_linear_weights(block: nn.Module) -> list[tuple[nn.Module, str]]:
    """Walk a block's modules and yield (module, 'weight') for Linear-like layers.

    Matches any module whose class name ends with 'Linear', which covers
    nn.Linear, bitsandbytes Linear8bitLt, and Linear4bit. Only yields
    modules that have a 'weight' attribute.

    Why class-name matching instead of isinstance:
        bitsandbytes linear classes are not always importable (optional dep).
        The musubi-tuner uses the same class-name check for compatibility.
    """
    results = []
    for module in block.modules():
        if module.__class__.__name__.endswith("Linear") and hasattr(module, "weight"):
            if module.weight is not None:
                results.append((module, "weight"))
    return results


class BlockSwapOffloader:
    """Hook-based transformer block offloader for VRAM reduction.

    Registers forward_pre_hook, forward_hook, and full_backward_hook on
    transformer blocks to move their Linear weights between GPU and CPU
    during training. Uses a secondary CUDA stream and pinned memory for
    efficient async transfers.

    Args:
        blocks: The transformer's nn.ModuleList of blocks.
        blocks_to_swap: Number of blocks to offload to CPU. Clamped to
            num_blocks - 1 (at least one block must remain on GPU).
        device: The GPU device for training (e.g., torch.device("cuda:0")).

    Usage:
        offloader = BlockSwapOffloader(model.blocks, blocks_to_swap=20, device=device)
        # ... training loop runs; hooks handle weight transfers automatically ...
        offloader.remove_hooks()  # cleanup when done
    """

    def __init__(
        self,
        blocks: nn.ModuleList,
        blocks_to_swap: int,
        device: torch.device,
    ) -> None:
        self.num_blocks: int = len(blocks)
        self.device: torch.device = device
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._pinned_buffers: dict[int, list[torch.Tensor]] = {}

        # Clamp to num_blocks - 1 (can't swap ALL blocks)
        self.blocks_to_swap: int = min(blocks_to_swap, self.num_blocks - 1)

        if self.blocks_to_swap <= 0:
            # No swapping -- nothing to do
            self.blocks_to_swap = 0
            return

        if self.blocks_to_swap != blocks_to_swap:
            logger.warning(
                "blocks_to_swap=%d exceeds maximum (%d). Clamped to %d.",
                blocks_to_swap,
                self.num_blocks - 1,
                self.blocks_to_swap,
            )

        if self.blocks_to_swap > _EXTREME_SWAP_THRESHOLD:
            logger.warning(
                "blocks_to_swap=%d is very high (>%d). Expect significant "
                "training speed reduction.",
                self.blocks_to_swap,
                _EXTREME_SWAP_THRESHOLD,
            )

        # Number of blocks that permanently stay on GPU
        self._num_resident: int = self.num_blocks - self.blocks_to_swap

        # Create secondary CUDA stream for async weight transfers
        self._stream: torch.cuda.Stream = torch.cuda.Stream(device=device)

        # Allocate pinned memory buffers for swappable blocks (one-time cost)
        for i in range(self._num_resident, self.num_blocks):
            self._create_pinned_buffers(blocks[i], i)

        # Apply initial device placement
        self._prepare_block_devices(blocks)

        # Register hooks on all blocks
        self._register_hooks(blocks)

        logger.info(
            "Block swap: %d/%d blocks offloaded, %d resident on GPU",
            self.blocks_to_swap,
            self.num_blocks,
            self._num_resident,
        )

    def _create_pinned_buffers(self, block: nn.Module, block_idx: int) -> None:
        """Allocate pinned CPU tensors for each Linear weight in a block.

        Pinned memory allows direct DMA access by the GPU, making
        CPU<->GPU transfers 2-3x faster than pageable memory. These
        buffers are allocated once and reused across all training steps.
        """
        buffers = []
        for module, _ in _get_linear_weights(block):
            buf = torch.empty_like(module.weight, device="cpu")
            try:
                buf = buf.pin_memory(device=self.device)
            except Exception:
                # pin_memory may fail on some systems (e.g., CPU-only).
                # Fall back to unpinned memory.
                pass
            buffers.append(buf)
        self._pinned_buffers[block_idx] = buffers

    def _prepare_block_devices(self, blocks: nn.ModuleList) -> None:
        """Set initial device placement for all blocks.

        Resident blocks (indices 0..num_resident-1) stay on GPU.
        Swappable blocks (indices num_resident..num_blocks-1) have their
        Linear weights moved to CPU. Non-Linear params (LayerNorm, etc.)
        stay on GPU so they're always ready.

        This method can be called after state_dict swap (expert switch)
        to re-prepare block placement.
        """
        for i in range(self._num_resident, self.num_blocks):
            self._move_block_to_cpu(blocks[i], i)

    def prepare_block_devices(self, blocks: nn.ModuleList) -> None:
        """Re-prepare initial device placement (public API).

        Called after state_dict swap during expert switching to ensure
        pinned buffers and device placement are consistent with the
        new weights.
        """
        if self.blocks_to_swap <= 0:
            return
        self._prepare_block_devices(blocks)

    def _move_block_to_cpu(self, block: nn.Module, block_idx: int) -> None:
        """Move only Linear weights of a block to CPU pinned buffers.

        Non-Linear params (LayerNorm, biases of non-Linear modules)
        stay on their current device.
        """
        linear_weights = _get_linear_weights(block)
        pinned_bufs = self._pinned_buffers.get(block_idx, [])
        for (module, _), pinned_buf in zip(linear_weights, pinned_bufs):
            pinned_buf.copy_(module.weight.data)
            module.weight.data = pinned_buf

    def _move_block_to_gpu(self, blocks: nn.ModuleList, block_idx: int) -> None:
        """Move Linear weights of a block to GPU using async stream.

        Uses non_blocking=True within the secondary CUDA stream context
        for overlapped transfers.
        """
        block = blocks[block_idx]
        with torch.cuda.stream(self._stream):
            for module, _ in _get_linear_weights(block):
                module.weight.data = module.weight.data.to(
                    self.device, non_blocking=True
                )

    def _submit_move_to_cpu(self, blocks: nn.ModuleList, block_idx: int) -> None:
        """Async move of a block's Linear weights to CPU pinned buffers."""
        block = blocks[block_idx]
        linear_weights = _get_linear_weights(block)
        pinned_bufs = self._pinned_buffers.get(block_idx, [])
        with torch.cuda.stream(self._stream):
            for (module, _), pinned_buf in zip(linear_weights, pinned_bufs):
                pinned_buf.copy_(module.weight.data, non_blocking=True)
                module.weight.data = pinned_buf

    def _wait_for_stream(self) -> None:
        """Synchronize: make main stream wait for secondary stream transfers."""
        torch.cuda.current_stream().wait_stream(self._stream)

    def _register_hooks(self, blocks: nn.ModuleList) -> None:
        """Register forward_pre_hook, forward_hook, and backward hooks.

        Forward pre-hooks are registered on ALL blocks to synchronize
        the async stream before each block executes. Forward hooks and
        backward hooks are registered for the swapping logic.
        """
        for i in range(self.num_blocks):
            # Forward pre-hook: synchronize stream before block executes
            h = blocks[i].register_forward_pre_hook(
                self._make_forward_pre_hook(blocks, i)
            )
            self._handles.append(h)

            # Forward hook: after block forward, swap weights if needed
            fwd_hook = self._make_forward_hook(blocks, i)
            if fwd_hook is not None:
                h = blocks[i].register_forward_hook(fwd_hook)
                self._handles.append(h)

            # Backward hook: reverse-order weight swapping
            bwd_hook = self._make_backward_hook(blocks, i)
            if bwd_hook is not None:
                h = blocks[i].register_full_backward_hook(bwd_hook)
                self._handles.append(h)

    def _make_forward_pre_hook(
        self, blocks: nn.ModuleList, block_idx: int
    ) -> Any:
        """Create hook that synchronizes stream before block forward.

        Ensures any pending async transfer of this block's weights is
        complete before the main stream executes the block. Prevents
        NaN/illegal-memory-access from using partially-transferred weights.
        """

        def hook(module: nn.Module, args: Any) -> None:
            self._wait_for_stream()

        return hook

    def _make_forward_hook(
        self, blocks: nn.ModuleList, block_idx: int
    ) -> Any | None:
        """Create hook that swaps weights after block forward completes.

        For blocks in the swappable range (after the resident blocks):
        - Move this block's Linear weights to CPU (pinned buffers)
        - Prefetch the next block that needs to come from CPU

        Returns None for resident blocks (no swap needed).
        """
        # During forward, blocks execute 0..N-1 in order.
        # After block[i] finishes forward:
        #   - If i >= num_resident, block[i] was brought from CPU, send it back
        #   - Prefetch block[i + 1] if it's a swappable block
        if block_idx < self._num_resident:
            # Resident block. But we may need to prefetch the first
            # swappable block when the last resident block finishes.
            if block_idx == self._num_resident - 1 and self.blocks_to_swap > 0:
                next_idx = self._num_resident

                def hook(module: nn.Module, args: Any, output: Any) -> None:
                    self._move_block_to_gpu(blocks, next_idx)

                return hook
            return None

        # Swappable block: offload to CPU, prefetch next if exists
        next_swap_idx = block_idx + 1 if block_idx + 1 < self.num_blocks else None

        def hook(module: nn.Module, args: Any, output: Any) -> None:
            self._submit_move_to_cpu(blocks, block_idx)
            if next_swap_idx is not None:
                self._move_block_to_gpu(blocks, next_swap_idx)

        return hook

    def _make_backward_hook(
        self, blocks: nn.ModuleList, block_idx: int
    ) -> Any | None:
        """Create hook for backward-pass weight swapping.

        Backward traverses blocks N-1..0 (reverse order). After backward
        through block[i], if block[i] was a swappable block, offload it
        to CPU and prefetch block[i-1] if it needs to come from CPU.

        Returns None for blocks that don't need backward swapping.
        """
        if block_idx < self._num_resident:
            return None

        # After backward through this swappable block, send it to CPU
        # and prefetch the previous block if it's also swappable
        prev_idx = block_idx - 1 if block_idx - 1 >= self._num_resident else None

        def hook(
            module: nn.Module, grad_input: Any, grad_output: Any
        ) -> None:
            self._submit_move_to_cpu(blocks, block_idx)
            if prev_idx is not None:
                self._move_block_to_gpu(blocks, prev_idx)

        return hook

    def remove_hooks(self) -> None:
        """Remove all registered hook handles.

        Call this when block swapping is no longer needed (e.g., at the
        end of training or before re-registering hooks after a disk
        reload expert switch).
        """
        for h in self._handles:
            h.remove()
        self._handles.clear()
