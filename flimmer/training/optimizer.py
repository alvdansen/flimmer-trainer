"""Per-phase optimizer and scheduler construction.

Builds the right optimizer and learning rate scheduler for each training
phase. Each expert phase gets its own fresh optimizer — unified momentum
doesn't transfer to expert-specific gradient landscapes.

Optimizer support:
    adamw, adamw8bit, adafactor, came, prodigy, ademamix, schedule_free_adamw,
    cpu_offload, adam_mini

Scheduler support:
    cosine_with_min_lr, constant, constant_with_warmup, polynomial, rex

LoRA+ parameter groups: A-matrix and B-matrix get different learning rates.
"""

from __future__ import annotations

import math
from typing import Any

from flimmer.training.errors import PhaseConfigError


# ---------------------------------------------------------------------------
# Optimizer construction
# ---------------------------------------------------------------------------

def build_optimizer(
    params: list[dict[str, Any]],
    optimizer_type: str,
    learning_rate: float,
    weight_decay: float = 0.01,
    betas: list[float] | None = None,
    eps: float | list[float] = 1e-8,
    optimizer_args: dict | None = None,
    model: Any = None,
) -> Any:
    """Build an optimizer instance for a training phase.

    Creates the appropriate optimizer based on type string. Each optimizer
    type maps to a specific PyTorch or third-party implementation.

    Args:
        params: Parameter groups (from build_parameter_groups or raw params).
            Each group is a dict with 'params' key (list of tensors or names).
        optimizer_type: Optimizer type name (from VALID_OPTIMIZERS).
        learning_rate: Base learning rate (used as default if not in groups).
        weight_decay: Weight decay coefficient.
        betas: Adam beta parameters [beta1, beta2] or [beta1, beta2, beta3].
        eps: Adam epsilon for numerical stability.
        optimizer_args: Additional kwargs passed to the optimizer constructor.
        model: Model reference. Needed only for adam_mini (named_parameters
            access for Hessian-aware parameter partitioning). Ignored by
            all other optimizer types.

    Returns:
        Optimizer instance.

    Raises:
        PhaseConfigError: If the optimizer type is unknown or import fails.
    """
    if betas is None:
        betas = [0.9, 0.999]
    if optimizer_args is None:
        optimizer_args = {}

    # Common kwargs
    common = {
        "lr": learning_rate,
        "weight_decay": weight_decay,
    }

    try:
        if optimizer_type == "adamw":
            import torch.optim
            return torch.optim.AdamW(
                params,
                betas=tuple(betas[:2]),
                eps=eps if isinstance(eps, float) else eps[0],
                **common,
                **optimizer_args,
            )

        elif optimizer_type == "adamw8bit":
            try:
                import bitsandbytes as bnb
                return bnb.optim.AdamW8bit(
                    params,
                    betas=tuple(betas[:2]),
                    eps=eps if isinstance(eps, float) else eps[0],
                    **common,
                    **optimizer_args,
                )
            except ImportError:
                raise PhaseConfigError(
                    "adamw8bit requires bitsandbytes. "
                    "Install with: pip install bitsandbytes"
                )

        elif optimizer_type == "adafactor":
            try:
                from transformers.optimization import Adafactor
                return Adafactor(
                    params,
                    scale_parameter=False,
                    relative_step=False,
                    warmup_init=False,
                    **common,
                    **optimizer_args,
                )
            except ImportError:
                raise PhaseConfigError(
                    "adafactor requires transformers. "
                    "Install with: pip install transformers"
                )

        elif optimizer_type == "came":
            try:
                from came_pytorch import CAME
                eps_vals = eps if isinstance(eps, list) else [eps, eps]
                return CAME(
                    params,
                    betas=tuple(betas[:3]) if len(betas) >= 3 else tuple(betas[:2]) + (0.999,),
                    eps=tuple(eps_vals[:2]),
                    **common,
                    **optimizer_args,
                )
            except ImportError:
                raise PhaseConfigError(
                    "CAME optimizer requires came-pytorch. "
                    "Install with: pip install came-pytorch"
                )

        elif optimizer_type == "prodigy":
            try:
                from prodigyopt import Prodigy
                return Prodigy(
                    params,
                    betas=tuple(betas[:2]),
                    **common,
                    **optimizer_args,
                )
            except ImportError:
                raise PhaseConfigError(
                    "Prodigy optimizer requires prodigyopt. "
                    "Install with: pip install prodigyopt"
                )

        elif optimizer_type == "ademamix":
            try:
                from ademamix_optimizer import AdEMAMix
                return AdEMAMix(
                    params,
                    betas=tuple(betas[:3]) if len(betas) >= 3 else tuple(betas[:2]) + (0.9999,),
                    eps=eps if isinstance(eps, float) else eps[0],
                    **common,
                    **optimizer_args,
                )
            except ImportError:
                raise PhaseConfigError(
                    "AdEMAMix requires ademamix-optimizer. "
                    "Install with: pip install ademamix-optimizer"
                )

        elif optimizer_type == "schedule_free_adamw":
            try:
                from schedulefree import AdamWScheduleFree
                return AdamWScheduleFree(
                    params,
                    betas=tuple(betas[:2]),
                    eps=eps if isinstance(eps, float) else eps[0],
                    **common,
                    **optimizer_args,
                )
            except ImportError:
                raise PhaseConfigError(
                    "schedule_free_adamw requires schedulefree. "
                    "Install with: pip install schedulefree"
                )

        elif optimizer_type == "cpu_offload":
            try:
                from torchao.prototype.low_bit_optim import CPUOffloadOptimizer
                import torch.optim
                return CPUOffloadOptimizer(
                    params,
                    torch.optim.AdamW,
                    offload_gradients=False,  # True breaks gradient accumulation
                    lr=learning_rate,
                    weight_decay=weight_decay,
                    betas=tuple(betas[:2]),
                    eps=eps if isinstance(eps, float) else eps[0],
                    **optimizer_args,
                )
            except ImportError:
                raise PhaseConfigError(
                    "cpu_offload requires torchao. "
                    "Install with: pip install torchao"
                )

        elif optimizer_type == "adam_mini":
            try:
                from adam_mini import Adam_mini
            except ImportError:
                raise PhaseConfigError(
                    "adam_mini requires adam-mini. "
                    "Install with: pip install adam-mini"
                )
            if model is None:
                raise PhaseConfigError(
                    "adam_mini requires model reference for named parameter access. "
                    "This is a bug — please report it."
                )
            # Filter to trainable params only
            trainable_named = [
                (n, p) for n, p in model.named_parameters() if p.requires_grad
            ]
            optimizer = Adam_mini(
                named_parameters=trainable_named,
                lr=learning_rate,
                betas=tuple(betas[:2]),
                eps=eps if isinstance(eps, float) else eps[0],
                weight_decay=weight_decay,
                # Wan 2.2 architecture dimensions for Hessian partitioning
                dim=5120,       # WAN_HIDDEN_DIM
                n_heads=40,     # WAN_NUM_HEADS
                n_kv_heads=40,  # Same as n_heads for Wan (no GQA)
            )
            # Add Wan-specific module names for correct parameter classification.
            # Adam-mini lowercases param names and does substring matching.
            optimizer.wqk_names.update({"to_q", "to_k"})
            optimizer.wv_names.update({"to_v"})
            optimizer.attn_proj_names.update({"to_out"})
            optimizer.mlp_names.add("ffn")
            return optimizer

        else:
            raise PhaseConfigError(
                f"Unknown optimizer type '{optimizer_type}'. "
                f"This should have been caught by config validation."
            )

    except PhaseConfigError:
        raise
    except ImportError as e:
        raise PhaseConfigError(
            f"Failed to import optimizer '{optimizer_type}': {e}"
        ) from e


# ---------------------------------------------------------------------------
# Scheduler construction
# ---------------------------------------------------------------------------

def build_scheduler(
    optimizer: Any,
    scheduler_type: str,
    total_steps: int,
    warmup_steps: int = 0,
    min_lr_ratio: float = 0.01,
    min_lr: float | None = None,
    rex_alpha: float = 0.1,
    rex_beta: float = 0.9,
) -> Any:
    """Build a learning rate scheduler for a training phase.

    Args:
        optimizer: The optimizer instance.
        scheduler_type: Scheduler type name (from VALID_SCHEDULERS).
        total_steps: Total training steps in this phase.
        warmup_steps: Steps of linear warmup from 0 to peak LR.
        min_lr_ratio: Minimum LR as fraction of peak LR (0.0-1.0).
        min_lr: Absolute minimum LR (overrides min_lr_ratio if set).
        rex_alpha: Rex scheduler alpha parameter.
        rex_beta: Rex scheduler beta parameter.

    Returns:
        Learning rate scheduler instance.

    Raises:
        PhaseConfigError: If scheduler type is unknown or import fails.
    """
    try:
        import torch.optim.lr_scheduler as lr_sched

        if scheduler_type == "constant":
            return lr_sched.LambdaLR(optimizer, lr_lambda=lambda step: 1.0)

        elif scheduler_type == "constant_with_warmup":
            return lr_sched.LambdaLR(
                optimizer,
                lr_lambda=_warmup_lambda(warmup_steps),
            )

        elif scheduler_type == "cosine_with_min_lr":
            return lr_sched.LambdaLR(
                optimizer,
                lr_lambda=_cosine_with_min_lr_lambda(
                    total_steps=total_steps,
                    warmup_steps=warmup_steps,
                    min_lr_ratio=min_lr_ratio,
                ),
            )

        elif scheduler_type == "polynomial":
            return lr_sched.LambdaLR(
                optimizer,
                lr_lambda=_polynomial_lambda(
                    total_steps=total_steps,
                    warmup_steps=warmup_steps,
                    min_lr_ratio=min_lr_ratio,
                    power=2.0,
                ),
            )

        elif scheduler_type == "rex":
            return lr_sched.LambdaLR(
                optimizer,
                lr_lambda=_rex_lambda(
                    total_steps=total_steps,
                    warmup_steps=warmup_steps,
                    alpha=rex_alpha,
                    beta=rex_beta,
                ),
            )

        else:
            raise PhaseConfigError(
                f"Unknown scheduler type '{scheduler_type}'. "
                f"This should have been caught by config validation."
            )

    except PhaseConfigError:
        raise
    except ImportError as e:
        raise PhaseConfigError(
            f"torch is required for scheduler construction: {e}"
        ) from e


# ---------------------------------------------------------------------------
# Lambda functions for LR schedulers
# ---------------------------------------------------------------------------

def _warmup_lambda(warmup_steps: int):
    """Linear warmup from 0 to 1 over warmup_steps, then constant 1.0."""
    def lr_lambda(step: int) -> float:
        if warmup_steps <= 0:
            return 1.0
        if step < warmup_steps:
            return float(step) / float(warmup_steps)
        return 1.0
    return lr_lambda


def _cosine_with_min_lr_lambda(
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float,
):
    """Cosine decay with warmup and minimum LR floor.

    Goes from 1.0 → min_lr_ratio following a cosine curve after warmup.
    The min_lr_ratio prevents the LR from reaching zero, keeping the
    optimizer active for late-training detail refinement.
    """
    def lr_lambda(step: int) -> float:
        # Warmup phase
        if step < warmup_steps:
            if warmup_steps <= 0:
                return 1.0
            return float(step) / float(warmup_steps)

        # Cosine decay phase
        remaining = total_steps - warmup_steps
        if remaining <= 0:
            return 1.0

        progress = float(step - warmup_steps) / float(remaining)
        progress = min(progress, 1.0)

        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay
    return lr_lambda


def _polynomial_lambda(
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float,
    power: float = 2.0,
):
    """Polynomial decay with warmup and minimum LR floor."""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            if warmup_steps <= 0:
                return 1.0
            return float(step) / float(warmup_steps)

        remaining = total_steps - warmup_steps
        if remaining <= 0:
            return 1.0

        progress = float(step - warmup_steps) / float(remaining)
        progress = min(progress, 1.0)

        decay = (1.0 - progress) ** power
        return min_lr_ratio + (1.0 - min_lr_ratio) * decay
    return lr_lambda


def _rex_lambda(
    total_steps: int,
    warmup_steps: int,
    alpha: float = 0.1,
    beta: float = 0.9,
):
    """Rex scheduler: aggressive early, stable late.

    Rex (Revised Exponential) uses an exponential decay controlled by
    alpha (early aggressiveness) and beta (late stability).
    """
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            if warmup_steps <= 0:
                return 1.0
            return float(step) / float(warmup_steps)

        remaining = total_steps - warmup_steps
        if remaining <= 0:
            return 1.0

        progress = float(step - warmup_steps) / float(remaining)
        progress = min(progress, 1.0)

        # Rex formula: lr = 1 / (1 + alpha * (exp(beta * progress) - 1))
        exp_term = math.exp(beta * progress) - 1.0
        return 1.0 / (1.0 + alpha * exp_term)
    return lr_lambda


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def compute_total_steps(
    num_samples: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    max_epochs: int,
) -> int:
    """Compute total optimizer steps for a training phase.

    Total steps = (samples_per_epoch / batch_size / grad_accum) * epochs

    Args:
        num_samples: Total samples in the dataset.
        batch_size: Training batch size.
        gradient_accumulation_steps: Steps between optimizer updates.
        max_epochs: Number of training epochs.

    Returns:
        Total optimizer steps (integer, at least 1).
    """
    steps_per_epoch = max(1, num_samples // (batch_size * gradient_accumulation_steps))
    return steps_per_epoch * max_epochs
