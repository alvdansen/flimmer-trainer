"""Flimmer training pipeline errors.

Custom exceptions for the training infrastructure. Every error message
says what went wrong AND how to fix it.

Hierarchy:
    FlimmerTrainingError          ← base for all training errors
    ├── PhaseConfigError          ← invalid phase config (bad override chain)
    ├── CheckpointError           ← save/load/resume failures
    ├── ModelBackendError         ← model loading/forward failures
    ├── LoRAError                 ← LoRA creation/fork/merge failures
    ├── SamplingError             ← preview generation failures
    └── ResumptionError           ← checkpoint resumption failures
"""

from __future__ import annotations


class FlimmerTrainingError(Exception):
    """Base exception for all training pipeline errors.

    Covers phase resolution, training loop, checkpoint management,
    LoRA operations, and sampling. Separate from FlimmerEncodingError
    and FlimmerVideoError because the training pipeline operates on
    pre-encoded cached tensors, not raw files.
    """
    pass


class PhaseConfigError(FlimmerTrainingError):
    """Raised when training phase configuration is invalid.

    Phase resolution turns the user's config into concrete training
    phases. This fails when expert overrides conflict with base config,
    when required fields are missing, or when the mode combination
    (fork_enabled + unified_epochs) doesn't make sense.
    """

    def __init__(self, detail: str) -> None:
        self.detail = detail
        super().__init__(f"Phase config error: {detail}")


class CheckpointError(FlimmerTrainingError):
    """Raised when checkpoint save/load fails.

    Common causes: disk full, permission errors, corrupt safetensors,
    incompatible checkpoint format, missing checkpoint directory.
    """

    def __init__(self, detail: str) -> None:
        self.detail = detail
        super().__init__(f"Checkpoint error: {detail}")


class ModelBackendError(FlimmerTrainingError):
    """Raised when the model backend fails.

    Common causes: model not found, wrong model format, out of VRAM,
    incompatible model architecture, forward pass failures.
    """

    def __init__(self, detail: str) -> None:
        self.detail = detail
        super().__init__(f"Model backend error: {detail}")


class LoRAError(FlimmerTrainingError):
    """Raised when LoRA operations fail.

    Common causes: incompatible state dicts during merge, rank mismatch
    during fork, missing parameters, corrupt safetensors files.
    """

    def __init__(self, detail: str) -> None:
        self.detail = detail
        super().__init__(f"LoRA error: {detail}")


class SamplingError(FlimmerTrainingError):
    """Raised when preview generation fails.

    Common causes: inference pipeline not available, out of VRAM during
    generation, invalid prompts, file write errors for output videos.
    """

    def __init__(self, detail: str) -> None:
        self.detail = detail
        super().__init__(f"Sampling error: {detail}")


class ResumptionError(FlimmerTrainingError):
    """Raised when training cannot be resumed from a checkpoint.

    Common causes: training_state.json missing or corrupt, checkpoint
    files referenced in state don't exist, config changed incompatibly
    since the checkpoint was saved.
    """

    def __init__(self, detail: str) -> None:
        self.detail = detail
        super().__init__(f"Resumption error: {detail}")
