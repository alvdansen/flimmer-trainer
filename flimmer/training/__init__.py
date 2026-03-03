"""Flimmer training infrastructure.

The training loop as a state machine: unified phase → fork → expert phases.
Model-agnostic core with Wan-specific backend in the wan/ subpackage.

Key modules:
    protocols   — ModelBackend, NoiseSchedule, InferencePipeline protocols
    errors      — FlimmerTrainingError hierarchy
    phase       — TrainingPhase resolution from config
    noise       — Flow matching math (pure functions + protocol impl)
    lora        — LoRA state management (fork, merge, save/load)
    optimizer   — Per-phase optimizer + scheduler factory
    checkpoint  — Phase-organized checkpoint save/load/prune
    sampler     — Preview generation framework
    logger      — Console, TensorBoard, W&B logging
    metrics     — Per-phase loss EMA, gradient stats
    loop        — TrainingOrchestrator state machine
    wan/        — Wan model backend (ModelBackend + InferencePipeline impls)
"""

from flimmer.training.errors import (
    CheckpointError,
    FlimmerTrainingError,
    LoRAError,
    ModelBackendError,
    PhaseConfigError,
    ResumptionError,
    SamplingError,
)
from flimmer.training.phase import PhaseType, TrainingPhase, resolve_phases
from flimmer.training.protocols import InferencePipeline, ModelBackend, NoiseSchedule

__all__ = [
    # Errors
    "FlimmerTrainingError",
    "PhaseConfigError",
    "CheckpointError",
    "ModelBackendError",
    "LoRAError",
    "SamplingError",
    "ResumptionError",
    # Phase resolution
    "PhaseType",
    "TrainingPhase",
    "resolve_phases",
    # Protocols
    "NoiseSchedule",
    "ModelBackend",
    "InferencePipeline",
]
