"""Training phase resolution — the heart of Flimmer's state machine.

Turns a FlimmerTrainingConfig into a sequence of concrete TrainingPhase
objects. Each phase has fully resolved parameters — no None values, no
"inherit from parent" logic. The inner training loop reads a TrainingPhase
and knows exactly what to do.

Three training modes from two config switches:

    | Mode                    | fork_enabled | unified_epochs | Phases                    |
    |-------------------------|-------------|----------------|---------------------------|
    | Fork-and-specialize     | true        | > 0            | full_noise → expert1 → expert2 |
    | Full noise only         | false       | any            | full_noise                |
    | Expert from scratch     | true        | 0              | expert1 → expert2         |

Expert order is configurable via moe.expert_order (default: low_noise first).
Either expert can be skipped with enabled: false.

Single-expert models (Wan 2.1, non-MoE) always produce one full_noise phase
with no expert masking — the degenerate case of the state machine.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from flimmer.training.errors import PhaseConfigError


# ---------------------------------------------------------------------------
# Phase type enum
# ---------------------------------------------------------------------------

class PhaseType(str, Enum):
    """What kind of training phase this is.

    UNIFIED: full noise range, both experts share one LoRA. No expert masking.
        User-facing name: "full_noise". Accepts "unified" for backward compat.
    HIGH_NOISE: training the high-noise expert only. Loss masked to
        high-noise timesteps.
    LOW_NOISE: training the low-noise expert only. Loss masked to
        low-noise timesteps.
    """
    UNIFIED = "full_noise"
    HIGH_NOISE = "high_noise"
    LOW_NOISE = "low_noise"

    @classmethod
    def _missing_(cls, value: object) -> PhaseType | None:
        """Accept legacy phase type names (e.g. 'unified' → UNIFIED)."""
        _ALIASES = {"unified": cls.UNIFIED}
        if isinstance(value, str):
            return _ALIASES.get(value)
        return None


# Default expert training order per variant.
# Low-noise first because it converges faster (from task vector analysis):
# - T2V: low-noise moved 0.04% (≈Wan 2.1), high-noise moved 46%
# - I2V: low-noise moved 36%, high-noise moved 50%
DEFAULT_EXPERT_ORDER: list[str] = ["low_noise", "high_noise"]

# Map expert name strings to PhaseType enum values
_EXPERT_NAME_TO_PHASE: dict[str, PhaseType] = {
    "high_noise": PhaseType.HIGH_NOISE,
    "low_noise": PhaseType.LOW_NOISE,
}


# ---------------------------------------------------------------------------
# TrainingPhase — fully resolved parameters for one training segment
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrainingPhase:
    """Fully resolved parameters for one training segment.

    Every field has a concrete value — no None values for overridable
    fields, no "inherit from parent" logic. All override resolution
    happens in resolve_phases() BEFORE training starts.

    This is INFRASTRUCTURE — users never see or edit this object.
    They write YAML config; resolve_phases() produces these.

    Why frozen dataclass (not Pydantic): internal resolved object,
    not user config. Lighter weight, signals "infrastructure."
    """
    phase_type: PhaseType
    """Which phase: UNIFIED, HIGH_NOISE, or LOW_NOISE."""

    max_epochs: int
    """Concrete epoch count for this phase."""

    learning_rate: float
    """Resolved from expert override or optimizer.learning_rate."""

    weight_decay: float
    """Resolved from expert override or optimizer.weight_decay."""

    optimizer_type: str
    """Resolved from expert override or optimizer.type."""

    scheduler_type: str
    """Resolved from expert override or scheduler.type."""

    min_lr_ratio: float
    """Resolved from expert override or scheduler.min_lr_ratio."""

    warmup_steps: int
    """From scheduler.warmup_steps."""

    batch_size: int
    """Resolved from expert override or training.batch_size."""

    gradient_accumulation_steps: int
    """Resolved from expert override or training.gradient_accumulation_steps."""

    caption_dropout_rate: float
    """Resolved from expert override or training.caption_dropout_rate."""

    first_frame_dropout_rate: float
    """Probability of dropping first-frame conditioning. Independent from caption dropout. 0.0 = never drop."""

    lora_dropout: float
    """Resolved from expert override (dropout) or lora.dropout."""

    fork_targets: list[str] | None
    """None = all LoRA targets. List = only these targets are trainable."""

    block_targets: str | None
    """None = all blocks. String like '0-11' = only these blocks."""

    resume_from: str | None
    """Path to existing LoRA to start from. None = start fresh or from fork."""

    boundary_ratio: float | None
    """For expert masking. None = no masking (unified or single-expert)."""

    active_expert: str | None
    """'high_noise' or 'low_noise' for expert phases. None for unified."""


# ---------------------------------------------------------------------------
# Phase resolution
# ---------------------------------------------------------------------------

def resolve_phases(config: object) -> list[TrainingPhase]:
    """Turn a FlimmerTrainingConfig into concrete training phases.

    This is where the three training modes manifest. All expert overrides
    are resolved against base config here — the returned phases have no
    None values for any overridable field.

    Args:
        config: A FlimmerTrainingConfig instance.

    Returns:
        List of TrainingPhase objects in execution order.

    Raises:
        PhaseConfigError: If the config produces an invalid phase sequence
            (e.g., no phases at all, or expert-only with no expert epochs).
    """
    # Access config sub-objects
    training = config.training  # type: ignore[attr-defined]
    optimizer = config.optimizer  # type: ignore[attr-defined]
    scheduler = config.scheduler  # type: ignore[attr-defined]
    lora = config.lora  # type: ignore[attr-defined]
    moe = config.moe  # type: ignore[attr-defined]
    model = config.model  # type: ignore[attr-defined]

    phases: list[TrainingPhase] = []

    # Determine effective boundary_ratio: moe override > model default
    boundary_ratio = moe.boundary_ratio if moe.boundary_ratio is not None else model.boundary_ratio

    # Determine the expert order
    expert_order = getattr(moe, "expert_order", None) or DEFAULT_EXPERT_ORDER

    # Validate expert order entries
    for name in expert_order:
        if name not in _EXPERT_NAME_TO_PHASE:
            valid = ", ".join(sorted(_EXPERT_NAME_TO_PHASE.keys()))
            raise PhaseConfigError(
                f"Invalid expert name '{name}' in expert_order. "
                f"Valid names: {valid}."
            )

    # ── Case 1: Non-MoE model (e.g., Wan 2.1) ──
    # Single unified phase, no expert masking, no fork.
    if not moe.enabled:
        epochs = training.unified_epochs
        if epochs <= 0:
            raise PhaseConfigError(
                "Non-MoE model requires unified_epochs > 0. "
                "Set training.unified_epochs to a positive value."
            )
        phases.append(_build_unified_phase(
            epochs=epochs,
            training=training,
            optimizer=optimizer,
            scheduler=scheduler,
            lora=lora,
        ))
        return phases

    # ── Case 2: MoE model, fork disabled (unified-only) ──
    if not moe.fork_enabled:
        epochs = training.unified_epochs
        if epochs <= 0:
            raise PhaseConfigError(
                "Unified-only mode (fork_enabled=false) requires "
                "unified_epochs > 0. Set training.unified_epochs to a "
                "positive value."
            )
        phases.append(_build_unified_phase(
            epochs=epochs,
            training=training,
            optimizer=optimizer,
            scheduler=scheduler,
            lora=lora,
        ))
        return phases

    # ── Case 3: MoE model, fork enabled ──
    # Determine which experts are enabled
    experts_config = {
        "high_noise": moe.high_noise,
        "low_noise": moe.low_noise,
    }

    enabled_experts = [
        name for name in expert_order
        if experts_config[name].enabled
    ]

    # Add unified phase if unified_epochs > 0
    if training.unified_epochs > 0:
        phases.append(_build_unified_phase(
            epochs=training.unified_epochs,
            training=training,
            optimizer=optimizer,
            scheduler=scheduler,
            lora=lora,
        ))

    # Add expert phases in configured order.
    # Skip experts with no max_epochs — they're configured via project.yaml.
    for expert_name in enabled_experts:
        expert_overrides = experts_config[expert_name]
        if expert_overrides.max_epochs is None:  # type: ignore[attr-defined]
            continue
        phase = _build_expert_phase(
            expert_name=expert_name,
            overrides=expert_overrides,
            training=training,
            optimizer=optimizer,
            scheduler=scheduler,
            lora=lora,
            boundary_ratio=boundary_ratio,
        )
        phases.append(phase)

    # Validate we have at least one phase
    if not phases:
        raise PhaseConfigError(
            "No training phases resolved. This happens when "
            "unified_epochs=0 and all experts are disabled. "
            "Enable at least one expert or set unified_epochs > 0."
        )

    return phases


# ---------------------------------------------------------------------------
# Phase builders
# ---------------------------------------------------------------------------

def _build_unified_phase(
    *,
    epochs: int,
    training: object,
    optimizer: object,
    scheduler: object,
    lora: object,
) -> TrainingPhase:
    """Build the unified training phase.

    Uses base config values directly — no expert overrides apply here.
    """
    return TrainingPhase(
        phase_type=PhaseType.UNIFIED,
        max_epochs=epochs,
        learning_rate=optimizer.learning_rate,  # type: ignore[attr-defined]
        weight_decay=optimizer.weight_decay,  # type: ignore[attr-defined]
        optimizer_type=optimizer.type,  # type: ignore[attr-defined]
        scheduler_type=scheduler.type,  # type: ignore[attr-defined]
        min_lr_ratio=scheduler.min_lr_ratio,  # type: ignore[attr-defined]
        warmup_steps=scheduler.warmup_steps,  # type: ignore[attr-defined]
        batch_size=training.batch_size,  # type: ignore[attr-defined]
        gradient_accumulation_steps=training.gradient_accumulation_steps,  # type: ignore[attr-defined]
        caption_dropout_rate=training.caption_dropout_rate,  # type: ignore[attr-defined]
        first_frame_dropout_rate=getattr(training, 'first_frame_dropout_rate', 0.0),
        lora_dropout=lora.dropout,  # type: ignore[attr-defined]
        fork_targets=training.unified_targets,  # type: ignore[attr-defined]
        block_targets=training.unified_block_targets,  # type: ignore[attr-defined]
        resume_from=training.resume_from,  # type: ignore[attr-defined]
        boundary_ratio=None,  # No expert masking during unified
        active_expert=None,
    )


def _build_expert_phase(
    *,
    expert_name: str,
    overrides: object,
    training: object,
    optimizer: object,
    scheduler: object,
    lora: object,
    boundary_ratio: float | None,
) -> TrainingPhase:
    """Build one expert training phase with override resolution.

    For each overridable field: use expert override if set (not None),
    otherwise fall back to the base config value.
    """
    phase_type = _EXPERT_NAME_TO_PHASE[expert_name]

    # Resolve max_epochs — required for expert phases
    max_epochs = overrides.max_epochs  # type: ignore[attr-defined]
    if max_epochs is None:
        raise PhaseConfigError(
            f"Expert '{expert_name}' has no max_epochs set. "
            f"Each expert needs its own training duration. "
            f"Set moe.{expert_name}.max_epochs in your config."
        )

    # Helper: use override if not None, else base value
    def _resolve(override_val: object, base_val: object) -> object:
        return override_val if override_val is not None else base_val

    return TrainingPhase(
        phase_type=phase_type,
        max_epochs=max_epochs,
        learning_rate=_resolve(overrides.learning_rate, optimizer.learning_rate),  # type: ignore[attr-defined]
        weight_decay=_resolve(overrides.weight_decay, optimizer.weight_decay),  # type: ignore[attr-defined]
        optimizer_type=_resolve(overrides.optimizer_type, optimizer.type),  # type: ignore[attr-defined]
        scheduler_type=_resolve(overrides.scheduler_type, scheduler.type),  # type: ignore[attr-defined]
        min_lr_ratio=_resolve(overrides.min_lr_ratio, scheduler.min_lr_ratio),  # type: ignore[attr-defined]
        warmup_steps=scheduler.warmup_steps,  # type: ignore[attr-defined]  # Not overridable per-expert
        batch_size=_resolve(overrides.batch_size, training.batch_size),  # type: ignore[attr-defined]
        gradient_accumulation_steps=_resolve(
            overrides.gradient_accumulation_steps,  # type: ignore[attr-defined]
            training.gradient_accumulation_steps,  # type: ignore[attr-defined]
        ),
        caption_dropout_rate=_resolve(
            overrides.caption_dropout_rate,  # type: ignore[attr-defined]
            training.caption_dropout_rate,  # type: ignore[attr-defined]
        ),
        first_frame_dropout_rate=_resolve(
            getattr(overrides, 'first_frame_dropout_rate', None),
            getattr(training, 'first_frame_dropout_rate', 0.0),
        ),
        lora_dropout=_resolve(overrides.dropout, lora.dropout),  # type: ignore[attr-defined]
        fork_targets=_resolve(overrides.fork_targets, training.unified_targets),  # type: ignore[attr-defined]
        block_targets=_resolve(overrides.block_targets, training.unified_block_targets),  # type: ignore[attr-defined]
        resume_from=overrides.resume_from,  # type: ignore[attr-defined]
        boundary_ratio=boundary_ratio,
        active_expert=expert_name,
    )
