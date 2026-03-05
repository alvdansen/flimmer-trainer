"""Phase-organized checkpoint management.

Handles saving, loading, pruning, and resumption of training checkpoints.
Checkpoints are organized by phase:

    {output_dir}/
        training_state.json              ← phase, epoch, step (for resumption)
        full_noise/
            {name}_full_noise_epoch005.safetensors
        high_noise/
            {name}_high_epoch015.safetensors
        low_noise/
            {name}_low_epoch025.safetensors
        final/
            {name}_merged.safetensors    ← final merged LoRA for inference

Six resumption scenarios:
    1. Crashed mid-full-noise → resume at last full_noise checkpoint
    2. Crashed mid-high-noise → resume at last high-noise checkpoint
    3. Crashed mid-low-noise → resume at last low-noise checkpoint
    4. Completed full_noise → resume at fork point
    5. Expert-from-scratch (no full_noise) → start expert directly
    6. Expert has resume_from → load that LoRA as starting point
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from flimmer.training.errors import CheckpointError, ResumptionError
from flimmer.training.phase import PhaseType


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TRAINING_STATE_FILENAME = "training_state.json"

# Phase type → subdirectory name
PHASE_DIRS: dict[PhaseType, str] = {
    PhaseType.UNIFIED: "full_noise",
    PhaseType.HIGH_NOISE: "high_noise",
    PhaseType.LOW_NOISE: "low_noise",
}

# Phase type → filename abbreviation
PHASE_ABBREV: dict[PhaseType, str] = {
    PhaseType.UNIFIED: "full_noise",
    PhaseType.HIGH_NOISE: "high",
    PhaseType.LOW_NOISE: "low",
}

# Legacy directory names (pre-rename) for backward-compatible checkpoint discovery
_LEGACY_DIRS: dict[PhaseType, str] = {
    PhaseType.UNIFIED: "unified",
}

# Legacy filename abbreviations for backward-compatible checkpoint discovery
_LEGACY_ABBREV: dict[PhaseType, str] = {
    PhaseType.UNIFIED: "unified",
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CheckpointMetadata:
    """Metadata for a single saved checkpoint.

    Stored in the training_state.json alongside the checkpoint paths.
    """
    phase: str
    """Phase type value string ('full_noise', 'high_noise', 'low_noise')."""

    epoch: int
    """Epoch number when this checkpoint was saved."""

    global_step: int
    """Global training step count when this checkpoint was saved."""

    loss: float
    """Loss value at save time (EMA or raw)."""


@dataclass
class TrainingState:
    """Complete training state for resumption.

    Serialized to training_state.json. Contains enough information to
    determine where to resume training — which phase, which epoch,
    and where the relevant checkpoints are.
    """
    phase_index: int = 0
    """Index into the resolved phases list (0-based)."""

    phase_type: str = "full_noise"
    """Current phase type value string."""

    epoch: int = 0
    """Current epoch within the active phase (0-based, 0 = not started)."""

    global_step: int = 0
    """Global step counter across all phases."""

    unified_lora_path: str | None = None
    """Path to the last unified checkpoint (for fork point)."""

    high_noise_lora_path: str | None = None
    """Path to the last high-noise expert checkpoint."""

    low_noise_lora_path: str | None = None
    """Path to the last low-noise expert checkpoint."""

    optimizer_path: str | None = None
    """Path to the last optimizer state checkpoint."""

    wandb_run_id: str | None = None
    """W&B run ID for resume continuity (same run after restart).

    When training is interrupted and resumed, this ID lets W&B append
    to the existing run instead of creating a new one — keeping the
    loss curve continuous and all metrics in one place.
    """

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "phase_index": self.phase_index,
            "phase_type": self.phase_type,
            "epoch": self.epoch,
            "global_step": self.global_step,
            "unified_lora_path": self.unified_lora_path,
            "high_noise_lora_path": self.high_noise_lora_path,
            "low_noise_lora_path": self.low_noise_lora_path,
            "optimizer_path": self.optimizer_path,
            "wandb_run_id": self.wandb_run_id,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> TrainingState:
        """Deserialize from a JSON dict.

        Backward compatible: missing fields default to None/0 so
        old training_state.json files (without wandb_run_id) still load.
        """
        # Backward compat: accept "unified" from old training_state.json files
        raw_phase = data.get("phase_type", "full_noise")
        if raw_phase == "unified":
            raw_phase = "full_noise"

        return TrainingState(
            phase_index=data.get("phase_index", 0),
            phase_type=raw_phase,
            epoch=data.get("epoch", 0),
            global_step=data.get("global_step", 0),
            unified_lora_path=data.get("unified_lora_path"),
            high_noise_lora_path=data.get("high_noise_lora_path"),
            low_noise_lora_path=data.get("low_noise_lora_path"),
            optimizer_path=data.get("optimizer_path"),
            wandb_run_id=data.get("wandb_run_id"),
        )


# ---------------------------------------------------------------------------
# Checkpoint manager
# ---------------------------------------------------------------------------

class CheckpointManager:
    """Manages phase-organized checkpoints for training.

    Handles the directory structure, filename conventions, save/load
    operations, pruning old checkpoints, and training state persistence.

    Args:
        output_dir: Root output directory for all checkpoints.
        name: Base name for checkpoint files (e.g. 'flimmer_lora').
        max_checkpoints: Maximum checkpoints per phase (None = unlimited).
    """

    def __init__(
        self,
        output_dir: str | Path,
        name: str = "flimmer_lora",
        max_checkpoints: int | None = None,
    ) -> None:
        self._output_dir = Path(output_dir)
        self._name = name
        self._max_checkpoints = max_checkpoints

    @property
    def output_dir(self) -> Path:
        """Root output directory."""
        return self._output_dir

    def ensure_dirs(self) -> None:
        """Create the output directory structure.

        Creates:
            {output_dir}/
            {output_dir}/full_noise/
            {output_dir}/high_noise/
            {output_dir}/low_noise/
            {output_dir}/final/
            {output_dir}/samples/
        """
        self._output_dir.mkdir(parents=True, exist_ok=True)
        for subdir in ["full_noise", "high_noise", "low_noise", "final", "samples"]:
            (self._output_dir / subdir).mkdir(exist_ok=True)

    def checkpoint_path(
        self,
        phase_type: PhaseType,
        epoch: int,
    ) -> Path:
        """Generate the checkpoint file path for a given phase and epoch.

        Filename format: {name}_{phase_abbrev}_epoch{N:03d}.safetensors

        Args:
            phase_type: Training phase.
            epoch: Epoch number.

        Returns:
            Full path to the checkpoint file.
        """
        subdir = PHASE_DIRS[phase_type]
        abbrev = PHASE_ABBREV[phase_type]
        filename = f"{self._name}_{abbrev}_epoch{epoch:03d}.safetensors"
        return self._output_dir / subdir / filename

    def final_path(self) -> Path:
        """Path for the final merged LoRA checkpoint.

        Returns:
            Full path to {output_dir}/final/{name}_merged.safetensors.
        """
        return self._output_dir / "final" / f"{self._name}_merged.safetensors"

    def sample_dir(self, phase_type: PhaseType, epoch: int) -> Path:
        """Directory for sample outputs at a specific phase and epoch.

        Args:
            phase_type: Training phase.
            epoch: Epoch number.

        Returns:
            Path to {output_dir}/samples/{phase_abbrev}_epoch{N:03d}/.
        """
        abbrev = PHASE_ABBREV[phase_type]
        return self._output_dir / "samples" / f"{abbrev}_epoch{epoch:03d}"

    def save_training_state(self, state: TrainingState) -> Path:
        """Save training state to JSON for resumption.

        Args:
            state: Current training state.

        Returns:
            Path to the saved state file.
        """
        state_path = self._output_dir / TRAINING_STATE_FILENAME
        self._output_dir.mkdir(parents=True, exist_ok=True)

        tmp_path = state_path.with_suffix(".json.tmp")
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(state.to_dict(), f, indent=2)
            tmp_path.replace(state_path)
        except Exception as e:
            # Clean up temp file on failure
            tmp_path.unlink(missing_ok=True)
            raise CheckpointError(
                f"Failed to save training state to '{state_path}': {e}"
            ) from e

        return state_path

    def load_training_state(self) -> TrainingState | None:
        """Load training state from JSON.

        Returns:
            TrainingState if the file exists, None otherwise.

        Raises:
            ResumptionError: If the file exists but is corrupt.
        """
        state_path = self._output_dir / TRAINING_STATE_FILENAME
        if not state_path.is_file():
            return None

        try:
            with open(state_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return TrainingState.from_dict(data)
        except json.JSONDecodeError as e:
            raise ResumptionError(
                f"Corrupt training state at '{state_path}': {e}\n"
                f"Delete the file to start fresh, or fix the JSON."
            ) from e
        except Exception as e:
            raise ResumptionError(
                f"Failed to load training state from '{state_path}': {e}"
            ) from e

    def find_latest_checkpoint(self, phase_type: PhaseType) -> Path | None:
        """Find the most recent checkpoint for a phase.

        Searches the phase subdirectory for safetensors files matching
        the naming convention and returns the one with the highest epoch.
        Also checks legacy directory/abbreviation names for backward compat.

        Args:
            phase_type: Training phase to search.

        Returns:
            Path to the latest checkpoint, or None if none exist.
        """
        def _epoch_from_path(p: Path) -> int:
            match = re.search(r"epoch(\d+)", p.stem)
            return int(match.group(1)) if match else 0

        all_checkpoints: list[Path] = []

        # Search current directory + abbreviation
        subdir = self._output_dir / PHASE_DIRS[phase_type]
        abbrev = PHASE_ABBREV[phase_type]
        if subdir.is_dir():
            pattern = f"{self._name}_{abbrev}_epoch*.safetensors"
            all_checkpoints.extend(subdir.glob(pattern))

        # Search legacy directory + abbreviation (backward compat)
        legacy_dir = _LEGACY_DIRS.get(phase_type)
        legacy_abbrev = _LEGACY_ABBREV.get(phase_type)
        if legacy_dir:
            legacy_subdir = self._output_dir / legacy_dir
            if legacy_subdir.is_dir() and legacy_abbrev:
                pattern = f"{self._name}_{legacy_abbrev}_epoch*.safetensors"
                all_checkpoints.extend(legacy_subdir.glob(pattern))

        if not all_checkpoints:
            return None

        all_checkpoints.sort(key=_epoch_from_path)
        return all_checkpoints[-1]

    def list_checkpoints(self, phase_type: PhaseType) -> list[Path]:
        """List all checkpoints for a phase, sorted by epoch.

        Includes checkpoints from legacy directory names for backward compat.

        Args:
            phase_type: Training phase.

        Returns:
            List of checkpoint paths, oldest first.
        """
        def _epoch_from_path(p: Path) -> int:
            match = re.search(r"epoch(\d+)", p.stem)
            return int(match.group(1)) if match else 0

        all_checkpoints: list[Path] = []

        # Current directory + abbreviation
        subdir = self._output_dir / PHASE_DIRS[phase_type]
        abbrev = PHASE_ABBREV[phase_type]
        if subdir.is_dir():
            pattern = f"{self._name}_{abbrev}_epoch*.safetensors"
            all_checkpoints.extend(subdir.glob(pattern))

        # Legacy directory + abbreviation (backward compat)
        legacy_dir = _LEGACY_DIRS.get(phase_type)
        legacy_abbrev = _LEGACY_ABBREV.get(phase_type)
        if legacy_dir:
            legacy_subdir = self._output_dir / legacy_dir
            if legacy_subdir.is_dir() and legacy_abbrev:
                pattern = f"{self._name}_{legacy_abbrev}_epoch*.safetensors"
                all_checkpoints.extend(legacy_subdir.glob(pattern))

        all_checkpoints.sort(key=_epoch_from_path)
        return all_checkpoints

    def prune_checkpoints(self, phase_type: PhaseType) -> list[Path]:
        """Remove old checkpoints exceeding the max_checkpoints limit.

        Keeps the most recent checkpoints and deletes the oldest ones.

        Args:
            phase_type: Training phase to prune.

        Returns:
            List of paths that were deleted.
        """
        if self._max_checkpoints is None:
            return []

        checkpoints = self.list_checkpoints(phase_type)
        if len(checkpoints) <= self._max_checkpoints:
            return []

        to_delete = checkpoints[: len(checkpoints) - self._max_checkpoints]
        deleted: list[Path] = []
        for path in to_delete:
            try:
                path.unlink()
                deleted.append(path)
            except OSError:
                pass  # Best effort — don't fail training for cleanup issues

        return deleted

    def find_resume_point(
        self,
        phases: list[Any],
    ) -> tuple[int, int, TrainingState] | None:
        """Determine where to resume training.

        Examines the saved training state and checkpoint files to find
        the right resumption point.

        Args:
            phases: List of resolved TrainingPhase objects.

        Returns:
            Tuple of (phase_index, epoch, state) if resumable, None if fresh start.
        """
        state = self.load_training_state()
        if state is None:
            return None

        # Validate phase_index is within range
        if state.phase_index >= len(phases):
            return None

        # Validate that the saved phase type matches the phase at that index.
        # Project-based runs invoke each phase as a separate training call,
        # so stale state from a previous phase (e.g. full_noise) must not
        # be applied to the current phase (e.g. high_noise).
        current_phase = phases[state.phase_index]
        if hasattr(current_phase, "phase_type") and state.phase_type:
            if current_phase.phase_type.value != state.phase_type:
                return None

        return (state.phase_index, state.epoch, state)
