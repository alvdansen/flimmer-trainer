"""Project lifecycle model: create, save, load, modify phases, status locking.

The Project is the central orchestrator that owns:
- Model selection (model_id linked to the registry)
- Run-level params (locked at creation)
- An ordered list of phases with lifecycle status tracking

Public API:
    - PhaseStatus: Enum with PENDING, RUNNING, COMPLETED
    - PhaseEntry: Pydantic model wrapping PhaseConfig + PhaseStatus
    - Project: Pydantic model with create/save/load/add_phase/modify_phase/
      remove_phase/reorder_phases/mark_phase_running/mark_phase_completed/validate
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from .errors import PhaseConfigError
from .phase_model import PhaseConfig
from .registry import get_model_definition

if TYPE_CHECKING:
    from .validation import ValidationResult

PROJECT_FILENAME = "flimmer_project.json"


class PhaseStatus(str, Enum):
    """Lifecycle status of a phase within a project."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"


class PhaseEntry(BaseModel):
    """A phase in the project: config plus its lifecycle status.

    Attributes:
        config: The user-facing phase configuration.
        status: Current lifecycle status (default: PENDING).
    """

    config: PhaseConfig
    status: PhaseStatus = PhaseStatus.PENDING


class Project(BaseModel):
    """Central project model managing model selection, params, and phases.

    Attributes:
        name: User-friendly project name.
        model_id: Registry model identifier (validated at creation).
        run_level_params: Params locked at project creation.
        phases: Ordered list of phase entries with status tracking.
        created_at: ISO timestamp of project creation.
    """

    name: str
    model_id: str
    run_level_params: dict[str, float | int | str | bool] = Field(
        default_factory=dict
    )
    phases: list[PhaseEntry] = Field(default_factory=list)
    created_at: str = ""

    # ---- Factory methods ----

    @classmethod
    def create(
        cls,
        name: str,
        model_id: str,
        run_level_params: dict[str, float | int | str | bool] | None = None,
    ) -> Project:
        """Create a new project, validating model exists in the registry.

        Builds default run-level params from the model's run_level_params
        (phase_level=False ParamSpecs), then merges user-provided overrides.

        Args:
            name: Project name.
            model_id: Must exist in the registry.
            run_level_params: Optional overrides for run-level param defaults.

        Returns:
            A new Project instance.

        Raises:
            ModelNotFoundError: If model_id is not registered.
        """
        model_def = get_model_definition(model_id)

        # Build defaults from model's run-level params
        defaults: dict[str, float | int | str | bool] = {
            p.name: p.default
            for p in model_def.run_level_params
        }

        # Merge user-provided values on top
        if run_level_params is not None:
            defaults.update(run_level_params)

        return cls(
            name=name,
            model_id=model_id,
            run_level_params=defaults,
            phases=[],
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    @classmethod
    def load(cls, directory: Path) -> Project:
        """Load a project from a directory containing flimmer_project.json.

        Args:
            directory: Path to the directory with the project file.

        Returns:
            The loaded Project instance.
        """
        file_path = Path(directory) / PROJECT_FILENAME
        with open(file_path) as f:
            data = json.load(f)
        return cls.model_validate(data)

    # ---- Persistence ----

    def save(self, directory: Path) -> None:
        """Save the project to directory/flimmer_project.json.

        Creates the directory if it doesn't exist.

        Args:
            directory: Path to write the project file.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        file_path = directory / PROJECT_FILENAME
        with open(file_path, "w") as f:
            json.dump(self.model_dump(), f, indent=2)

    # ---- Phase management ----

    def add_phase(
        self, config: PhaseConfig, index: int | None = None
    ) -> int:
        """Add a phase to the project with PENDING status.

        Args:
            config: The phase configuration.
            index: Insert position. None = append to end.

        Returns:
            The index where the phase was inserted.
        """
        entry = PhaseEntry(config=config, status=PhaseStatus.PENDING)
        if index is None:
            self.phases.append(entry)
            return len(self.phases) - 1
        self.phases.insert(index, entry)
        return index

    def modify_phase(self, index: int, config: PhaseConfig) -> None:
        """Replace the config of a PENDING phase.

        Args:
            index: Phase index to modify.
            config: New phase configuration.

        Raises:
            PhaseConfigError: If the phase is not PENDING.
        """
        self._assert_phase_status(index, PhaseStatus.PENDING, "modify")
        self.phases[index].config = config

    def remove_phase(self, index: int) -> PhaseConfig:
        """Remove a PENDING phase and return its config.

        Args:
            index: Phase index to remove.

        Returns:
            The removed phase's configuration.

        Raises:
            PhaseConfigError: If the phase is not PENDING.
        """
        self._assert_phase_status(index, PhaseStatus.PENDING, "remove")
        entry = self.phases.pop(index)
        return entry.config

    def reorder_phases(self, new_order: list[int]) -> None:
        """Reorder phases by index. All referenced phases must be PENDING.

        Args:
            new_order: List of current indices in desired new order.
                Must include all valid indices.

        Raises:
            PhaseConfigError: If any referenced phase is not PENDING,
                or if new_order doesn't cover all indices.
        """
        n = len(self.phases)
        if sorted(new_order) != list(range(n)):
            raise PhaseConfigError(
                f"new_order must be a permutation of [0..{n - 1}], "
                f"got {new_order}"
            )

        for idx in new_order:
            if self.phases[idx].status != PhaseStatus.PENDING:
                raise PhaseConfigError(
                    f"Cannot reorder phase {idx}: status is "
                    f"'{self.phases[idx].status.value}', must be 'pending'"
                )

        self.phases = [self.phases[i] for i in new_order]

    # ---- Status transitions ----

    def mark_phase_running(self, index: int) -> None:
        """Transition a phase from PENDING to RUNNING.

        Args:
            index: Phase index.

        Raises:
            PhaseConfigError: If phase is not PENDING.
        """
        self._assert_phase_status(index, PhaseStatus.PENDING, "start")
        self.phases[index].status = PhaseStatus.RUNNING

    def mark_phase_completed(self, index: int) -> None:
        """Transition a phase from RUNNING to COMPLETED.

        Args:
            index: Phase index.

        Raises:
            PhaseConfigError: If phase is not RUNNING.
        """
        self._assert_phase_status(index, PhaseStatus.RUNNING, "complete")
        self.phases[index].status = PhaseStatus.COMPLETED

    # ---- Validation ----

    def validate(self) -> ValidationResult:
        """Validate the project using the batch validation system.

        Returns:
            ValidationResult with all collected issues.
        """
        from .validation import validate_project

        return validate_project(self)

    # ---- Internal helpers ----

    def _assert_phase_status(
        self, index: int, expected: PhaseStatus, action: str
    ) -> None:
        """Check that a phase has the expected status.

        Args:
            index: Phase index.
            expected: Required status.
            action: Human-readable action name for error message.

        Raises:
            PhaseConfigError: If status doesn't match.
        """
        actual = self.phases[index].status
        if actual != expected:
            raise PhaseConfigError(
                f"Cannot {action} phase {index}: status is "
                f"'{actual.value}', must be '{expected.value}'"
            )
