"""Custom exception hierarchy for flimmer.phases."""

from __future__ import annotations


class FlimmerPhasesError(Exception):
    """Base exception for all flimmer.phases errors."""


class ModelNotFoundError(FlimmerPhasesError):
    """Raised when a model ID is not found in the registry.

    Attributes:
        model_id: The ID that was looked up.
        valid_ids: Sorted list of valid model IDs.
    """

    def __init__(self, model_id: str, valid_ids: list[str]) -> None:
        self.model_id = model_id
        self.valid_ids = valid_ids
        valid_str = ", ".join(valid_ids) if valid_ids else "(none registered)"
        super().__init__(
            f"Unknown model '{model_id}'. Valid model IDs: {valid_str}."
        )


class PhaseConfigError(FlimmerPhasesError):
    """Raised when a phase configuration is invalid.

    Attributes:
        detail: Description of what went wrong.
    """

    def __init__(self, detail: str) -> None:
        self.detail = detail
        super().__init__(f"Phase config error: {detail}")
