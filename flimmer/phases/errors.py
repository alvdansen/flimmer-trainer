"""Custom exception hierarchy for dimljus_phases."""

from __future__ import annotations


class DimljusPhasesError(Exception):
    """Base exception for all dimljus_phases errors."""


class ModelNotFoundError(DimljusPhasesError):
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


class PhaseConfigError(DimljusPhasesError):
    """Raised when a phase configuration is invalid.

    Attributes:
        detail: Description of what went wrong.
    """

    def __init__(self, detail: str) -> None:
        self.detail = detail
        super().__init__(f"Phase config error: {detail}")
