"""Flimmer dataset validation errors.

Custom exceptions for dataset-level operations (validation, discovery,
manifest generation). Follows the same pattern as flimmer.video.errors:
every error message says what went wrong AND how to fix it.
"""

from __future__ import annotations


class FlimmerDatasetError(Exception):
    """Base exception for all dataset-level errors.

    Separate from FlimmerVideoError because dataset operations work at
    the cross-file level (pairing, completeness) not the single-file level.
    """
    pass


class DatasetValidationError(FlimmerDatasetError):
    """Raised when dataset validation encounters a fatal error.

    This is for structural problems that prevent validation from running
    at all — not for individual sample issues (those are collected as
    ValidationIssue objects, not exceptions).
    """

    def __init__(self, detail: str) -> None:
        self.detail = detail
        super().__init__(f"Dataset validation failed: {detail}")


class OrganizeError(FlimmerDatasetError):
    """Raised when dataset organize encounters a fatal error.

    Examples: source directory doesn't exist, zero valid samples after
    filtering. Individual sample skips are NOT exceptions — they're
    recorded as OrganizedSample with skipped=True.
    """

    def __init__(self, detail: str) -> None:
        self.detail = detail
        super().__init__(f"Organize failed: {detail}")
