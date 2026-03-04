"""Batch validation: collect ALL errors across all phases without stopping at the first.

Wraps the existing per-phase validate_against() in a collector pattern.
Distinguishes errors (blocks training) from warnings (suspicious but allowed).

Public API:
    - ValidationIssue: frozen dataclass describing a single validation finding
    - ValidationResult: aggregated result with valid property and format()
    - validate_project(project) -> ValidationResult
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .errors import PhaseConfigError
from .registry import get_model_definition

if TYPE_CHECKING:
    from .definitions import ModelDefinition


@dataclass(frozen=True)
class ValidationIssue:
    """A single validation finding for a specific phase.

    Attributes:
        phase_index: Position of the phase in the project's phase list.
        phase_type: The phase_type string from the phase config.
        param: Parameter name involved, or None if not param-specific.
        message: Human-readable description of the issue.
        severity: Either "error" (blocks training) or "warning" (suspicious).
    """

    phase_index: int
    phase_type: str
    param: str | None
    message: str
    severity: str  # "error" or "warning"


@dataclass
class ValidationResult:
    """Aggregated validation result for a project.

    Attributes:
        issues: All collected validation issues (errors and warnings).
    """

    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        """True only when no issues have severity='error'."""
        return not any(issue.severity == "error" for issue in self.issues)

    def format(self) -> str:
        """Return human-readable validation output.

        If no issues: "Validation passed: no issues found."
        Otherwise, each issue on its own line:
            {ERROR|WARNING}: Phase {index} ({phase_type})[{param}]: {message}
        Omit the [{param}] bracket when param is None.
        """
        if not self.issues:
            return "Validation passed: no issues found."

        lines: list[str] = []
        for issue in self.issues:
            prefix = issue.severity.upper()
            param_part = f"[{issue.param}]" if issue.param is not None else ""
            lines.append(
                f"  {prefix}: Phase {issue.phase_index} "
                f"({issue.phase_type}){param_part}: {issue.message}"
            )
        return "\n".join(lines)


def _extract_param_name(detail: str) -> str | None:
    """Try to extract a parameter name from a PhaseConfigError detail string.

    Looks for patterns like: "'param_name'" or "for 'param_name'"
    in messages such as "Override value for 'learning_rate' (999.0) is above maximum".
    """
    # Pattern: "for 'param_name'" or "Parameter 'param_name'"
    match = re.search(r"(?:for|Parameter)\s+'([^']+)'", detail)
    if match:
        return match.group(1)
    return None


def _check_cross_phase_warnings(
    project: object,
    model_def: ModelDefinition,
) -> list[ValidationIssue]:
    """Run cross-phase and suspicious-value checks that produce warnings.

    Current checks:
    - Warning if learning_rate override equals ParamSpec.min_value (suspiciously low).
    """
    warnings: list[ValidationIssue] = []

    lr_param = model_def.get_param("learning_rate")

    for idx, entry in enumerate(project.phases):  # type: ignore[attr-defined]
        config = entry.config

        # Check: learning_rate at minimum bound
        if lr_param is not None and lr_param.min_value is not None:
            lr_override = config.overrides.get("learning_rate")
            if lr_override is not None and lr_override == lr_param.min_value:
                warnings.append(ValidationIssue(
                    phase_index=idx,
                    phase_type=config.phase_type,
                    param="learning_rate",
                    message=(
                        f"learning_rate ({lr_override}) equals the minimum "
                        f"allowed value ({lr_param.min_value}); "
                        f"this may be suspiciously low"
                    ),
                    severity="warning",
                ))

    return warnings


def validate_project(project: object) -> ValidationResult:
    """Validate an entire project, collecting ALL issues across all phases.

    Gets the ModelDefinition from the registry via the project's model_id.
    For each phase, calls validate_against() and catches PhaseConfigError to
    create error-severity ValidationIssues. Then runs cross-phase warning checks.

    Args:
        project: A Project (or any object with model_id: str and
            phases: list of objects with .config: PhaseConfig).

    Returns:
        ValidationResult with all collected issues.
    """
    model_def = get_model_definition(project.model_id)  # type: ignore[attr-defined]
    all_issues: list[ValidationIssue] = []

    # Per-phase validation via existing validate_against()
    for idx, entry in enumerate(project.phases):  # type: ignore[attr-defined]
        config = entry.config
        try:
            config.validate_against(model_def)
        except PhaseConfigError as exc:
            param_name = _extract_param_name(exc.detail)
            all_issues.append(ValidationIssue(
                phase_index=idx,
                phase_type=config.phase_type,
                param=param_name,
                message=exc.detail,
                severity="error",
            ))

    # Cross-phase / suspicious-value warnings
    all_issues.extend(_check_cross_phase_warnings(project, model_def))

    return ValidationResult(issues=all_issues)
