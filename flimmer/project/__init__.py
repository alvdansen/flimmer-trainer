"""flimmer.project: Project-based training management.

Bridges project YAML files to the existing training CLI. Users define
multi-phase training projects (unified -> expert fork) in a project
YAML file. This module converts that to a Project instance, generates
per-phase merged configs, and drives execution via the training CLI.

Public API:
    - load_project_yaml: Parse a project YAML file
    - project_from_yaml: Convert project YAML to a Project instance
    - merge_phase_config: Merge base training config with phase overrides
"""

from .loader import load_project_yaml, merge_phase_config, project_from_yaml

__all__ = [
    "load_project_yaml",
    "project_from_yaml",
    "merge_phase_config",
]
