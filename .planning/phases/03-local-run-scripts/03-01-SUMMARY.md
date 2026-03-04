---
phase: 03-local-run-scripts
plan: 01
subsystem: cli
tags: [yaml, project-management, phases, subprocess, argparse, pydantic]

# Dependency graph
requires:
  - phase: 01-phase-integration
    provides: "Project class with phase lifecycle (PENDING/RUNNING/COMPLETED), PhaseConfig, registry"
provides:
  - "flimmer.project module: project_from_yaml, merge_phase_config, load_project_yaml"
  - "CLI: python -m flimmer.project (run, status, plan subcommands)"
  - "Bridge from project YAML to training CLI via per-phase config merging"
affects: [03-local-run-scripts, training-launcher]

# Tech tracking
tech-stack:
  added: []
  patterns: ["Project YAML -> merge base config with phase overrides -> invoke training CLI via subprocess"]

key-files:
  created:
    - flimmer/project/__init__.py
    - flimmer/project/loader.py
    - flimmer/project/__main__.py
    - tests/test_project_loader.py
  modified: []

key-decisions:
  - "merge_phase_config reads full base YAML and patches values (not export_yaml) to preserve model paths and all config sections"
  - "MoE expert overrides placed under moe.{expert_type} section with fork_enabled=True"
  - "unified phase maps max_epochs to training.unified_epochs"
  - "CLI delegates to training CLI via subprocess.run with sys.executable for venv compatibility"

patterns-established:
  - "Project YAML schema: name, model_id, base_config, run_level_params, phases with type/name/overrides/extras"
  - "Config merging: deep copy base, apply run-level params, then phase-specific overrides by section"
  - "autouse fixture for re-registering models when registry cleared by phase test isolation"

requirements-completed: [RUN-02, RUN-04]

# Metrics
duration: 5min
completed: 2026-03-04
---

# Phase 3 Plan 1: Project Module Summary

**Project YAML loader and CLI bridging multi-phase training projects to the existing training CLI via per-phase config merging**

## Performance

- **Duration:** 5 min
- **Started:** 2026-03-04T03:23:43Z
- **Completed:** 2026-03-04T03:28:24Z
- **Tasks:** 1 (TDD: RED -> GREEN)
- **Files modified:** 4

## Accomplishments
- Created flimmer.project module with loader, CLI, and public API
- merge_phase_config() correctly bridges base training configs with phase overrides for unified and MoE expert phases
- project_from_yaml() creates/loads projects with resume behavior via flimmer_project.json
- CLI accepts project YAML files with run, status, and plan subcommands
- 30 unit tests covering loader, merger, CLI, and parser

## Task Commits

Each task was committed atomically:

1. **Task 1 (RED): Failing tests** - `ee5ab23` (test)
2. **Task 1 (GREEN): Implementation** - `4e1f2a0` (feat)

_TDD task: RED committed first with failing tests, GREEN committed with passing implementation._

## Files Created/Modified
- `flimmer/project/__init__.py` - Public API: load_project_yaml, project_from_yaml, merge_phase_config
- `flimmer/project/loader.py` - Core logic: YAML loading, Project creation, base config merging with phase overrides
- `flimmer/project/__main__.py` - CLI entry point with run/status/plan subcommands and argparse parser
- `tests/test_project_loader.py` - 30 unit tests for loader, merger, CLI status/plan, parser, and public API

## Decisions Made
- Used deep copy + patch approach for merge_phase_config instead of export_yaml, because export_yaml generates config from resolved phases but does NOT include model paths, data_config, save config, or other base config fields (Pitfall 2 from research)
- MoE expert phases place overrides under moe.{phase_type} section (e.g., moe.high_noise.learning_rate) matching the base training config format
- Unified phases map max_epochs to training.unified_epochs to match the existing config schema
- CLI uses subprocess.run with sys.executable to ensure same Python interpreter and venv are used

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Added autouse fixture for model registry re-population**
- **Found during:** Task 1 (GREEN phase, full test suite run)
- **Issue:** tests/phases/conftest.py clean_registry fixture clears the global MODEL_REGISTRY after phase tests. Since Python caches module imports, the auto-registration from flimmer.phases.__init__ does not re-run. Our tests failed with ModelNotFoundError when running after phase tests.
- **Fix:** Added autouse fixture _ensure_models_registered() that re-registers all model definitions if the registry is empty
- **Files modified:** tests/test_project_loader.py
- **Verification:** Full test suite passes (2255 tests)
- **Committed in:** 4e1f2a0 (GREEN commit)

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Single auto-fix necessary for test isolation correctness. No scope creep.

## Issues Encountered
None -- plan executed smoothly after the registry fixture fix.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- flimmer.project module ready for shell script wrappers (train.sh, prepare.sh)
- CLI can be invoked as `python -m flimmer.project run --project path/to/project.yaml`
- merge_phase_config generates per-phase configs that the training CLI can consume directly

## Self-Check: PASSED

- All 4 created files exist
- Both commit hashes (ee5ab23, 4e1f2a0) verified in git log
- 30/30 unit tests passing
- 2255/2255 full suite tests passing

---
*Phase: 03-local-run-scripts*
*Completed: 2026-03-04*
