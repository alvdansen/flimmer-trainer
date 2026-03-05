---
phase: 05-training-correctness
plan: 01
subsystem: training
tags: [gradient-checkpointing, cuda-allocator, peft, lora, vram, use_reentrant]

# Dependency graph
requires: []
provides:
  - "Gradient checkpointing with use_reentrant=False enforced on all API paths"
  - "CUDA memory allocator configured with expandable_segments:True before torch import"
affects: [06-image-training, 07-block-swap, 08-quantized-optimizers, 09-integration-validation]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Pre-torch env var configuration pattern in CLI entry point"
    - "Module-level logger in wan backend for training diagnostics"

key-files:
  created: []
  modified:
    - "flimmer/training/wan/backend.py"
    - "flimmer/training/__main__.py"
    - "tests/test_wan_backend.py"
    - "tests/test_training_cli.py"

key-decisions:
  - "Diffusers path left as-is (already defaults to use_reentrant=False internally)"
  - "Transformers fallback gets explicit gradient_checkpointing_kwargs"
  - "_configure_cuda_allocator placed before all torch-related imports in cmd_train"

patterns-established:
  - "TDD red-green for correctness fixes: failing tests committed before implementation"
  - "Pre-import env var configuration for CUDA allocator settings"

requirements-completed: [FIX-01, FIX-02]

# Metrics
duration: 3min
completed: 2026-03-05
---

# Phase 5 Plan 1: Training Correctness Fixes Summary

**Enforced use_reentrant=False in gradient checkpointing for PEFT LoRA and added expandable_segments CUDA allocator config to prevent fragmentation OOMs**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-05T03:27:11Z
- **Completed:** 2026-03-05T03:30:23Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- FIX-01: Gradient checkpointing now enforces use_reentrant=False on the transformers fallback path, preventing silent zero gradients with PEFT LoRA adapters
- FIX-02: CUDA memory allocator configured with expandable_segments:True before any torch import, reducing fragmentation-induced OOMs during variable-resolution video training
- Both fixes verified with TDD (red-green) and full test suite (2276 tests pass)

## Task Commits

Each task was committed atomically:

1. **Task 1: Fix gradient checkpointing (FIX-01)**
   - `31abe73` (test: failing tests for use_reentrant=False)
   - `fcaf4fc` (fix: enforce use_reentrant=False in gradient checkpointing)
2. **Task 2: Configure CUDA allocator (FIX-02)**
   - `a7826a3` (test: failing tests for _configure_cuda_allocator)
   - `d7192b6` (fix: add CUDA allocator config with expandable_segments)

_Note: TDD tasks have two commits each (RED test then GREEN implementation)_

## Files Created/Modified
- `flimmer/training/wan/backend.py` - Added logging, use_reentrant=False kwargs on transformers path, updated docstring
- `flimmer/training/__main__.py` - Added _configure_cuda_allocator() helper, called before torch imports in cmd_train
- `tests/test_wan_backend.py` - Updated 1 existing test, added 2 new tests (6 total in TestSetupGradientCheckpointing)
- `tests/test_training_cli.py` - Added 5 new tests in TestConfigureCudaAllocator class (15 total in file)

## Decisions Made
- Diffusers API path (enable_gradient_checkpointing) left without explicit kwargs since diffusers 0.35.x ModelMixin already defaults to use_reentrant=False internally
- Transformers API fallback (gradient_checkpointing_enable) receives explicit gradient_checkpointing_kwargs={"use_reentrant": False}
- _configure_cuda_allocator() placed as the very first line of cmd_train(), before any from-imports that could transitively import torch

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Both prerequisite correctness fixes are in place for all v1.1 features
- Gradient checkpointing is now safe for PEFT LoRA training
- CUDA allocator is configured for variable-resolution video training
- Ready for Phase 5 Plan 2 and subsequent phases (block swap, quantized optimizers, etc.)

## Self-Check: PASSED

- All 4 modified files exist on disk
- All 4 task commits verified in git history (31abe73, fcaf4fc, a7826a3, d7192b6)
- use_reentrant present in backend.py (6 occurrences)
- expandable_segments present in __main__.py (5 occurrences)
- Full test suite: 2276 passed, 0 failed

---
*Phase: 05-training-correctness*
*Completed: 2026-03-05*
