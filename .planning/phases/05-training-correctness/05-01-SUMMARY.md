---
phase: 05-training-correctness
plan: 01
subsystem: training
tags: [gradient-checkpointing, cuda-allocator, peft, lora, use_reentrant, expandable_segments]

# Dependency graph
requires:
  - phase: v1.0
    provides: "WanModelBackend with setup_gradient_checkpointing, training CLI entry point"
provides:
  - "Gradient checkpointing with use_reentrant=False on all API paths"
  - "_configure_cuda_allocator helper for expandable_segments before torch import"
affects: [06-image-training-support, 07-block-swap, 08-optimizer-quantization]

# Tech tracking
tech-stack:
  added: []
  patterns: ["env-var-before-import pattern for CUDA allocator config"]

key-files:
  created: []
  modified:
    - flimmer/training/wan/backend.py
    - flimmer/training/__main__.py
    - tests/test_wan_backend.py
    - tests/test_training_cli.py

key-decisions:
  - "Diffusers enable_gradient_checkpointing() left as-is (already defaults to use_reentrant=False since 0.35.x)"
  - "Transformers gradient_checkpointing_enable() gets explicit kwargs (does not default to False)"
  - "_configure_cuda_allocator appends to existing PYTORCH_CUDA_ALLOC_CONF rather than overwriting"

patterns-established:
  - "Pre-torch env config: set CUDA allocator env vars before any torch import in CLI entry points"

requirements-completed: [FIX-01, FIX-02]

# Metrics
duration: 3min
completed: 2026-03-05
---

# Phase 5 Plan 01: Training Correctness Fixes Summary

**Gradient checkpointing enforces use_reentrant=False for PEFT LoRA, CUDA allocator configured with expandable_segments before torch import**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-05T04:36:22Z
- **Completed:** 2026-03-05T04:39:39Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- FIX-01: Gradient checkpointing now correctly passes `use_reentrant=False` on the transformers fallback path, preventing silent zero gradients with PEFT LoRA adapters
- FIX-02: CUDA memory allocator is configured with `expandable_segments:True` as the very first action in `cmd_train()`, before any torch imports can initialize CUDA
- Both fixes include info-level logging for observability
- Full test suite passes (2276 tests, 0 failures)

## Task Commits

Each task was committed atomically (TDD: test then feat):

1. **Task 1: Fix gradient checkpointing use_reentrant=False (FIX-01)**
   - `7340032` (test) - Add failing tests for use_reentrant=False
   - `bdb6818` (fix) - Enforce use_reentrant=False in gradient checkpointing

2. **Task 2: Configure CUDA allocator with expandable_segments (FIX-02)**
   - `72b5d70` (test) - Add failing tests for _configure_cuda_allocator
   - `c2cf141` (fix) - Add CUDA allocator config with expandable_segments

## Files Created/Modified
- `flimmer/training/wan/backend.py` - Added logging, updated setup_gradient_checkpointing with use_reentrant=False kwargs on transformers path, expanded docstring
- `flimmer/training/__main__.py` - Added _configure_cuda_allocator() helper, called as first line of cmd_train()
- `tests/test_wan_backend.py` - Updated 1 existing test, added 2 new tests (6 total in TestSetupGradientCheckpointing)
- `tests/test_training_cli.py` - Added TestConfigureCudaAllocator class with 5 tests (15 total in file)

## Decisions Made
- Diffusers `enable_gradient_checkpointing()` left without explicit kwargs since diffusers 0.35.x already defaults to `use_reentrant=False` internally
- Transformers `gradient_checkpointing_enable()` gets explicit `gradient_checkpointing_kwargs={"use_reentrant": False}` since it does not default to False
- `_configure_cuda_allocator` appends to existing `PYTORCH_CUDA_ALLOC_CONF` value rather than overwriting, respecting user/script settings

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Both correctness prerequisites (FIX-01, FIX-02) are now in place for all v1.1 features
- Gradient checkpointing is safe for PEFT LoRA across all three call sites in training loop
- CUDA allocator is configured even when users bypass shell wrapper scripts
- Ready for Phase 6 (image training), Phase 7 (block swap), Phase 8 (optimizer quantization)

## Self-Check: PASSED

All 4 modified files exist on disk. All 4 commit hashes verified in git log. Key content confirmed: `use_reentrant` in backend.py, `expandable_segments` in __main__.py.

---
*Phase: 05-training-correctness*
*Completed: 2026-03-05*
