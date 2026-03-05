---
gsd_state_version: 1.0
milestone: v1.1
milestone_name: Low VRAM Training
status: active
stopped_at: Completed 08-01-PLAN.md
last_updated: "2026-03-05T21:34:43.100Z"
last_activity: 2026-03-05 — Executed 08-01 (CPU offload and Adam-Mini optimizer dispatch)
progress:
  total_phases: 5
  completed_phases: 4
  total_plans: 6
  completed_plans: 6
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-05)

**Core value:** Complete video LoRA training pipeline — raw footage to trained LoRA checkpoint, now targeting 24GB consumer GPUs
**Current focus:** Phase 8 complete — Optimizer improvements. Ready for Phase 9 (Integration Testing).

## Current Position

Phase: 8 of 9 (Optimizer Improvements) — v1.1
Plan: 1 of 1 in current phase (08-01 complete, phase done)
Status: CPU offload and Adam-Mini optimizer dispatch added. Phase 08 complete.
Last activity: 2026-03-05 — Executed 08-01 (CPU offload and Adam-Mini optimizer dispatch)

Progress: [██████████] 100% (v1.1 milestone)

## Performance Metrics

**v1.0 Summary:**
- 4 phases, 12 plans, 90 commits
- Timeline: 2 days (2026-03-03 to 2026-03-04)
- 139 files modified, 57,752 LOC Python

**Plan Execution Times:**

| Phase | Plan | Duration | Tasks | Files |
|-------|------|----------|-------|-------|
| 01 | P01 | 7min | 2 | 15 |
| 01 | P02 | 9min | 2 | 15 |
| 01 | P03 | 5min | 2 | 1 |
| 02 | P01 | 2min | 2 | 3 |
| 02 | P02 | 4min | 2 | 9 |
| 02 | P03 | 5min | 2 | 5 |
| 02 | P04 | 3min | 2 | 6 |
| 03 | P01 | 5min | 1 | 4 |
| 03 | P02 | 3min | 2 | 4 |
| 03 | P03 | 2min | 2 | 2 |
| 04 | P01 | 2min | 2 | 2 |
| 04 | P02 | 3min | 2 | 4 |
| 05 | P01 | 3min | 2 | 4 |
| 06 | P01 | 7min | 2 | 5 |
| 06 | P02 | 5min | 2 | 11 |
| 07 | P01 | 4min | 2 | 4 |
| 07 | P02 | 7min | 2 | 4 |
| 08 | P01 | 4min | 2 | 5 |

## Accumulated Context

### Decisions

- [v1.0]: fp8 default for I2V templates (required to fit 720p on H100)
- [v1.1]: Block swap via native PyTorch hooks, no new required deps
- [v1.1]: torchao and adam-mini optional behind import guards
- [v1.1]: Phase 6 (image training) independent of Phase 7 (block swap) -- can parallelize
- [05-01]: FIX-01 use_reentrant=False for PEFT LoRA gradient correctness; FIX-02 expandable_segments before torch import
- [06-01]: PIL-based image encoding via _encode_image_as_latent, auto_self_reference for I2V on stills, I2V detection via reference.source != "none"
- [06-02]: Repeat expansion at sampler level (index duplication) -- no extra disk/VRAM. CacheEntry.repeats defaults to 1 for backwards compat.
- [07-01]: blocks_to_swap defaults to 0 (opt-in). Only Linear weights swapped; clamped at runtime to num_blocks - 1. Warning at >35 blocks.
- [07-02]: Block swap NOT re-registered after PEFT wrapping (hooks persist through PEFT wrap). hasattr guard for non-Wan backend compat.
- [08-01]: offload_gradients=False for cpu_offload (True breaks gradient accumulation). adam_mini bypasses LoRA+ A/B grouping. Wan constants hardcoded in dispatch.

### Blockers/Concerns

- (RESOLVED) FIX-01 and FIX-02 applied in Phase 5 Plan 01
- Block swap + PEFT + bitsandbytes three-way interaction untested in Flimmer
- Need RTX 3090/4090 hardware access for empirical VRAM validation (Phase 9)

## Session Continuity

Last session: 2026-03-05T21:34:43Z
Stopped at: Completed 08-01-PLAN.md
Resume file: .planning/phases/08-optimizer-improvements/08-01-SUMMARY.md

## Git Workflow

- **dev remote** (private): `github.com/alvdansen/flimmer-trainer-dev` — ALL development pushes here
- **origin remote** (public): `github.com/alvdansen/flimmer-trainer` — tested releases only
- NEVER push to origin during development — releasing to public is a separate milestone
- Push to `dev` after each phase execution
- Release cycle: develop → test on hardware → validate → then push to `origin`
