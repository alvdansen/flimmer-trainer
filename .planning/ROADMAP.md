# Roadmap: Flimmer Phase Integration & I2V Support

## Overview

This milestone integrates the dimljus-phases registry-driven phase system into flimmer-trainer (with full rename), adds the Wan 2.2 I2V training backend, creates self-contained local run scripts for an A6000 machine, and updates documentation. The work flows linearly: integration must land before I2V can build on it, local run scripts are a parallel sub-project, and documentation wraps everything up.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Phase System Integration** - Copy dimljus_phases into flimmer/phases/, rename all references, surface genuine code diffs, get all tests passing (completed 2026-03-04)
- [ ] **Phase 2: I2V Backend (Wan 2.1 + 2.2)** - Build the I2V training backend with reference image conditioning for both Wan 2.1 (non-MoE) and Wan 2.2 (MoE), inheriting from existing T2V code
- [ ] **Phase 3: Local Run Scripts** - Self-contained setup, launcher, and example configs for A6000 training machine
- [ ] **Phase 4: Documentation** - Update README, docs, and usage examples for phase system, I2V support, and local scripts

## Phase Details

### Phase 1: Phase System Integration
**Goal**: The phase system lives in flimmer/phases/ with zero "dimljus" references, genuine code differences are surfaced for user decision, and the full test suite passes
**Depends on**: Nothing (first phase)
**Requirements**: INTG-01, INTG-02, INTG-03, INTG-04, INTG-05, TEST-01, TEST-02, TEST-03
**Success Criteria** (what must be TRUE):
  1. `import flimmer.phases` succeeds and auto-registers all model definitions
  2. `grep -r "dimljus" flimmer/phases/` returns zero hits (excluding historical attribution comments)
  3. A report of genuine code differences (not name-only diffs) between dimljus_phases and existing flimmer code is presented to the user with clear options for which version to keep
  4. Wan 2.1 T2V backend completeness verified — any gaps flagged for Phase 2
  5. `pytest tests/` passes with all existing flimmer tests AND all migrated phase system tests (registry isolation intact)
**Plans:** 3/3 plans complete

Plans:
- [x] 01-01-PLAN.md — Copy dimljus_phases source into flimmer/phases/, convert imports, rename all dimljus references
- [x] 01-02-PLAN.md — Copy test files into tests/phases/, rewrite imports, run full test suite
- [x] 01-03-PLAN.md — Generate diff report of genuine code differences for user review

### Phase 2: I2V Backend (Wan 2.1 + 2.2)
**Goal**: A user can configure and launch I2V training for both Wan 2.1 (non-MoE) and Wan 2.2 (MoE) through the phase system, with reference image conditioning handled end-to-end
**Depends on**: Phase 1
**Requirements**: I2V-01, I2V-02, I2V-03, I2V-04, I2V-05, I2V-06
**Success Criteria** (what must be TRUE):
  1. Wan 2.1 I2V (non-MoE) and Wan 2.2 I2V (MoE) model definitions both registered in MODEL_REGISTRY
  2. Wan 2.1 I2V variant added to WAN_VARIANTS with correct architecture params (non-MoE, I2V channels, single subfolder)
  3. I2V training backend in flimmer/training/wan/ accepts a reference image and produces conditioning tensors (shared logic for both variants)
  4. I2V backend shares core training logic with T2V (no duplicated training loop code)
  5. I2V backend tests for both 2.1 and 2.2 variants pass without GPU (mocked)
**Plans:** 3 plans

Plans:
- [ ] 02-01-PLAN.md — Add 2.1_i2v variant to WAN_VARIANTS, resolution subtypes to all variants, fix I2V model definitions
- [ ] 02-02-PLAN.md — Fix hardcoded config repo in load_model(), add first-frame dropout to training loop
- [ ] 02-03-PLAN.md — Update and add tests for 2.1 I2V variant, modality rename, first-frame dropout

### Phase 3: Local Run Scripts
**Goal**: A fresh A6000 machine can go from git clone to running a training job with a single setup script and launcher
**Depends on**: Phase 1
**Requirements**: RUN-01, RUN-02, RUN-03, RUN-04
**Success Criteria** (what must be TRUE):
  1. Setup script handles clone, venv creation, dependency installation, and GPU verification in one invocation
  2. Training launcher accepts a config path argument and starts a training run (T2V or I2V)
  3. Example YAML configs are provided for both T2V and I2V training scenarios
  4. Scripts run successfully on a fresh Linux machine with only CUDA and PyTorch pre-installed (no manual steps beyond git clone)
**Plans**: TBD

Plans:
- [ ] 03-01: TBD
- [ ] 03-02: TBD

### Phase 4: Documentation
**Goal**: A new user can understand the phase system, configure I2V training, and use local run scripts from the docs alone
**Depends on**: Phase 1, Phase 2, Phase 3
**Requirements**: DOC-01, DOC-02, DOC-03, DOC-04
**Success Criteria** (what must be TRUE):
  1. README includes a phase system overview explaining what it is, how to define phases, and how model definitions work
  2. docs/ covers Wan 2.2 I2V model support including reference image requirements and example configs
  3. Local run scripts have documented usage examples showing setup and training launch commands
  4. Beta caveat is prominently displayed (not buried) in both README and script docs for local run scripts
**Plans**: TBD

Plans:
- [ ] 04-01: TBD
- [ ] 04-02: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 2 -> 3 -> 4
(Phase 2 and Phase 3 can execute in parallel after Phase 1 completes)

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Phase System Integration | 3/3 | Complete    | 2026-03-04 |
| 2. I2V Backend (Wan 2.1 + 2.2) | 0/3 | Planned | - |
| 3. Local Run Scripts | 0/0 | Not started | - |
| 4. Documentation | 0/0 | Not started | - |
