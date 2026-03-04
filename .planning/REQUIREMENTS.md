# Requirements: Flimmer Phase Integration & I2V Support

**Defined:** 2026-03-03
**Core Value:** Phase system and I2V integration work end-to-end — user can define multi-phase training projects for the full Wan family and run them locally via scripts.

## v1 Requirements

### Code Integration

- [x] **INTG-01**: Copy dimljus_phases source into `flimmer/phases/` subpackage with relative imports
- [x] **INTG-02**: Rename all "dimljus" references to "flimmer" (classes, constants, filenames, strings)
- [x] **INTG-03**: Verify zero `dimljus` references remain via grep
- [x] **INTG-04**: `import flimmer.phases` succeeds without errors
- [x] **INTG-05**: Identify and surface genuine code differences between dimljus_phases and existing flimmer code (not name-only diffs) for user decision on which version to keep

### Testing

- [x] **TEST-01**: All existing flimmer-trainer tests pass after integration
- [x] **TEST-02**: All phase system tests pass under `flimmer/` namespace
- [x] **TEST-03**: Conftest fixtures properly scoped for registry isolation

### I2V Backend (Wan 2.1 + 2.2)

- [x] **I2V-01**: Wan 2.1 I2V model definition registered in phase system (non-MoE, unified phase)
- [x] **I2V-02**: Wan 2.2 I2V model definition registered in phase system (MoE, expert phases)
- [x] **I2V-03**: Wan 2.1 I2V variant added to WAN_VARIANTS registry with architecture params
- [x] **I2V-04**: I2V training backend in `flimmer/training/wan/` with reference image conditioning (serves both 2.1 and 2.2)
- [x] **I2V-05**: I2V inherits shared code from existing T2V backend (no duplicated loop code)
- [x] **I2V-06**: I2V backend tests for both variants (mocked, no GPU required)

### Local Run Scripts

- [x] **RUN-01**: Setup script: clone repo, create venv, install deps, verify GPU
- [x] **RUN-02**: Training launcher script with CLI args for config selection
- [x] **RUN-03**: Example YAML configs for T2V and I2V training runs
- [x] **RUN-04**: Scripts work on fresh Linux machine with CUDA/PyTorch

### Documentation

- [x] **DOC-01**: Update README with phase system overview
- [x] **DOC-02**: Update docs/ with I2V model support
- [x] **DOC-03**: Local run scripts documented with usage examples
- [x] **DOC-04**: Beta caveat prominently displayed for local run scripts

## v2 Requirements

### Registry Sync

- **SYNC-01**: Automated sync enforcement between MODEL_REGISTRY and WAN_VARIANTS

### Run Scripts

- **RUN-05**: Production-hardened run scripts with error recovery and logging
- **RUN-06**: Cloud deployment scripts (RunPod, Lambda, etc.)

## Out of Scope

| Feature | Reason |
|---------|--------|
| GUI for phase management | CLI-first, GUI deferred |
| Wan 2.1 T2V modifications | Existing T2V backend works as-is |
| Cloud deployment scripts | Local-only this milestone |
| Full local script QA | Explicitly beta — documented as needing further testing |
| Domain/stack research | User has deep domain expertise |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| INTG-01 | Phase 1 | Complete |
| INTG-02 | Phase 1 | Complete |
| INTG-03 | Phase 1 | Complete |
| INTG-04 | Phase 1 | Complete |
| INTG-05 | Phase 1 | Complete |
| TEST-01 | Phase 1 | Complete |
| TEST-02 | Phase 1 | Complete |
| TEST-03 | Phase 1 | Complete |
| I2V-01 | Phase 2 | Complete |
| I2V-02 | Phase 2 | Complete |
| I2V-03 | Phase 2 | Complete |
| I2V-04 | Phase 2 | Complete |
| I2V-05 | Phase 2 | Complete |
| I2V-06 | Phase 2 | Complete |
| RUN-01 | Phase 3 | Complete |
| RUN-02 | Phase 3 | Complete |
| RUN-03 | Phase 3 | Complete |
| RUN-04 | Phase 3 | Complete |
| DOC-01 | Phase 4 | Complete |
| DOC-02 | Phase 4 | Complete |
| DOC-03 | Phase 4 | Complete |
| DOC-04 | Phase 4 | Complete |

**Coverage:**
- v1 requirements: 22 total
- Mapped to phases: 22
- Unmapped: 0

---
*Requirements defined: 2026-03-03*
*Last updated: 2026-03-03 after roadmap creation*
