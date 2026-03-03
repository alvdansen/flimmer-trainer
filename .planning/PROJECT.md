# Flimmer Trainer — Phase System Integration & Wan 2.2 I2V

## What This Is

Flimmer is a video LoRA training toolkit for diffusion transformer models (Wan 2.1/2.2 T2V and I2V). This milestone integrates the dimljus-phases registry-driven phase configuration system into flimmer-trainer (renaming all references from "dimljus" to "flimmer"), adds Wan 2.2 I2V model support, and creates self-contained local run scripts for training on a remote A6000 machine.

## Core Value

The phase system and I2V integration must work end-to-end: a user can define a multi-phase training project, select any registered model (including Wan 2.2 I2V), and run training locally via shell scripts — all without manual setup beyond `git clone` and `pip install`.

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] Integrate dimljus_phases code into flimmer/phases/ subpackage with all "dimljus" references renamed to "flimmer"
- [ ] All existing tests pass after integration
- [ ] Wan 2.2 I2V model definition registered and functional in the phase system
- [ ] Wan 2.2 I2V backend integrated into flimmer/training/wan/ (net-new, inheriting from existing T2V code)
- [ ] Local run scripts: setup script (clone, install, verify GPU), training launcher, example configs
- [ ] Scripts are self-contained and portable — designed for A6000 machine running via git pull
- [ ] Documentation updated to reflect new phase system, I2V support, and local run scripts
- [ ] Local run scripts marked as beta in docs with debugging caveat

### Out of Scope

- Full production hardening of local run scripts — explicitly beta, may need further debugging
- Wan 2.1 changes — existing T2V/I2V definitions carry over as-is
- Cloud deployment scripts — local-only for now
- GUI/web interface for phase management

## Context

- **Source code:** `C:\Users\minta\Projects\dimljus training phases\src\dimljus_phases\` contains the complete phase system (15 Python files + 13 test files)
- **Target location:** New `flimmer/phases/` subpackage in flimmer-trainer
- **Existing architecture:** flimmer-trainer already has `flimmer/training/wan/` with T2V backend, constants, registry, checkpoint I/O, inference
- **I2V work:** Wan 2.2 I2V model definition already exists in phases code (`wan22_i2v.py`), but the actual training backend (image conditioning, CLIP encoding, reference image handling) is net-new
- **Remote training:** A6000 machine is fresh (CUDA/PyTorch only). Scripts need to handle full setup from `git clone` through training launch
- **Workflow:** Git-based — push code from dev machine, A6000 pulls via repo, Claude Code supervises on both ends

## Constraints

- **Naming:** All "dimljus" references must become "flimmer" — package names, imports, file names, constants, project file names
- **Compatibility:** Must not break existing 70 tests in flimmer-trainer
- **Dependencies:** Phase system only adds pydantic (already a dep) and pyyaml (already a dep)
- **A6000 target:** Scripts must work on Linux with NVIDIA GPU, PyTorch, and CUDA pre-installed

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Phase code goes in `flimmer/phases/` subpackage | Keeps phase system cleanly separated from training loop, mirrors original standalone package structure | — Pending |
| Local scripts as shell scripts + example YAML configs | Most portable and future-proofed — works with any CLI, easy to modify | — Pending |
| Git-based A6000 workflow | Both machines have Claude Code; git push/pull is simplest coordination | — Pending |
| I2V inherits from existing T2V backend | Wan 2.2 I2V shares architecture with T2V, just adds reference image conditioning | — Pending |
| Local run scripts marked beta | Not heavily tested yet, honest about maturity level | — Pending |

---
*Last updated: 2026-03-03 after initialization*
