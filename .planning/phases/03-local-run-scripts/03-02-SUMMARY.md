---
phase: 03-local-run-scripts
plan: 02
subsystem: infra
tags: [bash, shell-script, yaml, huggingface, model-weights, venv, gpu-verification]

requires:
  - phase: 01-phase-system-integration
    provides: "Phase system, WAN_VARIANTS registry, training config schema"
  - phase: 02-i2v-backend
    provides: "I2V model support, first-frame encoding, boundary_ratio for I2V"
provides:
  - "One-command setup script (venv + deps + GPU check + weight downloads)"
  - "I2V training config examples for Wan 2.1 and 2.2"
  - "Multi-phase project YAML example for MoE fork-and-specialize workflow"
affects: [03-local-run-scripts, docs]

tech-stack:
  added: [huggingface_hub (hf_hub_download for weight downloads)]
  patterns: [flat models/ directory for all weights, bash script with flag parsing, hf_hub_download + shutil.copy2 for flat downloads]

key-files:
  created:
    - scripts/setup.sh
    - examples/i2v_wan21.yaml
    - examples/i2v_wan22.yaml
    - examples/project_moe.yaml
  modified: []

key-decisions:
  - "Used hf_hub_download + shutil.copy2 for flat ./models/ structure (avoids huggingface-cli --local-dir preserving repo subdirectories)"
  - "Variant flag is required unless --skip-downloads (prevents accidental 60+ GB download)"
  - "New I2V configs use safetensors T5 format (umt5_xxl_fp16.safetensors) matching setup.sh downloads"
  - "Project YAML references base_config for all fixed settings, phases only override variable params"

patterns-established:
  - "Weight download pattern: hf_hub_download() to HF cache, then shutil.copy2 to flat ./models/"
  - "Example config convention: header comment block explaining usage, variant differences, and setup command"
  - "Project YAML schema: name, model_id, base_config, run_level_params, ordered phases with type/overrides/extras"

requirements-completed: [RUN-01, RUN-03, RUN-04]

duration: 3min
completed: 2026-03-04
---

# Phase 3 Plan 2: Setup Script and Example Configs Summary

**One-command setup.sh with venv/GPU/weight-download pipeline, plus I2V training configs for Wan 2.1/2.2 and a multi-phase MoE project YAML**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-04T03:23:47Z
- **Completed:** 2026-03-04T03:27:36Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- Setup script handles full pipeline: venv creation, pip install, GPU verification (nvidia-smi + torch.cuda), and weight downloads with variant selection
- I2V training configs for both Wan 2.1 (non-MoE, single DiT) and Wan 2.2 (MoE, expert fork) with proper ./models/ paths
- Multi-phase project YAML demonstrating the unified warmup -> expert fork workflow with phase lifecycle tracking

## Task Commits

Each task was committed atomically:

1. **Task 1: Create setup.sh** - `5516805` (feat)
2. **Task 2: Create I2V and project example configs** - `5ffce6e` (feat)

## Files Created/Modified
- `scripts/setup.sh` - One-command environment setup: venv, deps, GPU check, weight downloads with --variant/--skip-downloads/--venv flags
- `examples/i2v_wan21.yaml` - I2V training config for Wan 2.1 non-MoE (single unified phase, single DiT)
- `examples/i2v_wan22.yaml` - I2V training config for Wan 2.2 MoE (expert fork workflow, dual DiTs)
- `examples/project_moe.yaml` - Multi-phase project YAML showing unified -> high-noise -> low-noise workflow

## Decisions Made
- Used `hf_hub_download()` + `shutil.copy2` instead of `huggingface-cli download` to avoid the `--local-dir` pitfall that preserves repo subdirectory structure (Pitfall 5 from research)
- Made `--variant` required unless `--skip-downloads` to prevent users from accidentally starting a 60+ GB download without choosing
- New I2V configs reference `umt5_xxl_fp16.safetensors` (Comfy-Org safetensors format) instead of `models_t5_umt5-xxl-enc-bf16.pth` (Wan-AI .pth format) to match setup.sh downloads
- Project YAML uses `base_config` field pointing to a full training YAML, with phases only overriding variable params (learning_rate, max_epochs)

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Setup script is ready for users to run on Linux machines with CUDA
- Example configs demonstrate both non-MoE and MoE I2V workflows
- Remaining Phase 3 plans (prepare.sh, train.sh) can reference setup.sh's venv and weight paths
- Project YAML schema is established for the project CLI (python -m flimmer.project) to consume

## Self-Check: PASSED

All 4 created files verified present. Both task commits (5516805, 5ffce6e) verified in git log.

---
*Phase: 03-local-run-scripts*
*Completed: 2026-03-04*
