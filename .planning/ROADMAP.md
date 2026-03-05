# Roadmap: Flimmer Video LoRA Training Toolkit

## Milestones

- ✅ **v1.0 MVP** — Phases 1-4 (shipped 2026-03-05)
- 🚧 **v1.1 Low VRAM Training** — Phases 5-9 (in progress)

## Phases

<details>
<summary>✅ v1.0 MVP (Phases 1-4) — SHIPPED 2026-03-05</summary>

- [x] Phase 1: Phase System Integration (3/3 plans) — completed 2026-03-04
- [x] Phase 2: I2V Backend (4/4 plans) — completed 2026-03-04
- [x] Phase 3: Local Run Scripts (3/3 plans) — completed 2026-03-04
- [x] Phase 4: Documentation (2/2 plans) — completed 2026-03-04

</details>

### v1.1 Low VRAM Training

- [ ] **Phase 5: Training Correctness** - Fix gradient checkpointing and memory allocator before adding new features
- [ ] **Phase 6: Image Training Support** - Enable mixed stills+video datasets for all users
- [ ] **Phase 7: Block Swapping** - Offload transformer blocks to CPU to fit 14B models on 24GB GPUs
- [ ] **Phase 8: Optimizer Improvements** - CPU offloading and memory-efficient optimizers for additional VRAM savings
- [ ] **Phase 9: VRAM Estimation and User Experience** - Pre-flight memory prediction, 24GB config templates, and training guide

## Phase Details

### Phase 5: Training Correctness
**Goal**: Training produces correct gradients and avoids fragmentation OOMs regardless of PEFT adapter configuration
**Depends on**: Phase 4 (v1.0 complete)
**Requirements**: FIX-01, FIX-02
**Success Criteria** (what must be TRUE):
  1. LoRA adapter parameters receive non-zero gradients during training with gradient checkpointing enabled
  2. Training runs that previously fragmentation-OOMed at allocation boundaries complete without memory errors
  3. Existing H100/A100 training workflows produce identical results (no regression)
**Plans:** 1 plan
Plans:
- [ ] 05-01-PLAN.md — Fix gradient checkpointing use_reentrant and CUDA memory allocator config

### Phase 6: Image Training Support
**Goal**: Users can train on mixed datasets containing both video clips and still images, with images treated as single-frame training samples
**Depends on**: Phase 5
**Requirements**: IMG-01, IMG-02, IMG-03, IMG-04
**Success Criteria** (what must be TRUE):
  1. User can add .jpg/.png files alongside .mp4 clips in their dataset and run `cache-latents` without errors
  2. Training batches contain a configurable mix of image and video samples controlled by `image_repeat` in the data config
  3. In I2V mode, single-frame image samples use themselves as the first-frame conditioning input (no separate reference required)
  4. A dataset with only images trains successfully (edge case: pure stills dataset)
**Plans:** 2 plans
Plans:
- [ ] 06-01-PLAN.md — PIL-based image encoding in VAE + auto self-referencing for I2V
- [ ] 06-02-PLAN.md — image_repeat config field + BucketBatchSampler repeats support

### Phase 7: Block Swapping
**Goal**: Users can train Wan 2.2 I2V/T2V LoRA on a 24GB GPU by offloading inactive transformer blocks to CPU during forward/backward passes
**Depends on**: Phase 5
**Requirements**: SWAP-01, SWAP-02, SWAP-03, SWAP-04
**Success Criteria** (what must be TRUE):
  1. User can set `blocks_to_swap: 20` in their training config and training proceeds with reduced VRAM usage
  2. Block swap transfers use pinned memory and async CUDA streams (no synchronous stalls blocking the training loop)
  3. Block swap hooks are applied after model load and PEFT wrapping with no changes to the training loop or checkpoint code
  4. Training with block swap enabled produces LoRA checkpoints that load and generate correctly
  5. A Wan 2.2 I2V training run at 480p + rank 16 + fp8 + block swap fits within 24GB VRAM
**Plans**: TBD

### Phase 8: Optimizer Improvements
**Goal**: Users have additional VRAM savings options through CPU-offloaded optimizer state and memory-efficient optimizer algorithms
**Depends on**: Phase 7
**Requirements**: OPT-01, OPT-02, OPT-03
**Success Criteria** (what must be TRUE):
  1. User can set `optimizer: cpu_offload` in their training config to offload optimizer state to CPU via torchao
  2. User can set `optimizer: adam_mini` for reduced optimizer state memory (45-50% reduction vs AdamW)
  3. Both new optimizer options work correctly in combination with block swap and fp8 quantization
**Plans**: TBD

### Phase 9: VRAM Estimation and User Experience
**Goal**: Users know exactly what settings to use for their GPU tier and get warned before wasting time on configs that will OOM
**Depends on**: Phase 7, Phase 8
**Requirements**: VRAM-01, VRAM-02, VRAM-03, UX-01, UX-02, UX-03
**Success Criteria** (what must be TRUE):
  1. Running a pre-flight estimation command shows predicted VRAM usage for a given training config before any GPU work starts
  2. VRAM estimate accounts for block swap count, precision, resolution, frame count, and LoRA rank -- and warns when estimated usage exceeds available GPU memory
  3. 24GB config templates exist for both T2V and I2V with recommended block swap counts, resolutions, and frame counts
  4. A low VRAM training guide documents the full optimization stack, tradeoffs, and per-GPU-tier recommendations (24GB, 48GB, 80GB)
**Plans**: TBD

## Progress

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1. Phase System Integration | v1.0 | 3/3 | Complete | 2026-03-04 |
| 2. I2V Backend (Wan 2.1 + 2.2) | v1.0 | 4/4 | Complete | 2026-03-04 |
| 3. Local Run Scripts | v1.0 | 3/3 | Complete | 2026-03-04 |
| 4. Documentation | v1.0 | 2/2 | Complete | 2026-03-04 |
| 5. Training Correctness | v1.1 | 0/1 | Planned | - |
| 6. Image Training Support | v1.1 | 0/2 | Planned | - |
| 7. Block Swapping | v1.1 | 0/? | Not started | - |
| 8. Optimizer Improvements | v1.1 | 0/? | Not started | - |
| 9. VRAM Estimation and User Experience | v1.1 | 0/? | Not started | - |
