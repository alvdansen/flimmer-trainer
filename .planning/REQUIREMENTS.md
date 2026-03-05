# Requirements: Flimmer Low VRAM Training

**Defined:** 2026-03-05
**Core Value:** Enable Wan 2.2 I2V/T2V LoRA training on 24GB consumer GPUs (RTX 3090/4090) with the same techniques proven in kohya/musubi-tuner.

## v1.1 Requirements

### Training Correctness

- [x] **FIX-01**: Gradient checkpointing enforces `use_reentrant=False` to prevent silent zero gradients with PEFT LoRA adapters
- [x] **FIX-02**: PyTorch memory allocator configured with `expandable_segments:True` by default to prevent fragmentation OOMs

### Block Swapping

- [x] **SWAP-01**: Transformer block swapping offloads inactive blocks to CPU during forward/backward passes via PyTorch hooks
- [x] **SWAP-02**: Block swap count is configurable via `blocks_to_swap` field in training config
- [x] **SWAP-03**: Block swap uses pinned memory and async CUDA streams for efficient CPU<->GPU transfer
- [x] **SWAP-04**: Block swap applies after model load and PEFT wrapping (transparent to training loop)

### Mixed Image+Video Dataset

- [x] **IMG-01**: VAE encoder supports single-image target encoding as 1-frame latents
- [x] **IMG-02**: Cache pipeline dispatches correctly for `TARGET_IMAGE` role samples
- [x] **IMG-03**: `image_repeat` config field controls ratio of stills vs video clips in training batches
- [x] **IMG-04**: I2V mode uses image as its own first-frame reference when training on stills (self-referencing)

### Optimizer Improvements

- [ ] **OPT-01**: Optimizer state CPU offloading via torchao CPUOffloadOptimizer wrapper
- [ ] **OPT-02**: Adam-Mini optimizer option for 45-50% optimizer state reduction
- [ ] **OPT-03**: Memory-efficient optimizer selection via `optimizer` config field (adamw8bit, adam_mini, cpu_offload)

### VRAM Estimation

- [ ] **VRAM-01**: Pre-flight VRAM estimation predicts memory usage for a given config before training starts
- [ ] **VRAM-02**: VRAM estimation accounts for block swap count, precision, resolution, frame count, and LoRA rank
- [ ] **VRAM-03**: Clear warning or error when estimated VRAM exceeds available GPU memory

### User Experience

- [ ] **UX-01**: 24GB config templates for T2V and I2V training (RTX 3090/4090 settings)
- [ ] **UX-02**: Low VRAM training guide documenting settings, tradeoffs, and per-GPU-tier recommendations
- [ ] **UX-03**: Config templates include recommended block swap counts, resolution, frame counts per GPU tier

## v2 Requirements

### Advanced Memory

- **MEM-01**: Per-GPU VRAM presets (`vram_preset: 24gb`) that auto-tune block swap, precision, resolution
- **MEM-02**: Tiled VAE encoding for low VRAM encoding stage (separate from training VRAM)

### Hardware Support

- **HW-01**: RTX 3090 vs 4090 specific presets (fp8 tensor core availability, PCIe bandwidth)
- **HW-02**: 16GB GPU support (RTX 4080, RTX 4070 Ti Super)

## Out of Scope

| Feature | Reason |
|---------|--------|
| Multi-GPU / distributed training (FSDP/DDP) | Flimmer targets single-GPU. Block swap + quantization achieves 24GB on single GPU. |
| Dynamic resolution during training | Unpredictable VRAM spikes. Fixed resolution via config. |
| Automatic block swap count selection | VRAM too context-dependent to auto-detect. Provide estimation tool instead. |
| FP4 training computation | Experimental, produces quality artifacts. FP8 weights + bf16 compute is the sweet spot. |
| Model sharding (finer than block level) | Block swapping is the right granularity. Finer sharding adds complexity without benefit. |

## Constraints

- **Development:** Private staging repo (`flimmer-trainer-dev`) -- validate on real hardware before merging to public repo
- **Testing:** Must validate on RTX 3090/4090 (24GB) before public release
- **Compatibility:** Must not break existing H100/A100 training workflows
- **No new required deps:** Block swap is pure PyTorch. torchao and adam-mini are optional behind import guards.

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| FIX-01 | Phase 5 | Complete |
| FIX-02 | Phase 5 | Complete |
| SWAP-01 | Phase 7 | Complete |
| SWAP-02 | Phase 7 | Complete |
| SWAP-03 | Phase 7 | Complete |
| SWAP-04 | Phase 7 | Complete |
| IMG-01 | Phase 6 | Complete |
| IMG-02 | Phase 6 | Complete |
| IMG-03 | Phase 6 | Complete |
| IMG-04 | Phase 6 | Complete |
| OPT-01 | Phase 8 | Pending |
| OPT-02 | Phase 8 | Pending |
| OPT-03 | Phase 8 | Pending |
| VRAM-01 | Phase 9 | Pending |
| VRAM-02 | Phase 9 | Pending |
| VRAM-03 | Phase 9 | Pending |
| UX-01 | Phase 9 | Pending |
| UX-02 | Phase 9 | Pending |
| UX-03 | Phase 9 | Pending |

**Coverage:**
- v1.1 requirements: 19 total
- Mapped to phases: 19
- Unmapped: 0

---
*Requirements defined: 2026-03-05*
*Last updated: 2026-03-05 after roadmap creation*
