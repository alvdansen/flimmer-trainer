# Training Config Walkthrough

**Date:** 2026-03-03
**Status:** Verified against `wan22_training_master.py` and `full_train.yaml`

> **Beta — all Wan 2.2 MoE hyperparameters are experimental.** The defaults below represent our current best understanding, but they're actively being validated. Override anything in YAML to match your own experiments.

This document walks through the files that make up the Flimmer training config system and how they connect. It's a reference for understanding what exists, what decisions were made, and why.

---

## The YAML: What You Should Edit

The YAML is organized to follow the training flow: set up the model, configure fixed settings, configure the unified phase, configure expert overrides, then output settings.

### Section 1: Model & Data

```yaml
model:
  variant: 2.2_t2v                         # 2.2_t2v | 2.2_i2v

  # --- Individual weight files (recommended) ---
  dit_high: /workspace/models/wan2.2_t2v_high_noise_14B_fp16.safetensors
  dit_low: /workspace/models/wan2.2_t2v_low_noise_14B_fp16.safetensors
  vae: /workspace/models/wan_2.1_vae.safetensors
  t5: /workspace/models/models_t5_umt5-xxl-enc-bf16.pth
  # For non-MoE models (Wan 2.1), use dit instead of dit_high/dit_low:
  # dit: /path/to/transformer.safetensors

  # --- Diffusers directory (alternative) ---
  # path: Wan-AI/Wan2.2-T2V-14B-Diffusers

data_config: ./holly/flimmer_data.yaml
```

**`variant`** is the key field. It tells the loader which architecture defaults to apply. Setting `variant: 2.2_t2v` auto-fills: `is_moe: true`, `in_channels: 16`, `num_layers: 40`, `boundary_ratio: 0.875`, `flow_shift: 5.0`. The user can override any of these if needed — the commented-out fields show what's available.

**Two ways to point at model weights:**

1. **Individual files** (recommended): `dit_high`/`dit_low` for MoE experts (or `dit` for non-MoE), plus `vae` and `t5`. Each points to a single `.safetensors` or `.pth` file. This is how models are distributed by Comfy-Org and how cloud pod setup scripts download them.

2. **Diffusers directory**: `path` points to a local directory or HuggingFace ID. Components are loaded from standard subdirectories (transformer/, vae/, text_encoder/). When both are set, individual files take priority.

**`data_config`** points to the Flimmer data config YAML. The loader checks this file exists but doesn't load it — that happens at training time.

### Section 2: LoRA Structure (fixed)

```yaml
lora:
  rank: 16
  alpha: 16
  loraplus_lr_ratio: 4.0
  dropout: 0.05
  use_mua_init: false
  target_modules: null
  block_rank_overrides: null
```

Everything in this section is **locked for the entire training run**. Rank determines LoRA matrix dimensions — once created, the shape can't change. Alpha sets the scaling factor (alpha/rank = effective strength).

**`loraplus_lr_ratio`** — this is a LoRA+ feature that applies a higher learning rate to the B matrices (the "up" projection) relative to the A matrices (the "down" projection). A ratio of 4.0 means B matrices learn 4x faster than A. The intuition: B matrices are initialized to zero and need to move further from their starting point, while A matrices are initialized with useful structure and benefit from more conservative updates. Set to 1.0 to disable (equal learning rate for both). This is an advanced setting — the default of 4.0 works well in practice.

**Targeting** uses two levels:
- **Component level:** `"ffn"`, `"self_attn"`, `"cross_attn"` — trains all projections in that component
- **Projection level:** `"cross_attn.to_v"`, `"ffn.up_proj"` — trains one specific projection

`null` = all standard targets (all attention + FFN). The commented guide in the YAML explains the rules.

**Block rank overrides** let you give different LoRA capacity to different transformer blocks. Early blocks (0-11) handle composition, late blocks (30-39) handle detail — you might want more capacity where it matters most.

### Section 3: Optimizer

```yaml
optimizer:
  # Fixed for entire run
  type: adamw8bit
  betas: [0.9, 0.999]
  eps: 1e-8
  max_grad_norm: 1.0
  optimizer_args: {}

  # Unified starting values — overridable per expert after fork
  learning_rate: 5e-5
  weight_decay: 0.01
```

This section is split into **fixed** and **overridable** zones.

Fixed settings (optimizer type, betas, eps, gradient clipping) stay constant for the entire run. You pick your optimizer once.

Learning rate and weight decay are **starting values** — they apply during the unified phase and each expert inherits them after fork, unless overridden per-expert.

**Why 5e-5?** Community recommendations (2e-4) are too aggressive for Wan 2.2, especially for the low-noise expert which overfits rapidly at high LR. 5e-5 is conservative. Each expert overrides as needed.

### Section 4: Scheduler

```yaml
scheduler:
  # Fixed for entire run
  warmup_steps: 0

  # Unified starting values — overridable per expert after fork
  type: cosine_with_min_lr
  min_lr: null
  min_lr_ratio: 0.01
```

Same fixed/overridable split. Warmup runs once at training start (fixed). Scheduler type and min LR ratio are overridable per expert — high-noise might want faster decay, low-noise might want constant LR.

### Section 5: Training

```yaml
training:
  # Fixed for entire run
  mixed_precision: bf16
  base_model_precision: bf16
  timestep_sampling: shift
  discrete_flow_shift: null              # null = model default
  gradient_checkpointing: true
  seed: 42
  resume_from: null                      # Checkpoint path to resume training

  # Unified phase — starting values, overridable per expert after fork
  unified_epochs: 15                     # I2V: 15, T2V: 10
  unified_targets: null
  unified_block_targets: null
  batch_size: 1
  gradient_accumulation_steps: 1
  caption_dropout_rate: 0.15             # I2V: 0.15, T2V: 0.10
```

Fixed settings: precision (`bf16` for both training and frozen base model — no quantization shortcuts), timestep sampling (`shift` matches Wan's pretraining), gradient checkpointing (always on for video).

**`resume_from`** points to a checkpoint directory to resume training from. The system restores optimizer state, scheduler state, RNG, and training progress. Resume operates at epoch granularity — partial epochs are skipped and the next full epoch starts.

**`max_train_steps`** (not shown — advanced) stops training after a fixed number of optimizer steps. Used for testing checkpoint resume. `null` = train to completion.

Unified phase settings: `unified_epochs` controls how long both experts share a single LoRA before forking. `unified_targets` and `unified_block_targets` can narrow what gets trained during the shared phase.

**Why bf16 for base model?** Quality first. fp8 is available if VRAM-constrained but introduces quantization artifacts. The master file documents this tradeoff.

### Section 6: MoE Expert Fork

```yaml
moe:
  enabled: true
  fork_enabled: true
  boundary_ratio: null
  preload_experts: false                 # true = pre-load inactive expert to CPU RAM

  high_noise:
    learning_rate: 1e-4
    max_epochs: 30

  low_noise:
    learning_rate: 8e-5
    max_epochs: 50
```

**`enabled`** — this model has two noise-level experts. Both are active during training regardless of mode. Even in unified-only mode (fork_enabled: false), both experts route during the forward pass.

**`preload_experts`** — pre-load both expert weight sets into CPU RAM at startup for faster expert switching (~3s swap vs ~30s disk reload). Needs ~27GB extra RAM. Set `true` on cloud pods with plenty of RAM, `false` on local machines.

**`fork_enabled`** — the master switch for fork-and-specialize. Three training modes:

| Mode | Settings | What happens |
|------|----------|-------------|
| Fork-and-specialize | `fork_enabled: true` + `unified_epochs > 0` | Unified LoRA warmup → fork into per-expert copies → each trains independently |
| Unified only | `fork_enabled: false` | One LoRA on both experts, no forking |
| Expert from scratch | `fork_enabled: true` + `unified_epochs: 0` | Skip unified, go straight to per-expert |

**Per-expert overrides** — `null` means inherit from the unified/optimizer/scheduler defaults. Only set what you want to change. The full list of overridable fields per expert:

- `learning_rate`, `max_epochs`, `dropout`
- `fork_targets`, `block_targets` (narrow what gets trained per expert)
- `resume_from` (start from an existing LoRA file)
- `batch_size`, `gradient_accumulation_steps`, `caption_dropout_rate`
- `weight_decay`, `min_lr_ratio`
- `optimizer_type`, `scheduler_type`

### Section 7: Checkpoints

```yaml
save:
  save_every_n_epochs: 5
  output_dir: ./output/holly_i2v
  name: holly_lora
  save_last: true                        # Always save final checkpoint
  max_checkpoints: null                  # null = keep all
  format: safetensors
```

During fork-and-specialize, checkpoints are organized automatically:

```
{output_dir}/
  unified/
    {name}_epoch005.safetensors
    {name}_epoch010.safetensors
  high_noise/
    {name}_high_epoch015.safetensors
  low_noise/
    {name}_low_epoch020.safetensors
```

### Section 8: Logging

```yaml
logging:
  backends: [console, wandb]
  log_every_n_steps: 10
  wandb_project: flimmer-training
  wandb_entity: null
  wandb_run_name: holly_i2v_r16
```

W&B requires `wandb login` or `WANDB_API_KEY` env var before use. The validator catches missing `wandb_project` when wandb is in backends.

Additional logging fields (not shown above): `wandb_group` for clustering related runs, `wandb_tags` for filtering, `vram_sample_every_n_steps` (default 50) for GPU VRAM monitoring.

### Section 9: Sampling

```yaml
sampling:
  enabled: true
  every_n_epochs: 5
  prompts:
    - "Holly Golightly walks elegantly through a sunlit garden"
    - "Holly Golightly looks into the camera with a playful smile"
  neg: "blurry, low quality, distorted"
  seed: 42
  walk_seed: true
  sample_steps: 25
  guidance_scale: 4.0
  sample_dir: null
```

Off by default — inference passes are expensive. `walk_seed` increments the seed per prompt for variety while keeping each individual prompt reproducible across epochs.

`skip_phases` (not shown) lets you skip sampling during specific training phases (e.g. `['unified']` to only sample during expert phases).

### Section 10: Cache

```yaml
cache:
  cache_dir: ./cache
  dtype: bf16
  target_frames: [17, 33, 49, 81]
  frame_extraction: head
  include_head_frame: false
  reso_step: 16
```

Controls the latent pre-encoding cache. Pre-encoding separates the expensive VAE/T5 encoding from training — encode once, train many times.

**`cache_dir`** is separate from the dataset directory because cache files depend on the model (which VAE, which text encoder), not just the data.

**`target_frames`** lists the frame counts to cache per video. All must satisfy the 4n+1 constraint (Wan's 3D causal VAE temporal compression). Each video produces one cached latent per frame count that fits within its duration. Default `[17, 33, 49, 81]` covers ~1s to ~5s at 16fps.

**`frame_extraction`** controls how frames are pulled from videos: `head` = first N frames (matches Wan pretraining), `uniform` = evenly spaced across duration.

---

## The Master File: What's Inside

`wan22_training_master.py` has three parts.

### Part 1: Valid Options (lines 23-153)

Vocabulary — what names are legal in config fields. Eight sets:

| Set | Count | Examples |
|-----|-------|---------|
| `VALID_OPTIMIZERS` | 7 | adamw, adamw8bit, adafactor, came, prodigy, ademamix, schedule_free_adamw |
| `VALID_SCHEDULERS` | 11 | constant, cosine, cosine_with_min_lr, polynomial, rex, warmup_stable_decay... |
| `VALID_MIXED_PRECISION` | 3 | bf16, fp16, no |
| `VALID_BASE_PRECISION` | 5 | fp8, fp8_scaled, bf16, fp16, fp32 |
| `VALID_TIMESTEP_SAMPLING` | 4 | uniform, shift, logit_normal, sigmoid |
| `VALID_LOG_BACKENDS` | 3 | console, tensorboard, wandb |
| `VALID_CHECKPOINT_FORMATS` | 2 | safetensors, diffusers |
| `VALID_FORK_TARGETS` | 12 | Component: ffn, self_attn, cross_attn. Projection: cross_attn.to_q, ffn.up_proj... |

Each has a docstring explaining what every option does and when to use it.

### Part 2: Default Values (lines 155-577)

All constants with `T2V_` or `I2V_` prefixes. Organized into zones:

**Architecture** — model family, variant, channels, layers, boundary ratio, flow shift. These get applied by the loader when a user sets `variant: 2.2_t2v`.

**Training Strategy** — `T2V_FORK_ENABLED = True`. The master switch. One constant, three training modes via YAML settings.

**Fixed Settings** — rank (16), alpha (16), loraplus (4.0), optimizer type (adamw8bit), betas, eps, grad norm, warmup (0), mixed precision (bf16), base model precision (bf16), timestep sampling (shift). These don't change between phases.

**Unified Foundation** — unified_epochs (10), learning_rate (5e-5), weight_decay (0.01), scheduler (cosine_with_min_lr), min_lr_ratio (0.01), lora_dropout (0.0), batch_size (1), gradient_accumulation (1), caption_dropout (0.10). Starting values that experts can override.

**Expert Overrides** — Per-expert fields for high-noise and low-noise. All `None` by default = inherit from unified. `max_epochs` has a value (50 for both).

**Sampling** — Default prompts, guidance scale (4.0), sample steps (30), seed (42). OFF by default.

**I2V Differences** — Only the values that differ from T2V:
- `I2V_IN_CHANNELS = 36` (16 noise + 20 reference image)
- `I2V_BOUNDARY_RATIO = 0.900` (vs 0.875)
- `I2V_UNIFIED_EPOCHS = 15` (longer — more shared signal)
- `I2V_CAPTION_DROPOUT_RATE = 0.15` (higher — reference image carries conditioning)

**Variant Defaults Map** — `VARIANT_DEFAULTS` dict maps `"2.2_t2v"` and `"2.2_i2v"` to their override dicts. The loader deep-merges these under the user's YAML.

### Part 3: Pydantic Schema (lines 579-end)

12 models that define the YAML structure and validation:

| Model | Key fields |
|-------|-----------|
| `ModelConfig` | dit, dit_high, dit_low, vae, t5, path, family, variant, is_moe, in_channels, boundary_ratio, flow_shift |
| `LoraConfig` | rank, alpha, dropout, loraplus_lr_ratio, target_modules, block_rank_overrides, use_mua_init |
| `OptimizerConfig` | type, learning_rate, weight_decay, betas, eps, max_grad_norm |
| `SchedulerConfig` | type, warmup_steps, min_lr, min_lr_ratio, rex_alpha, rex_beta |
| `MoeExpertOverrides` | enabled, learning_rate, dropout, max_epochs, fork_targets, block_targets, resume_from, batch_size, gradient_accumulation_steps, caption_dropout_rate, weight_decay, min_lr_ratio, optimizer_type, scheduler_type |
| `MoeConfig` | enabled, fork_enabled, expert_order, preload_experts, high_noise, low_noise, boundary_ratio |
| `TrainingLoopConfig` | unified_epochs, unified_targets, unified_block_targets, batch_size, gradient_accumulation_steps, mixed_precision, base_model_precision, caption_dropout_rate, timestep_sampling, discrete_flow_shift, seed, max_train_steps, resume_from |
| `SaveConfig` | output_dir, name, save_every_n_epochs, save_last, max_checkpoints, format |
| `LoggingConfig` | backends, log_every_n_steps, wandb_project, wandb_entity, wandb_run_name, wandb_group, wandb_tags, vram_sample_every_n_steps |
| `SamplingConfig` | enabled, every_n_epochs, prompts, neg, seed, walk_seed, sample_steps, guidance_scale, sample_dir, skip_phases |
| `CacheConfig` | cache_dir, dtype, target_frames, frame_extraction, include_head_frame, reso_step |
| `FlimmerTrainingConfig` | Root — assembles all of the above |

**Root validators** (cross-field checks):
1. `check_moe_consistency` — error if MoE enabled on non-MoE model
2. `check_prodigy_lr` — error if Prodigy with lr != 1.0
3. `check_wandb_project` — error if wandb backend but no project name
4. `check_mua_alpha` — auto-set alpha=rank when muA enabled
5. `check_fork_without_moe` — error if fork_enabled but MoE disabled
6. `warn_aggressive_low_noise` — soft warning if low-noise LR > 2e-4

---

## The Loader: How They Connect

`training_loader.py` runs this sequence:

```
1. Find YAML (file path or directory with flimmer_train.yaml)
2. Load YAML → dict
3. Look up model.variant in VARIANT_DEFAULTS
4. Deep-merge: variant defaults as base, user YAML on top (user wins)
5. Auto-enable moe.enabled if is_moe is true
6. Resolve all relative paths (data_config, model.path, save.output_dir,
   training.resume_from, expert resume_from paths, sampling.sample_dir)
7. Validate through Pydantic → FlimmerTrainingConfig
8. Check data_config file exists
```

**Deep merge rules:**
- Scalars: user wins
- Dicts: recurse (nested merge)
- Lists: user replaces entirely (no list merging)

This means a user who sets `optimizer.learning_rate: 1e-4` overrides the default `5e-5` without affecting any other optimizer field.

---

## Current Defaults (Wan 2.2 T2V)

> These are experimental beta defaults. They reflect our current best understanding from ongoing training experiments, but they haven't been fully validated. Override anything that doesn't work for your use case.

| Setting | Default | Notes |
|---------|---------|-------|
| LoRA rank | 16 | Matrix size — locked for entire run |
| LoRA alpha | 16 | 1.0x neutral scaling (alpha/rank) |
| LoRA+ lr ratio | 4.0 | B matrices learn 4x faster than A |
| Learning rate | 5e-5 | Conservative — community 2e-4 is too aggressive for Wan 2.2 |
| Optimizer | adamw8bit | 8-bit saves VRAM, minimal quality loss |
| Weight decay | 0.01 | Standard regularization |
| Scheduler | cosine_with_min_lr | With min_lr_ratio 0.01 |
| Unified epochs | 10 (T2V) / 15 (I2V) | Shared LoRA warmup before fork |
| Expert epochs | 50 (both) | Per-expert after fork |
| Batch size | 1 | Video is memory-heavy |
| Mixed precision | bf16 | Training precision |
| Base model precision | bf16 | Quality first — fp8 available if VRAM-constrained |
| Flow shift | 5.0 | Wan 2.2 flow matching parameter |
| Timestep boundary | 0.875 | Where high-noise expert hands off to low-noise |
| Caption dropout | 0.10 (T2V) / 0.15 (I2V) | Forces reliance on visual conditioning |
| Save interval | 5 epochs | Checkpoint frequency |
| Seed | 42 | Reproducibility |
