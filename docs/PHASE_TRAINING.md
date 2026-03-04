# Phase Training Guide

## The Idea

Normal training uses one config file: you point it at your dataset, set your hyperparameters, and train a LoRA from scratch to done.

Phase training breaks that into stages. Each stage (phase) trains the same LoRA a bit further, but you can change things between phases — different datasets, different learning rates, different epoch counts. The LoRA checkpoint carries forward automatically.

Think of it like painting. You don't try to paint the whole picture in one pass. You block in the composition first, then add detail, then refine. Each pass builds on the last one.

This approach was pioneered for Stable Diffusion training by [TheLastBen](https://github.com/TheLastBen) in [fast-stable-diffusion](https://github.com/TheLastBen/fast-stable-diffusion). Flimmer adapts the concept for video diffusion models with native support for Wan 2.2's MoE expert architecture.

## When to Use Phases

**You don't always need phases.** If you have one dataset and one set of hyperparameters, a single training config is simpler and works fine.

Phases help when:

- **You have different types of footage.** Close-ups teach the model what a face looks like. Full-body shots teach posture and motion. Training on close-ups first, then expanding to full-body, produces better results than mixing everything together — because the model locks in identity before it has to juggle everything else.

- **You're using Wan 2.2 MoE.** The model has two internal experts (high-noise and low-noise). The recommended workflow is to train them together first, then fork into separate LoRAs with different hyperparameters per expert. This is hard to do without phases.

- **You want safe rollback.** Each phase saves a checkpoint. If Phase 3 makes things worse, you can go back to the Phase 2 checkpoint and try different settings.

## The Config Files

Phase training uses three types of files. Here's how they fit together:

```
my_project/
├── flimmer_train.yaml      ← Training config (the base)
├── project.yaml            ← Project config (the orchestrator)
├── data_closeups.yaml      ← Data config for Phase 1
├── data_fullbody.yaml      ← Data config for Phase 2
└── data_all.yaml           ← Data config for Phase 3
```

### Training config (the base)

This is a normal training YAML — the same one you'd use for single-run training. It has everything: model paths, LoRA rank, optimizer settings, save directory, logging, etc.

```yaml
# flimmer_train.yaml — everything about HOW to train
model:
  variant: 2.1_t2v
  dit: ./models/wan2.1_t2v_14B_fp16.safetensors
  vae: ./models/wan_2.1_vae.safetensors
  t5: ./models/umt5_xxl_fp16.safetensors

data_config: ./data_closeups.yaml   # default dataset (Phase 1 will use this)

lora:
  rank: 16
  alpha: 16

optimizer:
  type: adamw8bit
  learning_rate: 5e-5

training:
  mixed_precision: bf16
  gradient_checkpointing: true
  seed: 42
  unified_epochs: 10
  batch_size: 1

save:
  output_dir: ./output/my_character
  name: my_lora
  save_every_n_epochs: 5
```

This file is your source of truth. Everything in it applies to all phases unless a phase explicitly overrides it.

### Data configs (one per dataset)

Each data config describes a different slice of your footage:

```yaml
# data_closeups.yaml — tight shots, face and expressions
dataset:
  name: holly_closeups
  path: ./clips/closeups

controls:
  text:
    anchor_word: holly
```

```yaml
# data_fullbody.yaml — wider shots, motion and outfits
dataset:
  name: holly_fullbody
  path: ./clips/fullbody

controls:
  text:
    anchor_word: holly
```

These are normal data configs — nothing phase-specific about them.

### Project config (the orchestrator)

This is the only new file type. It tells Flimmer: "take that training config, and run it multiple times in sequence with these changes."

```yaml
# project.yaml — what to change at each stage
name: holly_character
model_id: wan-2.1-t2v-14b

base_config: ./flimmer_train.yaml   # ← points to the training config above

# These are locked — can't change between phases
run_level_params:
  lora_rank: 16
  lora_alpha: 16
  mixed_precision: bf16

# These execute in order, top to bottom
phases:
  - type: unified
    name: "Close-ups"
    overrides:
      data_config: ./data_closeups.yaml
      learning_rate: 5e-5
      max_epochs: 10

  - type: unified
    name: "Full Body"
    overrides:
      data_config: ./data_fullbody.yaml
      learning_rate: 3e-5        # lower — don't overwrite Phase 1
      max_epochs: 15

  - type: unified
    name: "Mixed Fine-tune"
    overrides:
      data_config: ./data_all.yaml
      learning_rate: 1e-5        # lowest — just polish
      max_epochs: 8
```

The `overrides` section is the key. Each phase starts from the base training config and patches in only what's listed under `overrides`. Everything else stays the same.

## What You Can Override Per Phase

| Parameter | What it does |
|-----------|-------------|
| `data_config` | Point to a different dataset |
| `learning_rate` | How fast the LoRA updates |
| `max_epochs` | How many passes through the dataset |
| `caption_dropout_rate` | How often to drop text conditioning |
| `batch_size` | Samples per step |
| `gradient_accumulation_steps` | Virtual batch size multiplier |

Things you **cannot** change between phases (they're locked in `run_level_params`): LoRA rank, alpha, precision. Changing these mid-project would break the checkpoint chain.

## Running a Project

```bash
# Run the next pending phase
python -m flimmer.project run --project project.yaml

# Run ALL pending phases back-to-back
python -m flimmer.project run --project project.yaml --all

# Check which phases are done
python -m flimmer.project status --project project.yaml

# Preview without actually training
python -m flimmer.project run --project project.yaml --dry-run
```

Or via the shell scripts:
```bash
bash scripts/train.sh --project project.yaml
bash scripts/train.sh --project project.yaml --all
bash scripts/train.sh --project project.yaml --status
```

### What happens when you run it

1. Flimmer reads `project.yaml` and checks `flimmer_project.json` for phase status
2. It finds the first phase that isn't COMPLETED
3. It merges the base training config with that phase's overrides
4. It runs training — the LoRA trains for `max_epochs` on the phase's dataset
5. It saves a checkpoint and marks the phase COMPLETED
6. Next time you run it, it skips to the next phase

If training crashes mid-phase, re-running picks up from the last checkpoint within that phase.

## MoE Phases (Wan 2.2)

Wan 2.2 has two experts that handle different noise levels. The phase types change:

```yaml
phases:
  # Both experts train together
  - type: unified
    name: "Unified Warmup"
    overrides:
      learning_rate: 5e-5
      max_epochs: 15

  # High-noise expert gets its own LoRA fork
  - type: high_noise
    name: "High Noise Expert"
    overrides:
      learning_rate: 1e-4     # higher — coarse features are easy
      max_epochs: 30

  # Low-noise expert gets its own LoRA fork
  - type: low_noise
    name: "Low Noise Expert"
    overrides:
      learning_rate: 8e-5     # lower — fine detail needs patience
      max_epochs: 50
```

The only difference from the dataset-progression example is the `type` field. Instead of all phases being `unified`, you use `high_noise` and `low_noise` to tell Flimmer which expert to train.

See `examples/projects/i2v_moe_phases.yaml` for a complete MoE project config.

## Examples

| Example | What it shows |
|---------|--------------|
| `examples/projects/t2v_phases.yaml` | Dataset progression — close-ups → full-body → mixed |
| `examples/projects/i2v_moe_phases.yaml` | MoE fork-and-specialize — unified → high-noise → low-noise |
