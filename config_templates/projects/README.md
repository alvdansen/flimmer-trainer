# Project Config Templates

Projects are **optional**. They chain multiple training phases into one workflow with automatic checkpoint handoff between phases. If you just want to train one LoRA, use a [training config](../training/) directly — no project needed.

## When to Use a Project

Use a project when you want **multi-phase training** — either:

1. **MoE fork-and-specialize** — unified warmup, then per-expert LoRAs (`i2v_moe_phases.yaml`)
2. **Dataset progression** — train on different datasets in sequence (`t2v_phases.yaml`)

If you only need the standard MoE fork (unified → high-noise → low-noise) with one dataset, you may not need a project at all — the `moe:` section in your training config handles that automatically. Projects give you more control: different datasets per phase, manual epoch/LR tuning per phase, and the ability to resume individual phases.

## Which Template?

| Template | Phase strategy | Use case |
|---|---|---|
| `i2v_moe_phases.yaml` | full_noise → high_noise → low_noise | MoE expert specialization |
| `t2v_phases.yaml` | full_noise → full_noise → full_noise | Dataset progression (close-ups → full body → mixed) |

**Both templates work with any Wan variant.** The model is determined by `base_config`, not the project file. To use the MoE fork template for T2V instead of I2V, just point `base_config` at a T2V training config. Same with dataset progression — it works for I2V, T2V, MoE, and non-MoE.

## How It Works

```yaml
base_config: ./my_train.yaml     # Full training config with all defaults

phases:
  - type: full_noise              # Phase type (full_noise, high_noise, low_noise)
    name: "Unified Warmup"
    overrides:                    # Only override what changes
      learning_rate: 5e-5
      max_epochs: 15
```

- **`base_config`** provides everything — model paths, LoRA settings, optimizer, save location. Phases inherit it all.
- Each phase **only overrides** what changes (learning rate, epochs, dataset, etc.).
- **Checkpoint handoff** is automatic — each phase starts from where the previous one left off.
- Phase status is tracked in `flimmer_project.json` (created automatically). Re-running the project skips completed phases.

## Running

```bash
# Preview the fully resolved plan (verify overrides are applied)
python -m flimmer.training plan --project my_project.yaml
# or: bash scripts/train.sh --project my_project.yaml --dry-run

# Run all pending phases
bash scripts/train.sh --project my_project.yaml --all

# Run just the next pending phase
bash scripts/train.sh --project my_project.yaml

# Check progress
bash scripts/train.sh --project my_project.yaml --status
```

**Always preview your plan first.** The plan command merges your project overrides with the base config and shows the actual epochs, learning rates, and settings for each phase. If values look wrong, your overrides aren't being applied.

## Adapting

**Add more phases:** Append to the `phases:` list. The project runner executes them in order.

**Different datasets per phase:** Override `data_config` in each phase's `overrides:` block (see `t2v_phases.yaml` for an example).

**MoE with dataset progression:** Combine both strategies — use `full_noise` phases with different datasets for warmup, then `high_noise`/`low_noise` phases for expert specialization.

**Adjust hyperparameters:** Any training parameter can be overridden per phase — `learning_rate`, `max_epochs`, `batch_size`, `caption_dropout_rate`, etc.

For a conceptual overview of phase training, see the [Phase Training Guide](../../docs/PHASE_TRAINING.md).
