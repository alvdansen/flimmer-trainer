# Examples

## Folder Structure

```
examples/
├── data/               Data configs — what your dataset looks like
│   ├── minimal.yaml    Just a path. Start here.
│   ├── standard.yaml   Anchor word + first frame extraction
│   └── full.yaml       Every option documented
│
├── training/           Training configs — how to train a single LoRA
│   ├── t2v_wan22.yaml  Text-to-Video (Wan 2.2 MoE) — full reference config
│   ├── i2v_wan21.yaml  Image-to-Video (Wan 2.1, non-MoE)
│   └── i2v_wan22.yaml  Image-to-Video (Wan 2.2 MoE)
│
└── projects/           Project configs — multi-phase training workflows
    ├── t2v_phases.yaml       Dataset progression (Wan 2.1 T2V)
    └── i2v_moe_phases.yaml   Fork-and-specialize (Wan 2.2 I2V MoE)
```

**Data configs** describe your clips — where they are, what resolution, how captions work.
**Training configs** describe a single training run — model, LoRA rank, optimizer, epochs.
**Project configs** chain multiple training phases together into one workflow.

## What is Phase Training?

A single training config runs one LoRA training from start to finish. That works well for simple cases.

Phase training splits the process into stages that execute in sequence, with the LoRA checkpoint carrying forward between them. Each phase can use different datasets, learning rates, epoch counts, or training strategies.

### Why use phases?

**Dataset progression** — Train on close-ups first to lock in identity, then expand to full-body footage. Mixing everything from the start dilutes the signal. See `projects/t2v_phases.yaml`.

**MoE fork-and-specialize** — Wan 2.2 has two noise-level experts. Train them together first (unified phase), then fork into separate LoRAs with per-expert hyperparameters. See `projects/i2v_moe_phases.yaml`.

**Checkpointing** — Each phase saves a checkpoint. If Phase 2 goes wrong, revert to Phase 1's output and try again with different settings.

### How it works

A project YAML points to a base training config (model paths, LoRA structure, save settings) and defines a list of phases. Each phase only overrides what changes — typically `data_config`, `learning_rate`, and `max_epochs`.

```yaml
base_config: ./my_training.yaml

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
      learning_rate: 3e-5
      max_epochs: 15
```

Run with:
```bash
python -m flimmer.project run --project my_project.yaml
python -m flimmer.project status --project my_project.yaml
```

The project system tracks phase status in `flimmer_project.json`. Re-running skips completed phases and picks up where it left off.

## Where to Start

| Goal | Start with |
|------|-----------|
| First time, just want to train | `data/minimal.yaml` + `training/t2v_wan22.yaml` |
| I2V training | `training/i2v_wan21.yaml` or `training/i2v_wan22.yaml` |
| Multi-phase character training | `projects/t2v_phases.yaml` |
| MoE expert specialization | `projects/i2v_moe_phases.yaml` |
