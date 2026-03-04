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

## Phase Training

Phase training breaks LoRA training into stages — different datasets, learning rates, or strategies at each stage, with checkpoints carrying forward automatically. For a full walkthrough of the concept, config file structure, and examples, see [Phase Training Guide](../docs/PHASE_TRAINING.md).

## Where to Start

| Goal | Start with |
|------|-----------|
| First time, just want to train | `data/minimal.yaml` + `training/t2v_wan22.yaml` |
| I2V training | `training/i2v_wan21.yaml` or `training/i2v_wan22.yaml` |
| Multi-phase character training | `projects/t2v_phases.yaml` |
| MoE expert specialization | `projects/i2v_moe_phases.yaml` |
