# Config Templates

Starting points for Flimmer configs. Copy a template, edit the paths, and train.

## How Configs Work

Flimmer uses three types of config files that layer together:

```
data config          training config          project config (optional)
  "what my clips       "how to train            "chain multiple training
   look like"           one LoRA"                runs together"

  flimmer_data.yaml ─── referenced by ──→ train.yaml ←── referenced by ── project.yaml
```

**Data config** describes your dataset — where clips are, resolution, captions, anchor words. It does NOT control training.

**Training config** is a complete training run — model paths, LoRA rank, optimizer, epochs, checkpoints. It references a data config via `data_config:`. This is all you need to train.

**Project config** (optional) chains multiple training phases together — different datasets, learning rates, or expert specialization at each stage. It references a training config via `base_config:` and only overrides what changes per phase.

## Recommended Project Layout

Keep your config files alongside your dataset in one folder. All paths in configs resolve relative to the config file's location, so this keeps paths simple:

```
my_project/
  flimmer_data.yaml        # data config (copied from data/standard.yaml)
  flimmer_train.yaml       # training config (copied from training/)
  project.yaml             # optional — only if using multi-phase
  video_clips/             # your clips + sidecar caption .txt files
    clip_001.mp4
    clip_001.txt
    clip_002.mp4
    clip_002.txt
```

With this layout, your training config just says `data_config: ./flimmer_data.yaml` and your data config says `path: ./video_clips` — no absolute paths needed.

## Do I Need a Project?

**No.** Most users start with just a data config + training config:

```bash
bash scripts/prepare.sh --config my_train.yaml
bash scripts/train.sh --config my_train.yaml
```

Projects are for when you want multi-phase training — either MoE fork-and-specialize or dataset progression. See [projects/](projects/) for when that makes sense.

## Which Template Do I Start With?

| I want to... | Data | Training | Project |
|---|---|---|---|
| Train my first LoRA (T2V, Wan 2.2) | `data/minimal.yaml` | `training/t2v_wan22.yaml` | - |
| Train I2V with Wan 2.2 MoE | `data/standard.yaml` | `training/i2v_wan22.yaml` | - |
| Train I2V with Wan 2.1 (simpler) | `data/standard.yaml` | `training/i2v_wan21.yaml` | - |
| Fork MoE experts into separate LoRAs | `data/standard.yaml` | `training/i2v_wan22.yaml` | `projects/i2v_moe_phases.yaml` |
| Train on different datasets in stages | `data/standard.yaml` | any training config | `projects/t2v_phases.yaml` |
| See every available option | `data/full.yaml` | `training/t2v_wan22.yaml` | - |

## Folder Structure

```
config_templates/
├── data/                   Dataset configs — what your clips look like
│   ├── minimal.yaml          Just a path. Start here.
│   ├── standard.yaml         Anchor word + first frame extraction
│   └── full.yaml             Every option documented
│
├── training/               Training configs — one complete training run
│   ├── t2v_wan22.yaml        T2V Wan 2.2 MoE — full reference (every option)
│   ├── i2v_wan22.yaml        I2V Wan 2.2 MoE — streamlined
│   └── i2v_wan21.yaml        I2V Wan 2.1 non-MoE — simplest
│
└── projects/               Project configs — multi-phase workflows (optional)
    ├── i2v_moe_phases.yaml   MoE fork-and-specialize (works for T2V too)
    └── t2v_phases.yaml       Dataset progression (works for any model)
```

Each subfolder has its own README with details on adapting templates.

## Phase Training

For a conceptual overview of the phase system — what it is, why you'd use it, and how to configure it — see the [Phase Training Guide](../docs/PHASE_TRAINING.md).
