# Example: Phased Training Project

Multi-phase training — break training into stages with different hyperparameters, datasets, or expert targets. Copy this folder, add your clips, and train.

## Layout

```
my_project/
  flimmer_data.yaml      <- data config (what clips, what format)
  flimmer_train.yaml     <- base training config (model, LoRA, optimizer, output)
  project.yaml           <- phase definitions (what changes per stage)
  video_clips/           <- your clips + sidecar .txt captions
    clip_001.mp4
    clip_001.txt
    ...
```

## Steps

1. Copy this folder to your project location
2. Put your video clips and caption `.txt` files in `video_clips/`
3. Edit `flimmer_data.yaml` — set your project name and anchor word
4. Edit `flimmer_train.yaml` — set your model variant, weight paths, and output dir
5. Edit `project.yaml` — define your phases (see examples inside)
6. Encode and train:

```bash
bash scripts/prepare.sh --config flimmer_train.yaml
bash scripts/train.sh --project project.yaml --all
```

## When to use phases

- **Dataset progression** — close-ups first, then full-body, then mixed
- **MoE expert specialization** — unified warmup, then per-expert training (Wan 2.2)
- **Learning rate scheduling** — aggressive early, conservative late
- **Any time you want checkpoint control** between training stages

All paths in the configs are relative, so the project is self-contained and portable.
