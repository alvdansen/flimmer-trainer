# Example: Phased Training Project

Multi-phase training — break training into stages with different hyperparameters, datasets, or expert targets. Copy this folder, add your clips, and train.

## Layout

You set up your configs and dataset. Flimmer creates the rest.

```
my_project/
  flimmer_data.yaml        <- you create: data config
  flimmer_train.yaml       <- you create: base training config
  project.yaml             <- you create: phase definitions
  video_clips/             <- you create: clips + .txt captions
    clip_001.mp4
    clip_001.txt
    clip_002.mp4
    clip_002.txt
  cache/                   <- created by prepare.sh (pre-encoded latents)
    cache_manifest.json
    latents/
    text/
  output/                  <- created by training
    full_noise/               checkpoints per phase
      my_lora_epoch005.safetensors
    high_noise/
      my_lora_high_epoch015.safetensors
    low_noise/
      my_lora_low_epoch025.safetensors
    final/                    merged LoRA for inference
      my_lora_merged.safetensors
    samples/                  sample videos (if sampling enabled)
      fn_epoch005/
      hn_epoch015/
    training_state.json       resume state
    resolved_config.yaml      snapshot of config used
```

## Steps

1. Copy this folder to your project location
2. Put your video clips and caption `.txt` files in `video_clips/`
3. Edit `flimmer_data.yaml` — set your project name and anchor word
4. Edit `flimmer_train.yaml` — set your model variant, weight paths, and output dir
5. Edit `project.yaml` — define your phases (see examples inside)
6. Preview, encode, and train:

```bash
# Preview the plan — verify your project overrides are applied
python -m flimmer.training plan --project project.yaml

# Encode latents
bash scripts/prepare.sh --config flimmer_train.yaml

# Train all phases
bash scripts/train.sh --project project.yaml --all
```

**Tip:** Always run the plan command before training. It shows the fully resolved parameters (actual epochs, LR, etc.) for each phase after merging your project overrides with the base config. If something looks wrong, fix it before burning GPU time.

## When to use phases

- **Dataset progression** — close-ups first, then full-body, then mixed
- **MoE expert specialization** — unified warmup, then per-expert training (Wan 2.2)
- **Learning rate scheduling** — aggressive early, conservative late
- **Any time you want checkpoint control** between training stages

All paths in the configs are relative, so the project is self-contained and portable.
