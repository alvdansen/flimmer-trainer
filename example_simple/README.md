# Example: Simple Training Project

Single-config training — one pass, no phases. Copy this folder, add your clips, and train.

## Layout

You set up your configs and dataset. Flimmer creates the rest.

```
my_project/
  flimmer_data.yaml        <- you create: data config
  flimmer_train.yaml       <- you create: training config
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
      my_lora_epoch010.safetensors
    final/                    merged LoRA for inference
      my_lora_merged.safetensors
    samples/                  sample videos (if sampling enabled)
      fn_epoch005/
    training_state.json       resume state
    resolved_config.yaml      snapshot of config used
```

## Steps

1. Copy this folder to your project location
2. Put your video clips and caption `.txt` files in `video_clips/`
3. Edit `flimmer_data.yaml` — set your project name and anchor word
4. Edit `flimmer_train.yaml` — set your model variant, weight paths, and output dir
5. Encode and train:

```bash
bash scripts/prepare.sh --config flimmer_train.yaml
bash scripts/train.sh --config flimmer_train.yaml
```

All paths in the configs are relative, so the project is self-contained and portable.
