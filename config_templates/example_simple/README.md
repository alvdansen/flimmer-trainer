# Example: Simple Training Project

Single-config training — one pass, no phases. Copy this folder, add your clips, and train.

## Layout

```
my_project/
  flimmer_data.yaml      <- data config (what clips, what format)
  flimmer_train.yaml     <- training config (model, hyperparameters, output)
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
5. Encode and train:

```bash
bash scripts/prepare.sh --config flimmer_train.yaml
bash scripts/train.sh --config flimmer_train.yaml
```

All paths in the configs are relative, so the project is self-contained and portable.
