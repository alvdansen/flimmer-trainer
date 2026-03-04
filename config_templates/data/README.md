# Data Config Templates

Data configs describe your dataset — where your video clips are, what resolution they should be, how captions work. They don't control training at all. Your training config references a data config via the `data_config:` field.

## Three Tiers

These are progressive — start minimal and add fields as you need them.

### `minimal.yaml` — Start here

Just a path to your clips. Everything else uses sensible defaults (16fps, 480p, sidecar .txt captions).

```yaml
dataset:
  path: ./video_clips    # flat layout — .mp4 + .txt side by side
  # path: .              # flimmer layout — training/targets/ + training/signals/
```

**Flat layout:** Your folder has pairs like `clip_001.mp4` + `clip_001.txt` side by side.

**Flimmer layout:** Used when you ran `flimmer.video` pipeline. Clips in `training/targets/`, captions in `training/signals/captions/`. Flimmer auto-detects which layout you have.

### `standard.yaml` — Most users land here

Adds an anchor word (prepended to every caption so the model associates a trigger with your subject) and first-frame extraction for I2V training.

Good for character LoRAs where you want a consistent trigger word.

### `full.yaml` — Reference for power users

Every available option documented with comments. You probably won't need most of these, but it's useful as a reference when you want to know what's possible.

## Key Fields

| Field | What it does | When you need it |
|---|---|---|
| `dataset.path` | Where your clips are | Always |
| `dataset.use_case` | character / style / motion / object | Helps the captioner make better decisions |
| `controls.text.anchor_word` | Trigger word prepended to captions | Character/style LoRAs |
| `datasets[].repeats` | How many times to repeat a dataset per epoch | Balancing multiple datasets |
| `video.resolution` | 480 or 720 | When training at higher resolution |
| `controls.images.reference.source` | `first_frame` to auto-extract | I2V training |

## Adapting

1. Copy the tier that fits your needs
2. Edit `path` to point to your clips
3. Add `anchor_word` if training a character or style
4. The training config controls everything else (model, optimizer, epochs)
