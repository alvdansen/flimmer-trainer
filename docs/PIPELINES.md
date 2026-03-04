# Flimmer Pipelines

Order of operations for the most common workflows. For dataset preparation philosophy and captioning strategies (character vs style vs motion), see [Klippbok](link-tbd).

---

## I already have a dataset

Clips with captions, ready to go? Validate, encode, train.

```bash
# 1. Check everything is complete and correctly formatted
python -m flimmer.dataset validate "path/to/dataset"

# 2. Pre-encode (VAE + T5, run separately so they don't compete for VRAM)
python -m flimmer.encoding cache-latents -c train.yaml
python -m flimmer.encoding cache-text -c train.yaml

# 3. Train
python -m flimmer.training train -c train.yaml
```

If validate reports issues (wrong resolution, missing captions, bad frame counts):

```bash
python -m flimmer.video normalize "path/to/dataset" -o fixed
python -m flimmer.dataset validate fixed
```

Normalize copies existing `.txt` caption files alongside the re-encoded clips — you don't lose your captions.

---

## I have raw uncut video

Long footage that needs splitting into training clips.

```bash
# 1. Split at scene boundaries, normalize to training specs
python -m flimmer.video ingest "path/to/video.mp4" -o clips

# 2. Caption each clip
python -m flimmer.video caption clips -p gemini -a "anchor word"

# 3. Extract first frames (for I2V training — skip for T2V)
#    See the I2V Guide for details: docs/I2V_GUIDE.md
python -m flimmer.video extract clips -o clips/references

# 4. Validate
python -m flimmer.dataset validate clips

# 5. Encode + train
python -m flimmer.encoding cache-latents -c train.yaml
python -m flimmer.encoding cache-text -c train.yaml
python -m flimmer.training train -c train.yaml
```

**Optional: triage first** — if you only want clips of a specific subject, use triage to find matching scenes before ingesting:

```bash
python -m flimmer.video triage "path/to/footage" -s concepts/
# review scene_triage_manifest.json, then:
python -m flimmer.video ingest "path/to/footage" -o clips --triage scene_triage_manifest.json
```

---

## I have pre-cut clips

Clips are already trimmed but might be wrong resolution, fps, or format.

```bash
# 1. Check what needs fixing
python -m flimmer.video scan "path/to/clips"

# 2. Normalize to training specs (16fps, correct resolution, valid frame counts)
python -m flimmer.video normalize "path/to/clips" -o ready

# 3. Caption (skip if you already have .txt sidecar files — normalize copies them)
python -m flimmer.video caption ready -p gemini -a "anchor word"

# 4. Extract first frames (for I2V — skip for T2V)
python -m flimmer.video extract ready -o ready/references

# 5. Validate
python -m flimmer.dataset validate ready

# 6. Encode + train
python -m flimmer.encoding cache-latents -c train.yaml
python -m flimmer.encoding cache-text -c train.yaml
python -m flimmer.training train -c train.yaml
```

---

## I2V Training (Image-to-Video)

Same workflow as T2V, with one extra step: extract first frames from your clips.

```bash
# 1. Prepare clips (same as T2V)
python -m flimmer.video ingest "path/to/video.mp4" -o clips
python -m flimmer.video caption clips -p gemini -a "anchor word"

# 2. Extract first frames
python -m flimmer.video extract clips -o clips/references

# 3. Validate, encode, train (same as T2V)
python -m flimmer.dataset validate clips
python -m flimmer.encoding cache-latents -c i2v_train.yaml
python -m flimmer.encoding cache-text -c i2v_train.yaml
python -m flimmer.training train -c i2v_train.yaml
```

Key differences from T2V:
- First frames are VAE-encoded and concatenated to video latents (in_channels: 36 vs 16)
- Higher caption dropout (0.15 vs 0.10) to strengthen image conditioning
- `first_frame_dropout_rate` available to reduce first frame reliance

For a deep dive on I2V configuration, see [I2V Training Guide](I2V_GUIDE.md).

---

## I need to fix an existing dataset

Wrong specs, inherited from another project, migrating between trainers.

```bash
# 1. See what's wrong
python -m flimmer.video scan "path/to/dataset"

# 2. Fix specs (copies existing captions alongside)
python -m flimmer.video normalize "path/to/dataset" -o fixed

# 3. Score caption quality (optional, no API — runs locally)
python -m flimmer.video score fixed

# 4. Validate
python -m flimmer.dataset validate fixed
```

---

## Organize for a different trainer

After validation, structure your dataset for other trainers:

```bash
python -m flimmer.dataset organize clips -o organized              # flat layout (any trainer)
python -m flimmer.dataset organize clips -o organized -t musubi     # kohya/sd-scripts config
python -m flimmer.dataset organize clips -o organized -t ai-toolkit # ai-toolkit config
```

---

## CLI Quick Reference

| Tool | Command | What it does |
|------|---------|-------------|
| **scan** | `python -m flimmer.video scan <dir>` | Check clips, report what needs normalizing |
| **ingest** | `python -m flimmer.video ingest <path> -o <out>` | Scene detect + split + normalize raw video |
| **normalize** | `python -m flimmer.video normalize <dir> -o <out>` | Fix fps, resolution, frame count. Copies captions. |
| **triage** | `python -m flimmer.video triage <dir> -s <concepts>` | Find clips matching reference images via CLIP |
| **caption** | `python -m flimmer.video caption <dir> -p <provider>` | Generate captions via VLM (Gemini, Replicate, Ollama) |
| **score** | `python -m flimmer.video score <dir>` | Score caption quality locally (no API) |
| **extract** | `python -m flimmer.video extract <dir> -o <out>` | Extract first frames from clips |
| **audit** | `python -m flimmer.video audit <dir>` | Compare existing captions against fresh VLM output |
| **validate** | `python -m flimmer.dataset validate <dir>` | Check dataset completeness and quality |
| **organize** | `python -m flimmer.dataset organize <dir> -o <out>` | Structure dataset for a specific trainer |
| **info** | `python -m flimmer.encoding info -c <config>` | Show cache status (no GPU) |
| **cache-latents** | `python -m flimmer.encoding cache-latents -c <config>` | Encode through VAE and cache to disk (GPU) |
| **cache-text** | `python -m flimmer.encoding cache-text -c <config>` | Encode captions through T5 and cache to disk (GPU) |
| **plan** | `python -m flimmer.training plan -c <config>` | Preview training plan (dry run) |
| **train** | `python -m flimmer.training train -c <config>` | Run training |

For config details: [Data Config Walkthrough](DATA_CONFIG_WALKTHROUGH.md) | [Training Config Walkthrough](TRAINING_CONFIG_WALKTHROUGH.md)
