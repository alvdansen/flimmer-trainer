# Flimmer

Video LoRA training toolkit for diffusion transformer models. Starting with Wan 2.1 and Wan 2.2 (T2V and I2V), with more model integrations planned.

> **Beta release** — Flimmer is functional and producing results, but this is an early release. Training defaults, config schemas, and CLI interfaces may change. MoE hyperparameters are actively being validated. Local scripts are included but lightly tested. The phase system has also been lightly tested, please flag any issues.

Built by [Alvdansen Labs](https://github.com/alvdansen).

## What is Flimmer?

Flimmer takes you from raw video footage to a trained LoRA checkpoint. It covers the full pipeline: cutting and preparing clips, generating captions, validating your dataset, pre-encoding for fast training, and running the actual training loop.

The data preparation tools are **standalone** — they produce standard output formats that work with any trainer (kohya/sd-scripts, ai-toolkit, etc.), not just Flimmer.

## Quick Start

### Using scripts (recommended for getting started)

```bash
bash scripts/setup.sh --variant 2.2_i2v         # install deps + download weights
bash scripts/prepare.sh --config train.yaml      # pre-encode latents + text
bash scripts/train.sh --config train.yaml        # train
```

See [Local Setup Guide](docs/LOCAL_SETUP.md) for full options and project-based workflows.

### Using Python commands 

**Already have a prepared dataset?**

```
your_dataset/           encode              train
  clip.mp4        →     VAE + T5        →   LoRA checkpoint
  clip.txt              cached to disk       .safetensors
```

```bash
python -m flimmer.encoding cache-latents -c train.yaml
python -m flimmer.encoding cache-text -c train.yaml
python -m flimmer.training train -c train.yaml
```

**Starting from raw footage?**

```
raw_video.mp4 → split into clips → caption → validate → encode → train
```

```bash
python -m flimmer.video ingest video.mp4 -o clips           # scene detect + split
python -m flimmer.video caption clips -p gemini -a "Holly"   # generate captions
python -m flimmer.dataset validate clips                     # check everything looks right
python -m flimmer.encoding cache-latents -c train.yaml       # encode through VAE
python -m flimmer.encoding cache-text -c train.yaml          # encode captions through T5
python -m flimmer.training train -c train.yaml               # train
```

**Have pre-cut clips that need normalizing?**

```
messy_clips/ → fix fps/resolution → caption → validate → encode → train
```

```bash
python -m flimmer.video normalize clips/ -o ready/           # fix to training specs
python -m flimmer.video caption ready/ -p gemini -a "Holly"  # caption
python -m flimmer.dataset validate ready/                    # validate
# then encode + train as above
```

For a deep dive on dataset preparation strategies, see [Klippbok](link-tbd).

## What You Can Do

### Prepare your data

Take raw footage (movie rips, YouTube downloads, client assets) and turn it into clean training clips. Flimmer handles scene detection, splitting, frame rate normalization, and format conversion so every clip meets the model's requirements.

```bash
python -m flimmer.video scan "path/to/clips"                              # check what needs fixing
python -m flimmer.video ingest "path/to/video.mp4" -o clips               # split at scene boundaries
python -m flimmer.video normalize "path/to/clips" -o normalized           # fix fps, resolution, frame counts
python -m flimmer.video caption clips -p gemini -u character -a "Holly"   # generate captions
python -m flimmer.video extract clips -o clips/references                 # extract first frames for I2V
python -m flimmer.video triage "path/to/footage" -s concepts/             # find clips of a specific person
```

### Validate and organize

Check that your dataset is complete and correctly formatted before you spend GPU time on it. Catches missing captions, resolution mismatches, and other problems early.

```bash
python -m flimmer.dataset validate "path/to/dataset"                 # check everything
python -m flimmer.dataset organize "path/to/dataset" -o organized    # clean layout for training
```

### Pre-encode for training

Convert your videos and captions into the latent representations the model actually trains on. This is done once and cached to disk so training doesn't repeat the expensive encoding step every epoch.

```bash
python -m flimmer.encoding info -c train.yaml              # preview what will be cached
python -m flimmer.encoding cache-latents -c train.yaml     # encode videos through VAE (GPU)
python -m flimmer.encoding cache-text -c train.yaml        # encode captions through T5 (GPU)
```

### Train

Run training from a single YAML config. Supports checkpoint resume, W&B logging, and video sampling during training so you can see how your LoRA is progressing.

```bash
python -m flimmer.training plan -c train.yaml    # preview training plan (no GPU)
python -m flimmer.training train -c train.yaml   # train
```

## Training Modes

Flimmer supports both standard and experimental MoE training for Wan models:

**Standard training** — a single LoRA trained across the full model. Works with Wan 2.1 (single transformer) and Wan 2.2 (both experts trained together as one). This is the conventional approach used by all trainers.

**MoE phased training (experimental)** — for Wan 2.2's dual-expert architecture. The model has two transformer experts that specialize by noise level: one handles early denoising (global composition, motion) and the other handles late denoising (fine detail, texture).

Phased training lets you:
1. **Unified phase** — train a single LoRA across both experts together (warmup)
2. **Fork** — split into two separate LoRAs, one per expert
3. **Specialize** — train each expert independently with its own learning rate, epochs, and other settings

This is an experimental approach — the theory is that each expert benefits from different training intensities, but this is still being validated. You can configure everything in YAML:

```yaml
moe:
  fork_enabled: true    # set to false for standard unified-only training

  high_noise:
    learning_rate: 1e-4
    max_epochs: 30

  low_noise:
    learning_rate: 8e-5
    max_epochs: 50
```

> **All Wan 2.2 MoE hyperparameters are experimental.** This is a beta — defaults will evolve as we validate results. Override anything in YAML to match your own experiments.

**Base strategy (open question):** For MoE phased training, the unified phase produces a shared LoRA that gets forked. What the low-noise expert starts from after the fork matters — should it inherit the merged unified weights, or start from a low-noise-only baseline? We're actively running experiments to answer this. A merge script is included in `runpod/merge_expert_weights.py`.

## Phase System

For Wan 2.2 MoE models, Flimmer supports multi-phase training through project configs. A project defines a sequence of training phases — typically unified warmup followed by per-expert specialization:

```yaml
name: holly_i2v
model_id: wan-2.2-i2v-14b
base_config: ./holly/flimmer_train.yaml

run_level_params:
  lora_rank: 16
  lora_alpha: 16
  mixed_precision: bf16

phases:
  - type: unified
    name: "Unified Warmup"
    overrides:
      learning_rate: 5e-5
      max_epochs: 15

  - type: high_noise
    name: "High Noise Expert"
    overrides:
      learning_rate: 1e-4
      max_epochs: 30

  - type: low_noise
    name: "Low Noise Expert"
    overrides:
      learning_rate: 8e-5
      max_epochs: 50
```

Phases are tracked automatically. Re-running a project skips completed phases and picks up where it left off.

```bash
bash scripts/train.sh --project project.yaml --status    # check phase progress
bash scripts/train.sh --project project.yaml --all       # run all pending phases
```

For single-config training (no phases), use `--config` instead of `--project`. See `examples/` for both approaches.

## Running

**RunPod (recommended):** Setup scripts are included in `runpod/` for cloud GPU training. See the configs there for tested pod configurations.

**Local:** Setup, encoding, and training scripts are included for local GPU machines. See [Local Setup Guide](docs/LOCAL_SETUP.md) for the full workflow.

## CLI Quick Reference

| Module | Command | What it does |
|--------|---------|-------------|
| video | `python -m flimmer.video scan <dir>` | Check clips and report what needs fixing |
| video | `python -m flimmer.video ingest <path> -o <dir>` | Split raw video into training clips |
| video | `python -m flimmer.video normalize <dir> -o <dir>` | Fix fps, resolution, frame counts |
| video | `python -m flimmer.video caption <dir> -p <provider>` | Generate captions with a vision model |
| video | `python -m flimmer.video audit <dir> -p <provider>` | Compare existing captions against fresh output |
| video | `python -m flimmer.video score <dir>` | Score caption quality locally (no API) |
| video | `python -m flimmer.video extract <dir> -o <dir>` | Extract first frames from clips |
| video | `python -m flimmer.video triage <dir> -s <concepts>` | Find clips matching reference photos |
| dataset | `python -m flimmer.dataset validate <path>` | Check dataset completeness and quality |
| dataset | `python -m flimmer.dataset organize <path> -o <dir>` | Organize into trainer-ready layout |
| encoding | `python -m flimmer.encoding info -c <config>` | Show what would be cached |
| encoding | `python -m flimmer.encoding cache-latents -c <config>` | Encode videos/images through VAE |
| encoding | `python -m flimmer.encoding cache-text -c <config>` | Encode captions through T5 |
| training | `python -m flimmer.training plan -c <config>` | Preview training plan (dry run) |
| training | `python -m flimmer.training train -c <config>` | Run training |

## Installation

```bash
# Core only
pip install -e .

# With specific modules
pip install -e ".[video]"       # Video processing (scene detection, ffmpeg)
pip install -e ".[caption]"     # VLM captioning (Gemini, Replicate backends)
pip install -e ".[dataset]"     # Dataset validation and organization
pip install -e ".[triage]"      # CLIP-based scene matching
pip install -e ".[encoding]"    # Latent pre-encoding (VAE + T5)
pip install -e ".[training]"    # Training loop
pip install -e ".[wan]"         # Wan model backend (diffusers, transformers, peft)

# Everything
pip install -e ".[all]"

# Development
pip install -e ".[dev]"
```

**Requires:** ffmpeg and ffprobe for video operations. Install via `winget install ffmpeg` (Windows) or your system package manager.

## Project Structure

```
flimmer/
  config/          Config schemas and validation
  video/           Video processing, scene detection, captioning CLI
  caption/         VLM captioning backends (Gemini, Replicate, OpenAI-compatible)
  triage/          CLIP-based scene matching
  dataset/         Dataset validation and organization
  encoding/        Latent pre-encoding and caching
  training/        Training loop, LoRA injection, checkpointing
  training/wan/    Wan model backend (2.1/2.2 T2V/I2V)
  phases/          Phase system: model definitions and phase resolution
  project/         Multi-phase project runner
scripts/           Local run scripts (setup, prepare, train)
examples/          Example YAML configs (data, training, projects)
docs/              Architecture, pipelines, config reference, guides
```


## Documentation

- [Technical Architecture](docs/TECHNICAL_ARCHITECTURE.md) — How video LoRA training works and Flimmer's design
- [Pipelines](docs/PIPELINES.md) — Practical guides for common training scenarios
- [Training Config Walkthrough](docs/TRAINING_CONFIG_WALKTHROUGH.md) — Training config reference
- [Local Setup Guide](docs/LOCAL_SETUP.md) — Setting up and running on a local GPU machine
- [I2V Training Guide](docs/I2V_GUIDE.md) — Image-to-Video training with Wan models

## License

Apache-2.0
