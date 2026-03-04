# Local Setup Guide

This guide covers the three-script workflow for setting up and running Flimmer on a local GPU machine.

---

## Prerequisites

- **Linux** with NVIDIA GPU (tested on A6000, should work on any Ampere+ with sufficient VRAM)
- **CUDA drivers** installed (`nvidia-smi` should report your GPU)
- **Python 3.10+**
- **~24 GB VRAM minimum** for training (Wan 14B with LoRA + gradient checkpointing)
- **Disk space** for model weights (44-69 GB depending on variant — see table below)
- **tmux** recommended for background training sessions (optional — scripts gracefully fall back to foreground)

---

## Three-Script Workflow

Flimmer's local workflow is three scripts run in sequence. Each script can be run independently — if you already have a venv and weights, skip straight to `prepare.sh`.

```
setup.sh  →  prepare.sh  →  train.sh
 venv        encode          train
 deps        VAE latents     LoRA
 GPU check   T5 text         checkpoint
 weights
```

---

### 1. Setup (`scripts/setup.sh`)

Creates a Python virtual environment, installs all dependencies, verifies GPU access, and downloads model weights.

**Usage:**

```bash
bash scripts/setup.sh --variant 2.2_i2v           # full setup with Wan 2.2 I2V weights
bash scripts/setup.sh --variant 2.1_i2v_480p       # full setup with Wan 2.1 I2V 480p
bash scripts/setup.sh --skip-downloads             # venv + deps + GPU check only (no weights)
bash scripts/setup.sh --variant 2.2_t2v --venv /path/to/venv   # custom venv location
```

**Flags:**

| Flag | Required | Description |
|------|----------|-------------|
| `--variant VARIANT` | Yes (unless `--skip-downloads`) | Which model weights to download |
| `--skip-downloads` | No | Skip weight downloads — set up venv and dependencies only |
| `--venv PATH` | No | Override venv location (default: `.venv` in repo root) |
| `-h`, `--help` | No | Show help message |

**Available variants:**

| Variant | Model | Architecture | Download Size |
|---------|-------|-------------|---------------|
| `2.2_t2v` | Wan 2.2 Text-to-Video | MoE (2 expert DiTs) | ~69 GB |
| `2.2_i2v` | Wan 2.2 Image-to-Video | MoE (2 expert DiTs) | ~69 GB |
| `2.1_i2v_480p` | Wan 2.1 I2V 480p | Non-MoE (1 DiT) | ~44 GB |
| `2.1_i2v_720p` | Wan 2.1 I2V 720p | Non-MoE (1 DiT) | ~44 GB |

All variants download shared components (VAE ~254 MB + T5 ~11.4 GB) plus variant-specific DiT weights. Files that already exist in `models/` are skipped.

---

### 2. Prepare (`scripts/prepare.sh`)

Runs VAE latent pre-encoding followed by T5 text pre-encoding. This is done once and cached to disk so training does not repeat the expensive encoding step every epoch.

**Usage:**

```bash
bash scripts/prepare.sh --config path/to/train.yaml             # encode for a single config
bash scripts/prepare.sh --project path/to/project.yaml           # encode for a project (uses base_config)
bash scripts/prepare.sh --config path/to/train.yaml --dry-run    # preview without running
bash scripts/prepare.sh --config path/to/train.yaml --force      # re-encode even if cache exists
```

**Flags:**

| Flag | Required | Description |
|------|----------|-------------|
| `--config PATH` | One of these | Direct path to a training YAML |
| `--project PATH` | is required | Path to a project YAML (extracts `base_config` from it) |
| `--dry-run` | No | Preview what would be encoded without running |
| `--force` | No | Re-encode latents even if cached versions exist |
| `-h`, `--help` | No | Show help message |

Exactly one of `--config` or `--project` is required. When using `--project`, the script extracts the `base_config` path from the project YAML and uses that for encoding.

The `FLIMMER_VENV` environment variable can override the venv location (see [Environment Variables](#environment-variables) below).

---

### 3. Train (`scripts/train.sh`)

Launches training with either a single config or a project-based multi-phase workflow.

**Usage:**

```bash
# Single config
bash scripts/train.sh --config path/to/train.yaml
bash scripts/train.sh --config path/to/train.yaml --dry-run

# Project mode
bash scripts/train.sh --project path/to/project.yaml              # run next pending phase
bash scripts/train.sh --project path/to/project.yaml --all        # run all pending phases
bash scripts/train.sh --project path/to/project.yaml --status     # check phase progress
bash scripts/train.sh --project path/to/project.yaml --dry-run    # preview training plan

# Background session
bash scripts/train.sh --project path/to/project.yaml --all --tmux
```

**Flags:**

| Flag | Required | Description |
|------|----------|-------------|
| `--config PATH` | One of these | Direct path to a training YAML (single run) |
| `--project PATH` | is required | Path to a project YAML (multi-phase training) |
| `--tmux` | No | Start training in a named tmux session |
| `--all` | No | Run all pending phases (only with `--project`) |
| `--dry-run` | No | Show what would happen without training |
| `--status` | No | Show project phase statuses (only with `--project`) |
| `-h`, `--help` | No | Show help message |

Exactly one of `--config` or `--project` is required.

**tmux sessions:** When using `--tmux`, the script creates a tmux session named `flimmer-<config-name>`. If tmux is not installed, it prints a warning and runs in the foreground instead.

```bash
# Attach to a running session
tmux attach -t flimmer-project_moe
```

---

## Project-Based Training

A project YAML defines a multi-phase training workflow — typically unified warmup followed by per-expert specialization for Wan 2.2 MoE models. The project runner executes phases in order and tracks their status.

See `config_templates/projects/` for project configs — `i2v_moe_phases.yaml` for MoE fork-and-specialize, `t2v_phases.yaml` for multi-dataset progression.

**Basic commands:**

```bash
# Preview the fully resolved plan (verify your overrides are applied)
bash scripts/train.sh --project config_templates/projects/i2v_moe_phases.yaml --dry-run

# Run the next pending phase
bash scripts/train.sh --project config_templates/projects/i2v_moe_phases.yaml

# Run all pending phases sequentially
bash scripts/train.sh --project config_templates/projects/i2v_moe_phases.yaml --all

# Check which phases are completed/pending
bash scripts/train.sh --project config_templates/projects/i2v_moe_phases.yaml --status
```

You can also preview the plan directly via Python:

```bash
python -m flimmer.training plan --project project.yaml
```

**Important:** Always preview your plan before training. The `--dry-run` flag shows the fully resolved parameters (actual epochs, learning rates, etc.) after merging your project overrides with the base config. If the values look wrong, your overrides aren't being applied — check that `base_config` in your project YAML points to the right file.

Re-running a project skips completed phases and picks up where it left off. Phase status is tracked in a `flimmer_project.json` file alongside the project YAML. If you edit `project.yaml` after a run, Flimmer detects the change and re-reads from the YAML automatically.

For a conceptual overview of the phase system, see the [Phase System](../README.md#phase-system) section in the README.

---

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `FLIMMER_VENV` | `.venv` in repo root | Override venv location for `prepare.sh` and `train.sh` |

**Example:**

```bash
export FLIMMER_VENV=/home/user/my-venvs/flimmer
bash scripts/prepare.sh --config train.yaml
bash scripts/train.sh --config train.yaml
```

---

## Troubleshooting

### "No GPU detected" or "nvidia-smi not found"

Your NVIDIA drivers are not installed or not accessible. Check:

```bash
nvidia-smi                    # should show your GPU
python -c "import torch; print(torch.cuda.is_available())"   # should print True
```

If `nvidia-smi` works but PyTorch cannot see CUDA, you may need to reinstall PyTorch with CUDA support:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### "Virtual environment not found"

Run `setup.sh` first to create the venv, or set `FLIMMER_VENV` to point to an existing environment:

```bash
bash scripts/setup.sh --variant 2.2_i2v       # creates .venv/
# or
export FLIMMER_VENV=/path/to/existing/venv
```

### Weight download fails

- Check disk space — Wan 2.2 variants need ~69 GB free
- Check network connectivity
- For gated HuggingFace models, authenticate first: `huggingface-cli login`
- Use `--skip-downloads` to set up dependencies separately, then download weights manually

### "tmux not found"

`train.sh` runs in the foreground if tmux is not installed. This is expected — you will see a warning but training proceeds normally. Install tmux if you want background sessions:

```bash
sudo apt install tmux          # Debian/Ubuntu
sudo dnf install tmux          # Fedora
```
