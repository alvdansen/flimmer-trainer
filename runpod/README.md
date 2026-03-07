# Flimmer — RunPod Quickstart

Quick-start guide for training Wan 2.1/2.2 video LoRAs on RunPod.

## Setup

### 1. Create a RunPod Pod

- **GPU**: H100 80GB (recommended) or A100 80GB
- **Template**: RunPod PyTorch 2.x
- **Container Disk**: **50 GB** (default 20 GB will run out)
- **Volume Disk**: 200 GB (models + datasets + outputs)
- **Environment Variables**: Set `HF_TOKEN` to your HuggingFace token

### 2. Clone and Setup

Open a terminal in Jupyter Lab:

```bash
# Clone the repo
cd /workspace
git clone https://github.com/alvdansen/flimmer-trainer.git

# Run setup (installs packages, downloads models)
# Pick your variant:
bash /workspace/flimmer-trainer/runpod/setup.sh --variant 2.2_i2v       # Wan 2.2 I2V (~69 GB)
bash /workspace/flimmer-trainer/runpod/setup.sh --variant 2.2_t2v       # Wan 2.2 T2V (~69 GB)
bash /workspace/flimmer-trainer/runpod/setup.sh --variant 2.1_i2v_480p  # Wan 2.1 I2V 480p (~44 GB)
bash /workspace/flimmer-trainer/runpod/setup.sh --variant 2.1_i2v_720p  # Wan 2.1 I2V 720p (~44 GB)
```

### 3. Upload Your Dataset

Via Jupyter Lab file browser, upload to:

```
/workspace/datasets/my_dataset/
    clip_001.mp4          <- training video clips
    clip_001.txt          <- caption sidecar (same stem as video)
    clip_002.mp4
    clip_002.txt
    ...
```

You also need a `flimmer_data.yaml` in the dataset directory. Use a minimal one:

```yaml
# /workspace/datasets/my_dataset/flimmer_data.yaml
video:
  fps: 16
  resolution: 480
```

See `config_templates/data/` for more data config examples.

### 4. Create a Training Config

```bash
cp /workspace/flimmer-trainer/runpod/test-train.yaml /workspace/my_train.yaml
```

Edit `/workspace/my_train.yaml`:
- Set `data_config` to point to your dataset's `flimmer_data.yaml`
- Adjust epochs, learning rates, and output paths as needed
- Model file paths are already set for RunPod (`/workspace/models/...`)

See `config_templates/training/` for variant-specific config templates.

## Training

Always run training inside tmux (survives browser disconnects):

```bash
tmux new -s train
```

### Full Run (Encode + Train)

```bash
python /workspace/flimmer-trainer/runpod/train.py --config /workspace/my_train.yaml
```

This runs all three steps automatically:
1. **Cache latents** -- encode videos through VAE
2. **Cache text** -- encode captions through T5
3. **Train** -- run the flimmer training loop

### Dry Run (Validate Config)

```bash
python /workspace/flimmer-trainer/runpod/train.py --config /workspace/my_train.yaml --dry-run
```

Validates your config and prints the training plan without using the GPU.

### Encode Only

```bash
python /workspace/flimmer-trainer/runpod/train.py --config /workspace/my_train.yaml --encode-only
```

Build latent and text caches without starting training. Useful for
verifying encoding works before committing to a long training run.

### Skip Encoding

```bash
python /workspace/flimmer-trainer/runpod/train.py --config /workspace/my_train.yaml --skip-encoding
```

Skip encoding steps (use existing caches). Useful when re-running
training with different hyperparameters on the same dataset.

## Download Results

Results are saved to `/workspace/outputs/` (or wherever `save.output_dir` points).
Download via:
- Jupyter Lab file browser
- `scp -P PORT root@HOST:/workspace/outputs/*.safetensors .`

## After Pod Restart

Run setup again to reinstall Python packages (models stay cached on `/workspace`):

```bash
bash /workspace/flimmer-trainer/runpod/setup.sh --variant <your-variant>
```

## Available Variants

| Variant | Model | Architecture | Download Size |
|---------|-------|-------------|---------------|
| `2.2_t2v` | Wan 2.2 Text-to-Video | MoE (2 expert DiTs) | ~69 GB |
| `2.2_i2v` | Wan 2.2 Image-to-Video | MoE (2 expert DiTs) | ~69 GB |
| `2.1_i2v_480p` | Wan 2.1 I2V 480p | Non-MoE (1 DiT) | ~44 GB |
| `2.1_i2v_720p` | Wan 2.1 I2V 720p | Non-MoE (1 DiT) | ~44 GB |

## Default Hyperparameters

All training configuration lives in the YAML config file. See
`config_templates/training/` for the full reference with all options documented.

Key defaults for fork-and-specialize MoE training:

| Setting | Unified | High-Noise Expert | Low-Noise Expert |
|---------|---------|-------------------|------------------|
| Learning Rate | 5e-5 | 1e-4 | 8e-5 |
| Epochs | 15 | 30 | 50 |
| LoRA Rank | 16 | 16 | 16 |
| LoRA Alpha | 16 | 16 | 16 |

Shared: adamw8bit optimizer, cosine_with_min_lr scheduler, 0.01 weight decay, seed 42.
