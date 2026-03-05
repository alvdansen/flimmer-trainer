# Training Config Templates

A training config is everything needed for one complete training run — model paths, LoRA settings, optimizer, epochs, checkpoints. This is the main config file you'll work with.

**You do NOT need a project config.** A training config + data config is the simplest path to a trained LoRA:

```bash
bash scripts/prepare.sh --config my_train.yaml
bash scripts/train.sh --config my_train.yaml
```

## Which Template?

| Template | Model | GPU Tier | Best for |
|---|---|---|---|
| `t2v_wan22.yaml` | Wan 2.2 T2V 14B | 80GB | Full reference — every option documented with comments |
| `i2v_wan22.yaml` | Wan 2.2 I2V 14B | 80GB | I2V with MoE expert fork |
| `t2v_wan22_48gb.yaml` | Wan 2.2 T2V 14B | 48GB | T2V on A6000 / RTX 6000 — fp8 + moderate block swap |
| `i2v_wan22_48gb.yaml` | Wan 2.2 I2V 14B | 48GB | I2V on A6000 / RTX 6000 — fp8 + moderate block swap |
| `t2v_wan22_24gb.yaml` | Wan 2.2 T2V 14B | 24GB | T2V on RTX 3090/4090 — fp8 + aggressive block swap |
| `i2v_wan22_24gb.yaml` | Wan 2.2 I2V 14B | 24GB | I2V on RTX 3090/4090 — fp8 + aggressive block swap |
| `i2v_wan21.yaml` | Wan 2.1 I2V 14B | 80GB | Simplest — no expert fork, one DiT file |

**Tip:** `t2v_wan22.yaml` is the most complete reference config. Even if you're training I2V, it's worth reading for the comments explaining each field.

## GPU Tier Selection

Pick the template matching your GPU's VRAM:

- **80GB (H100, A100):** Use the default templates (`t2v_wan22.yaml`, `i2v_wan22.yaml`). No quantization needed for T2V; fp8 for I2V. Full 720p resolution, up to 81 frames.
- **48GB (A6000, RTX 6000 Ada, L40S):** Use `*_48gb.yaml` templates. fp8 quantization + 10-15 blocks swapped to CPU. 720p for T2V, 480p for I2V (720p possible with more block swap).
- **24GB (RTX 3090, RTX 4090):** Use `*_24gb.yaml` templates. fp8 + 25-30 blocks swapped to CPU + 480p resolution. Training works but is slower due to block swap overhead.

Not sure if your config fits? Run the VRAM estimator before training:

```bash
python -m flimmer.training estimate-vram --config your_config.yaml --gpu-memory 24
```

For a detailed breakdown of each optimization and its VRAM savings, see [docs/LOW_VRAM_GUIDE.md](../../docs/LOW_VRAM_GUIDE.md).

## What to Customize

After copying a template, you'll typically change:

1. **`model.dit` / `model.dit_high` / `model.dit_low`** — paths to your model weight files
2. **`model.vae` / `model.t5`** — paths to shared components
3. **`data_config`** — path to your data config YAML
4. **`save.output_dir`** — where checkpoints are saved
5. **`training.unified_epochs`** — how many epochs to train
6. **`lora.rank`** — LoRA capacity (16 is a good default, 32 for complex subjects)

Everything else has reasonable defaults.

## MoE Fork-and-Specialize

Wan 2.2 models have two expert DiTs — one for high-noise timesteps (composition, motion) and one for low-noise (detail, texture). The `moe:` section controls whether they get separate LoRAs:

- **`fork_enabled: true`** (default in MoE templates) — after the unified phase, the LoRA splits into per-expert copies with independent hyperparameters. This is the recommended approach for MoE models.
- **`fork_enabled: false`** — train both experts together with shared weights for the entire run. Simpler, fewer knobs, but the experts can't specialize independently.

The fork happens automatically at the end of `unified_epochs`. No project config needed — the training config handles it.

**When do you need a project config for MoE?** Only if you want manual control over the phase sequence, need different datasets per phase, or want to resume individual phases independently. For standard fork-and-specialize, the training config's `moe:` section is sufficient.

## Per-Expert Overrides

In the `moe:` section, each expert can override epoch count and other training parameters:

```yaml
moe:
  high_noise:
    max_epochs: 30
  low_noise:
    max_epochs: 50
```

Any field not overridden inherits from the main optimizer/training sections. You can also override `learning_rate`, `batch_size`, `caption_dropout_rate`, `weight_decay`, and more per expert. See `t2v_wan22.yaml` for the full list of overridable fields.

## Adapting for Your Model

The templates are organized by model variant, but you can adapt any of them:

- **Wan 2.1 T2V:** Copy `i2v_wan21.yaml`, change `variant` to your 2.1 T2V variant, remove I2V-specific comments.
- **Different resolution:** Change `model.dit` path and `video.resolution` in your data config. Training config doesn't need resolution changes — bucketing handles it automatically.
- **Non-MoE on MoE template:** Set `moe.fork_enabled: false` and `moe.enabled: false`. Remove the `high_noise`/`low_noise` sections.
