# Low VRAM Training Guide

Flimmer supports training Wan 2.2 14B LoRA on GPUs from 24GB to 80GB using a combination of quantization, block swapping, and optimizer memory reduction. This guide explains each optimization, its VRAM savings, and its tradeoffs so you can pick the right settings for your hardware.

---

## Quick Reference Table

The fast path: find your GPU tier, pick the matching template, and start training.

### T2V (Text-to-Video)

| GPU Tier | GPU Examples | Resolution | Max Frames | Precision | Block Swap | Optimizer | LoRA Rank | Template |
|----------|-------------|-----------|-----------|-----------|-----------|-----------|-----------|----------|
| 24GB | RTX 3090, RTX 4090 | 480p | 49 (33 for tight fit) | fp8 | 25 | adamw8bit | 16 | `t2v_wan22_24gb.yaml` |
| 48GB | A6000, RTX 6000, L40S | 720p | 49 | fp8 | 10 | adamw8bit | 16 | `t2v_wan22_48gb.yaml` |
| 80GB | H100, A100 | 720p | 81 | bf16 | 0 | adamw8bit | 16-32 | `t2v_wan22.yaml` |

### I2V (Image-to-Video)

| GPU Tier | GPU Examples | Resolution | Max Frames | Precision | Block Swap | Optimizer | LoRA Rank | Template |
|----------|-------------|-----------|-----------|-----------|-----------|-----------|-----------|----------|
| 24GB | RTX 3090, RTX 4090 | 480p | 33 | fp8 | 30 | adamw8bit | 16 (8 if tight) | `i2v_wan22_24gb.yaml` |
| 48GB | A6000, RTX 6000, L40S | 480p | 49 | fp8 | 15 | adamw8bit | 16 | `i2v_wan22_48gb.yaml` |
| 80GB | H100, A100 | 720p | 49-81 | fp8 | 0 | adamw8bit | 16 | `i2v_wan22.yaml` |

All templates are in `config_templates/training/`. Copy the one matching your GPU and mode, then customize model paths, data config, and output directory.

---

## Pre-Flight VRAM Check

Before starting a training run, verify your config fits on your GPU:

```bash
# Basic check (uses default GPU memory)
python -m flimmer.training estimate-vram --config your_train.yaml

# Check against a specific GPU size
python -m flimmer.training estimate-vram --config your_train.yaml --gpu-memory 24
```

The estimator breaks down VRAM usage by component:

```
VRAM Estimate for your_train.yaml
==================================
Model weights (fp8):         14.0 GB
LoRA parameters:              0.1 GB
Optimizer state (adamw8bit):  0.2 GB
Gradients:                    0.1 GB
Activations (480p, 49f):      4.5 GB
Block swap savings (-25):    -8.8 GB
----------------------------------
Estimated total:             10.1 GB
GPU memory:                  24.0 GB
Headroom:                    13.9 GB   OK
```

- **Model weights:** The frozen base model. fp8 halves this from ~28 GB to ~14 GB.
- **LoRA parameters:** Trainable adapter weights. Scales with rank.
- **Optimizer state:** Per-parameter momentum and variance. adamw8bit uses 2 bytes/param instead of 8.
- **Gradients:** Gradient tensors for trainable parameters.
- **Activations:** Forward pass intermediate values. Scales with resolution x frame count.
- **Block swap savings:** VRAM freed by offloading transformer blocks to CPU.

If the estimate shows you are over your GPU memory, increase `blocks_to_swap`, reduce resolution, or reduce frame count in your data config.

---

## Optimization Stack

Each optimization reduces VRAM in a different way. You can stack them — the savings are roughly additive.

### FP8 Quantization

```yaml
training:
  base_model_precision: fp8
```

Cuts model weight VRAM roughly in half (~28 GB to ~14 GB for Wan 14B). Only the frozen base model weights are quantized — LoRA parameters and gradients remain in full precision. Minimal quality impact for LoRA training since the quantization only affects frozen weights that you are not updating.

Required for 24GB and 48GB tiers. Optional on 80GB (T2V trains fine at bf16 on H100).

### Block Swapping

```yaml
training:
  blocks_to_swap: 25    # Number of transformer blocks to offload to CPU
```

Offloads N of the 40 transformer blocks to CPU during the forward and backward pass. Each block is ~175 MB at fp8 (~350 MB at bf16). Only one block is on GPU at a time during the swap region — the rest stay in CPU RAM.

VRAM savings: `N * ~175 MB` at fp8 (e.g., 25 blocks = ~4.4 GB weights + activation savings).

Tradeoff: ~15-30% slower training depending on PCIe bandwidth. RTX 4090 (PCIe 4.0 x16) has roughly 2x the bandwidth of RTX 3090 (PCIe 3.0 x16), so block swap overhead is lower on newer cards.

Recommended settings:

| GPU Tier | T2V | I2V |
|----------|-----|-----|
| 24GB | 25 | 30 |
| 48GB | 10 | 15 |
| 80GB | 0 | 0 |

The maximum useful value is 39 (one block must always remain on GPU). A warning is printed if you set more than 35.

### 8-bit Optimizer (adamw8bit)

```yaml
optimizer:
  type: adamw8bit
```

Reduces optimizer state from 8 bytes per parameter to 2 bytes per parameter using dynamic quantization. For a rank 16 LoRA, this saves ~300 MB. Minimal quality impact — adamw8bit is battle-tested across thousands of LoRA training runs.

This is the default in all Flimmer templates. No reason not to use it unless you need exact optimizer state reproducibility.

### CPU Offload Optimizer

```yaml
optimizer:
  type: cpu_offload
```

Moves ALL optimizer state to CPU RAM via torchao. This frees all optimizer VRAM — the GPU only holds gradients temporarily during the optimizer step.

Slightly slower step time compared to adamw8bit since optimizer state must be transferred to/from CPU each step. Use this if adamw8bit is not enough to fit your config.

Requires `torchao` package (optional dependency, installed automatically if available).

### Adam-Mini

```yaml
optimizer:
  type: adam_mini
```

45-50% optimizer state reduction via Hessian-aware partitioning. Adam-Mini groups parameters by their Hessian structure and shares momentum/variance within groups, dramatically reducing the number of optimizer states.

Requires the `adam-mini` package (`pip install adam-mini`). This is an optional dependency.

### Resolution and Frame Count

Configured in your data config (`flimmer_data.yaml`), not the training config:

```yaml
video:
  resolution: 480       # 480 or 720
  max_frames: 49        # 33, 49, or 81
```

Lower resolution and fewer frames dramatically reduce activation VRAM. These have the biggest impact on activation memory because activation size scales with `resolution^2 * frames`.

Going from 720p to 480p roughly halves activation memory. Going from 81 to 49 frames saves another ~40%.

---

## Stacking Optimizations

Each optimization stacks on top of the previous. This table shows the cumulative effect for T2V training:

| Starting Point | Optimization | Config Field | Approx VRAM | Saved |
|----------------|-------------|-------------|-------------|-------|
| Base bf16 model | None | `base_model_precision: bf16` | ~42 GB | - |
| + fp8 weights | Quantize frozen weights | `base_model_precision: fp8` | ~28 GB | ~14 GB |
| + block swap 20 | Offload 20 blocks to CPU | `blocks_to_swap: 20` | ~21 GB | ~7 GB |
| + 8-bit optimizer | Quantize optimizer state | `optimizer: adamw8bit` | ~20.5 GB | ~0.5 GB |
| + 480p resolution | Reduce activation memory | `video.resolution: 480` (data config) | ~14 GB | ~6.5 GB |
| + fewer frames | Reduce activation memory | `video.max_frames: 33` (data config) | ~11 GB | ~3 GB |

The key insight: **fp8 and resolution have the biggest individual impact.** Block swap and optimizer are smaller but essential for fitting on 24GB. Stack all four to go from "needs an H100" to "runs on an RTX 3090."

---

## I2V vs T2V Memory Differences

I2V training uses significantly more VRAM than T2V at the same resolution and frame count. Here is why:

**36-channel input vs 16-channel.** T2V feeds 16 channels of noisy video latents into the transformer. I2V feeds 36 channels: 16 noisy video latents + 16 first-frame latents + 4 mask channels. This means the input embedding layer processes ~2.25x more data.

**After the first layer, both modes are identical.** The transformer's hidden dimension is 5120 regardless of input channels. So the extra memory is concentrated in the input embedding and the first few layers, not spread across all 40 blocks.

**Net effect: I2V adds ~3-5 GB over T2V** at the same resolution. The exact amount depends on resolution, frame count, and batch size.

**Practical impact:**

- T2V at 720p fits on 48GB with moderate block swap. I2V at 720p on 48GB needs aggressive block swap (20-25) or dropping to 480p.
- I2V on 24GB needs `blocks_to_swap: 30` (vs 25 for T2V) and should use 480p / 33 frames.
- On 80GB, T2V can use bf16 (no quantization). I2V still needs fp8 because the extra channels push past 80GB at bf16.

---

## Per-GPU Tier Details

### 24GB (RTX 3090 / RTX 4090)

This is the most constrained tier. Every optimization must be active:

- **Precision:** `base_model_precision: fp8` (required)
- **Block swap:** 25 for T2V, 30 for I2V
- **Optimizer:** `adamw8bit` (default). Use `cpu_offload` if still tight.
- **Resolution:** 480p only
- **Frames:** 49 for T2V, 33 for I2V
- **LoRA rank:** 16 for T2V. 16 for I2V (drop to 8 if OOM).
- **Batch size:** 1 (always)

**RTX 4090 vs RTX 3090:** Both have 24GB, but the RTX 4090 has PCIe 4.0 x16 (~32 GB/s) vs the RTX 3090's PCIe 3.0 x16 (~16 GB/s). Block swap overhead is roughly 2x lower on the 4090, so training is noticeably faster despite the same VRAM.

Templates: `t2v_wan22_24gb.yaml`, `i2v_wan22_24gb.yaml`

### 48GB (A6000 / RTX 6000 Ada / L40S)

A comfortable tier with room for either higher resolution or lower block swap:

- **Precision:** `base_model_precision: fp8`
- **Block swap:** 10 for T2V, 15 for I2V
- **Optimizer:** `adamw8bit`
- **Resolution:** 720p for T2V, 480p for I2V. I2V can do 720p by increasing block swap to 20-25.
- **Frames:** 49
- **LoRA rank:** 16
- **Batch size:** 1

Training speed is close to 80GB since only 10-15 blocks are swapped (vs 25-30 on 24GB).

Templates: `t2v_wan22_48gb.yaml`, `i2v_wan22_48gb.yaml`

### 80GB (H100 / A100)

The default tier. No memory-saving tradeoffs needed for T2V:

- **Precision:** `bf16` for T2V (full precision, no quantization). `fp8` for I2V (36-channel input needs it even on 80GB).
- **Block swap:** 0 (none needed)
- **Optimizer:** `adamw8bit`
- **Resolution:** 720p
- **Frames:** Up to 81 for T2V, 49-81 for I2V (test with `estimate-vram`)
- **LoRA rank:** 16-32

Templates: `t2v_wan22.yaml`, `i2v_wan22.yaml`

---

## Troubleshooting OOM

### "I set blocks_to_swap but still OOM"

Increase the count. If you are at 25, try 30. If at 30, try 35. Also check your frame count in the data config — a single 81-frame clip can push you over the edge even with block swap.

### "Training starts fine but OOMs mid-training"

This usually means a large resolution/frame bucket was hit. The bucketing system groups clips by resolution and frame count. If one bucket is much larger than others (e.g., 81 frames while the rest are 49), it can spike activation memory.

Fix: delete the large cached latent files. Look in your `cache/latents/` directory for files with the larger frame count in the filename (e.g., `*81x*.safetensors`) and remove them. Training will skip those buckets.

### "How do I know my actual VRAM usage?"

Flimmer logs peak VRAM usage to the console during training via the VRAMTracker. Look for lines like:

```
[VRAMTracker] Peak VRAM: 21.3 GB / 24.0 GB (88.8%)
```

You can also run the `estimate-vram` command before training to get a component-by-component breakdown.

### "Can I use fp8_scaled (NF4) for even more savings?"

Yes. Setting `base_model_precision: fp8_scaled` uses 4-bit NF4 quantization via bitsandbytes, which further halves the model weight VRAM to ~7 GB. This is more aggressive than fp8 and may have a small quality impact, but it is still only quantizing the frozen base model weights — your LoRA parameters remain in full precision.

Use fp8_scaled if you need every last GB — for example, to fit I2V at higher resolution on 24GB, or to enable rank 32 on 48GB.

---

## Related Resources

- [I2V Training Guide](I2V_GUIDE.md) -- I2V-specific training workflows and first-frame handling
- [Training Config Templates](../config_templates/training/README.md) -- template selection guide with GPU tier table
- [Training Config Walkthrough](TRAINING_CONFIG_WALKTHROUGH.md) -- full config reference with field-by-field explanations
