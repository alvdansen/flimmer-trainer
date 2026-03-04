# Weights & Biases Setup

Track your training runs with W&B — loss curves, VRAM usage, sample videos, and full config snapshots all in one dashboard.

## Quick Setup

1. Create a free account at [wandb.ai](https://wandb.ai)
2. Get your API key from [wandb.ai/authorize](https://wandb.ai/authorize)
3. Log in (one time):

```bash
wandb login
# paste your API key when prompted
```

That's it. W&B is already installed as a Flimmer dependency.

## Enable in Your Config

Add `wandb` to your logging backends and set a project name:

```yaml
logging:
  backends: [console, wandb]
  wandb_project: flimmer-training     # required — your W&B project name
  log_every_n_steps: 10
```

## Optional Settings

```yaml
logging:
  backends: [console, wandb]
  wandb_project: flimmer-training
  wandb_entity: my-team               # team/org name (default: your personal account)
  wandb_run_name: holly_i2v_v3        # custom run name (default: auto-generated)
  wandb_group: holly_experiments      # group related runs together
  wandb_tags: [i2v, wan22, rank16]    # tags for filtering in the dashboard
  log_every_n_steps: 10
  vram_sample_interval: 50            # log GPU VRAM usage every N steps
```

## What Gets Logged

- **Loss curves** — per-step and per-epoch, broken out by phase (full_noise, high_noise, low_noise)
- **Learning rate** — tracks scheduler progression
- **VRAM usage** — GPU memory over time (if `vram_sample_interval` is set)
- **Sample videos** — generated during training (if `sampling.enabled: true`)
- **Full config snapshot** — your resolved training config in the W&B config tab
- **Phase transitions** — clear markers when training moves between phases

## RunPod / Headless Servers

On a headless machine (RunPod, Lambda, etc.), set your API key as an environment variable instead of interactive login:

```bash
export WANDB_API_KEY=your_key_here
```

Or add it to your pod's environment variables in the RunPod dashboard so it persists across restarts.

## Offline Mode

Training somewhere without internet? W&B can log locally and sync later:

```bash
export WANDB_MODE=offline
# ... run training ...

# When you have internet again:
wandb sync ./wandb/offline-run-*
```

## Dashboard Tips

- Use **Groups** (`wandb_group`) to cluster runs by dataset or experiment — e.g., all runs training "Holly" together
- Use **Tags** to filter — e.g., find all `rank16` runs or all `i2v` runs
- Pin the loss chart and sample videos panel for quick comparison across runs
- The config tab shows every setting that was active for that run, so you can always reproduce it
