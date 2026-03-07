"""
Flimmer — RunPod Training Runner
=================================
Thin wrapper that sequences flimmer encoding and training on RunPod.

Three steps:
  1. Cache latents  — encode videos through VAE (one-time, skippable)
  2. Cache text     — encode captions through T5 (one-time, skippable)
  3. Train          — run the flimmer training loop

All training configuration lives in the YAML config file. This script
just validates prerequisites and calls the flimmer CLI modules in order.

Usage:
  # Full run (encode + train)
  python /workspace/flimmer-trainer/runpod/train.py --config /workspace/my_train.yaml

  # Skip encoding (latents and text already cached)
  python /workspace/flimmer-trainer/runpod/train.py --config /workspace/my_train.yaml --skip-encoding

  # Dry run (validate config + print plan, no GPU work)
  python /workspace/flimmer-trainer/runpod/train.py --config /workspace/my_train.yaml --dry-run

  # Encoding only (cache latents + text, no training)
  python /workspace/flimmer-trainer/runpod/train.py --config /workspace/my_train.yaml --encode-only
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Reduce CUDA memory fragmentation for large models (14B+ params).
# Without this, gradient checkpointing recompute can OOM on 81-frame samples
# even with 95GB VRAM due to reserved-but-unallocated memory blocks.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


# =============================================================================
# Validation
# =============================================================================

def validate_environment() -> list[str]:
    """Check that the RunPod environment is ready for training.

    Returns a list of error messages. Empty list = all good.
    """
    errors: list[str] = []

    # Check we're on a machine with a GPU
    try:
        import torch
        if not torch.cuda.is_available():
            errors.append(
                "No GPU detected. Training requires a CUDA-capable GPU.\n"
                "  Check that you selected a GPU pod template on RunPod."
            )
        else:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"  GPU: {gpu_name} ({gpu_mem:.0f} GB)")
    except ImportError:
        errors.append(
            "PyTorch not installed. Run setup.sh first:\n"
            "  bash /workspace/flimmer-trainer/runpod/setup.sh --variant <variant>"
        )

    # Check flimmer is importable
    try:
        import flimmer
    except ImportError:
        errors.append(
            "flimmer not installed. Run setup.sh first:\n"
            "  bash /workspace/flimmer-trainer/runpod/setup.sh --variant <variant>"
        )

    return errors


def validate_config(config_path: str) -> None:
    """Load and validate the training config. Prints a summary.

    Exits with an error message if the config is invalid or if
    referenced model files don't exist.

    Args:
        config_path: Path to the Flimmer training YAML config.
    """
    from flimmer.config.training_loader import load_training_config

    try:
        config = load_training_config(config_path)
    except Exception as e:
        print(f"\nERROR: Invalid training config: {e}")
        sys.exit(1)

    # Check model files exist
    missing: list[str] = []
    model = config.model

    # Individual weight files
    for field, label in [
        ("dit", "DiT (non-MoE)"),
        ("dit_high", "DiT high-noise expert"),
        ("dit_low", "DiT low-noise expert"),
        ("vae", "VAE"),
        ("t5", "T5 text encoder"),
    ]:
        path = getattr(model, field, None)
        if path is not None and not Path(path).is_file():
            missing.append(f"  {label}: {path}")

    # Diffusers directory fallback
    if model.path and not any([model.dit, model.dit_high, model.dit_low]):
        p = Path(model.path)
        if not p.is_dir() and "/" not in model.path:
            missing.append(f"  Model path: {model.path}")

    if missing:
        print("\nERROR: Model files not found:")
        for m in missing:
            print(m)
        print("\nRun setup.sh to download models:")
        print("  bash /workspace/flimmer-trainer/runpod/setup.sh --variant <variant>")
        sys.exit(1)

    # Check data_config exists
    if not Path(config.data_config).is_file():
        print(f"\nERROR: Data config not found: {config.data_config}")
        print("Create a dataset config or update data_config in your training YAML.")
        sys.exit(1)

    # Print summary
    print(f"\n  Config:    {config_path}")
    print(f"  Variant:   {model.variant}")

    # Show model source
    if model.dit_high and model.dit_low:
        print(f"  DiT high:  {model.dit_high}")
        print(f"  DiT low:   {model.dit_low}")
    elif model.dit:
        print(f"  DiT:       {model.dit}")
    elif model.path:
        print(f"  Model:     {model.path}")

    if model.vae:
        print(f"  VAE:       {model.vae}")
    if model.t5:
        print(f"  T5:        {model.t5}")

    print(f"  Data:      {config.data_config}")
    print(f"  Output:    {config.save.output_dir}")

    # Training summary
    training = config.training
    lora = config.lora
    print(f"  Rank:      {lora.rank} (alpha {lora.alpha})")
    print(f"  Precision: {training.mixed_precision}")

    # MoE info
    moe = config.moe
    if moe.enabled and moe.fork_enabled:
        print(f"  Mode:      fork-and-specialize (MoE)")
        print(f"  Unified:   {training.unified_epochs} epochs")
        if moe.high_noise:
            hn = moe.high_noise
            print(f"  High:      LR {hn.learning_rate or config.optimizer.learning_rate}, "
                  f"{hn.max_epochs} epochs")
        if moe.low_noise:
            ln = moe.low_noise
            print(f"  Low:       LR {ln.learning_rate or config.optimizer.learning_rate}, "
                  f"{ln.max_epochs} epochs")
    elif moe.enabled:
        print(f"  Mode:      unified only (MoE, no fork)")
        print(f"  Epochs:    {training.unified_epochs}")
    else:
        print(f"  Mode:      single transformer")
        print(f"  Epochs:    {training.unified_epochs}")


# =============================================================================
# Step runners
# =============================================================================

def run_cache_latents(config_path: str, dry_run: bool = False) -> None:
    """Run latent pre-encoding (VAE) in a subprocess.

    Running encoding as a separate process ensures the OS fully reclaims
    all GPU memory when encoding finishes. This is critical because Python's
    gc.collect() + torch.cuda.empty_cache() cannot guarantee full VRAM
    reclamation — leaked references or fragmentation can leave the GPU
    partially occupied. Subprocess isolation is the safest approach.
    """
    import subprocess
    cmd = [
        sys.executable, "-m", "flimmer.encoding", "cache-latents",
        "--config", config_path,
    ]
    if dry_run:
        cmd.append("--dry-run")

    result = subprocess.run(cmd, env={**os.environ, "PYTHONUNBUFFERED": "1"})
    if result.returncode != 0:
        print(f"\nERROR: Latent encoding failed (exit code {result.returncode})")
        sys.exit(result.returncode)


def run_cache_text(config_path: str, dry_run: bool = False) -> None:
    """Run text pre-encoding (T5) in a subprocess.

    Same rationale as run_cache_latents — subprocess isolation ensures
    T5's ~10GB VRAM is fully reclaimed before training loads the ~27GB DiT.
    """
    import subprocess
    cmd = [
        sys.executable, "-m", "flimmer.encoding", "cache-text",
        "--config", config_path,
    ]
    if dry_run:
        cmd.append("--dry-run")

    result = subprocess.run(cmd, env={**os.environ, "PYTHONUNBUFFERED": "1"})
    if result.returncode != 0:
        print(f"\nERROR: Text encoding failed (exit code {result.returncode})")
        sys.exit(result.returncode)


def run_training(config_path: str, dry_run: bool = False) -> None:
    """Run the flimmer training loop."""
    from argparse import Namespace
    from flimmer.training.__main__ import cmd_train

    args = Namespace(config=config_path, dry_run=dry_run)
    cmd_train(args)


def run_plan(config_path: str) -> None:
    """Print the resolved training plan (no GPU needed)."""
    from argparse import Namespace
    from flimmer.training.__main__ import cmd_plan

    cmd_plan(Namespace(config=config_path))


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Flimmer — RunPod Training Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full run: encode + train
  python train.py --config /workspace/my_train.yaml

  # Skip encoding (caches already built)
  python train.py --config /workspace/my_train.yaml --skip-encoding

  # Encoding only (build caches, don't train)
  python train.py --config /workspace/my_train.yaml --encode-only

  # Dry run (validate config, print plan)
  python train.py --config /workspace/my_train.yaml --dry-run
        """,
    )

    parser.add_argument(
        "--config", "-c",
        required=True,
        help="Path to the Flimmer training config YAML.",
    )
    parser.add_argument(
        "--skip-encoding",
        action="store_true",
        help="Skip latent and text encoding (use existing caches).",
    )
    parser.add_argument(
        "--encode-only",
        action="store_true",
        help="Only run encoding (cache latents + text), skip training.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and print plan. No GPU work.",
    )

    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("  Flimmer — RunPod Training Runner")
    print("=" * 60)

    # --- Validate environment ---
    print("\nChecking environment...")
    if not args.dry_run:
        env_errors = validate_environment()
        if env_errors:
            print("\nEnvironment errors:")
            for err in env_errors:
                print(f"  {err}")
            sys.exit(1)
    else:
        print("  Dry run — skipping GPU check")

    # --- Validate config ---
    print("\nValidating config...")
    validate_config(args.config)

    # --- Dry run: just print the plan ---
    if args.dry_run:
        print(f"\n{'=' * 60}")
        print("  Training Plan (dry run)")
        print(f"{'=' * 60}\n")
        run_plan(args.config)
        print("\nDry run complete. No GPU work was done.")
        return

    # --- Step 1: Cache latents ---
    if not args.skip_encoding:
        print(f"\n{'=' * 60}")
        print("  Step 1/3: Caching latents (VAE encoding)")
        print(f"{'=' * 60}\n")
        t0 = time.time()
        run_cache_latents(args.config)
        elapsed = time.time() - t0
        print(f"\n  Latent encoding took {elapsed / 60:.1f} minutes")
    else:
        print("\n  Skipping latent encoding (--skip-encoding)")

    # --- Step 2: Cache text ---
    if not args.skip_encoding:
        print(f"\n{'=' * 60}")
        print("  Step 2/3: Caching text (T5 encoding)")
        print(f"{'=' * 60}\n")
        t0 = time.time()
        run_cache_text(args.config)
        elapsed = time.time() - t0
        print(f"\n  Text encoding took {elapsed / 60:.1f} minutes")
    else:
        print("\n  Skipping text encoding (--skip-encoding)")

    # --- Stop here if encode-only ---
    if args.encode_only:
        print(f"\n{'=' * 60}")
        print("  Encoding complete (--encode-only)")
        print(f"{'=' * 60}")
        print("\n  Caches are ready. Run again without --encode-only to train.")
        return

    # --- Step 3: Train ---
    print(f"\n{'=' * 60}")
    print("  Step 3/3: Training")
    print(f"{'=' * 60}\n")
    t0 = time.time()
    run_training(args.config)
    elapsed = time.time() - t0

    print(f"\n{'=' * 60}")
    print(f"  Training complete! ({elapsed / 3600:.1f} hours)")
    print(f"{'=' * 60}")
    print("\n  Download results from /workspace/outputs/ via Jupyter Lab.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.", file=sys.stderr)
        sys.exit(130)
    except SystemExit:
        raise
    except Exception as e:
        print(f"\nFatal error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
