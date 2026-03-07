"""CLI for the Flimmer training pipeline.

Commands:
    train       Run training from a config file
    plan        Print the resolved training plan without training (dry run)

Usage::

    # Full training run
    python -m flimmer.training train --config path/to/train.yaml

    # Dry run — resolve phases and print plan
    python -m flimmer.training plan --config path/to/train.yaml
"""

from __future__ import annotations

import argparse
import sys


def cmd_train(args: argparse.Namespace) -> None:
    """Run training from a config file."""
    from flimmer.config.training_loader import load_training_config
    from flimmer.training.loop import TrainingOrchestrator

    config = load_training_config(args.config)

    # Resolve model backend from config variant
    backend = _resolve_backend(config)

    # Optionally create inference pipeline for sampling
    inference_pipeline = None
    if config.sampling.enabled:
        inference_pipeline = _resolve_inference_pipeline(config)

    # Load cached dataset (unless dry run)
    dataset = None
    if not args.dry_run:
        dataset = _load_dataset(config)

    orchestrator = TrainingOrchestrator(
        config=config,
        model_backend=backend,
        inference_pipeline=inference_pipeline,
    )
    orchestrator.run(dataset=dataset, dry_run=args.dry_run)


def cmd_plan(args: argparse.Namespace) -> None:
    """Print the resolved training plan without training.

    Accepts either --config (single training config) or --project
    (project YAML with phase overrides). When given --project, delegates
    to the project plan command which merges overrides before resolving.
    """
    if getattr(args, "project", None):
        # Delegate to project plan — it merges overrides and resolves
        from flimmer.project.__main__ import cmd_plan as project_cmd_plan
        project_cmd_plan(args)
        return

    from flimmer.config.training_loader import load_training_config
    from flimmer.training.loop import TrainingOrchestrator

    config = load_training_config(args.config)

    # Plan mode uses stub backend — no GPU needed
    orchestrator = TrainingOrchestrator(
        config=config,
        model_backend=_StubBackend(),
        inference_pipeline=None,
    )
    orchestrator.run(dry_run=True)


def _load_dataset(config: object) -> object:
    """Load the CachedLatentDataset from the encoding cache.

    Reads the cache manifest and builds a map-style dataset that the
    training loop can iterate over.

    Args:
        config: FlimmerTrainingConfig instance.

    Returns:
        CachedLatentDataset ready for DataLoader.
    """
    from pathlib import Path
    from flimmer.encoding.cache import load_cache_manifest
    from flimmer.encoding.dataset import CachedLatentDataset

    cache_dir = Path(config.cache.cache_dir)
    print(f"Loading cached dataset from: {cache_dir}")

    manifest = load_cache_manifest(cache_dir)
    dataset = CachedLatentDataset(manifest=manifest, cache_dir=cache_dir)

    print(f"  Loaded: {len(dataset)} training samples")
    return dataset


def _resolve_backend(config: object) -> object:
    """Resolve the model backend from config.

    Routes to the appropriate backend based on model family or variant prefix:
    - family='ltx' or variant starting with '2.3' → LTX backend
    - Everything else → Wan backend (default)

    Falls back to stub if the required packages are not installed
    or the variant is unknown.

    Args:
        config: FlimmerTrainingConfig instance.

    Returns:
        ModelBackend implementation (LtxModelBackend, WanModelBackend,
        or _StubBackend).
    """
    model = getattr(config, "model", None)
    variant = getattr(model, "variant", None) if model else None
    family = getattr(model, "family", None) if model else None

    if variant is None:
        print(
            "Warning: No model variant set. Using stub backend.\n"
            "Set model.variant in your config (e.g. '2.2_t2v')."
        )
        return _StubBackend()

    # Route by family (explicit) or infer from variant prefix
    if family == "ltx" or (variant and variant.startswith("2.3")):
        try:
            from flimmer.training.ltx.registry import get_ltx_backend
            print(f"Loading LTX model backend for variant '{variant}'...")
            return get_ltx_backend(config)
        except ImportError as e:
            print(
                f"Warning: Cannot load LTX backend ({e}).\n"
                f"Install with: pip install 'flimmer[ltx]'\n"
                f"Falling back to stub backend."
            )
            return _StubBackend()
        except ValueError as e:
            print(f"Warning: {e}\nFalling back to stub backend.")
            return _StubBackend()

    # Default: Wan backend
    try:
        from flimmer.training.wan.registry import get_wan_backend
        print(f"Loading Wan model backend for variant '{variant}'...")
        return get_wan_backend(config)
    except ImportError as e:
        print(
            f"Warning: Cannot load Wan backend ({e}).\n"
            f"Install with: pip install 'flimmer[wan]'\n"
            f"Falling back to stub backend."
        )
        return _StubBackend()
    except ValueError as e:
        print(f"Warning: {e}\nFalling back to stub backend.")
        return _StubBackend()


def _resolve_inference_pipeline(config: object) -> object | None:
    """Create inference pipeline for sampling during training.

    Supports both individual safetensors files (model.vae, model.t5)
    and Diffusers directories (model.path). Individual file paths take
    priority, with model.path as fallback.

    Args:
        config: FlimmerTrainingConfig instance.

    Returns:
        WanInferencePipeline or None if not available.
    """
    model = getattr(config, "model", None)
    if model is None:
        return None

    # LTX inference pipeline not yet implemented
    family = getattr(model, "family", None)
    variant = getattr(model, "variant", "")
    if family == "ltx" or (variant and variant.startswith("2.3")):
        print("Note: LTX inference pipeline not yet available. Sampling disabled.")
        return None

    try:
        from flimmer.training.wan.inference import WanInferencePipeline

        is_i2v = getattr(model, "variant", "") == "2.2_i2v"
        dtype = getattr(
            getattr(config, "training", None),
            "mixed_precision",
            "bf16",
        )

        # Resolve component paths — individual files take priority
        vae_path = getattr(model, "vae", None)
        t5_path = getattr(model, "t5", None)
        diffusers_path = getattr(model, "path", None)

        # Need at least one loading method available
        has_individual = bool(vae_path and t5_path)
        has_diffusers = bool(diffusers_path)
        if not has_individual and not has_diffusers:
            print(
                "Warning: Cannot create inference pipeline — no model paths "
                "available for sampling.\n"
                "Set model.vae + model.t5 (individual files) or model.path "
                "(Diffusers directory) in your config."
            )
            return None

        return WanInferencePipeline(
            vae_path=vae_path,
            t5_path=t5_path,
            diffusers_path=diffusers_path,
            is_i2v=is_i2v,
            dtype=dtype,
        )
    except ImportError:
        return None


class _StubBackend:
    """Minimal stub backend for dry-run and plan commands.

    Satisfies the ModelBackend protocol with no-ops so the orchestrator
    can resolve phases and print plans without a real model.
    """

    @property
    def model_id(self) -> str:
        return "stub"

    @property
    def supports_moe(self) -> bool:
        return True

    @property
    def supports_reference_image(self) -> bool:
        return False

    def load_model(self, config):
        return None

    def get_lora_target_modules(self) -> list[str]:
        return []

    def get_expert_mask(self, timesteps, boundary_ratio):
        return (None, None)

    def prepare_model_inputs(self, batch, timesteps, noisy_latents):
        return {}

    def forward(self, model, **inputs):
        return None

    def setup_gradient_checkpointing(self, model):
        pass

    def get_noise_schedule(self):
        from flimmer.training.noise import FlowMatchingSchedule
        return FlowMatchingSchedule()


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the training CLI."""
    parser = argparse.ArgumentParser(
        prog="python -m flimmer.training",
        description="Flimmer training pipeline — video LoRA training with differential MoE.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ─── train ───
    train_parser = subparsers.add_parser(
        "train",
        help="Run training from a config file.",
    )
    train_parser.add_argument(
        "--config", "-c",
        required=True,
        help="Path to the Flimmer training config YAML.",
    )
    train_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve phases and print plan without training.",
    )

    # ─── plan ───
    plan_parser = subparsers.add_parser(
        "plan",
        help="Print resolved training plan (dry run).",
    )
    plan_group = plan_parser.add_mutually_exclusive_group(required=True)
    plan_group.add_argument(
        "--config", "-c",
        help="Path to the Flimmer training config YAML.",
    )
    plan_group.add_argument(
        "--project", "-p",
        help="Path to a project YAML (shows plan with phase overrides applied).",
    )

    return parser


def main() -> None:
    """Entry point for the training CLI."""
    parser = build_parser()
    args = parser.parse_args()

    commands = {
        "train": cmd_train,
        "plan": cmd_plan,
    }

    try:
        commands[args.command](args)
    except KeyboardInterrupt:
        print("\nTraining interrupted.", file=sys.stderr)
        sys.exit(130)
    except SystemExit:
        raise
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
