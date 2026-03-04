"""Multi-backend training logger with phase awareness.

Provides consistent logging across console, TensorBoard, and W&B.
Each backend is optional — missing libraries are handled gracefully.

Key features:
    - Phase-prefixed logging (unified/, high_noise/, low_noise/)
    - W&B define_metric() for per-phase panels (METR-01, METR-04)
    - Auto-descriptive run naming from config
    - VRAM metric logging (METR-03)
    - End-of-run summary with timing, loss, VRAM (METR-02)
    - Full resolved config save to disk and W&B config tab
    - Training plan pretty-print before training starts
    - Phase transition events (start, end, fork)
    - GPU-free (mocked backends for testing)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml

from flimmer.training.phase import PhaseType, TrainingPhase


def generate_run_name(config: Any) -> str:
    """Auto-generate a descriptive W&B run name from training config.

    Format: {variant}-{dataset_name}-{mode}-r{rank}-lr{lr}

    Examples:
        wan22t2v-holly-fork-r16-lr1e-04
        wan22t2v-default-unified-r16-lr5e-05
        wan22t2v-annika-expert-r24-lr8e-05

    Args:
        config: FlimmerTrainingConfig instance.

    Returns:
        A human-readable run name string.
    """
    parts: list[str] = []

    # Model variant — compact form: family + variant with dots/underscores removed.
    # e.g. family="wan" + variant="2.2_t2v" -> "wan22t2v"
    model = getattr(config, "model", None)
    family = getattr(model, "family", "") or ""
    variant = getattr(model, "variant", "") or ""
    variant_compact = variant.replace(".", "").replace("_", "")
    # Avoid duplicating family if variant already starts with it
    if variant_compact.startswith(family):
        parts.append(variant_compact)
    else:
        parts.append(f"{family}{variant_compact}")

    # Dataset name — from save.name, replacing the default with "default"
    save_name = getattr(getattr(config, "save", None), "name", "default") or "default"
    parts.append(save_name.replace("flimmer_lora", "default"))

    # Training mode — fork, expert, or unified
    moe = getattr(config, "moe", None)
    training = getattr(config, "training", None)
    moe_enabled = getattr(moe, "enabled", False) if moe else False
    fork_enabled = getattr(moe, "fork_enabled", False) if moe else False
    unified_epochs = getattr(training, "unified_epochs", 0) if training else 0

    if moe_enabled and fork_enabled:
        if unified_epochs > 0:
            parts.append("fork")
        else:
            parts.append("expert")
    else:
        parts.append("unified")

    # LoRA rank
    lora = getattr(config, "lora", None)
    rank = getattr(lora, "rank", 16) if lora else 16
    parts.append(f"r{rank}")

    # Learning rate in scientific notation
    optimizer = getattr(config, "optimizer", None)
    lr = getattr(optimizer, "learning_rate", 5e-5) if optimizer else 5e-5
    lr_str = f"{lr:.0e}".replace("+", "")
    parts.append(f"lr{lr_str}")

    return "-".join(parts)


def save_resolved_config(config: Any, output_dir: Path) -> Path:
    """Save the fully resolved training config as YAML to disk.

    This ensures every training run can be reproduced from its saved
    config — no ambiguity about what defaults were applied.

    Args:
        config: FlimmerTrainingConfig instance (must support model_dump()).
        output_dir: Directory to save the config file into.

    Returns:
        Path to the saved resolved_config.yaml file.
    """
    config_dict = config.model_dump() if hasattr(config, "model_dump") else {}
    config_path = Path(output_dir) / "resolved_config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
    return config_path


class TrainingLogger:
    """Multi-backend logger for the training loop.

    Initializes requested backends and dispatches log calls to all active
    ones. Console is always available; TensorBoard and W&B are optional.

    Args:
        backends: List of backend names ('console', 'tensorboard', 'wandb').
        output_dir: Directory for TensorBoard logs.
        wandb_project: W&B project name (required if 'wandb' in backends).
        wandb_entity: W&B team/org name.
        wandb_run_name: W&B run name (None = auto-generated externally).
        log_every_n_steps: How often to log metrics.
        wandb_group: W&B run group for clustering related runs.
        wandb_tags: W&B tags for filtering runs.
        resolved_config: Full resolved config dict for W&B config tab.
        wandb_run_id: W&B run ID for resume continuity. None = new run,
            saved ID = resume existing run (appends to same loss curve).
    """

    def __init__(
        self,
        backends: list[str] | None = None,
        output_dir: str | Path | None = None,
        wandb_project: str | None = None,
        wandb_entity: str | None = None,
        wandb_run_name: str | None = None,
        log_every_n_steps: int = 10,
        wandb_group: str | None = None,
        wandb_tags: list[str] | None = None,
        resolved_config: dict[str, Any] | None = None,
        wandb_run_id: str | None = None,
    ) -> None:
        self._backends: list[str] = backends or ["console"]
        self._output_dir = Path(output_dir) if output_dir else None
        self._log_every_n_steps = log_every_n_steps
        self._step_count = 0

        # Initialize backends
        self._tb_writer: Any = None
        self._wandb_run: Any = None
        self._wandb_run_id: str | None = None
        self._wandb_log_warned: bool = False
        self._tb_close_warned: bool = False
        self._wandb_close_warned: bool = False

        if "tensorboard" in self._backends and self._output_dir is not None:
            self._init_tensorboard()

        if "wandb" in self._backends:
            self._init_wandb(
                project=wandb_project,
                entity=wandb_entity,
                run_name=wandb_run_name,
                group=wandb_group,
                tags=wandb_tags or [],
                resolved_config=resolved_config,
                wandb_run_id=wandb_run_id,
            )

    def _init_tensorboard(self) -> None:
        """Initialize TensorBoard writer (optional import)."""
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_dir = self._output_dir / "tensorboard"  # type: ignore[operator]
            tb_dir.mkdir(parents=True, exist_ok=True)
            self._tb_writer = SummaryWriter(log_dir=str(tb_dir))
        except ImportError:
            print(
                "Warning: tensorboard not installed. "
                "Install with: pip install tensorboard",
                file=sys.stderr,
            )

    def _init_wandb(
        self,
        project: str | None,
        entity: str | None,
        run_name: str | None,
        group: str | None = None,
        tags: list[str] | None = None,
        resolved_config: dict[str, Any] | None = None,
        wandb_run_id: str | None = None,
    ) -> None:
        """Initialize W&B run with define_metric() for per-phase panels.

        Sets up W&B with proper metric organization so each phase type
        (unified, high_noise, low_noise) gets its own panel in the W&B
        dashboard. Also configures system metrics (VRAM) with appropriate
        summary aggregations.

        When wandb_run_id is provided (from a saved TrainingState), W&B
        resumes the existing run — appending new metrics to the same loss
        curve instead of creating a new run. The resume="allow" mode
        creates a new run if the ID doesn't exist (fresh start) and
        resumes if it does (interrupted resume).

        This is called once at logger creation and must complete before
        any wandb.log() calls happen in the training loop.
        """
        try:
            import wandb
            self._wandb_run = wandb.init(
                project=project or "flimmer",
                entity=entity,
                name=run_name,
                group=group,
                tags=tags or [],
                reinit=True,
                id=wandb_run_id,
                resume="allow",
            )

            # Store the actual run ID so the orchestrator can save it
            # to TrainingState for future resume.
            self._wandb_run_id = (
                wandb.run.id if wandb.run else None
            )

            # Log resolved config to W&B config tab for reproducibility.
            # allow_val_change=True is needed for resume: the config dict
            # may differ in key ordering or include defaults (e.g.
            # optimizer_args: {}) that weren't present in the original run.
            if resolved_config and self._wandb_run is not None:
                wandb.config.update(resolved_config, allow_val_change=True)

            # Define metric axes — each phase prefix gets its own W&B panel.
            # This is the key to METR-01 (per-phase loss curves) and METR-04
            # (per-expert divergence tracking).
            run = self._wandb_run
            if run is not None:
                run.define_metric("global_step")

                # Per-phase metrics with their own panels
                run.define_metric("unified/*", step_metric="global_step")
                run.define_metric("unified/loss_ema", summary="min")

                run.define_metric("high_noise/*", step_metric="global_step")
                run.define_metric("high_noise/loss_ema", summary="min")

                run.define_metric("low_noise/*", step_metric="global_step")
                run.define_metric("low_noise/loss_ema", summary="min")

                # System metrics (VRAM) — track maximum as summary
                run.define_metric("system/*", step_metric="global_step")
                run.define_metric(
                    "system/vram_allocated_gb", summary="max",
                )

        except ImportError:
            print(
                "Warning: wandb not installed. "
                "Install with: pip install wandb",
                file=sys.stderr,
            )

    @property
    def wandb_run_id(self) -> str | None:
        """The active W&B run ID (for saving to TrainingState).

        Returns None if W&B is not active. On resume, this returns the
        same ID that was passed in — on fresh start, it returns the
        newly assigned ID from W&B.
        """
        return self._wandb_run_id

    def print_training_plan(self, phases: list[TrainingPhase]) -> None:
        """Pretty-print the resolved training plan before training starts.

        Shows each phase with its key parameters so the user can review
        the setup before committing GPU time.

        Args:
            phases: List of resolved TrainingPhase objects.
        """
        print("\n" + "=" * 60)
        print("  FLIMMER TRAINING PLAN")
        print("=" * 60)

        for i, phase in enumerate(phases):
            phase_label = phase.phase_type.value.upper()
            if phase.active_expert:
                phase_label += f" (expert: {phase.active_expert})"

            print(f"\n  Phase {i + 1}: {phase_label}")
            print(f"  {'─' * 40}")
            print(f"    Epochs:        {phase.max_epochs}")
            print(f"    Learning rate:  {phase.learning_rate:.2e}")
            print(f"    Optimizer:      {phase.optimizer_type}")
            print(f"    Scheduler:      {phase.scheduler_type}")
            print(f"    Batch size:     {phase.batch_size}")
            print(f"    Grad accum:     {phase.gradient_accumulation_steps}")
            print(f"    Caption dropout: {phase.caption_dropout_rate:.1%}")
            if hasattr(phase, 'first_frame_dropout_rate') and phase.first_frame_dropout_rate > 0:
                print(f"    First-frame dropout: {phase.first_frame_dropout_rate:.1%}")
            print(f"    LoRA dropout:   {phase.lora_dropout:.1%}")

            if phase.fork_targets:
                print(f"    Fork targets:   {', '.join(phase.fork_targets)}")
            if phase.block_targets:
                print(f"    Block targets:  {phase.block_targets}")
            if phase.boundary_ratio is not None:
                print(f"    Boundary ratio: {phase.boundary_ratio}")
            if phase.resume_from:
                print(f"    Resume from:    {phase.resume_from}")

        print(f"\n  Total phases: {len(phases)}")
        total_epochs = sum(p.max_epochs for p in phases)
        print(f"  Total epochs: {total_epochs}")
        print("=" * 60 + "\n")

    def log_phase_start(self, phase: TrainingPhase, phase_index: int) -> None:
        """Log the start of a training phase.

        Args:
            phase: The phase starting.
            phase_index: 0-based index into the phase list.
        """
        label = phase.phase_type.value.upper()
        msg = f"Starting phase {phase_index + 1}: {label} ({phase.max_epochs} epochs)"

        if "console" in self._backends:
            print(f"\n{'─' * 50}")
            print(f"  {msg}")
            print(f"  LR: {phase.learning_rate:.2e} | Optimizer: {phase.optimizer_type}")
            print(f"{'─' * 50}")

    def log_phase_end(self, phase: TrainingPhase, phase_index: int) -> None:
        """Log the end of a training phase.

        Args:
            phase: The phase ending.
            phase_index: 0-based index into the phase list.
        """
        label = phase.phase_type.value.upper()
        if "console" in self._backends:
            print(f"\n  Phase {phase_index + 1} ({label}) complete.")

    def log_fork(self) -> None:
        """Log the fork event (unified -> expert phases)."""
        if "console" in self._backends:
            print(f"\n{'━' * 50}")
            print("  FORK: Splitting unified LoRA into expert copies")
            print(f"{'━' * 50}")

    def log_step(
        self,
        metrics: dict[str, float],
        global_step: int,
        phase_type: PhaseType | None = None,
    ) -> None:
        """Log training metrics for one step.

        Respects log_every_n_steps — only actually logs at the configured
        interval. All backends receive the same metrics dict.

        Args:
            metrics: Flat dict of metric names -> values.
            global_step: Global step counter.
            phase_type: Optional phase for prefixing.
        """
        self._step_count = global_step

        if global_step % self._log_every_n_steps != 0:
            return

        # Prefix metrics with phase name if provided
        if phase_type is not None:
            prefix = f"{phase_type.value}/"
            prefixed = {f"{prefix}{k}": v for k, v in metrics.items()}
        else:
            prefixed = metrics

        # Console
        if "console" in self._backends:
            loss = metrics.get("loss_ema", metrics.get("loss_raw", 0.0))
            lr = metrics.get("learning_rate", 0.0)
            epoch = metrics.get("epoch", 0)
            print(
                f"  step {global_step:>6d} | "
                f"epoch {int(epoch):>3d} | "
                f"loss {loss:.4f} | "
                f"lr {lr:.2e}"
            )

        # TensorBoard
        if self._tb_writer is not None:
            for key, value in prefixed.items():
                self._tb_writer.add_scalar(key, value, global_step)

        # W&B — use define_metric(step_metric="global_step") x-axis.
        # Do NOT pass step= to wandb.log() — it conflicts with define_metric
        # and causes empty graphs when log_every_n_steps skips steps.
        if self._wandb_run is not None:
            try:
                import wandb
                prefixed["global_step"] = global_step
                wandb.log(prefixed)
            except Exception as e:
                if not self._wandb_log_warned:
                    print(
                        f"  Warning: W&B logging failed ({e}). "
                        f"Subsequent W&B errors will be suppressed.",
                        file=sys.stderr,
                    )
                    self._wandb_log_warned = True

    def log_vram(self, metrics: dict[str, float], global_step: int) -> None:
        """Log VRAM metrics to TensorBoard and W&B backends.

        Console is intentionally skipped — VRAM metrics are too frequent
        for terminal output. They are visible in W&B and TensorBoard
        dashboards.

        Args:
            metrics: Dict with keys like 'system/vram_allocated_gb'.
            global_step: Global step counter.
        """
        # TensorBoard
        if self._tb_writer is not None:
            for key, value in metrics.items():
                self._tb_writer.add_scalar(key, value, global_step)

        # W&B — same pattern as log_step: no step= parameter.
        if self._wandb_run is not None:
            try:
                import wandb
                metrics["global_step"] = global_step
                wandb.log(metrics)
            except Exception as e:
                if not self._wandb_log_warned:
                    print(
                        f"  Warning: W&B VRAM logging failed ({e}).",
                        file=sys.stderr,
                    )
                    self._wandb_log_warned = True

    def log_run_summary(
        self,
        total_time: float,
        phase_times: dict[str, float],
        peak_vram_gb: float,
        phase_losses: dict[str, float],
        frozen_checks: dict[str, bool] | None = None,
    ) -> None:
        """Log the end-of-run summary to console and W&B.

        Prints a formatted summary block to the console with timing,
        loss, and VRAM information. Logs key summary values to W&B's
        run summary for cross-run comparison.

        Args:
            total_time: Total wall-clock seconds for the run.
            phase_times: Per-phase wall-clock seconds (name -> seconds).
            peak_vram_gb: Peak VRAM allocation in GB.
            phase_losses: Final EMA loss per phase (name -> loss).
            frozen_checks: Frozen-expert verification results (future use).
        """
        # Console summary
        if "console" in self._backends:
            print("\n" + "=" * 60)
            print("  FLIMMER TRAINING COMPLETE")
            print("=" * 60)

            # Timing
            print(f"\n  Total time: {total_time / 60:.1f} min ({total_time:.0f}s)")
            for name, t in phase_times.items():
                print(f"    {name}: {t / 60:.1f} min")

            # Loss
            if phase_losses:
                print(f"\n  Final loss (EMA):")
                for name, loss in phase_losses.items():
                    print(f"    {name}: {loss:.6f}")

            # VRAM
            if peak_vram_gb > 0:
                print(f"\n  Peak VRAM: {peak_vram_gb:.2f} GB")

            # Frozen expert check (reserved for Phase 2 Plan 02)
            if frozen_checks:
                print(f"\n  Frozen expert verification:")
                for name, passed in frozen_checks.items():
                    status = "PASS" if passed else "FAIL (weights changed!)"
                    print(f"    {name}: {status}")

            print("=" * 60 + "\n")

        # W&B summary — these show as summary metrics in the runs table,
        # useful for cross-experiment comparison at a glance
        if self._wandb_run is not None:
            try:
                import wandb
                summary_data = {
                    "total_time_min": total_time / 60,
                    "peak_vram_gb": peak_vram_gb,
                }
                for name, loss in phase_losses.items():
                    summary_data[f"{name}/final_loss_ema"] = loss
                for name, t in phase_times.items():
                    summary_data[f"{name}/wall_clock_sec"] = t
                wandb.run.summary.update(summary_data)
            except Exception as e:
                if not self._wandb_log_warned:
                    print(
                        f"  Warning: W&B summary update failed ({e}).",
                        file=sys.stderr,
                    )
                    self._wandb_log_warned = True

    def log_samples_to_wandb(
        self,
        sample_paths: list[Path],
        phase_type: str,
        epoch: int,
        global_step: int,
    ) -> None:
        """Log sample videos and keyframe grids to W&B.

        For each video path, logs:
          - samples/{phase_type}/prompt_{i} as wandb.Video
          - grids/{phase_type}/prompt_{i} as wandb.Image (if .grid.png exists)

        This is the primary way Minta evaluates convergence during training.
        Videos show temporal coherence and identity preservation; keyframe
        grids give instant visual feedback without downloading video files.

        Args:
            sample_paths: List of paths to generated .mp4 sample files.
            phase_type: Phase type value string (e.g. 'unified', 'high_noise').
            epoch: Current epoch number (used in caption).
            global_step: Global training step (used as W&B step).
        """
        if self._wandb_run is None:
            return

        try:
            import wandb
            log_dict: dict[str, Any] = {}

            for i, video_path in enumerate(sample_paths):
                if video_path.is_file():
                    log_dict[f"samples/{phase_type}/prompt_{i}"] = wandb.Video(
                        str(video_path), caption=f"epoch {epoch}", fps=16,
                    )
                # Check for keyframe grid alongside the video
                grid_path = video_path.with_suffix(".grid.png")
                if grid_path.is_file():
                    log_dict[f"grids/{phase_type}/prompt_{i}"] = wandb.Image(
                        str(grid_path), caption=f"epoch {epoch}",
                    )

            if log_dict:
                log_dict["global_step"] = global_step
                wandb.log(log_dict)

        except Exception as e:
            if not self._wandb_log_warned:
                print(
                    f"  Warning: W&B sample logging failed ({e}).",
                    file=sys.stderr,
                )
                self._wandb_log_warned = True

    def log_frozen_check(self, result: Any) -> None:
        """Log frozen-expert verification result to console and W&B.

        Prints a clear PASS/FAIL message to the console so the user
        immediately knows whether frozen expert integrity held. Also
        logs to W&B summary for cross-run comparison.

        Args:
            result: VerificationResult from WeightVerifier.verify().
                Must have .expert_name, .passed, and .details attributes.
        """
        expert_name = result.expert_name
        passed = result.passed

        # Console output -- always print, this is critical info
        if "console" in self._backends:
            if passed:
                print(f"  Frozen expert check: {expert_name}: PASS")
            else:
                print(
                    f"  Frozen expert check: {expert_name}: "
                    f"FAIL (weights changed!)",
                    file=sys.stderr,
                )

        # W&B summary -- visible in runs table for cross-run comparison
        if self._wandb_run is not None:
            try:
                import wandb
                status = "pass" if passed else "FAIL"
                wandb.run.summary.update({
                    f"frozen_check/{expert_name}": status,
                })
            except Exception as e:
                if not self._wandb_log_warned:
                    print(
                        f"  Warning: W&B frozen check logging failed ({e}).",
                        file=sys.stderr,
                    )
                    self._wandb_log_warned = True

    def log_checkpoint_saved(self, path: Path, phase_type: PhaseType, epoch: int) -> None:
        """Log a checkpoint save event.

        Args:
            path: Where the checkpoint was saved.
            phase_type: Phase that produced the checkpoint.
            epoch: Epoch number.
        """
        if "console" in self._backends:
            print(f"  Checkpoint saved: {path.name} (epoch {epoch})")

    def log_sample_generated(self, path: Path, prompt_index: int) -> None:
        """Log a sample generation event.

        Args:
            path: Where the sample was saved.
            prompt_index: Index of the prompt that generated this sample.
        """
        if "console" in self._backends:
            print(f"  Sample generated: {path.name} (prompt {prompt_index})")

    def close(self) -> None:
        """Close all logging backends and flush pending writes."""
        if self._tb_writer is not None:
            try:
                self._tb_writer.close()
            except Exception as e:
                if not self._tb_close_warned:
                    print(
                        f"  Warning: TensorBoard close failed ({e}).",
                        file=sys.stderr,
                    )
                    self._tb_close_warned = True

        if self._wandb_run is not None:
            try:
                import wandb
                wandb.finish()
            except Exception as e:
                if not self._wandb_close_warned:
                    print(
                        f"  Warning: W&B finish failed ({e}).",
                        file=sys.stderr,
                    )
                    self._wandb_close_warned = True
