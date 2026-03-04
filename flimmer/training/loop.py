"""Training orchestrator — the main state machine.

Sequences through training phases: unified → fork → expert_1 → expert_2.
Within each phase, the inner loop is identical: sample batch → noise →
forward → loss → backward → step.

The orchestrator manages:
    - Phase sequencing from resolved TrainingPhase list
    - LoRA creation, fork, and management via PEFT bridge
    - Per-phase optimizer and scheduler lifecycle
    - Checkpoint saving and resumption
    - Sampling orchestration
    - Logging and metrics

What changes between phases:
    - Which LoRA is being trained
    - Which optimizer/scheduler are active
    - Which timesteps contribute to loss (expert masking)
    - The resolved hyperparameters

Model-specific operations (load model, forward pass, inference) are
delegated to ModelBackend and InferencePipeline protocols.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from flimmer.training.checkpoint import CheckpointManager, TrainingState
from flimmer.training.errors import (
    FlimmerTrainingError,
    ModelBackendError,
    SamplingError,
)
from flimmer.training.logger import TrainingLogger, generate_run_name, save_resolved_config
from flimmer.training.lora import LoRAState
from flimmer.training.metrics import MetricsTracker, RunTimer
from flimmer.training.vram import VRAMTracker
from flimmer.training.optimizer import build_optimizer, build_scheduler, compute_total_steps
from flimmer.training.phase import PhaseType, TrainingPhase, resolve_phases, _EXPERT_NAME_TO_PHASE
from flimmer.training.sampler import SamplingEngine
from flimmer.training.verification import WeightVerifier


# Default gradient norm limit — prevents training instability from spikes
_DEFAULT_MAX_GRAD_NORM = 1.0


class TrainingOrchestrator:
    """The main training state machine.

    Orchestrates the full training pipeline:
    1. Load config → resolve phases
    2. Print training plan
    3. Load model via ModelBackend
    4. Create LoRA via PEFT bridge
    5. Load cached dataset via DataLoader
    6. Iterate through phases (unified → fork → experts)
    7. Save final merged LoRA

    Two execution modes:
    - Real training (dataset provided): full GPU pipeline with PEFT LoRA,
      mixed precision, gradient accumulation, optimizer/scheduler.
    - Mock/dry run (dataset=None): phase counting only, no GPU operations.
      Used by Phase 7 tests and the CLI 'plan' command.

    Args:
        config: FlimmerTrainingConfig instance.
        model_backend: ModelBackend protocol implementation.
        inference_pipeline: InferencePipeline for sampling (optional).
    """

    def __init__(
        self,
        config: Any,
        model_backend: Any,
        inference_pipeline: Any = None,
    ) -> None:
        self._config = config
        self._backend = model_backend
        self._pipeline = inference_pipeline

        # Resolve phases upfront — catch config errors before training
        self._phases = resolve_phases(config)

        # Initialize subsystems
        self._checkpoint_mgr = CheckpointManager(
            output_dir=config.save.output_dir,
            name=config.save.name,
            max_checkpoints=config.save.max_checkpoints,
        )

        # Check for W&B run ID from a previous run (for resume continuity).
        # This must happen BEFORE creating the logger so we can pass the
        # saved run ID into wandb.init(id=, resume="allow"). Without this,
        # resumed training creates a new W&B run → broken loss curve.
        _prior_state = self._checkpoint_mgr.load_training_state()
        _wandb_run_id = _prior_state.wandb_run_id if _prior_state else None

        # Auto-generate a descriptive run name if the user didn't specify one.
        # The name encodes model, dataset, training mode, rank, and LR so
        # runs are distinguishable at a glance in W&B.
        run_name = config.logging.wandb_run_name
        if run_name is None and "wandb" in config.logging.backends:
            run_name = generate_run_name(config)

        # Build the resolved config dict once — used for both W&B config
        # tab and YAML disk save
        resolved_config_dict: dict | None = None
        if hasattr(config, "model_dump"):
            resolved_config_dict = config.model_dump()

        self._logger = TrainingLogger(
            backends=config.logging.backends,
            output_dir=config.save.output_dir,
            wandb_project=config.logging.wandb_project,
            wandb_entity=config.logging.wandb_entity,
            wandb_run_name=run_name,
            log_every_n_steps=config.logging.log_every_n_steps,
            wandb_group=getattr(config.logging, "wandb_group", None),
            wandb_tags=getattr(config.logging, "wandb_tags", None),
            resolved_config=resolved_config_dict,
            wandb_run_id=_wandb_run_id,
        )
        self._metrics = MetricsTracker()

        # VRAM tracker — samples GPU memory at configurable intervals
        vram_interval = getattr(config.logging, "vram_sample_every_n_steps", 50)
        self._vram_tracker = VRAMTracker(sample_every_n_steps=vram_interval)

        # Wall-clock timer — records per-phase and total training time
        self._timer = RunTimer()
        self._sampler = SamplingEngine(
            enabled=config.sampling.enabled,
            every_n_epochs=config.sampling.every_n_epochs,
            prompts=config.sampling.prompts,
            negative_prompt=config.sampling.neg,
            seed=config.sampling.seed,
            walk_seed=config.sampling.walk_seed,
            num_inference_steps=config.sampling.sample_steps,
            guidance_scale=config.sampling.guidance_scale,
            sample_dir=config.sampling.sample_dir or str(
                Path(config.save.output_dir) / "samples"
            ),
            skip_phases=getattr(config.sampling, "skip_phases", None),
        )

        # Weight verifier — checksums frozen experts to catch silent corruption
        self._weight_verifier = WeightVerifier()
        self._frozen_results: dict[str, bool] = {}

        # State
        self._global_step = 0
        self._model: Any = None
        self._unified_lora: LoRAState | None = None
        self._high_noise_lora: LoRAState | None = None
        self._low_noise_lora: LoRAState | None = None

        # Resume fidelity flags
        self._resuming = False
        """True on first phase after resume — triggers optimizer/scheduler/RNG restore."""
        self._stop_requested = False
        """Set to True when max_train_steps is reached — triggers clean shutdown."""

    @property
    def phases(self) -> list[TrainingPhase]:
        """The resolved training phases."""
        return self._phases

    @property
    def global_step(self) -> int:
        """Global training step counter."""
        return self._global_step

    def run(self, dataset: Any = None, dry_run: bool = False) -> None:
        """Execute the full training pipeline.

        Args:
            dataset: CachedLatentDataset instance. Required unless dry_run.
            dry_run: If True, resolve phases and print plan without training.
        """
        # Print the training plan
        self._logger.print_training_plan(self._phases)

        if dry_run:
            return

        # Start wall-clock timer for the entire training run
        self._timer.start_run()

        # Ensure output directories exist
        self._checkpoint_mgr.ensure_dirs()

        # Save the fully resolved config to disk as YAML for reproducibility.
        # This happens early so the config is captured even if training crashes.
        try:
            save_resolved_config(
                self._config,
                Path(self._config.save.output_dir),
            )
        except Exception as e:
            print(f"  Warning: Failed to save resolved config ({e}).")

        # Check for resumption
        resume_point = self._checkpoint_mgr.find_resume_point(self._phases)
        start_phase_idx = 0
        start_epoch = 0

        if resume_point is not None:
            start_phase_idx, start_epoch, state = resume_point
            self._global_step = state.global_step

            # Restore LoRA states from saved paths
            if state.unified_lora_path:
                self._unified_lora = LoRAState.load(state.unified_lora_path)
            if state.high_noise_lora_path:
                self._high_noise_lora = LoRAState.load(state.high_noise_lora_path)
            if state.low_noise_lora_path:
                self._low_noise_lora = LoRAState.load(state.low_noise_lora_path)

            # Flag for optimizer/scheduler/RNG restore on first resumed phase
            self._resuming = True

        # Load model and move to GPU
        self._model = self._backend.load_model(self._config.model)
        if hasattr(self._model, "to"):
            import torch
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._model.to(device)
        if self._config.training.gradient_checkpointing:
            self._backend.setup_gradient_checkpointing(self._model)

        # Execute phases
        for phase_idx in range(start_phase_idx, len(self._phases)):
            phase = self._phases[phase_idx]
            epoch_start = start_epoch if phase_idx == start_phase_idx else 0

            self._execute_phase(
                phase=phase,
                phase_index=phase_idx,
                dataset=dataset,
                start_epoch=epoch_start,
            )

            # Check if max_train_steps triggered a clean stop
            if self._stop_requested:
                break  # Exit phase loop — proceed to summary/cleanup

            # Fork after unified phase if next phase is an expert phase
            if (
                phase.phase_type == PhaseType.UNIFIED
                and phase_idx + 1 < len(self._phases)
                and self._phases[phase_idx + 1].active_expert is not None
                and self._unified_lora is not None
            ):
                self._logger.log_fork()
                high, low = self._unified_lora.fork()
                self._high_noise_lora = high
                self._low_noise_lora = low

        # Log end-of-run summary with timing, loss, and VRAM data.
        # Collects final loss EMA from each tracked phase for the summary.
        phase_losses: dict[str, float] = {}
        for pt in self._metrics.tracked_phases:
            phase_metrics = self._metrics.get_phase(pt)
            if phase_metrics is not None and phase_metrics.step_count > 0:
                phase_losses[pt.value] = phase_metrics.loss_ema

        self._logger.log_run_summary(
            total_time=self._timer.total_elapsed(),
            phase_times=self._timer.phase_times,
            peak_vram_gb=self._vram_tracker.peak(),
            phase_losses=phase_losses,
            frozen_checks=self._frozen_results if self._frozen_results else None,
        )

        # Cleanup inference pipeline if loaded (free VRAM)
        if self._pipeline is not None and hasattr(self._pipeline, "cleanup"):
            self._pipeline.cleanup()

        # Close logger
        self._logger.close()

    # ------------------------------------------------------------------
    # Phase execution
    # ------------------------------------------------------------------

    def _execute_phase(
        self,
        phase: TrainingPhase,
        phase_index: int,
        dataset: Any,
        start_epoch: int = 0,
    ) -> None:
        """Execute one training phase.

        For real training (dataset provided): creates LoRA via PEFT bridge,
        builds optimizer/scheduler, runs epochs with DataLoader, extracts
        LoRA weights after completion.

        For mock/dry run (dataset=None): runs through epoch counting with
        zero loss, no GPU operations.

        Args:
            phase: The resolved TrainingPhase.
            phase_index: Index into self._phases.
            dataset: CachedLatentDataset (None for dry run).
            start_epoch: Epoch to start from (for resumption).
        """
        self._logger.log_phase_start(phase, phase_index)
        self._metrics.start_phase(phase.phase_type)
        self._timer.start_phase(phase.phase_type.value)

        # Get the active LoRA state for this phase (may be None for first phase)
        active_lora = self._get_active_lora(phase)

        # Get noise schedule from model backend
        noise_schedule = self._backend.get_noise_schedule()

        # Snapshot frozen expert weights BEFORE this phase trains.
        # During high_noise training, low_noise should be frozen (and vice versa).
        # We checksum the frozen expert's checkpoint file on disk so we can
        # verify it didn't change after the phase completes.
        frozen_expert = self._get_frozen_expert_name(phase)
        if frozen_expert is not None:
            frozen_phase_type = _EXPERT_NAME_TO_PHASE[frozen_expert]
            frozen_ckpt = self._checkpoint_mgr.find_latest_checkpoint(frozen_phase_type)
            if frozen_ckpt is not None:
                try:
                    self._weight_verifier.snapshot(frozen_expert, checkpoint_path=frozen_ckpt)
                except Exception as e:
                    print(f"  Warning: Could not snapshot frozen expert '{frozen_expert}': {e}")

        # GPU training setup (only when real dataset provided)
        optimizer = None
        scheduler = None
        if dataset is not None:
            # Switch expert model if needed (MoE phases)
            self._ensure_expert_model(phase)

            # Create LoRA adapter on model, inject weights if available
            active_lora = self._setup_phase_lora(phase, active_lora)

            # Build optimizer and scheduler with real model parameters
            optimizer, scheduler = self._build_phase_optimizer(phase, dataset)

            # Restore optimizer/scheduler/RNG state on resume.
            # This must happen AFTER _build_phase_optimizer() because the
            # optimizer needs the correct parameter groups before loading
            # state_dict into it.
            if self._resuming:
                self._restore_full_state(optimizer, scheduler)
                self._resuming = False  # Only restore once (first phase after resume)

        # Training loop
        for epoch in range(start_epoch + 1, phase.max_epochs + 1):
            self._metrics.set_epoch(epoch)

            # Run one epoch
            epoch_loss = self._run_epoch(
                phase=phase,
                dataset=dataset,
                active_lora=active_lora,
                noise_schedule=noise_schedule,
                optimizer=optimizer,
                scheduler=scheduler,
            )

            # Check if max_train_steps triggered a clean stop
            if self._stop_requested:
                # Save checkpoint and training state before exiting
                self._save_checkpoint(phase, epoch, active_lora)
                self._save_training_state(
                    phase, phase_index, epoch, optimizer, scheduler,
                )
                break  # Exit the epoch loop

            # Save checkpoint at interval
            if (
                epoch % self._config.save.save_every_n_epochs == 0
                or epoch == phase.max_epochs
            ):
                self._save_checkpoint(phase, epoch, active_lora)

            # Generate samples
            if self._sampler.should_sample(epoch, phase.phase_type):
                self._generate_samples(phase, epoch, active_lora)

            # Save training state for resumption (includes optimizer/RNG state)
            self._save_training_state(
                phase, phase_index, epoch, optimizer, scheduler,
            )

        # GPU teardown: extract LoRA weights, remove PEFT wrapper
        if dataset is not None:
            active_lora = self._teardown_phase_lora(phase, active_lora)

        # Verify frozen expert weights unchanged after this phase.
        # This catches silent bugs where the frozen expert's checkpoint
        # gets corrupted during the other expert's training.
        if frozen_expert is not None and frozen_expert in self._weight_verifier._snapshots:
            frozen_phase_type = _EXPERT_NAME_TO_PHASE[frozen_expert]
            frozen_ckpt = self._checkpoint_mgr.find_latest_checkpoint(frozen_phase_type)
            if frozen_ckpt is not None:
                try:
                    result = self._weight_verifier.verify(
                        frozen_expert, checkpoint_path=frozen_ckpt,
                    )
                    self._logger.log_frozen_check(result)
                    self._frozen_results[result.expert_name] = result.passed
                    if not result.passed:
                        import sys
                        print(
                            f"  WARNING: Frozen expert '{frozen_expert}' weights changed "
                            f"during {phase.phase_type.value} training! This is a bug.",
                            file=sys.stderr,
                        )
                except Exception as e:
                    print(f"  Warning: Frozen expert verification failed: {e}")

        # Record phase wall-clock time
        phase_elapsed = self._timer.end_phase(phase.phase_type.value)

        self._logger.log_phase_end(phase, phase_index)

        # Update stored LoRA reference
        self._update_lora_state(phase, active_lora)

    # ------------------------------------------------------------------
    # Frozen expert helpers
    # ------------------------------------------------------------------

    def _get_frozen_expert_name(self, phase: TrainingPhase) -> str | None:
        """Get the name of the expert that should be frozen during this phase.

        During high_noise training, low_noise is frozen (and vice versa).
        During unified training, no expert is frozen.

        Args:
            phase: Current training phase.

        Returns:
            Expert name string ('high_noise' or 'low_noise'), or None.
        """
        if phase.active_expert == "high_noise":
            return "low_noise"
        elif phase.active_expert == "low_noise":
            return "high_noise"
        return None

    # ------------------------------------------------------------------
    # LoRA lifecycle (GPU — PEFT bridge)
    # ------------------------------------------------------------------

    def _ensure_expert_model(self, phase: TrainingPhase) -> None:
        """Ensure the correct expert model is loaded for MoE phases.

        For MoE models, each expert phase needs a specific transformer
        (high_noise vs low_noise subfolder). If the wrong expert is
        currently loaded, remove any LoRA, then switch experts.

        Two switching strategies (chosen by backend based on config):
        - State-dict swap (preload_experts=True): fast, in-place
        - Disk reload (default): slower, but no extra RAM

        For non-MoE models or unified phases, this is a no-op.
        """
        if phase.active_expert is None:
            return

        # Check if backend tracks current expert
        current_expert = getattr(self._backend, "current_expert", None)
        if current_expert == phase.active_expert:
            return

        # Remove any existing LoRA wrapper before switching
        try:
            from flimmer.training.wan.modules import remove_lora_from_model
            self._model = remove_lora_from_model(self._model)
        except ImportError:
            pass

        # Switch expert via the backend (handles both swap and reload)
        if hasattr(self._backend, "switch_expert"):
            self._model = self._backend.switch_expert(
                self._model,
                new_expert=phase.active_expert,
                config=self._config.model,
            )
        else:
            # Fallback for non-Wan backends: load from scratch
            self._model = self._backend.load_model(
                self._config.model,
                expert=phase.active_expert,
            )

        # Ensure model is on GPU
        if hasattr(self._model, "to"):
            import torch
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._model.to(device)
        if self._config.training.gradient_checkpointing:
            self._backend.setup_gradient_checkpointing(self._model)

    def _setup_phase_lora(
        self,
        phase: TrainingPhase,
        active_lora: LoRAState | None,
    ) -> LoRAState:
        """Create LoRA adapter on model for this phase.

        Uses the PEFT bridge from flimmer.training.wan.modules:
        1. Resolve target modules (variant defaults + fork target filtering)
        2. Create LoRA via get_peft_model (wraps model)
        3. Inject existing weights if resuming or post-fork

        Args:
            phase: Current training phase.
            active_lora: Existing LoRA state (None for first phase).

        Returns:
            LoRAState for this phase (existing or newly created).
        """
        from flimmer.training.wan.modules import (
            create_lora_on_model,
            inject_lora_state_dict,
            resolve_target_modules,
        )

        # Resolve final LoRA target modules
        variant_targets = self._backend.get_lora_target_modules()
        target_modules = resolve_target_modules(
            variant_targets=variant_targets,
            fork_targets=phase.fork_targets,
        )

        # Create LoRA adapter (returns PEFT-wrapped model)
        self._model = create_lora_on_model(
            model=self._model,
            target_modules=target_modules,
            rank=self._config.lora.rank,
            alpha=self._config.lora.alpha,
            dropout=phase.lora_dropout,
        )

        # Re-enable gradient checkpointing after PEFT wrapping.
        # PEFT's get_peft_model() wraps the model in a PeftModel which
        # may reset the gradient checkpointing flag. We re-apply it
        # to the base model inside the PEFT wrapper.
        if self._config.training.gradient_checkpointing:
            self._backend.setup_gradient_checkpointing(self._model)

        # Inject existing weights (resumption or post-fork)
        if active_lora is not None and active_lora.state_dict:
            inject_lora_state_dict(self._model, active_lora.state_dict)
        else:
            # Create new empty LoRAState for this phase
            active_lora = LoRAState(
                state_dict={},
                rank=self._config.lora.rank,
                alpha=self._config.lora.alpha,
                phase_type=phase.phase_type,
            )

        return active_lora

    def _teardown_phase_lora(
        self,
        phase: TrainingPhase,
        active_lora: LoRAState,
    ) -> LoRAState:
        """Extract LoRA weights from model after phase completion.

        Updates the LoRAState with trained weights and removes the PEFT
        wrapper to restore the base model.

        Args:
            phase: Completed training phase.
            active_lora: LoRAState to update with extracted weights.

        Returns:
            Updated LoRAState with trained weights.
        """
        from flimmer.training.wan.modules import (
            extract_lora_state_dict,
            remove_lora_from_model,
        )

        # Extract trained LoRA weights from model
        state_dict = extract_lora_state_dict(self._model)
        active_lora = LoRAState(
            state_dict=state_dict,
            rank=active_lora.rank,
            alpha=active_lora.alpha,
            phase_type=phase.phase_type,
            metadata=active_lora.metadata,
        )

        # Remove PEFT wrapper (restores base model)
        self._model = remove_lora_from_model(self._model)

        return active_lora

    # ------------------------------------------------------------------
    # Optimizer / scheduler construction
    # ------------------------------------------------------------------

    def _build_phase_optimizer(
        self,
        phase: TrainingPhase,
        dataset: Any,
    ) -> tuple[Any, Any]:
        """Build optimizer and scheduler for a training phase.

        Groups parameters by LoRA A-matrix vs B-matrix for LoRA+ support
        (different learning rates). Uses the resolved phase hyperparameters.

        Args:
            phase: Resolved TrainingPhase.
            dataset: CachedLatentDataset (for total step computation).

        Returns:
            Tuple of (optimizer, scheduler).
        """
        # Group parameters: LoRA+ gives B-matrix higher LR
        loraplus_ratio = getattr(self._config.lora, "loraplus_lr_ratio", 1.0)
        a_params = []
        b_params = []
        for name, param in self._model.named_parameters():
            if not param.requires_grad:
                continue
            if ".lora_B." in name or ".lora_up." in name:
                b_params.append(param)
            else:
                a_params.append(param)

        param_groups: list[dict[str, Any]] = []
        if a_params:
            param_groups.append({"params": a_params, "lr": phase.learning_rate})
        if b_params:
            param_groups.append({
                "params": b_params,
                "lr": phase.learning_rate * loraplus_ratio,
            })

        # Build optimizer
        optimizer_cfg = self._config.optimizer
        optimizer = build_optimizer(
            params=param_groups,
            optimizer_type=phase.optimizer_type,
            learning_rate=phase.learning_rate,
            weight_decay=phase.weight_decay,
            betas=getattr(optimizer_cfg, "betas", None),
            eps=getattr(optimizer_cfg, "eps", 1e-8),
            optimizer_args=getattr(optimizer_cfg, "optimizer_args", None),
        )

        # Compute total optimizer steps for this phase
        num_samples = len(dataset) if hasattr(dataset, "__len__") else 0
        total_steps = compute_total_steps(
            num_samples=num_samples,
            batch_size=phase.batch_size,
            gradient_accumulation_steps=phase.gradient_accumulation_steps,
            max_epochs=phase.max_epochs,
        )

        # Build scheduler
        scheduler = build_scheduler(
            optimizer=optimizer,
            scheduler_type=phase.scheduler_type,
            total_steps=total_steps,
            warmup_steps=phase.warmup_steps,
            min_lr_ratio=phase.min_lr_ratio,
        )

        return optimizer, scheduler

    # ------------------------------------------------------------------
    # Training loop (inner loop)
    # ------------------------------------------------------------------

    def _run_epoch(
        self,
        phase: TrainingPhase,
        dataset: Any,
        active_lora: LoRAState | None,
        noise_schedule: Any,
        optimizer: Any = None,
        scheduler: Any = None,
    ) -> float:
        """Run one training epoch.

        When dataset is provided: uses DataLoader with BucketBatchSampler,
        gradient accumulation, mixed precision, and real gradient updates.

        When dataset is None: returns 0.0 (for dry run / mock testing).

        Args:
            phase: Current training phase.
            dataset: CachedLatentDataset (None for dry run).
            active_lora: Current LoRA state.
            noise_schedule: NoiseSchedule from model backend.
            optimizer: Phase optimizer (None for dry run).
            scheduler: Phase LR scheduler (None for dry run).

        Returns:
            Average loss for the epoch.
        """
        if dataset is None:
            return 0.0

        import torch
        from torch.utils.data import DataLoader

        from flimmer.encoding.dataset import BucketBatchSampler, collate_cached_batch

        # Resolve compute dtype from config
        dtype_map = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }
        mixed_precision = getattr(self._config.training, "mixed_precision", "bf16")
        compute_dtype = dtype_map.get(mixed_precision, torch.bfloat16)

        # Determine device from model parameters
        device = next(
            (p.device for p in self._model.parameters()),
            torch.device("cpu"),
        )

        # Create DataLoader with bucketed batching for uniform dimensions
        batch_sampler = BucketBatchSampler(
            dataset=dataset,
            batch_size=phase.batch_size,
            shuffle=True,
        )
        dataloader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate_cached_batch,
        )

        # Training settings
        grad_accum = phase.gradient_accumulation_steps
        max_grad_norm = getattr(
            self._config.optimizer, "max_grad_norm", _DEFAULT_MAX_GRAD_NORM,
        )

        total_loss = 0.0
        num_steps = 0
        accum_count = 0

        self._model.train()
        optimizer.zero_grad()

        for batch in dataloader:
            # Caption dropout: zero out text embeddings randomly
            batch = self._apply_caption_dropout(batch, phase.caption_dropout_rate)
            # First-frame dropout: zero out reference image conditioning randomly
            # Independent roll from caption dropout (sometimes drop text only,
            # sometimes first frame only, sometimes both, sometimes neither)
            batch = self._apply_first_frame_dropout(batch, phase.first_frame_dropout_rate)

            # Forward + loss + backward
            loss = self._training_step(
                phase=phase,
                batch=batch,
                noise_schedule=noise_schedule,
                compute_dtype=compute_dtype,
                device=device,
                grad_accum_steps=grad_accum,
            )

            total_loss += loss
            num_steps += 1
            accum_count += 1
            self._global_step += 1

            # Optimizer step after gradient accumulation
            if accum_count >= grad_accum:
                if max_grad_norm is not None and max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self._model.parameters() if p.requires_grad],
                        max_grad_norm,
                    )
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad()
                accum_count = 0

            # Update metrics
            current_lr = phase.learning_rate
            if scheduler is not None:
                try:
                    current_lr = scheduler.get_last_lr()[0]
                except Exception:
                    pass
            self._metrics.update(loss=loss, learning_rate=current_lr)

            # Log at interval
            current = self._metrics.get_current()
            if current is not None:
                self._logger.log_step(
                    metrics=current.to_dict(),
                    global_step=self._global_step,
                    phase_type=phase.phase_type,
                )

            # Sample VRAM at configured interval (post-optimizer-step
            # to capture steady-state working memory, not transient peaks)
            vram_metrics = self._vram_tracker.sample(self._global_step)
            if vram_metrics is not None:
                self._logger.log_vram(vram_metrics, self._global_step)

            # Check max_train_steps limit (for resume testing).
            # When reached, sets _stop_requested flag and breaks out of
            # the dataloader loop. _execute_phase() picks up the flag
            # and saves a checkpoint before exiting the epoch loop.
            max_steps = getattr(self._config.training, 'max_train_steps', None)
            if max_steps is not None and self._global_step >= max_steps:
                print(f"\n  Reached max_train_steps ({max_steps}). Stopping.")
                self._stop_requested = True
                break  # Exit the dataloader loop

        # Flush remaining accumulated gradients
        if accum_count > 0:
            if max_grad_norm is not None and max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self._model.parameters() if p.requires_grad],
                    max_grad_norm,
                )
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()

        return total_loss / max(num_steps, 1)

    def _training_step(
        self,
        phase: TrainingPhase,
        batch: dict[str, Any],
        noise_schedule: Any,
        compute_dtype: Any = None,
        device: Any = None,
        grad_accum_steps: int = 1,
    ) -> float:
        """Execute one training step: noise → forward → loss → backward.

        Flow matching training step:
        1. Sample noise and timesteps
        2. Compute noisy latents: (1-t)*clean + t*noise
        3. Compute velocity target: noise - clean
        4. Forward pass through model (with mixed precision)
        5. MSE loss (masked by expert for MoE phases)
        6. Backward pass (scaled for gradient accumulation)

        Args:
            phase: Current training phase.
            batch: Collated batch from DataLoader.
            noise_schedule: NoiseSchedule from model backend.
            compute_dtype: Torch dtype for mixed precision (e.g. torch.bfloat16).
            device: Target device (e.g. torch.device("cuda")).
            grad_accum_steps: Gradient accumulation divisor.

        Returns:
            Unscaled loss value for this step (float).
        """
        import torch
        import torch.nn.functional as F

        latents = batch.get("latent")
        if latents is None:
            return 0.0

        # Move all batch tensors to device and compute dtype
        if hasattr(latents, "to"):
            latents = latents.to(device=device, dtype=compute_dtype)
        for key in ("text_emb", "text_mask", "reference"):
            val = batch.get(key)
            if val is not None and hasattr(val, "to"):
                if key == "text_mask":
                    # Attention masks stay as int, just move device
                    batch[key] = val.to(device=device)
                else:
                    batch[key] = val.to(device=device, dtype=compute_dtype)

        batch_size = latents.shape[0]

        # 1. Sample pure noise (same shape and dtype as latents)
        noise = torch.randn_like(latents)

        # 2. Sample timesteps (numpy array in [0, 1])
        timesteps_np = noise_schedule.sample_timesteps(
            batch_size=batch_size,
            strategy=self._config.training.timestep_sampling,
            flow_shift=self._config.model.flow_shift or 3.0,
        )
        timesteps = torch.from_numpy(timesteps_np).to(
            device=device, dtype=compute_dtype,
        )

        # 3. Compute noisy latents: (1-t)*clean + t*noise
        t_broadcast = timesteps.reshape(-1, 1, 1, 1, 1)
        noisy_latents = (1.0 - t_broadcast) * latents + t_broadcast * noise

        # 4. Compute velocity target: noise - clean
        target = noise - latents

        # 5. Prepare model-specific inputs (handles text emb, ref image, etc.)
        model_inputs = self._backend.prepare_model_inputs(
            batch=batch,
            timesteps=timesteps,
            noisy_latents=noisy_latents,
        )

        # 6. Forward pass with mixed precision
        device_type = str(device).split(":")[0]
        if compute_dtype is not None and device_type != "cpu":
            with torch.amp.autocast(device_type=device_type, dtype=compute_dtype):
                prediction = self._backend.forward(self._model, **model_inputs)
        else:
            prediction = self._backend.forward(self._model, **model_inputs)

        # 7. Compute MSE loss in float32 for numerical stability
        prediction = prediction.float()
        target = target.float()
        loss = F.mse_loss(prediction, target, reduction="none")
        # Mean over spatial dims → per-sample loss [B]
        loss = loss.mean(dim=list(range(1, loss.ndim)))

        # 8. Apply expert mask (only for expert phases with boundary)
        if phase.boundary_ratio is not None and phase.active_expert is not None:
            high_mask, low_mask = self._backend.get_expert_mask(
                timesteps_np, phase.boundary_ratio,
            )
            if phase.phase_type == PhaseType.HIGH_NOISE:
                mask = torch.from_numpy(high_mask).float().to(device)
            else:
                mask = torch.from_numpy(low_mask).float().to(device)
            loss = loss * mask

        # 9. Mean across batch
        loss = loss.mean()

        # 10. Scale for gradient accumulation and backward
        scaled_loss = loss / grad_accum_steps
        scaled_loss.backward()

        return loss.item()

    # ------------------------------------------------------------------
    # Caption dropout
    # ------------------------------------------------------------------

    def _apply_caption_dropout(
        self,
        batch: dict[str, Any],
        dropout_rate: float,
    ) -> dict[str, Any]:
        """Apply caption dropout to a batch.

        Zeroes out text embeddings with probability dropout_rate, forcing
        the model to rely on visual control signals instead of text.

        Args:
            batch: Collated batch dict.
            dropout_rate: Probability of dropping each sample's caption.

        Returns:
            Batch with dropped captions (modified in-place).
        """
        if dropout_rate <= 0.0:
            return batch

        text_emb = batch.get("text_emb")
        text_mask = batch.get("text_mask")

        if text_emb is None or not hasattr(text_emb, "shape"):
            return batch

        # Per-sample dropout
        batch_size = text_emb.shape[0] if text_emb.ndim >= 2 else 1
        for i in range(batch_size):
            if random.random() < dropout_rate:
                text_emb[i].zero_()
                if text_mask is not None and hasattr(text_mask, "__setitem__"):
                    text_mask[i].zero_()

        batch["text_emb"] = text_emb
        if text_mask is not None:
            batch["text_mask"] = text_mask
        return batch

    # ------------------------------------------------------------------
    # First-frame dropout
    # ------------------------------------------------------------------

    def _apply_first_frame_dropout(
        self,
        batch: dict[str, Any],
        dropout_rate: float,
    ) -> dict[str, Any]:
        """Apply first-frame dropout to a batch.

        Zeroes out reference image conditioning with probability dropout_rate,
        forcing the model to generate video from text alone. Independent from
        caption dropout -- each is rolled separately (sometimes drop text only,
        sometimes first frame only, sometimes both, sometimes neither).

        Args:
            batch: Collated batch dict.
            dropout_rate: Probability of dropping each sample's reference.

        Returns:
            Batch with dropped references (modified in-place).
        """
        if dropout_rate <= 0.0:
            return batch

        reference = batch.get("reference")
        if reference is None or not hasattr(reference, "shape"):
            return batch

        batch_size = reference.shape[0] if reference.ndim >= 2 else 1
        for i in range(batch_size):
            if random.random() < dropout_rate:
                reference[i].zero_()

        batch["reference"] = reference
        return batch

    # ------------------------------------------------------------------
    # LoRA state helpers
    # ------------------------------------------------------------------

    def _get_active_lora(self, phase: TrainingPhase) -> LoRAState | None:
        """Get the LoRA state that should be trained in this phase.

        Args:
            phase: Current training phase.

        Returns:
            The active LoRA state, or None if not yet created.
        """
        if phase.phase_type == PhaseType.UNIFIED:
            return self._unified_lora
        elif phase.phase_type == PhaseType.HIGH_NOISE:
            return self._high_noise_lora
        elif phase.phase_type == PhaseType.LOW_NOISE:
            return self._low_noise_lora
        return None

    def _update_lora_state(self, phase: TrainingPhase, lora: LoRAState | None) -> None:
        """Update the stored LoRA reference after a phase completes.

        Args:
            phase: Completed training phase.
            lora: The LoRA state that was trained.
        """
        if phase.phase_type == PhaseType.UNIFIED:
            self._unified_lora = lora
        elif phase.phase_type == PhaseType.HIGH_NOISE:
            self._high_noise_lora = lora
        elif phase.phase_type == PhaseType.LOW_NOISE:
            self._low_noise_lora = lora

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(
        self,
        phase: TrainingPhase,
        epoch: int,
        lora: LoRAState | None,
    ) -> None:
        """Save a checkpoint at the current epoch.

        Extracts the current LoRA weights from the live PEFT model
        (not from the LoRAState, which may be stale or empty). This
        ensures checkpoints always contain the actual trained weights.

        Saves with diffusers component prefix so the file is directly
        loadable by pipeline.load_lora_weights():
            - UNIFIED/HIGH_NOISE → 'transformer.' prefix
            - LOW_NOISE → 'transformer_2.' prefix

        Args:
            phase: Current training phase.
            epoch: Current epoch number.
            lora: LoRA state (used for rank/alpha/metadata).
        """
        if lora is None:
            return

        # Extract live LoRA weights from the PEFT-wrapped model.
        # The LoRAState.state_dict may be empty (initial) or stale
        # (from phase start); the real trained weights live in the
        # model's PEFT layers.
        try:
            from flimmer.training.wan.modules import extract_lora_state_dict
            live_state_dict = extract_lora_state_dict(self._model)
        except Exception:
            # Fall back to whatever the LoRAState has
            live_state_dict = lora.state_dict

        # Build a fresh LoRAState with the live weights for saving
        checkpoint_lora = LoRAState(
            state_dict=live_state_dict,
            rank=lora.rank,
            alpha=lora.alpha,
            phase_type=lora.phase_type,
            metadata=lora.metadata,
        )

        # Determine diffusers prefix based on phase type.
        # Unified and high-noise both target 'transformer' (the first/only model).
        # Low-noise targets 'transformer_2' (the second model in dual-expert).
        if phase.phase_type == PhaseType.LOW_NOISE:
            diffusers_prefix = "transformer_2"
        else:
            diffusers_prefix = "transformer"

        path = self._checkpoint_mgr.checkpoint_path(phase.phase_type, epoch)
        checkpoint_lora.save(
            path,
            extra_metadata={
                "epoch": str(epoch),
                "global_step": str(self._global_step),
            },
            diffusers_prefix=diffusers_prefix,
        )
        self._logger.log_checkpoint_saved(path, phase.phase_type, epoch)

        # Prune old checkpoints
        self._checkpoint_mgr.prune_checkpoints(phase.phase_type)

    def _save_training_state(
        self,
        phase: TrainingPhase,
        phase_index: int,
        epoch: int,
        optimizer: Any = None,
        scheduler: Any = None,
    ) -> None:
        """Save training state for resumption.

        Saves both the lightweight JSON state (phase position, paths) and
        the full optimizer/scheduler/RNG bundle when an optimizer is provided.

        Args:
            phase: Current training phase.
            phase_index: Index into the phases list.
            epoch: Current epoch number.
            optimizer: Phase optimizer (None for dry run — skips full state).
            scheduler: Phase LR scheduler (None for dry run).
        """
        state = TrainingState(
            phase_index=phase_index,
            phase_type=phase.phase_type.value,
            epoch=epoch,
            global_step=self._global_step,
        )

        # Capture W&B run ID for resume continuity.
        # On resume, the orchestrator passes this ID back to the logger
        # so W&B appends to the existing run instead of creating a new one.
        if hasattr(self._logger, 'wandb_run_id'):
            state.wandb_run_id = self._logger.wandb_run_id

        # Record latest checkpoint paths
        for pt in PhaseType:
            latest = self._checkpoint_mgr.find_latest_checkpoint(pt)
            if latest is not None:
                path_str = str(latest)
                if pt == PhaseType.UNIFIED:
                    state.unified_lora_path = path_str
                elif pt == PhaseType.HIGH_NOISE:
                    state.high_noise_lora_path = path_str
                elif pt == PhaseType.LOW_NOISE:
                    state.low_noise_lora_path = path_str

        self._checkpoint_mgr.save_training_state(state)

        # Save full optimizer/scheduler/RNG state for resume fidelity.
        # This is the critical part — without it, resumed training gets
        # a cold optimizer (no momentum), wrong LR, and different noise.
        if optimizer is not None:
            try:
                ckpt_path = self._save_full_state(optimizer, scheduler)
                # Update the state with the checkpoint path reference
                state.optimizer_path = str(ckpt_path)
                self._checkpoint_mgr.save_training_state(state)
            except Exception as e:
                print(f"  Warning: Could not save full training state ({e})")

    def _save_full_state(
        self,
        optimizer: Any,
        scheduler: Any,
    ) -> Path:
        """Save optimizer, scheduler, and RNG state for resume fidelity.

        Saves everything into a single .pt file for atomicity — if the
        process is killed between writing separate files, the state would
        be inconsistent. torch.save() handles 8-bit optimizer state
        (bitsandbytes AdamW8bit) correctly via custom serialization hooks.

        Args:
            optimizer: Current phase optimizer.
            scheduler: Current phase LR scheduler.

        Returns:
            Path to the saved training_checkpoint.pt file.
        """
        import torch
        import numpy as np

        bundle = {
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "rng_python": random.getstate(),
            "rng_numpy": np.random.get_state(),
            "rng_torch_cpu": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            bundle["rng_torch_cuda"] = torch.cuda.get_rng_state()

        path = self._checkpoint_mgr.output_dir / "training_checkpoint.pt"
        torch.save(bundle, str(path))
        return path

    def _restore_full_state(
        self,
        optimizer: Any,
        scheduler: Any,
    ) -> bool:
        """Restore optimizer, scheduler, and RNG state from checkpoint.

        Called after _build_phase_optimizer() during resume. The optimizer
        must already exist with the correct parameter groups before loading
        state_dict into it. Uses weights_only=False because optimizer state
        contains non-tensor objects (step counters, momentum format info).

        Args:
            optimizer: Freshly built optimizer (correct param groups).
            scheduler: Freshly built scheduler.

        Returns:
            True if state was restored, False if no checkpoint found.
        """
        import torch
        import numpy as np

        path = self._checkpoint_mgr.output_dir / "training_checkpoint.pt"
        if not path.is_file():
            print("  No training_checkpoint.pt found -- starting optimizer fresh")
            return False

        print("  Restoring full training state from checkpoint...")
        bundle = torch.load(str(path), map_location="cpu", weights_only=False)

        # Optimizer — critical for loss continuity (preserves momentum)
        if "optimizer_state_dict" in bundle and bundle["optimizer_state_dict"]:
            try:
                optimizer.load_state_dict(bundle["optimizer_state_dict"])
                print("    Restored optimizer state")
            except Exception as e:
                print(f"    Warning: Could not restore optimizer state ({e})")
                print("    Training will continue with fresh optimizer (expect loss spike)")

        # Scheduler — preserves LR position (step counter)
        if scheduler and "scheduler_state_dict" in bundle and bundle["scheduler_state_dict"]:
            try:
                scheduler.load_state_dict(bundle["scheduler_state_dict"])
                print("    Restored scheduler state")
            except Exception as e:
                print(f"    Warning: Could not restore scheduler state ({e})")

        # RNG state — preserves noise reproducibility
        if "rng_python" in bundle:
            random.setstate(bundle["rng_python"])
        if "rng_numpy" in bundle:
            np.random.set_state(bundle["rng_numpy"])
        if "rng_torch_cpu" in bundle:
            torch.set_rng_state(bundle["rng_torch_cpu"])
        if "rng_torch_cuda" in bundle and torch.cuda.is_available():
            torch.cuda.set_rng_state(bundle["rng_torch_cuda"])

        print("    Restored RNG state")
        return True

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _generate_samples(
        self,
        phase: TrainingPhase,
        epoch: int,
        lora: LoRAState | None,
    ) -> None:
        """Generate sample previews.

        For expert phases (high_noise / low_noise), loads the partner
        expert's base model from disk so the pipeline can generate with
        both experts active — producing coherent dual-expert output.
        The partner model is freed immediately after sampling to reclaim
        VRAM.

        Args:
            phase: Current training phase.
            epoch: Current epoch number.
            lora: Current LoRA state for this phase.
        """
        if self._pipeline is None or lora is None:
            return

        # For expert phases, resolve partner LoRA
        partner_path = self._sampler.resolve_partner_lora(
            active_expert=phase.active_expert,
            high_noise_path=self._checkpoint_mgr.find_latest_checkpoint(PhaseType.HIGH_NOISE),
            low_noise_path=self._checkpoint_mgr.find_latest_checkpoint(PhaseType.LOW_NOISE),
            unified_path=self._checkpoint_mgr.find_latest_checkpoint(PhaseType.UNIFIED),
        )

        # Wan 2.2 MoE ALWAYS needs both experts for inference — even in
        # unified phase. Each expert is a specialist: the low-noise expert
        # only handles low-noise timesteps, the high-noise expert only
        # handles high-noise timesteps. Without both, the pipeline produces
        # pure noise because a single specialist can't handle the full
        # denoising trajectory.
        #
        # If the partner has a trained LoRA checkpoint, apply it so
        # samples reflect both experts' training — not just the active one.
        # Falls back to base weights (no LoRA) when no checkpoint exists
        # (e.g., high-noise phase before low-noise has trained).
        #
        # Decision tree:
        #   - Expert phase: partner is the OTHER expert (existing logic)
        #   - Unified phase on MoE: training model is low_noise (backend
        #     default for MoE unified), so partner is high_noise
        #   - Non-MoE (Wan 2.1): no partner needed (single transformer)
        partner_model = None
        active_expert_for_inference = phase.active_expert

        if phase.active_expert is not None:
            # Expert phase: load the partner expert + its LoRA if available
            partner_model = self._load_partner_model(
                phase.active_expert, partner_lora_path=partner_path,
            )
        elif self._backend.supports_moe:
            # Unified phase on MoE model: the training model is the
            # low-noise expert (backend._resolve_single_file_path(None)
            # returns dit_low_path for MoE). We need the high-noise
            # partner for coherent dual-expert inference.
            partner_model = self._load_partner_model("low_noise")
            # Tell the pipeline the training model is low_noise so it
            # wires transformer (high=partner) / transformer_2 (low=model).
            active_expert_for_inference = "low_noise"

        try:
            # Extract live LoRA weights from PEFT model (not stale lora.state_dict)
            try:
                from flimmer.training.wan.modules import extract_lora_state_dict
                live_state_dict = extract_lora_state_dict(self._model)
            except Exception:
                live_state_dict = lora.state_dict

            samples = self._sampler.generate_samples(
                pipeline=self._pipeline,
                model=self._model,
                lora_state_dict=live_state_dict,
                phase_type=phase.phase_type,
                epoch=epoch,
                partner_model=partner_model,
                active_expert=active_expert_for_inference,
            )
            for i, path in enumerate(samples):
                self._logger.log_sample_generated(path, i)

            # Log samples to W&B (video + keyframe grid) so Minta can
            # evaluate convergence directly from the W&B dashboard.
            self._logger.log_samples_to_wandb(
                sample_paths=samples,
                phase_type=phase.phase_type.value,
                epoch=epoch,
                global_step=self._global_step,
            )
        except Exception as e:
            # Sampling failure shouldn't crash training, but log a warning
            print(f"  Warning: Sampling failed (training continues): {e}")
        finally:
            # Free partner model VRAM regardless of success/failure
            if partner_model is not None:
                del partner_model
                import gc
                gc.collect()
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass

    def _load_partner_model(
        self,
        active_expert: str,
        partner_lora_path: str | Path | None = None,
    ) -> Any:
        """Load the partner expert's base model from disk for sampling.

        During expert training, only one expert is on GPU. For sampling
        we need both experts to produce coherent output. This loads the
        other expert temporarily from disk.

        If partner_lora_path is provided, applies the partner's trained
        LoRA so samples reflect both experts' training progress — not
        just the active expert. This matters most during low-noise
        training where the high-noise expert already has a fully trained
        LoRA from the previous phase.

        Diffusers convention:
            transformer   = HIGH-noise expert
            transformer_2 = LOW-noise expert

        So if training low_noise, the partner is high_noise (and vice versa).

        Args:
            active_expert: The expert currently being trained
                ('high_noise' or 'low_noise').
            partner_lora_path: Path to the partner's LoRA checkpoint
                (.safetensors). If None, partner runs with base weights.

        Returns:
            Loaded WanTransformer3DModel on GPU, with LoRA if available.
        """
        # Determine which expert to load as partner
        partner_expert = (
            "low_noise" if active_expert == "high_noise" else "high_noise"
        )

        # Get the file path from the backend's configured paths
        partner_file = self._backend._resolve_single_file_path(partner_expert)
        if partner_file is None:
            raise SamplingError(
                f"No file path configured for partner expert '{partner_expert}'. "
                f"Both experts are required for Wan 2.2 inference. "
                f"Set model.dit_high and model.dit_low in your training config."
            )

        try:
            import torch
            from diffusers import WanTransformer3DModel

            print(f"  Loading partner expert ({partner_expert}) for sampling...")

            # Use the same loading pattern as the backend — config= and
            # subfolder= are required for Wan 2.2 to avoid loading Wan 2.1
            # config silently (diffusers#12329).
            subfolder = self._backend._resolve_config_subfolder(partner_expert)
            model = WanTransformer3DModel.from_single_file(
                partner_file,
                torch_dtype=torch.bfloat16,
                device="cpu",
                config="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
                subfolder=subfolder,
            )

            # Apply partner's trained LoRA if checkpoint exists.
            # During low-noise training, this loads the high-noise expert's
            # completed LoRA. During high-noise training, this loads the
            # unified fork (low-noise hasn't trained yet).
            if partner_lora_path is not None:
                self._apply_partner_lora(model, partner_lora_path, partner_expert)

            # Move to GPU
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            model.eval()

            lora_msg = f" + LoRA from {Path(partner_lora_path).name}" if partner_lora_path else ""
            print(f"  Partner expert loaded ({partner_expert}{lora_msg}).")
            return model

        except Exception as e:
            raise SamplingError(
                f"Failed to load partner expert '{partner_expert}' from "
                f"'{partner_file}': {e}\n"
                f"Both experts are required for Wan 2.2 inference. "
                f"Check that the safetensors file exists and is valid."
            ) from e

    def _apply_partner_lora(
        self,
        model: Any,
        lora_path: str | Path,
        partner_expert: str,
    ) -> None:
        """Apply a saved LoRA checkpoint to the partner model via PEFT.

        Creates PEFT adapters on the partner model and injects the saved
        weights. The checkpoint file has diffusers prefixes (transformer.
        or transformer_2.) which are stripped before injection.

        Args:
            model: Bare WanTransformer3DModel (not yet PEFT-wrapped).
            lora_path: Path to the .safetensors LoRA checkpoint.
            partner_expert: Which expert this is ('high_noise' or 'low_noise').
        """
        from pathlib import Path as P
        from safetensors.torch import load_file

        from flimmer.training.wan.modules import (
            create_lora_on_model,
            inject_lora_state_dict,
            resolve_target_modules,
        )

        lora_file = P(lora_path)
        if not lora_file.is_file():
            print(f"  Warning: Partner LoRA not found at {lora_path}, using base weights")
            return

        # Load checkpoint and strip diffusers prefix to get clean LoRA keys
        raw_state = load_file(str(lora_file))
        clean_state: dict[str, Any] = {}
        for key, value in raw_state.items():
            # Strip 'transformer.' or 'transformer_2.' prefix
            clean_key = key
            if clean_key.startswith("transformer_2."):
                clean_key = clean_key[len("transformer_2."):]
            elif clean_key.startswith("transformer."):
                clean_key = clean_key[len("transformer."):]
            clean_state[clean_key] = value

        # Create PEFT adapters (same config as the training model)
        variant_targets = self._backend.get_lora_target_modules()
        target_modules = resolve_target_modules(
            variant_targets=variant_targets,
        )

        create_lora_on_model(
            model=model,
            target_modules=target_modules,
            rank=self._config.lora.rank,
            alpha=self._config.lora.alpha,
            adapter_name="partner",
        )

        # Inject the saved weights
        inject_lora_state_dict(model, clean_state)
        print(f"  Partner LoRA applied: {lora_file.name} ({len(clean_state)} keys)")

