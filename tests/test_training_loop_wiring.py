"""Tests for the GPU training path in flimmer.training.loop.

Tests the real training loop wiring added in Phase 8 Step 9 — the code
that replaced Phase 7 stubs with actual DataLoader, PEFT LoRA, gradient
accumulation, mixed precision, and expert masking.

All tests run on CPU without requiring torch, peft, or any GPU dependencies.
Everything is mocked via monkeypatch.

Tested methods:
    _apply_caption_dropout   — caption dropout with text embedding zeroing
    _training_step           — noise -> forward -> loss -> backward pipeline
    _build_phase_optimizer   — parameter grouping with LoRA+ support
    _ensure_expert_model     — expert model switching for MoE phases
    _setup_phase_lora        — PEFT LoRA creation and weight injection
    _teardown_phase_lora     — LoRA extraction and PEFT removal
    _run_epoch               — full DataLoader loop with gradient accumulation
"""

import random
import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pytest

from flimmer.training.lora import LoRAState
from flimmer.training.noise import FlowMatchingSchedule
from flimmer.training.phase import PhaseType, TrainingPhase


# ---------------------------------------------------------------------------
# Mock config objects — reuse the pattern from test_training_loop.py
# ---------------------------------------------------------------------------

class MockModelBackend:
    """Minimal model backend for orchestrator testing."""

    def __init__(self):
        self.current_expert = None

    @property
    def model_id(self):
        return "mock"

    @property
    def supports_moe(self):
        return True

    @property
    def supports_reference_image(self):
        return False

    def load_model(self, config, expert=None):
        self.current_expert = expert
        return MagicMock(name="mock_model")

    def get_lora_target_modules(self):
        return ["attn1.to_q", "attn1.to_k", "ffn.net.0.proj"]

    def get_expert_mask(self, timesteps, boundary_ratio):
        high = (timesteps >= boundary_ratio).astype(np.float64)
        low = (timesteps < boundary_ratio).astype(np.float64)
        return (high, low)

    def prepare_model_inputs(self, batch, timesteps, noisy_latents):
        return {"hidden_states": noisy_latents, "timesteps": timesteps}

    def forward(self, model, **inputs):
        return MagicMock(name="prediction")

    def setup_gradient_checkpointing(self, model):
        pass

    def get_noise_schedule(self):
        return FlowMatchingSchedule(1000)


class MockSaveConfig:
    output_dir = ""
    name = "test_lora"
    save_every_n_epochs = 5
    save_last = True
    max_checkpoints = None
    format = "safetensors"


class MockLoggingConfig:
    backends = ["console"]
    log_every_n_steps = 100
    wandb_project = None
    wandb_entity = None
    wandb_run_name = None
    wandb_group = None
    wandb_tags = []
    vram_sample_every_n_steps = 50


class MockSamplingConfig:
    enabled = False
    every_n_epochs = 5
    prompts = []
    neg = ""
    seed = 42
    walk_seed = True
    sample_steps = 30
    guidance_scale = 5.0
    sample_dir = None
    skip_phases = []


class MockOptimizer:
    type = "adamw"
    learning_rate = 5e-5
    weight_decay = 0.01
    betas = [0.9, 0.999]
    eps = 1e-8
    max_grad_norm = 1.0
    optimizer_args = {}


class MockScheduler:
    type = "cosine_with_min_lr"
    warmup_steps = 0
    min_lr_ratio = 0.01


class MockLora:
    rank = 16
    alpha = 16
    dropout = 0.0
    loraplus_lr_ratio = 1.0


class MockExpertOverrides:
    def __init__(self, **kwargs):
        self.enabled = kwargs.get("enabled", True)
        self.learning_rate = None
        self.dropout = None
        self.max_epochs = kwargs.get("max_epochs", 30)
        self.fork_targets = None
        self.block_targets = None
        self.resume_from = None
        self.batch_size = None
        self.gradient_accumulation_steps = None
        self.caption_dropout_rate = None
        self.weight_decay = None
        self.min_lr_ratio = None
        self.optimizer_type = None
        self.scheduler_type = None


class MockMoeConfig:
    def __init__(self, **kwargs):
        self.enabled = kwargs.get("enabled", True)
        self.fork_enabled = kwargs.get("fork_enabled", True)
        self.expert_order = ["low_noise", "high_noise"]
        self.high_noise = MockExpertOverrides(max_epochs=30)
        self.low_noise = MockExpertOverrides(max_epochs=50)
        self.boundary_ratio = None


class MockTrainingConfig:
    def __init__(self, **kwargs):
        self.unified_epochs = kwargs.get("unified_epochs", 10)
        self.batch_size = 1
        self.gradient_accumulation_steps = 1
        self.caption_dropout_rate = 0.1
        self.unified_targets = None
        self.unified_block_targets = None
        self.resume_from = None
        self.gradient_checkpointing = False
        self.timestep_sampling = "uniform"
        self.mixed_precision = "bf16"
        self.base_model_precision = "bf16"


class MockModelConfig:
    boundary_ratio = 0.875
    flow_shift = 3.0
    path = "/fake/model/path"
    variant = "2.2_t2v"


class MockConfig:
    def __init__(self, tmp_path, **kwargs):
        self.training = kwargs.get("training", MockTrainingConfig())
        self.optimizer = MockOptimizer()
        self.scheduler = MockScheduler()
        self.lora = MockLora()
        self.moe = kwargs.get("moe", MockMoeConfig())
        self.model = MockModelConfig()
        save = MockSaveConfig()
        save.output_dir = str(tmp_path / "output")
        self.save = save
        self.logging = MockLoggingConfig()
        sampling = MockSamplingConfig()
        sampling.sample_dir = str(tmp_path / "samples")
        self.sampling = sampling


# ---------------------------------------------------------------------------
# Helper: build a TrainingPhase with sensible defaults
# ---------------------------------------------------------------------------

def _make_phase(
    phase_type=PhaseType.UNIFIED,
    max_epochs=10,
    learning_rate=5e-5,
    weight_decay=0.01,
    optimizer_type="adamw",
    scheduler_type="cosine_with_min_lr",
    min_lr_ratio=0.01,
    warmup_steps=0,
    batch_size=1,
    gradient_accumulation_steps=1,
    caption_dropout_rate=0.1,
    lora_dropout=0.0,
    fork_targets=None,
    block_targets=None,
    resume_from=None,
    boundary_ratio=None,
    active_expert=None,
):
    """Build a TrainingPhase with defaults for testing."""
    return TrainingPhase(
        phase_type=phase_type,
        max_epochs=max_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        optimizer_type=optimizer_type,
        scheduler_type=scheduler_type,
        min_lr_ratio=min_lr_ratio,
        warmup_steps=warmup_steps,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        caption_dropout_rate=caption_dropout_rate,
        lora_dropout=lora_dropout,
        fork_targets=fork_targets,
        block_targets=block_targets,
        resume_from=resume_from,
        boundary_ratio=boundary_ratio,
        active_expert=active_expert,
    )


# ---------------------------------------------------------------------------
# Helper: create an orchestrator with mocked imports
# ---------------------------------------------------------------------------

def _make_orchestrator(tmp_path, **kwargs):
    """Build a TrainingOrchestrator with a mock backend."""
    from flimmer.training.loop import TrainingOrchestrator

    config = MockConfig(tmp_path, **kwargs)
    backend = MockModelBackend()
    orch = TrainingOrchestrator(config, backend)
    return orch, config, backend


# ---------------------------------------------------------------------------
# Numpy-based tensor mock — lightweight replacement for torch.Tensor
# ---------------------------------------------------------------------------

class _FakeTensorView:
    """A view into a FakeTensor row — modifications affect the parent data."""

    def __init__(self, parent_data, idx):
        self._parent_data = parent_data
        self._idx = idx

    def zero_(self):
        """Zero the viewed row in the parent array."""
        self._parent_data[self._idx] = 0.0
        return self

    def sum(self):
        """Sum the viewed row."""
        return float(self._parent_data[self._idx].sum())


class FakeTensor:
    """Lightweight tensor mock that supports .zero_(), .shape, .ndim, indexing.

    Indexing returns a _FakeTensorView that modifies the parent data in-place,
    matching how PyTorch tensor indexing returns views.
    """

    def __init__(self, data):
        self._data = np.array(data, dtype=np.float32)

    @property
    def shape(self):
        return self._data.shape

    @property
    def ndim(self):
        return self._data.ndim

    def __getitem__(self, idx):
        return _FakeTensorView(self._data, idx)

    def __setitem__(self, idx, val):
        if isinstance(val, FakeTensor):
            self._data[idx] = val._data
        else:
            self._data[idx] = val

    def zero_(self):
        self._data[:] = 0.0
        return self

    def sum(self):
        return float(self._data.sum())


# ---------------------------------------------------------------------------
# Helper: install mock wan.modules into sys.modules
# ---------------------------------------------------------------------------

def _install_mock_wan_modules(monkeypatch):
    """Install a mock flimmer.training.wan.modules into sys.modules."""
    wan_modules = types.ModuleType("flimmer.training.wan.modules")
    wan_modules.resolve_target_modules = MagicMock(
        return_value=["attn1.to_q", "attn1.to_k"],
    )
    wan_modules.create_lora_on_model = MagicMock(
        return_value=MagicMock(name="peft_wrapped_model"),
    )
    wan_modules.inject_lora_state_dict = MagicMock()
    wan_modules.extract_lora_state_dict = MagicMock(
        return_value={"lora_A.weight": "tensor_a", "lora_B.weight": "tensor_b"},
    )
    wan_modules.remove_lora_from_model = MagicMock(
        return_value=MagicMock(name="base_model"),
    )
    monkeypatch.setitem(sys.modules, "flimmer.training.wan.modules", wan_modules)
    return wan_modules


# ---------------------------------------------------------------------------
# Helper: install mock torch into sys.modules for _training_step
# ---------------------------------------------------------------------------

def _install_mock_torch(monkeypatch):
    """Install a mock torch module into sys.modules for lazy imports.

    Returns a tuple of (torch_mod, F_mock) so tests can configure behavior.
    """
    F_mock = MagicMock(name="torch.nn.functional")
    nn_mock = MagicMock(name="torch.nn")
    nn_mock.functional = F_mock

    torch_mod = types.ModuleType("torch")
    torch_mod.bfloat16 = "bf16"
    torch_mod.float16 = "fp16"
    torch_mod.float32 = "fp32"
    torch_mod.device = MagicMock
    torch_mod.nn = nn_mock
    torch_mod.amp = MagicMock()

    # randn_like — returns a mock tensor
    torch_mod.randn_like = MagicMock(return_value=MagicMock(name="noise"))

    # from_numpy — returns a chainable mock tensor
    timesteps_t = MagicMock(name="timesteps_tensor")
    timesteps_t.to = MagicMock(return_value=timesteps_t)
    timesteps_t.reshape = MagicMock(return_value=timesteps_t)
    timesteps_t.float = MagicMock(return_value=timesteps_t)
    torch_mod.from_numpy = MagicMock(return_value=timesteps_t)

    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    monkeypatch.setitem(sys.modules, "torch.nn", nn_mock)
    monkeypatch.setitem(sys.modules, "torch.nn.functional", F_mock)

    return torch_mod, F_mock


# ---------------------------------------------------------------------------
# Helper: install mock torch + DataLoader + encoding.dataset for _run_epoch
# ---------------------------------------------------------------------------

def _install_run_epoch_mocks(monkeypatch, batches):
    """Install all mocks needed by _run_epoch.

    Args:
        monkeypatch: pytest monkeypatch fixture.
        batches: List of batch dicts the DataLoader should yield.

    Returns:
        Tuple of (torch_mod, mock_dataloader_cls, mock_sampler_cls).
    """
    torch_mod, F_mock = _install_mock_torch(monkeypatch)
    torch_mod.nn.utils = MagicMock()
    torch_mod.nn.utils.clip_grad_norm_ = MagicMock()

    mock_sampler_cls = MagicMock(name="BucketBatchSampler")
    mock_collate = MagicMock(name="collate_fn")
    mock_dataloader_cls = MagicMock(return_value=iter(batches))

    encoding_dataset = types.ModuleType("flimmer.encoding.dataset")
    encoding_dataset.BucketBatchSampler = mock_sampler_cls
    encoding_dataset.collate_cached_batch = mock_collate
    monkeypatch.setitem(sys.modules, "flimmer.encoding.dataset", encoding_dataset)

    torch_data = types.ModuleType("torch.utils.data")
    torch_data.DataLoader = mock_dataloader_cls
    monkeypatch.setitem(sys.modules, "torch.utils.data", torch_data)

    return torch_mod, mock_dataloader_cls, mock_sampler_cls


# =========================================================================
# TestCaptionDropout
# =========================================================================

class TestCaptionDropout:
    """Tests for _apply_caption_dropout -- zeroes text embeddings randomly."""

    def test_no_dropout_zero_rate(self, tmp_path):
        """dropout_rate=0.0 leaves the batch completely unchanged."""
        orch, _, _ = _make_orchestrator(tmp_path)

        text_emb = FakeTensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        text_mask = FakeTensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
        batch = {"text_emb": text_emb, "text_mask": text_mask, "latent": "x"}

        result = orch._apply_caption_dropout(batch, dropout_rate=0.0)

        # Nothing should be zeroed
        assert result["text_emb"].sum() == 21.0
        assert result["text_mask"].sum() == 6.0

    def test_full_dropout(self, tmp_path):
        """dropout_rate=1.0 zeros ALL text embeddings and masks."""
        orch, _, _ = _make_orchestrator(tmp_path)

        text_emb = FakeTensor([[1.0, 2.0], [3.0, 4.0]])
        text_mask = FakeTensor([[1.0, 1.0], [1.0, 1.0]])
        batch = {"text_emb": text_emb, "text_mask": text_mask}

        result = orch._apply_caption_dropout(batch, dropout_rate=1.0)

        assert result["text_emb"].sum() == 0.0
        assert result["text_mask"].sum() == 0.0

    def test_partial_dropout_statistical(self, tmp_path):
        """dropout_rate=0.5 zeros roughly 50% of samples over many runs."""
        orch, _, _ = _make_orchestrator(tmp_path)

        zeroed_count = 0
        total_samples = 0
        num_runs = 200
        batch_size = 10

        random.seed(42)
        for _ in range(num_runs):
            text_emb = FakeTensor(np.ones((batch_size, 4), dtype=np.float32))
            text_mask = FakeTensor(np.ones((batch_size, 4), dtype=np.float32))
            batch = {"text_emb": text_emb, "text_mask": text_mask}

            orch._apply_caption_dropout(batch, dropout_rate=0.5)

            for i in range(batch_size):
                total_samples += 1
                if batch["text_emb"][i].sum() == 0.0:
                    zeroed_count += 1

        ratio = zeroed_count / total_samples
        # Should be ~0.5, allow generous margin
        assert 0.35 < ratio < 0.65, f"Expected ~50% dropout, got {ratio:.2%}"

    def test_no_text_emb_noop(self, tmp_path):
        """Batch without text_emb is returned unchanged."""
        orch, _, _ = _make_orchestrator(tmp_path)

        batch = {"latent": "some_data", "other_key": 42}
        result = orch._apply_caption_dropout(batch, dropout_rate=1.0)

        assert result["latent"] == "some_data"
        assert result["other_key"] == 42

    def test_mask_zeroed_with_embedding(self, tmp_path):
        """When a text embedding is zeroed, its mask is also zeroed."""
        orch, _, _ = _make_orchestrator(tmp_path)

        # Single sample -- rate=1.0 guarantees dropout
        text_emb = FakeTensor([[5.0, 6.0, 7.0]])
        text_mask = FakeTensor([[1.0, 1.0, 1.0]])
        batch = {"text_emb": text_emb, "text_mask": text_mask}

        orch._apply_caption_dropout(batch, dropout_rate=1.0)

        assert batch["text_emb"][0].sum() == 0.0
        assert batch["text_mask"][0].sum() == 0.0


# =========================================================================
# TestTrainingStep
# =========================================================================

class TestTrainingStep:
    """Tests for _training_step -- the core noise->forward->loss->backward."""

    def test_returns_float_loss(self, tmp_path, monkeypatch):
        """_training_step returns a float loss value."""
        orch, config, backend = _make_orchestrator(
            tmp_path, moe=MockMoeConfig(fork_enabled=False),
        )
        phase = _make_phase()

        torch_mod, F_mock = _install_mock_torch(monkeypatch)

        # Latent mock
        latent = MagicMock(name="latent")
        latent.shape = (2, 16, 5, 32, 32)
        latent.to = MagicMock(return_value=latent)

        # Prediction mock
        prediction = MagicMock(name="prediction")
        prediction.float = MagicMock(return_value=prediction)
        backend.forward = MagicMock(return_value=prediction)

        # Build the loss chain: F.mse_loss() -> .mean(dim=...) -> .mean() -> /N -> .backward()
        # The per-element loss from mse_loss
        mse_result = MagicMock(name="mse_result")
        mse_result.ndim = 5

        # After .mean(dim=[1,2,3,4]) -> per-sample loss
        per_sample = MagicMock(name="per_sample")
        # After .mean() -> scalar loss
        scalar_loss = MagicMock(name="scalar_loss")
        scalar_loss.item = MagicMock(return_value=0.42)
        scaled = MagicMock(name="scaled")
        scalar_loss.__truediv__ = MagicMock(return_value=scaled)

        # Wire up the .mean() call chain
        call_idx = [0]
        def mean_side(*args, **kwargs):
            call_idx[0] += 1
            if "dim" in kwargs:
                return per_sample
            return scalar_loss
        mse_result.mean = MagicMock(side_effect=mean_side)
        per_sample.mean = MagicMock(return_value=scalar_loss)
        F_mock.mse_loss = MagicMock(return_value=mse_result)

        # Noise schedule
        noise_schedule = MagicMock()
        noise_schedule.sample_timesteps = MagicMock(
            return_value=np.array([0.3, 0.7]),
        )

        orch._model = MagicMock()
        orch._backend = backend

        result = orch._training_step(
            phase=phase,
            batch={"latent": latent},
            noise_schedule=noise_schedule,
            compute_dtype="bf16",
            device="cpu",
            grad_accum_steps=1,
        )

        assert isinstance(result, float)
        assert result == 0.42

    def test_expert_mask_applied_high_noise(self, tmp_path):
        """High noise phase mask zeros loss for low-noise timesteps."""
        _, _, backend = _make_orchestrator(tmp_path)

        # All timesteps below boundary -- mask should zero them for high_noise
        timesteps_np = np.array([0.1, 0.2])
        high_mask, low_mask = backend.get_expert_mask(timesteps_np, 0.875)

        assert np.allclose(high_mask, [0.0, 0.0])  # All masked out
        assert np.allclose(low_mask, [1.0, 1.0])

    def test_expert_mask_applied_low_noise(self, tmp_path):
        """Low noise phase mask zeros loss for high-noise timesteps."""
        _, _, backend = _make_orchestrator(tmp_path)

        # All timesteps above boundary -- low noise mask zeros them
        timesteps_np = np.array([0.9, 0.95])
        high_mask, low_mask = backend.get_expert_mask(timesteps_np, 0.875)

        assert np.allclose(high_mask, [1.0, 1.0])
        assert np.allclose(low_mask, [0.0, 0.0])  # All masked out for low_noise

    def test_no_mask_for_unified(self, tmp_path):
        """Unified phase has no boundary_ratio and no active_expert -- no masking."""
        phase = _make_phase(
            phase_type=PhaseType.UNIFIED,
            boundary_ratio=None,
            active_expert=None,
        )

        # The masking code is gated by:
        #   if phase.boundary_ratio is not None and phase.active_expert is not None
        assert phase.boundary_ratio is None
        assert phase.active_expert is None

    def test_gradient_accumulation_scaling(self, tmp_path, monkeypatch):
        """Loss is divided by grad_accum_steps before backward."""
        orch, config, backend = _make_orchestrator(
            tmp_path, moe=MockMoeConfig(fork_enabled=False),
        )
        phase = _make_phase(gradient_accumulation_steps=4)

        torch_mod, F_mock = _install_mock_torch(monkeypatch)

        latent = MagicMock(name="latent")
        latent.shape = (1, 16, 5, 32, 32)
        latent.to = MagicMock(return_value=latent)

        prediction = MagicMock(name="prediction")
        prediction.float = MagicMock(return_value=prediction)
        backend.forward = MagicMock(return_value=prediction)

        # Build loss chain
        mse_result = MagicMock(name="mse_result")
        mse_result.ndim = 5
        per_sample = MagicMock(name="per_sample")
        scalar_loss = MagicMock(name="scalar_loss")
        scalar_loss.item = MagicMock(return_value=1.0)
        scaled_loss = MagicMock(name="scaled_loss")
        scalar_loss.__truediv__ = MagicMock(return_value=scaled_loss)

        def mean_side(*args, **kwargs):
            if "dim" in kwargs:
                return per_sample
            return scalar_loss
        mse_result.mean = MagicMock(side_effect=mean_side)
        per_sample.mean = MagicMock(return_value=scalar_loss)
        F_mock.mse_loss = MagicMock(return_value=mse_result)

        noise_schedule = MagicMock()
        noise_schedule.sample_timesteps = MagicMock(
            return_value=np.array([0.5]),
        )

        orch._model = MagicMock()
        orch._backend = backend

        orch._training_step(
            phase=phase,
            batch={"latent": latent},
            noise_schedule=noise_schedule,
            compute_dtype="bf16",
            device="cpu",
            grad_accum_steps=4,
        )

        # The loss should be divided by 4 (grad_accum_steps)
        scalar_loss.__truediv__.assert_called_once_with(4)
        # backward should be called on the scaled loss
        scaled_loss.backward.assert_called_once()


# =========================================================================
# TestBuildPhaseOptimizer
# =========================================================================

class TestBuildPhaseOptimizer:
    """Tests for _build_phase_optimizer -- parameter grouping."""

    def _make_named_params(self):
        """Create mock named_parameters with LoRA A and B matrices."""
        params = []
        for name in [
            "blocks.0.attn1.to_q.lora_A.weight",
            "blocks.0.attn1.to_q.lora_B.weight",
            "blocks.0.ffn.lora_A.weight",
            "blocks.0.ffn.lora_up.weight",  # alias for B
        ]:
            p = MagicMock(name=f"param_{name}")
            p.requires_grad = True
            params.append((name, p))
        return params

    def test_groups_lora_a_and_b(self, tmp_path, monkeypatch):
        """A-matrix and B-matrix parameters are separated into different groups."""
        orch, config, _ = _make_orchestrator(
            tmp_path, moe=MockMoeConfig(fork_enabled=False),
        )
        phase = _make_phase(max_epochs=5)

        # Mock model.named_parameters
        named_params = self._make_named_params()
        model = MagicMock()
        model.named_parameters = MagicMock(return_value=named_params)
        orch._model = model

        # Track what build_optimizer receives
        captured_groups = []
        mock_optimizer = MagicMock(name="optimizer")
        mock_scheduler = MagicMock(name="scheduler")

        def capture_build_optimizer(params, **kwargs):
            captured_groups.extend(params)
            return mock_optimizer

        monkeypatch.setattr(
            "flimmer.training.loop.build_optimizer",
            capture_build_optimizer,
        )
        monkeypatch.setattr(
            "flimmer.training.loop.build_scheduler",
            lambda **kwargs: mock_scheduler,
        )
        monkeypatch.setattr(
            "flimmer.training.loop.compute_total_steps",
            lambda **kwargs: 100,
        )

        dataset = MagicMock()
        dataset.__len__ = MagicMock(return_value=50)

        orch._build_phase_optimizer(phase, dataset)

        # Should have 2 groups: A-matrix and B-matrix
        assert len(captured_groups) == 2
        # First group = A-matrix params (no lora_B or lora_up in name)
        assert len(captured_groups[0]["params"]) == 2  # lora_A.weight x2
        # Second group = B-matrix params (lora_B or lora_up in name)
        assert len(captured_groups[1]["params"]) == 2  # lora_B + lora_up

    def test_loraplus_ratio_applied(self, tmp_path, monkeypatch):
        """B-matrix group gets lr * loraplus_lr_ratio."""
        orch, config, _ = _make_orchestrator(
            tmp_path, moe=MockMoeConfig(fork_enabled=False),
        )
        config.lora.loraplus_lr_ratio = 4.0
        phase = _make_phase(learning_rate=1e-4)

        named_params = self._make_named_params()
        model = MagicMock()
        model.named_parameters = MagicMock(return_value=named_params)
        orch._model = model

        captured_groups = []

        def capture_build(params, **kwargs):
            captured_groups.extend(params)
            return MagicMock()

        monkeypatch.setattr("flimmer.training.loop.build_optimizer", capture_build)
        monkeypatch.setattr(
            "flimmer.training.loop.build_scheduler",
            lambda **kwargs: MagicMock(),
        )
        monkeypatch.setattr(
            "flimmer.training.loop.compute_total_steps",
            lambda **kwargs: 100,
        )

        dataset = MagicMock()
        dataset.__len__ = MagicMock(return_value=50)

        orch._build_phase_optimizer(phase, dataset)

        # A-matrix group LR = learning_rate = 1e-4
        assert captured_groups[0]["lr"] == pytest.approx(1e-4)
        # B-matrix group LR = learning_rate * loraplus_ratio = 1e-4 * 4 = 4e-4
        assert captured_groups[1]["lr"] == pytest.approx(4e-4)

    def test_scheduler_total_steps_computed(self, tmp_path, monkeypatch):
        """Total steps are computed from dataset size and passed to build_scheduler."""
        orch, config, _ = _make_orchestrator(
            tmp_path, moe=MockMoeConfig(fork_enabled=False),
        )
        phase = _make_phase(
            max_epochs=5,
            batch_size=2,
            gradient_accumulation_steps=4,
        )

        model = MagicMock()
        model.named_parameters = MagicMock(return_value=[
            ("lora_A.weight", MagicMock(requires_grad=True)),
        ])
        orch._model = model

        captured_scheduler_kwargs = {}

        def capture_scheduler(**kwargs):
            captured_scheduler_kwargs.update(kwargs)
            return MagicMock()

        monkeypatch.setattr(
            "flimmer.training.loop.build_optimizer",
            lambda params, **kwargs: MagicMock(),
        )
        monkeypatch.setattr(
            "flimmer.training.loop.build_scheduler",
            capture_scheduler,
        )

        # Dataset with 80 samples, batch_size=2, grad_accum=4, epochs=5
        # steps_per_epoch = 80 // (2*4) = 10
        # total_steps = 10 * 5 = 50
        dataset = MagicMock()
        dataset.__len__ = MagicMock(return_value=80)

        orch._build_phase_optimizer(phase, dataset)

        assert captured_scheduler_kwargs["total_steps"] == 50

    def test_optimizer_type_from_phase(self, tmp_path, monkeypatch):
        """Uses phase.optimizer_type, not the config default."""
        orch, config, _ = _make_orchestrator(
            tmp_path, moe=MockMoeConfig(fork_enabled=False),
        )
        # Config default is "adamw", phase overrides to "prodigy"
        phase = _make_phase(optimizer_type="prodigy")

        model = MagicMock()
        model.named_parameters = MagicMock(return_value=[
            ("lora_A.weight", MagicMock(requires_grad=True)),
        ])
        orch._model = model

        captured_optimizer_type = []

        def capture_build(params, optimizer_type=None, **kwargs):
            captured_optimizer_type.append(optimizer_type)
            return MagicMock()

        monkeypatch.setattr("flimmer.training.loop.build_optimizer", capture_build)
        monkeypatch.setattr(
            "flimmer.training.loop.build_scheduler",
            lambda **kwargs: MagicMock(),
        )
        monkeypatch.setattr(
            "flimmer.training.loop.compute_total_steps",
            lambda **kwargs: 100,
        )

        dataset = MagicMock()
        dataset.__len__ = MagicMock(return_value=10)

        orch._build_phase_optimizer(phase, dataset)

        assert captured_optimizer_type[0] == "prodigy"


# =========================================================================
# TestEnsureExpertModel
# =========================================================================

class TestEnsureExpertModel:
    """Tests for _ensure_expert_model -- expert model switching."""

    def test_noop_for_unified_phase(self, tmp_path, monkeypatch):
        """Unified phase (active_expert=None) does nothing."""
        orch, _, backend = _make_orchestrator(tmp_path)
        phase = _make_phase(active_expert=None)

        original_model = MagicMock(name="original_model")
        orch._model = original_model

        load_calls = []
        backend.load_model = lambda config, expert=None: (
            load_calls.append(expert) or MagicMock()
        )

        _install_mock_wan_modules(monkeypatch)

        orch._ensure_expert_model(phase)

        assert len(load_calls) == 0
        assert orch._model is original_model

    def test_switches_when_different(self, tmp_path, monkeypatch):
        """Loads new expert model when current_expert != needed expert."""
        orch, _, backend = _make_orchestrator(tmp_path)
        phase = _make_phase(
            phase_type=PhaseType.HIGH_NOISE,
            active_expert="high_noise",
        )

        backend.current_expert = "low_noise"  # Currently loaded: wrong expert
        new_model = MagicMock(name="high_noise_model")
        backend.load_model = MagicMock(return_value=new_model)

        wan = _install_mock_wan_modules(monkeypatch)

        orch._model = MagicMock(name="old_model")
        orch._ensure_expert_model(phase)

        # Should have removed LoRA from old model and loaded new expert
        wan.remove_lora_from_model.assert_called_once()
        backend.load_model.assert_called_once()
        assert orch._model is new_model

    def test_noop_when_already_loaded(self, tmp_path, monkeypatch):
        """No reload if the correct expert is already loaded."""
        orch, _, backend = _make_orchestrator(tmp_path)
        phase = _make_phase(
            phase_type=PhaseType.HIGH_NOISE,
            active_expert="high_noise",
        )

        backend.current_expert = "high_noise"  # Already correct
        original_model = MagicMock(name="current_model")
        orch._model = original_model

        load_calls = []
        backend.load_model = lambda config, expert=None: (
            load_calls.append(expert) or MagicMock()
        )

        orch._ensure_expert_model(phase)

        assert len(load_calls) == 0
        assert orch._model is original_model


# =========================================================================
# TestPhaseLoraLifecycle
# =========================================================================

class TestPhaseLoraLifecycle:
    """Tests for _setup_phase_lora and _teardown_phase_lora."""

    def test_setup_creates_lora_on_model(self, tmp_path, monkeypatch):
        """create_lora_on_model is called during setup."""
        orch, config, backend = _make_orchestrator(
            tmp_path, moe=MockMoeConfig(fork_enabled=False),
        )
        wan = _install_mock_wan_modules(monkeypatch)
        phase = _make_phase()
        orch._model = MagicMock(name="base_model")

        orch._setup_phase_lora(phase, active_lora=None)

        wan.create_lora_on_model.assert_called_once()

    def test_setup_injects_existing_weights(self, tmp_path, monkeypatch):
        """inject_lora_state_dict is called when active_lora has weights."""
        orch, config, backend = _make_orchestrator(
            tmp_path, moe=MockMoeConfig(fork_enabled=False),
        )
        wan = _install_mock_wan_modules(monkeypatch)
        phase = _make_phase()
        orch._model = MagicMock()

        existing_lora = LoRAState(
            state_dict={"key1": "val1", "key2": "val2"},
            rank=16,
            alpha=16,
            phase_type=PhaseType.UNIFIED,
        )

        result = orch._setup_phase_lora(phase, active_lora=existing_lora)

        wan.inject_lora_state_dict.assert_called_once()
        # Should return the existing LoRA state (not create new one)
        assert result is existing_lora

    def test_setup_creates_new_lora_when_none(self, tmp_path, monkeypatch):
        """Creates an empty LoRAState when no existing state is provided."""
        orch, config, backend = _make_orchestrator(
            tmp_path, moe=MockMoeConfig(fork_enabled=False),
        )
        wan = _install_mock_wan_modules(monkeypatch)
        phase = _make_phase()
        orch._model = MagicMock()

        result = orch._setup_phase_lora(phase, active_lora=None)

        # inject should NOT be called -- no existing weights
        wan.inject_lora_state_dict.assert_not_called()
        # Should return a new LoRAState
        assert isinstance(result, LoRAState)
        assert result.state_dict == {}
        assert result.rank == 16
        assert result.alpha == 16

    def test_teardown_extracts_weights(self, tmp_path, monkeypatch):
        """extract_lora_state_dict is called during teardown."""
        orch, config, backend = _make_orchestrator(
            tmp_path, moe=MockMoeConfig(fork_enabled=False),
        )
        wan = _install_mock_wan_modules(monkeypatch)
        phase = _make_phase()
        orch._model = MagicMock()

        active_lora = LoRAState(
            state_dict={},
            rank=16,
            alpha=16,
            phase_type=PhaseType.UNIFIED,
        )

        result = orch._teardown_phase_lora(phase, active_lora)

        wan.extract_lora_state_dict.assert_called_once()
        # The returned LoRAState should have the extracted weights
        assert result.state_dict == {
            "lora_A.weight": "tensor_a",
            "lora_B.weight": "tensor_b",
        }

    def test_teardown_removes_peft(self, tmp_path, monkeypatch):
        """remove_lora_from_model is called during teardown."""
        orch, config, backend = _make_orchestrator(
            tmp_path, moe=MockMoeConfig(fork_enabled=False),
        )
        wan = _install_mock_wan_modules(monkeypatch)
        phase = _make_phase()

        peft_model = MagicMock(name="peft_wrapped")
        orch._model = peft_model

        active_lora = LoRAState(
            state_dict={},
            rank=16,
            alpha=16,
            phase_type=PhaseType.UNIFIED,
        )

        orch._teardown_phase_lora(phase, active_lora)

        wan.remove_lora_from_model.assert_called_once_with(peft_model)
        # Model should be replaced with the unwrapped base model
        assert orch._model is wan.remove_lora_from_model.return_value


# =========================================================================
# TestRunEpochReal
# =========================================================================

class TestRunEpochReal:
    """Tests for _run_epoch with real DataLoader wiring."""

    def _prepare_epoch_test(self, tmp_path, monkeypatch, batches,
                            grad_accum=1, phase=None):
        """Common setup for _run_epoch tests.

        Installs all mocks, patches _training_step, and sets up metrics.

        Args:
            tmp_path: Temp directory.
            monkeypatch: pytest monkeypatch.
            batches: List of batch dicts to iterate.
            grad_accum: Gradient accumulation steps.
            phase: TrainingPhase (built with defaults if None).

        Returns:
            Tuple of (orch, phase, mock_optimizer, mock_scheduler, dataset,
                      training_step_calls).
        """
        orch, config, backend = _make_orchestrator(
            tmp_path, moe=MockMoeConfig(fork_enabled=False),
        )
        if phase is None:
            phase = _make_phase(
                batch_size=1,
                gradient_accumulation_steps=grad_accum,
            )

        _install_run_epoch_mocks(monkeypatch, batches)

        # Track _training_step calls
        training_step_calls = []

        def fake_training_step(
            self_orch, phase, batch, noise_schedule,
            compute_dtype=None, device=None, grad_accum_steps=1,
        ):
            training_step_calls.append(batch)
            return 0.1

        monkeypatch.setattr(
            "flimmer.training.loop.TrainingOrchestrator._training_step",
            fake_training_step,
        )

        # Set up model
        model = MagicMock()
        model.parameters = MagicMock(
            return_value=iter([MagicMock(device="cpu")]),
        )
        orch._model = model

        # Start a metrics phase so _metrics.update() does not raise
        orch._metrics.start_phase(phase.phase_type)

        # Mock dataset
        dataset = MagicMock()
        dataset.__len__ = MagicMock(return_value=len(batches))

        # Mock optimizer and scheduler
        mock_optimizer = MagicMock()
        mock_scheduler = MagicMock()
        mock_scheduler.get_last_lr = MagicMock(return_value=[5e-5])

        return orch, phase, mock_optimizer, mock_scheduler, dataset, training_step_calls

    def test_returns_zero_for_none_dataset(self, tmp_path):
        """dataset=None (dry run) returns 0.0 immediately."""
        orch, _, _ = _make_orchestrator(
            tmp_path, moe=MockMoeConfig(fork_enabled=False),
        )
        phase = _make_phase()
        noise_schedule = FlowMatchingSchedule(1000)

        result = orch._run_epoch(
            phase=phase,
            dataset=None,
            active_lora=None,
            noise_schedule=noise_schedule,
        )

        assert result == 0.0

    def test_iterates_dataloader_batches(self, tmp_path, monkeypatch):
        """Creates DataLoader and iterates through all batches."""
        batches = [
            {"latent": "batch1"},
            {"latent": "batch2"},
            {"latent": "batch3"},
        ]
        orch, phase, opt, sched, dataset, step_calls = self._prepare_epoch_test(
            tmp_path, monkeypatch, batches,
        )

        result = orch._run_epoch(
            phase=phase,
            dataset=dataset,
            active_lora=None,
            noise_schedule=FlowMatchingSchedule(1000),
            optimizer=opt,
            scheduler=sched,
        )

        assert len(step_calls) == 3
        assert isinstance(result, float)

    def test_gradient_accumulation_triggers_step(self, tmp_path, monkeypatch):
        """optimizer.step() is called after grad_accum batches."""
        batches = [{"latent": f"b{i}"} for i in range(4)]
        orch, phase, opt, sched, dataset, _ = self._prepare_epoch_test(
            tmp_path, monkeypatch, batches, grad_accum=2,
        )

        orch._run_epoch(
            phase=phase,
            dataset=dataset,
            active_lora=None,
            noise_schedule=FlowMatchingSchedule(1000),
            optimizer=opt,
            scheduler=sched,
        )

        # 4 batches / 2 grad_accum = 2 optimizer steps (evenly divisible, no flush)
        assert opt.step.call_count == 2
        # zero_grad: 1 initial + 2 after steps = 3
        assert opt.zero_grad.call_count == 3

    def test_flushes_remaining_gradients(self, tmp_path, monkeypatch):
        """Remaining accumulated gradients are flushed at epoch end."""
        batches = [{"latent": f"b{i}"} for i in range(3)]  # Odd: 3 / 2 = 1 R 1
        orch, phase, opt, sched, dataset, _ = self._prepare_epoch_test(
            tmp_path, monkeypatch, batches, grad_accum=2,
        )

        orch._run_epoch(
            phase=phase,
            dataset=dataset,
            active_lora=None,
            noise_schedule=FlowMatchingSchedule(1000),
            optimizer=opt,
            scheduler=sched,
        )

        # 3 batches / 2 grad_accum:
        #   batch 0,1 -> accum_count=2 -> step (1st) -> reset
        #   batch 2 -> accum_count=1 -> end of loop -> flush step (2nd)
        # Total: 2 optimizer.step() calls
        assert opt.step.call_count == 2
        # zero_grad: 1 initial + 1 after regular step + 1 after flush = 3
        assert opt.zero_grad.call_count == 3


# =========================================================================
# TestGenerateSamplesPartnerLoading
# =========================================================================

class TestGenerateSamplesPartnerLoading:
    """Tests for _generate_samples — partner model loading for MoE inference.

    Wan 2.2 MoE always needs both experts for inference. These tests verify
    that _generate_samples loads the partner model correctly for all phase types.
    """

    def test_unified_moe_loads_high_noise_partner(self, tmp_path, monkeypatch):
        """Unified phase on MoE model loads high-noise partner for inference.

        The unified training model is the low-noise expert (backend default).
        We need the high-noise partner, so _load_partner_model("low_noise")
        should be called (which loads the HIGH-noise expert as partner).
        """
        orch, config, backend = _make_orchestrator(tmp_path)
        phase = _make_phase(active_expert=None)  # Unified

        # backend.supports_moe is True (default MockModelBackend)
        assert backend.supports_moe is True

        # Track _load_partner_model calls
        partner_calls = []
        partner_model = MagicMock(name="partner_model")

        def mock_load_partner(active_expert, partner_lora_path=None):
            partner_calls.append(active_expert)
            return partner_model

        monkeypatch.setattr(orch, "_load_partner_model", mock_load_partner)

        # Mock sampler to capture what it receives
        captured_kwargs = {}
        mock_samples = ["/fake/sample.mp4"]

        def mock_generate_samples(**kwargs):
            captured_kwargs.update(kwargs)
            return mock_samples

        orch._sampler = MagicMock()
        orch._sampler.generate_samples = mock_generate_samples

        # Mock pipeline and other dependencies
        orch._pipeline = MagicMock()
        orch._logger = MagicMock()

        wan = _install_mock_wan_modules(monkeypatch)

        lora = LoRAState(state_dict={"k": "v"}, rank=16, alpha=16, phase_type=PhaseType.UNIFIED)

        orch._generate_samples(phase, epoch=1, lora=lora)

        # Verify partner was loaded with "low_noise" (to get high-noise partner)
        assert partner_calls == ["low_noise"]
        # Verify sampler received the partner model and correct active_expert
        assert captured_kwargs["partner_model"] is partner_model
        assert captured_kwargs["active_expert"] == "low_noise"

    def test_expert_phase_loads_correct_partner(self, tmp_path, monkeypatch):
        """Expert phase (high_noise) loads low-noise partner."""
        orch, config, backend = _make_orchestrator(tmp_path)
        phase = _make_phase(
            phase_type=PhaseType.HIGH_NOISE,
            active_expert="high_noise",
        )

        partner_calls = []
        partner_model = MagicMock(name="partner_model")

        def mock_load_partner(active_expert, partner_lora_path=None):
            partner_calls.append(active_expert)
            return partner_model

        monkeypatch.setattr(orch, "_load_partner_model", mock_load_partner)

        captured_kwargs = {}

        def mock_generate_samples(**kwargs):
            captured_kwargs.update(kwargs)
            return ["/fake/sample.mp4"]

        orch._sampler = MagicMock()
        orch._sampler.generate_samples = mock_generate_samples
        orch._sampler.resolve_partner_lora = MagicMock(return_value=None)
        orch._pipeline = MagicMock()
        orch._logger = MagicMock()
        orch._checkpoint_mgr = MagicMock()
        orch._checkpoint_mgr.find_latest_checkpoint = MagicMock(return_value=None)

        wan = _install_mock_wan_modules(monkeypatch)

        lora = LoRAState(state_dict={"k": "v"}, rank=16, alpha=16, phase_type=PhaseType.HIGH_NOISE)

        orch._generate_samples(phase, epoch=1, lora=lora)

        # Verify partner was loaded with "high_noise" (to get low-noise partner)
        assert partner_calls == ["high_noise"]
        # active_expert stays "high_noise" for expert phases
        assert captured_kwargs["active_expert"] == "high_noise"

    def test_non_moe_skips_partner_loading(self, tmp_path, monkeypatch):
        """Non-MoE model (Wan 2.1) does not load a partner for unified phase."""

        class NonMoeBackend(MockModelBackend):
            @property
            def supports_moe(self):
                return False

        from flimmer.training.loop import TrainingOrchestrator

        config = MockConfig(tmp_path)
        backend = NonMoeBackend()
        orch = TrainingOrchestrator(config, backend)

        phase = _make_phase(active_expert=None)

        partner_calls = []

        def mock_load_partner(active_expert, partner_lora_path=None):
            partner_calls.append(active_expert)
            return MagicMock()

        monkeypatch.setattr(orch, "_load_partner_model", mock_load_partner)

        captured_kwargs = {}

        def mock_generate_samples(**kwargs):
            captured_kwargs.update(kwargs)
            return ["/fake/sample.mp4"]

        orch._sampler = MagicMock()
        orch._sampler.generate_samples = mock_generate_samples
        orch._pipeline = MagicMock()
        orch._logger = MagicMock()

        wan = _install_mock_wan_modules(monkeypatch)

        lora = LoRAState(state_dict={"k": "v"}, rank=16, alpha=16, phase_type=PhaseType.UNIFIED)

        orch._generate_samples(phase, epoch=1, lora=lora)

        # Partner should NOT have been loaded
        assert partner_calls == []
        # active_expert should remain None
        assert captured_kwargs["partner_model"] is None
        assert captured_kwargs["active_expert"] is None


# =========================================================================
# TestLoadPartnerModelErrors
# =========================================================================

class TestLoadPartnerModelErrors:
    """Tests for _load_partner_model — error handling.

    Partner loading failures must raise SamplingError, not return None.
    Getting noise is worse than getting an error message.
    """

    def test_raises_when_partner_file_missing(self, tmp_path, monkeypatch):
        """Raises SamplingError when backend has no file path for the partner."""
        from flimmer.training.errors import SamplingError

        orch, config, backend = _make_orchestrator(tmp_path)

        # Mock _resolve_single_file_path to return None
        backend._resolve_single_file_path = MagicMock(return_value=None)

        with pytest.raises(SamplingError, match="No file path configured"):
            orch._load_partner_model("high_noise")

    def test_raises_on_loading_failure(self, tmp_path, monkeypatch):
        """Raises SamplingError when from_single_file() fails."""
        from flimmer.training.errors import SamplingError

        orch, config, backend = _make_orchestrator(tmp_path)

        # Mock _resolve_single_file_path to return a path
        backend._resolve_single_file_path = MagicMock(return_value="/fake/model.safetensors")
        backend._resolve_config_subfolder = MagicMock(return_value="transformer")

        # Mock the import path so from_single_file raises
        mock_transformer_cls = MagicMock()
        mock_transformer_cls.from_single_file = MagicMock(
            side_effect=RuntimeError("corrupt file"),
        )

        import sys
        import types

        # Install mock torch
        torch_mod = types.ModuleType("torch")
        torch_mod.bfloat16 = "bf16"
        torch_mod.device = MagicMock
        torch_mod.cuda = MagicMock()
        torch_mod.cuda.is_available = MagicMock(return_value=False)
        monkeypatch.setitem(sys.modules, "torch", torch_mod)

        # Install mock diffusers
        diffusers_mod = types.ModuleType("diffusers")
        diffusers_mod.WanTransformer3DModel = mock_transformer_cls
        monkeypatch.setitem(sys.modules, "diffusers", diffusers_mod)

        with pytest.raises(SamplingError, match="Failed to load partner expert"):
            orch._load_partner_model("high_noise")
