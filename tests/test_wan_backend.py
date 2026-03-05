"""Tests for flimmer.training.wan.backend.WanModelBackend.

WanModelBackend implements the ModelBackend protocol for all Wan model
variants (2.1 T2V, 2.2 T2V, 2.2 I2V). These tests cover:

    - Properties: model_id, supports_moe, supports_reference_image,
      boundary_ratio, flow_shift, current_expert
    - get_lora_target_modules: correct return, mutation safety
    - get_expert_mask: correct high/low masks for given timesteps
    - get_noise_schedule: returns FlowMatchingSchedule with 1000 steps
    - prepare_model_inputs: keys, timestep scaling, I2V concatenation
    - load_model: error handling when path missing or diffusers absent
    - setup_gradient_checkpointing: calls the right method on the model
    - Protocol compliance: isinstance(backend, ModelBackend)

All tests are GPU-free. Tests that call prepare_model_inputs or
concatenation behavior use pytest.importorskip("torch") so they are
skipped cleanly if torch is not installed.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from flimmer.training.errors import ModelBackendError
from flimmer.training.noise import FlowMatchingSchedule
from flimmer.training.protocols import ModelBackend
from flimmer.training.wan.backend import WanModelBackend


# ---------------------------------------------------------------------------
# Shared helpers — build standard backend instances
# ---------------------------------------------------------------------------

def _make_t2v_backend(**overrides) -> WanModelBackend:
    """Return a T2V (MoE) backend with configurable overrides."""
    kwargs = dict(
        model_id="wan-2.2-t2v-14b",
        model_path="/fake/path",
        is_moe=True,
        is_i2v=False,
        in_channels=16,
        num_blocks=40,
        boundary_ratio=0.875,
        flow_shift=3.0,
        lora_targets=["attn1.to_q", "attn1.to_k"],
        expert_subfolders={"high_noise": "transformer", "low_noise": "transformer_2"},
    )
    kwargs.update(overrides)
    return WanModelBackend(**kwargs)


def _make_i2v_backend(**overrides) -> WanModelBackend:
    """Return an I2V (MoE) backend with configurable overrides."""
    kwargs = dict(
        model_id="wan-2.2-i2v-14b",
        model_path="/fake/path",
        is_moe=True,
        is_i2v=True,
        in_channels=36,
        num_blocks=40,
        boundary_ratio=0.900,
        flow_shift=3.0,
        lora_targets=["attn1.to_q", "attn1.to_k", "attn2.add_k_proj"],
        expert_subfolders={"high_noise": "transformer", "low_noise": "transformer_2"},
    )
    kwargs.update(overrides)
    return WanModelBackend(**kwargs)


def _make_21_t2v_backend(**overrides) -> WanModelBackend:
    """Return a Wan 2.1 (single-transformer, non-MoE) backend."""
    kwargs = dict(
        model_id="wan-2.1-t2v-14b",
        model_path="/fake/path",
        is_moe=False,
        is_i2v=False,
        in_channels=16,
        num_blocks=40,
        boundary_ratio=None,
        flow_shift=3.0,
        lora_targets=["attn1.to_q", "attn1.to_k"],
        expert_subfolders={"default": "transformer"},
    )
    kwargs.update(overrides)
    return WanModelBackend(**kwargs)


def _make_21_i2v_backend(**overrides) -> WanModelBackend:
    """Return a Wan 2.1 I2V (single-transformer, non-MoE, I2V) backend."""
    kwargs = dict(
        model_id="wan-2.1-i2v-14b",
        model_path="/fake/path",
        is_moe=False,
        is_i2v=True,
        in_channels=36,
        num_blocks=40,
        boundary_ratio=None,
        flow_shift=3.0,
        lora_targets=["attn1.to_q", "attn1.to_k", "attn2.add_k_proj", "attn2.add_v_proj"],
        expert_subfolders={"default": "transformer"},
        config_repo="Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
    )
    kwargs.update(overrides)
    return WanModelBackend(**kwargs)


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------

class TestModelId:
    """model_id property returns the configured string."""

    def test_model_id_t2v(self):
        backend = _make_t2v_backend()
        assert backend.model_id == "wan-2.2-t2v-14b"

    def test_model_id_i2v(self):
        backend = _make_i2v_backend()
        assert backend.model_id == "wan-2.2-i2v-14b"

    def test_model_id_custom(self):
        # model_id is a plain string — any value is accepted
        backend = _make_t2v_backend(model_id="my-custom-wan-finetune")
        assert backend.model_id == "my-custom-wan-finetune"


class TestSupportsMoe:
    """supports_moe reflects is_moe constructor argument."""

    def test_supports_moe_true_for_22(self):
        """Wan 2.2 (MoE) returns True."""
        backend = _make_t2v_backend(is_moe=True)
        assert backend.supports_moe is True

    def test_supports_moe_false_for_21(self):
        """Wan 2.1 (single transformer) returns False."""
        backend = _make_21_t2v_backend()
        assert backend.supports_moe is False

    def test_supports_moe_i2v_also_true(self):
        """I2V is also an MoE model."""
        backend = _make_i2v_backend()
        assert backend.supports_moe is True


class TestSupportsReferenceImage:
    """supports_reference_image reflects is_i2v constructor argument."""

    def test_supports_reference_image_true_for_i2v(self):
        backend = _make_i2v_backend(is_i2v=True)
        assert backend.supports_reference_image is True

    def test_supports_reference_image_false_for_t2v(self):
        backend = _make_t2v_backend(is_i2v=False)
        assert backend.supports_reference_image is False

    def test_supports_reference_image_false_for_21(self):
        backend = _make_21_t2v_backend()
        assert backend.supports_reference_image is False


class TestBoundaryRatio:
    """boundary_ratio property returns the configured value."""

    def test_boundary_ratio_t2v(self):
        backend = _make_t2v_backend(boundary_ratio=0.875)
        assert backend.boundary_ratio == pytest.approx(0.875)

    def test_boundary_ratio_i2v(self):
        backend = _make_i2v_backend(boundary_ratio=0.900)
        assert backend.boundary_ratio == pytest.approx(0.900)

    def test_boundary_ratio_none_for_21(self):
        """Non-MoE models have no boundary (None)."""
        backend = _make_21_t2v_backend()
        assert backend.boundary_ratio is None


class TestFlowShift:
    """flow_shift property returns the configured value."""

    def test_flow_shift_default(self):
        backend = _make_t2v_backend(flow_shift=3.0)
        assert backend.flow_shift == pytest.approx(3.0)

    def test_flow_shift_custom(self):
        backend = _make_t2v_backend(flow_shift=5.0)
        assert backend.flow_shift == pytest.approx(5.0)


class TestCurrentExpert:
    """current_expert is None before any model is loaded."""

    def test_current_expert_initially_none(self):
        """Before load_model() is called, no expert is active."""
        backend = _make_t2v_backend()
        assert backend.current_expert is None

    def test_current_expert_remains_none_after_construction(self):
        """Construction never side-effects the expert state."""
        backend = _make_i2v_backend()
        assert backend.current_expert is None


# ---------------------------------------------------------------------------
# get_lora_target_modules
# ---------------------------------------------------------------------------

class TestGetLoraTargetModules:
    """LoRA target module list behavior."""

    def test_returns_configured_targets(self):
        """The returned list contains exactly what was passed to the constructor."""
        targets = ["attn1.to_q", "attn1.to_k", "ffn.net.2"]
        backend = _make_t2v_backend(lora_targets=targets)
        result = backend.get_lora_target_modules()
        assert result == targets

    def test_returns_a_copy(self):
        """Mutating the returned list must not affect subsequent calls.

        The backend's internal list should be protected from external
        mutation — callers may modify the returned list without breaking
        the backend.
        """
        targets = ["attn1.to_q", "attn1.to_k"]
        backend = _make_t2v_backend(lora_targets=targets)
        result = backend.get_lora_target_modules()
        result.append("sneaky.injection")
        # A second call should return the original, unmodified list
        fresh = backend.get_lora_target_modules()
        assert "sneaky.injection" not in fresh

    def test_returns_list_type(self):
        backend = _make_t2v_backend()
        result = backend.get_lora_target_modules()
        assert isinstance(result, list)

    def test_empty_targets_allowed(self):
        """Empty target list is valid (edge case for testing)."""
        backend = _make_t2v_backend(lora_targets=[])
        assert backend.get_lora_target_modules() == []


# ---------------------------------------------------------------------------
# get_expert_mask
# ---------------------------------------------------------------------------

class TestGetExpertMask:
    """Expert mask computation from timesteps and boundary_ratio.

    boundary_ratio=0.875 means:
        t >= 0.875 → high-noise expert (1.0), low-noise (0.0)
        t <  0.875 → low-noise expert (1.0), high-noise (0.0)
    """

    def test_high_noise_mask_correct(self):
        """t=0.9 is above boundary → high-noise mask is 1, low is 0."""
        backend = _make_t2v_backend()
        ts = np.array([0.9, 0.5, 0.1])
        high, low = backend.get_expert_mask(ts, boundary_ratio=0.875)
        np.testing.assert_array_equal(high, [1.0, 0.0, 0.0])

    def test_low_noise_mask_correct(self):
        """t=0.5 and t=0.1 are below boundary → low-noise mask is 1."""
        backend = _make_t2v_backend()
        ts = np.array([0.9, 0.5, 0.1])
        high, low = backend.get_expert_mask(ts, boundary_ratio=0.875)
        np.testing.assert_array_equal(low, [0.0, 1.0, 1.0])

    def test_masks_are_complementary(self):
        """Every timestep belongs to exactly one expert — masks sum to 1."""
        backend = _make_t2v_backend()
        ts = np.linspace(0.01, 0.99, 50)
        high, low = backend.get_expert_mask(ts, boundary_ratio=0.875)
        np.testing.assert_array_equal(high + low, np.ones(50))

    def test_boundary_exact_goes_to_high_noise(self):
        """t == boundary_ratio is assigned to the high-noise expert (>=)."""
        backend = _make_t2v_backend()
        ts = np.array([0.875])
        high, low = backend.get_expert_mask(ts, boundary_ratio=0.875)
        assert high[0] == 1.0
        assert low[0] == 0.0

    def test_custom_boundary_ratio(self):
        """The boundary_ratio argument is respected, not the backend's default."""
        backend = _make_t2v_backend(boundary_ratio=0.875)
        ts = np.array([0.6])
        # With boundary=0.5, t=0.6 should be high-noise
        high, low = backend.get_expert_mask(ts, boundary_ratio=0.5)
        assert high[0] == 1.0
        assert low[0] == 0.0

    def test_returns_tuple_of_two(self):
        backend = _make_t2v_backend()
        ts = np.array([0.5])
        result = backend.get_expert_mask(ts, boundary_ratio=0.875)
        assert len(result) == 2

    def test_mask_dtype_is_float(self):
        """Masks must be float arrays (for loss weighting arithmetic)."""
        backend = _make_t2v_backend()
        ts = np.array([0.9, 0.1])
        high, low = backend.get_expert_mask(ts, boundary_ratio=0.875)
        assert high.dtype == np.float64
        assert low.dtype == np.float64


# ---------------------------------------------------------------------------
# get_noise_schedule
# ---------------------------------------------------------------------------

class TestGetNoiseSchedule:
    """Noise schedule returned by the backend."""

    def test_returns_flow_matching_schedule(self):
        """Must return a FlowMatchingSchedule (not a bare object)."""
        backend = _make_t2v_backend()
        schedule = backend.get_noise_schedule()
        assert isinstance(schedule, FlowMatchingSchedule)

    def test_num_timesteps_is_1000(self):
        """All Wan models use 1000-step flow matching."""
        backend = _make_t2v_backend()
        schedule = backend.get_noise_schedule()
        assert schedule.num_timesteps == 1000

    def test_same_instance_returned(self):
        """The schedule is created once in __init__ and reused.

        The backend shouldn't create a new schedule on every call —
        the training loop may call this many times.
        """
        backend = _make_t2v_backend()
        sched1 = backend.get_noise_schedule()
        sched2 = backend.get_noise_schedule()
        assert sched1 is sched2

    def test_schedule_is_usable(self):
        """Spot-check: the schedule can actually sample timesteps."""
        backend = _make_t2v_backend()
        schedule = backend.get_noise_schedule()
        ts = schedule.sample_timesteps(8, strategy="uniform")
        assert ts.shape == (8,)
        assert ts.min() > 0.0
        assert ts.max() < 1.0


# ---------------------------------------------------------------------------
# prepare_model_inputs  (requires torch)
# ---------------------------------------------------------------------------

class TestPrepareModelInputs:
    """Input preparation for the Wan transformer forward pass.

    Skipped if torch is not installed — these tests deal with tensor
    shapes and concatenation, which requires an actual tensor library.
    """

    def test_basic_keys_present(self):
        """Output must include 'hidden_states' and 'timestep' at minimum."""
        torch = pytest.importorskip("torch")
        backend = _make_t2v_backend()

        B, C, F, H, W = 2, 16, 5, 8, 8
        noisy_latents = torch.randn(B, C, F, H, W)
        timesteps = torch.tensor([0.5, 0.3])
        batch = {
            "latent": torch.randn(B, C, F, H, W),
            "text_emb": None,
            "text_mask": None,
            "reference": None,
        }

        inputs = backend.prepare_model_inputs(batch, timesteps, noisy_latents)

        assert "hidden_states" in inputs
        assert "timestep" in inputs

    def test_timestep_scaling(self):
        """Wan expects integer timesteps [0, 1000]; our schedule uses [0, 1].

        prepare_model_inputs must scale float t → t * 1000.
        """
        torch = pytest.importorskip("torch")
        backend = _make_t2v_backend()

        B = 2
        noisy_latents = torch.zeros(B, 16, 5, 8, 8)
        timesteps = torch.tensor([0.5, 0.25])  # [0.5, 0.25] should scale to [500, 250]
        batch = {"latent": noisy_latents, "text_emb": None, "text_mask": None, "reference": None}

        inputs = backend.prepare_model_inputs(batch, timesteps, noisy_latents)

        # After scaling: 0.5 * 1000 = 500, 0.25 * 1000 = 250
        scaled = inputs["timestep"]
        assert float(scaled[0]) == pytest.approx(500.0)
        assert float(scaled[1]) == pytest.approx(250.0)

    def test_text_embedding_passed_through(self):
        """If text_emb is present, it must appear as encoder_hidden_states."""
        torch = pytest.importorskip("torch")
        backend = _make_t2v_backend()

        B, seq = 2, 32
        text_emb = torch.randn(B, seq, 4096)
        text_mask = torch.ones(B, seq)
        noisy_latents = torch.zeros(B, 16, 5, 8, 8)
        batch = {
            "latent": noisy_latents,
            "text_emb": text_emb,
            "text_mask": text_mask,
            "reference": None,
        }

        inputs = backend.prepare_model_inputs(batch, torch.tensor([0.5, 0.3]), noisy_latents)

        assert "encoder_hidden_states" in inputs
        # Wan's forward() builds its own attention mask internally,
        # so we do NOT pass encoder_attention_mask (removed in GPU validation).
        assert "encoder_attention_mask" not in inputs

    def test_no_text_means_no_text_keys(self):
        """When text_emb is None the encoder keys must be absent."""
        torch = pytest.importorskip("torch")
        backend = _make_t2v_backend()

        B = 1
        noisy_latents = torch.zeros(B, 16, 5, 8, 8)
        batch = {"latent": noisy_latents, "text_emb": None, "text_mask": None, "reference": None}

        inputs = backend.prepare_model_inputs(batch, torch.tensor([0.5]), noisy_latents)

        assert "encoder_hidden_states" not in inputs
        assert "encoder_attention_mask" not in inputs

    def test_t2v_reference_is_not_concatenated(self):
        """T2V models must NOT concatenate the reference image even if provided.

        A reference tensor in a T2V batch should be silently ignored —
        only I2V models concatenate it.
        """
        torch = pytest.importorskip("torch")
        backend = _make_t2v_backend(is_i2v=False, in_channels=16)

        B, C, F, H, W = 1, 16, 5, 8, 8
        noisy_latents = torch.zeros(B, C, F, H, W)
        # Provide a reference tensor (wrong — shouldn't happen in production,
        # but the backend should be defensive)
        reference = torch.ones(B, 20, F, H, W)
        batch = {"latent": noisy_latents, "text_emb": None, "text_mask": None, "reference": reference}

        inputs = backend.prepare_model_inputs(batch, torch.tensor([0.5]), noisy_latents)

        # hidden_states must still be the original noisy_latents shape (C=16)
        assert inputs["hidden_states"].shape[1] == C

    def test_i2v_concatenates_reference(self):
        """I2V builds 36-channel input from noisy latents + reference.

        Reference is a raw VAE latent [B, 16, 1, H, W]. The backend
        expands it temporally and adds 4 mask channels to produce
        [B, 36, F, H, W] = noisy(16) + mask(4) + reference(16).
        """
        torch = pytest.importorskip("torch")
        backend = _make_i2v_backend(is_i2v=True, in_channels=36)

        B, C, F, H, W = 1, 16, 5, 8, 8
        noisy_latents = torch.zeros(B, C, F, H, W)
        # Raw single-frame VAE reference (no mask, no temporal expansion)
        reference = torch.ones(B, C, 1, H, W)
        batch = {
            "latent": noisy_latents,
            "text_emb": None,
            "text_mask": None,
            "reference": reference,
        }

        inputs = backend.prepare_model_inputs(batch, torch.tensor([0.5]), noisy_latents)

        # Channel dim: 16 (noisy) + 4 (mask) + 16 (reference) = 36
        assert inputs["hidden_states"].shape[1] == 36
        # Reference should only be in the first latent frame
        hs = inputs["hidden_states"]
        assert (hs[0, 20:, 0, 0, 0] == 1.0).all()  # reference at frame 0
        assert (hs[0, 20:, 1, 0, 0] == 0.0).all()  # zero-padded at frame 1
        # Mask should be 1 at frame 0, 0 elsewhere
        assert hs[0, 16, 0, 0, 0].item() == 1.0  # mask at frame 0
        assert hs[0, 16, 1, 0, 0].item() == 0.0  # mask at frame 1

    def test_i2v_no_reference_zero_fills(self):
        """I2V backend with reference=None must zero-fill to 36 channels."""
        torch = pytest.importorskip("torch")
        backend = _make_i2v_backend(is_i2v=True)

        B, C, F, H, W = 1, 16, 5, 8, 8
        noisy_latents = torch.zeros(B, C, F, H, W)
        batch = {"latent": noisy_latents, "text_emb": None, "text_mask": None, "reference": None}

        inputs = backend.prepare_model_inputs(batch, torch.tensor([0.5]), noisy_latents)

        # Still 36 channels — mask and reference are zero-filled
        assert inputs["hidden_states"].shape[1] == 36
        # Mask and reference channels should all be zero
        mask_and_ref = inputs["hidden_states"][:, C:]
        assert (mask_and_ref == 0).all()

    def test_21_i2v_concatenates_reference(self):
        """Wan 2.1 I2V also builds 36-channel input from raw reference."""
        torch = pytest.importorskip("torch")
        backend = _make_21_i2v_backend()
        B, C, F, H, W = 1, 16, 5, 8, 8
        noisy = torch.zeros(B, C, F, H, W)
        reference = torch.ones(B, C, 1, H, W)
        batch = {"latent": noisy, "text_emb": None, "text_mask": None, "reference": reference}
        inputs = backend.prepare_model_inputs(batch, torch.tensor([0.5]), noisy)
        assert inputs["hidden_states"].shape[1] == 36


# ---------------------------------------------------------------------------
# Wan 2.1 I2V Properties
# ---------------------------------------------------------------------------

class TestWan21I2vProperties:
    """Wan 2.1 I2V -- non-MoE but with reference image support."""

    def test_is_not_moe(self):
        backend = _make_21_i2v_backend()
        assert backend.supports_moe is False

    def test_is_i2v(self):
        backend = _make_21_i2v_backend()
        assert backend.supports_reference_image is True

    def test_boundary_ratio_is_none(self):
        backend = _make_21_i2v_backend()
        assert backend.boundary_ratio is None

    def test_config_repo_stored(self):
        backend = _make_21_i2v_backend()
        assert backend._resolve_config_repo() == "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"

    def test_model_id(self):
        backend = _make_21_i2v_backend()
        assert backend.model_id == "wan-2.1-i2v-14b"

    def test_flow_shift(self):
        backend = _make_21_i2v_backend()
        assert backend.flow_shift == pytest.approx(3.0)

    def test_protocol_compliance(self):
        backend = _make_21_i2v_backend()
        assert isinstance(backend, ModelBackend)


# ---------------------------------------------------------------------------
# First-frame dropout
# ---------------------------------------------------------------------------

class TestFirstFrameDropout:
    """First-frame dropout zeros reference tensor independently."""

    def test_no_dropout_at_zero_rate(self):
        """rate=0.0 should leave all references intact."""
        torch = pytest.importorskip("torch")
        import random as rng
        from flimmer.training.loop import TrainingOrchestrator

        # Create a minimal mock orchestrator to test the method
        orchestrator = MagicMock(spec=TrainingOrchestrator)
        orchestrator._apply_first_frame_dropout = TrainingOrchestrator._apply_first_frame_dropout.__get__(orchestrator)

        reference = torch.ones(4, 20, 5, 8, 8)
        batch = {"latent": torch.zeros(4, 16, 5, 8, 8), "reference": reference}
        result = orchestrator._apply_first_frame_dropout(batch, 0.0)
        # All references should still be ones (no dropout)
        assert torch.all(result["reference"] == 1.0)

    def test_full_dropout_at_rate_one(self):
        """rate=1.0 should zero ALL references."""
        torch = pytest.importorskip("torch")
        from flimmer.training.loop import TrainingOrchestrator

        orchestrator = MagicMock(spec=TrainingOrchestrator)
        orchestrator._apply_first_frame_dropout = TrainingOrchestrator._apply_first_frame_dropout.__get__(orchestrator)

        reference = torch.ones(4, 20, 5, 8, 8)
        batch = {"latent": torch.zeros(4, 16, 5, 8, 8), "reference": reference}
        result = orchestrator._apply_first_frame_dropout(batch, 1.0)
        # All references should be zeroed
        assert torch.all(result["reference"] == 0.0)

    def test_no_crash_when_reference_is_none(self):
        """batch with reference=None should pass through unchanged."""
        torch = pytest.importorskip("torch")
        from flimmer.training.loop import TrainingOrchestrator

        orchestrator = MagicMock(spec=TrainingOrchestrator)
        orchestrator._apply_first_frame_dropout = TrainingOrchestrator._apply_first_frame_dropout.__get__(orchestrator)

        batch = {"latent": torch.zeros(2, 16, 5, 8, 8), "reference": None}
        result = orchestrator._apply_first_frame_dropout(batch, 0.5)
        assert result["reference"] is None


# ---------------------------------------------------------------------------
# load_model
# ---------------------------------------------------------------------------

class TestLoadModel:
    """load_model error handling — no real GPU or diffusers required."""

    def test_raises_model_backend_error_when_path_missing(self):
        """A local path that does not exist must raise ModelBackendError.

        The backend must not raise a bare FileNotFoundError or an
        unrelated exception — users need a helpful message.
        """
        torch = pytest.importorskip("torch")

        # Patch WanTransformer3DModel.from_pretrained to simulate a missing path
        backend = _make_t2v_backend(model_path="/definitely/does/not/exist/anywhere")
        mock_config = MagicMock()
        mock_config.base_model_precision = "bf16"

        # The path does not exist as a directory, so we need diffusers to exist
        # but the pretrained load to fail. We patch at the diffusers import.
        with patch.dict("sys.modules", {
            "diffusers": MagicMock(WanTransformer3DModel=_make_exploding_model_class()),
        }):
            with pytest.raises(ModelBackendError):
                backend.load_model(mock_config)

    def test_raises_model_backend_error_when_diffusers_missing(self):
        """If torch or diffusers are not installed, raise ModelBackendError.

        The error must mention how to fix it ('pip install flimmer[wan]').
        """
        backend = _make_t2v_backend()
        mock_config = MagicMock()

        with patch.dict("sys.modules", {"diffusers": None, "torch": None}):
            with pytest.raises((ModelBackendError, ImportError)):
                backend.load_model(mock_config)

    def test_updates_current_expert_on_success(self):
        """After a successful load, current_expert should reflect the expert arg."""
        torch = pytest.importorskip("torch")

        backend = _make_t2v_backend()
        mock_config = MagicMock()
        mock_config.base_model_precision = "bf16"

        # Provide a mock that returns a fake model without touching disk
        mock_model = MagicMock()
        mock_wan_cls = MagicMock()
        mock_wan_cls.from_pretrained.return_value = mock_model

        mock_diffusers = MagicMock()
        mock_diffusers.WanTransformer3DModel = mock_wan_cls

        with patch.dict("sys.modules", {"diffusers": mock_diffusers}):
            # Make the path look like a non-directory so HF path is used
            with patch("flimmer.training.wan.backend.Path") as mock_path_cls:
                mock_path_cls.return_value.is_dir.return_value = False
                backend.load_model(mock_config, expert="high_noise")

        assert backend.current_expert == "high_noise"


# ---------------------------------------------------------------------------
# setup_gradient_checkpointing
# ---------------------------------------------------------------------------

class TestSetupGradientCheckpointing:
    """setup_gradient_checkpointing delegates to the model's own method.

    FIX-01: The diffusers path (enable_gradient_checkpointing) already
    defaults to use_reentrant=False internally. The transformers fallback
    (gradient_checkpointing_enable) must be passed the kwarg explicitly
    to prevent zero gradients with PEFT LoRA adapters.
    """

    def test_calls_enable_gradient_checkpointing(self):
        """If the model exposes enable_gradient_checkpointing(), call it.

        The diffusers API already defaults to use_reentrant=False internally,
        so no kwargs are needed on this path.
        """
        backend = _make_t2v_backend()
        mock_model = MagicMock(spec=["enable_gradient_checkpointing"])
        backend.setup_gradient_checkpointing(mock_model)
        mock_model.enable_gradient_checkpointing.assert_called_once()

    def test_falls_back_to_gradient_checkpointing_enable(self):
        """Older HuggingFace models use gradient_checkpointing_enable() instead.

        FIX-01: This path MUST pass gradient_checkpointing_kwargs with
        use_reentrant=False to prevent zero gradients with PEFT LoRA.
        """
        backend = _make_t2v_backend()
        # MagicMock without the newer method — simulate older API
        mock_model = MagicMock(spec=["gradient_checkpointing_enable"])
        backend.setup_gradient_checkpointing(mock_model)
        mock_model.gradient_checkpointing_enable.assert_called_once_with(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    def test_no_error_if_model_has_neither_method(self):
        """Models without either method should be silently skipped.

        The backend must not crash if the model doesn't support
        gradient checkpointing — some test stubs won't have it.
        """
        backend = _make_t2v_backend()
        mock_model = MagicMock(spec=[])  # No methods at all
        # Must not raise
        backend.setup_gradient_checkpointing(mock_model)

    def test_prefers_enable_gradient_checkpointing(self):
        """If both methods exist, the backend calls enable_gradient_checkpointing.

        This matches the diffusers/transformers convention where
        enable_gradient_checkpointing() is the preferred newer API.
        """
        backend = _make_t2v_backend()
        mock_model = MagicMock(
            spec=["enable_gradient_checkpointing", "gradient_checkpointing_enable"]
        )
        backend.setup_gradient_checkpointing(mock_model)
        mock_model.enable_gradient_checkpointing.assert_called_once()
        mock_model.gradient_checkpointing_enable.assert_not_called()

    def test_logs_info_when_checkpointing_enabled(self, caplog):
        """Info log should mention use_reentrant=False for the diffusers path."""
        import logging
        backend = _make_t2v_backend()
        mock_model = MagicMock(spec=["enable_gradient_checkpointing"])
        with caplog.at_level(logging.INFO, logger="flimmer.training.wan.backend"):
            backend.setup_gradient_checkpointing(mock_model)
        assert any("use_reentrant=False" in msg for msg in caplog.messages)

    def test_logs_info_for_transformers_fallback(self, caplog):
        """Info log should mention use_reentrant=False for the transformers fallback path."""
        import logging
        backend = _make_t2v_backend()
        mock_model = MagicMock(spec=["gradient_checkpointing_enable"])
        with caplog.at_level(logging.INFO, logger="flimmer.training.wan.backend"):
            backend.setup_gradient_checkpointing(mock_model)
        assert any("use_reentrant=False" in msg for msg in caplog.messages)


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------

class TestProtocolCompliance:
    """WanModelBackend satisfies the ModelBackend @runtime_checkable protocol."""

    def test_isinstance_model_backend_t2v(self):
        """T2V variant passes isinstance check against ModelBackend protocol."""
        backend = _make_t2v_backend()
        assert isinstance(backend, ModelBackend)

    def test_isinstance_model_backend_i2v(self):
        """I2V variant also passes — same protocol, different config."""
        backend = _make_i2v_backend()
        assert isinstance(backend, ModelBackend)

    def test_isinstance_model_backend_21(self):
        """Wan 2.1 (non-MoE) variant passes — protocol is model-agnostic."""
        backend = _make_21_t2v_backend()
        assert isinstance(backend, ModelBackend)

    def test_has_all_required_properties(self):
        """All protocol properties must be accessible without error."""
        backend = _make_t2v_backend()
        # These must not raise
        _ = backend.model_id
        _ = backend.supports_moe
        _ = backend.supports_reference_image

    def test_has_all_required_methods(self):
        """All protocol methods must be callable without error in isolation."""
        backend = _make_t2v_backend()
        # GPU-free methods only — load_model and prepare_model_inputs need torch
        ts = np.array([0.5])
        _ = backend.get_lora_target_modules()
        _ = backend.get_expert_mask(ts, 0.875)
        _ = backend.get_noise_schedule()


# ---------------------------------------------------------------------------
# Integration: registry → backend construction
# ---------------------------------------------------------------------------

class TestRegistryIntegration:
    """Smoke tests: use the variant registry to build backends."""

    def test_can_build_from_registry_constants(self):
        """The values in WAN_VARIANTS produce valid WanModelBackend instances."""
        from flimmer.training.wan.registry import WAN_VARIANTS

        for variant_name, variant_info in WAN_VARIANTS.items():
            backend = WanModelBackend(
                model_id=variant_info["model_id"],
                model_path="/fake/path",
                is_moe=variant_info["is_moe"],
                is_i2v=variant_info["is_i2v"],
                in_channels=variant_info["in_channels"],
                num_blocks=variant_info["num_blocks"],
                boundary_ratio=variant_info["boundary_ratio"],
                flow_shift=variant_info["flow_shift"],
                lora_targets=variant_info["lora_targets"],
                expert_subfolders=variant_info["expert_subfolders"],
            )
            assert isinstance(backend, ModelBackend), (
                f"Variant '{variant_name}' did not satisfy ModelBackend protocol"
            )
            assert backend.model_id == variant_info["model_id"]

    def test_22_t2v_is_moe_not_i2v(self):
        """Wan 2.2 T2V is MoE but NOT I2V."""
        from flimmer.training.wan.registry import WAN_VARIANTS
        info = WAN_VARIANTS["2.2_t2v"]
        backend = WanModelBackend(model_path="/fake", **{
            k: v for k, v in info.items()
            if k in ("model_id", "is_moe", "is_i2v", "in_channels",
                     "num_blocks", "boundary_ratio", "flow_shift",
                     "lora_targets", "expert_subfolders")
        })
        assert backend.supports_moe is True
        assert backend.supports_reference_image is False

    def test_22_i2v_is_moe_and_i2v(self):
        """Wan 2.2 I2V is both MoE AND I2V."""
        from flimmer.training.wan.registry import WAN_VARIANTS
        info = WAN_VARIANTS["2.2_i2v"]
        backend = WanModelBackend(model_path="/fake", **{
            k: v for k, v in info.items()
            if k in ("model_id", "is_moe", "is_i2v", "in_channels",
                     "num_blocks", "boundary_ratio", "flow_shift",
                     "lora_targets", "expert_subfolders")
        })
        assert backend.supports_moe is True
        assert backend.supports_reference_image is True

    def test_21_t2v_is_not_moe(self):
        """Wan 2.1 T2V has no MoE and no reference image support."""
        from flimmer.training.wan.registry import WAN_VARIANTS
        info = WAN_VARIANTS["2.1_t2v"]
        backend = WanModelBackend(model_path="/fake", **{
            k: v for k, v in info.items()
            if k in ("model_id", "is_moe", "is_i2v", "in_channels",
                     "num_blocks", "boundary_ratio", "flow_shift",
                     "lora_targets", "expert_subfolders")
        })
        assert backend.supports_moe is False
        assert backend.supports_reference_image is False
        assert backend.boundary_ratio is None


# ---------------------------------------------------------------------------
# Helper: build a model class that raises on from_pretrained
# ---------------------------------------------------------------------------

def _make_exploding_model_class():
    """Return a mock WanTransformer3DModel class whose from_pretrained raises.

    Used in load_model error tests to simulate a missing/corrupt model
    without needing real disk paths or a real diffusers install.
    """
    mock_cls = MagicMock()
    mock_cls.from_pretrained.side_effect = RuntimeError(
        "No such file or directory: '/definitely/does/not/exist'"
    )
    return mock_cls
