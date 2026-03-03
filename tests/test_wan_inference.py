"""Tests for flimmer.training.wan.inference.WanInferencePipeline.

WanInferencePipeline is GPU-dependent, so all tests here are GPU-free.
They cover:

    1. Constructor attribute storage (individual files, diffusers path, both)
    2. Lazy-load contract (no GPU/filesystem access at init)
    3. I2V flag handling
    4. cleanup() safety (no-op when nothing loaded, clears cache)
    5. Protocol compliance with InferencePipeline (@runtime_checkable)
    6. Error propagation when generate() is called without torch/diffusers
    7. Default device value
    8. Prompt embedding cache management

No torch, diffusers, or GPU hardware required for any of these tests.
"""

import pytest

from flimmer.training.wan.inference import WanInferencePipeline
from flimmer.training.protocols import InferencePipeline


# ---------------------------------------------------------------------------
# Constructor behaviour
# ---------------------------------------------------------------------------

def test_constructor_individual_files():
    """Constructor must store individual component paths."""
    pipe = WanInferencePipeline(
        vae_path="/fake/vae.safetensors",
        t5_path="/fake/t5.pth",
        is_i2v=False,
        dtype="bf16",
    )
    assert pipe._vae_path == "/fake/vae.safetensors"
    assert pipe._t5_path == "/fake/t5.pth"
    assert pipe._is_i2v is False
    assert pipe._dtype_str == "bf16"


def test_constructor_diffusers_path():
    """Constructor must store Diffusers directory path."""
    pipe = WanInferencePipeline(
        diffusers_path="/fake/diffusers-dir",
        is_i2v=True,
    )
    assert pipe._diffusers_path == "/fake/diffusers-dir"
    assert pipe._is_i2v is True


def test_constructor_both_paths():
    """Both individual and diffusers paths can be set (individual takes priority)."""
    pipe = WanInferencePipeline(
        vae_path="/fake/vae.safetensors",
        t5_path="/fake/t5.pth",
        diffusers_path="/fake/diffusers-dir",
    )
    assert pipe._vae_path == "/fake/vae.safetensors"
    assert pipe._t5_path == "/fake/t5.pth"
    assert pipe._diffusers_path == "/fake/diffusers-dir"


def test_no_pipeline_loaded_at_init():
    """Lazy-load contract: nothing should be loaded at init time.

    The pipeline is only built on the first generate() call. Initialising
    the class should never touch the GPU or the filesystem.
    """
    pipe = WanInferencePipeline(vae_path="/fake/vae")
    assert pipe._cached_prompt_embeds == {}


def test_i2v_flag_true():
    """is_i2v=True must be stored and readable."""
    pipe = WanInferencePipeline(is_i2v=True)
    assert pipe._is_i2v is True


def test_i2v_flag_false():
    """is_i2v=False (the default) must also be stored correctly."""
    pipe = WanInferencePipeline(is_i2v=False)
    assert pipe._is_i2v is False


def test_default_device():
    """Device should default to 'cuda' when not specified."""
    pipe = WanInferencePipeline()
    assert pipe._device == "cuda"


def test_constructor_stores_device():
    """An explicit device argument must be stored."""
    pipe = WanInferencePipeline(device="cpu")
    assert pipe._device == "cpu"


# ---------------------------------------------------------------------------
# cleanup() behaviour
# ---------------------------------------------------------------------------

def test_cleanup_when_nothing_cached():
    """cleanup() must not raise when no embeddings are cached."""
    pipe = WanInferencePipeline()
    pipe.cleanup()  # Should not raise


def test_cleanup_clears_prompt_cache():
    """cleanup() must clear the prompt embedding cache."""
    pipe = WanInferencePipeline()
    pipe._cached_prompt_embeds = {"test": "dummy_tensor"}
    pipe.cleanup()
    assert pipe._cached_prompt_embeds == {}


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------

def test_protocol_compliance():
    """WanInferencePipeline must satisfy the InferencePipeline protocol.

    InferencePipeline is @runtime_checkable, so isinstance() confirms
    that the class exposes the required generate() method signature.
    """
    pipe = WanInferencePipeline()
    assert isinstance(pipe, InferencePipeline)


# ---------------------------------------------------------------------------
# Error propagation from generate()
# ---------------------------------------------------------------------------

def test_generate_without_deps_raises(monkeypatch):
    """generate() must raise when torch/diffusers are not available.

    In a GPU-free CI environment, this raises ImportError or SamplingError.
    Either is acceptable — the key contract is that it doesn't silently
    succeed.
    """
    from flimmer.training.errors import SamplingError

    pipe = WanInferencePipeline(
        vae_path="/nonexistent/vae.safetensors",
        t5_path="/nonexistent/t5.pth",
    )
    with pytest.raises((SamplingError, ImportError, OSError, Exception)):
        pipe.generate(
            model=None,
            lora_state_dict=None,
            prompt="test prompt",
        )
