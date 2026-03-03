"""Tests for flimmer.encoding.encoder — encoder protocol and registry.

Tests cover:
    - ControlEncoder protocol checking
    - EncoderRegistry: register, get, has, signal_types, cleanup_all
    - Error cases: unregistered type, non-protocol object
"""

from __future__ import annotations

from typing import Any

import pytest

from flimmer.encoding.encoder import ControlEncoder, EncoderRegistry
from flimmer.encoding.errors import EncoderError


# ---------------------------------------------------------------------------
# Mock encoder for testing
# ---------------------------------------------------------------------------

class MockEncoder:
    """A minimal encoder that implements the ControlEncoder protocol."""

    def __init__(self, signal: str = "latent", eid: str = "mock-v1") -> None:
        self._signal = signal
        self._eid = eid
        self.cleaned_up = False

    @property
    def encoder_id(self) -> str:
        return self._eid

    @property
    def signal_type(self) -> str:
        return self._signal

    def encode(self, input_path: str, **kwargs: Any) -> dict[str, Any]:
        return {"mock_tensor": b"data"}

    def cleanup(self) -> None:
        self.cleaned_up = True


class NotAnEncoder:
    """Does NOT implement the ControlEncoder protocol."""
    pass


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

class TestControlEncoderProtocol:
    """Tests for the ControlEncoder protocol."""

    def test_mock_implements_protocol(self) -> None:
        """MockEncoder satisfies the ControlEncoder protocol."""
        encoder = MockEncoder()
        assert isinstance(encoder, ControlEncoder)

    def test_not_an_encoder(self) -> None:
        """NotAnEncoder does NOT satisfy the protocol."""
        obj = NotAnEncoder()
        assert not isinstance(obj, ControlEncoder)

    def test_mock_properties(self) -> None:
        encoder = MockEncoder(signal="text", eid="test-t5")
        assert encoder.encoder_id == "test-t5"
        assert encoder.signal_type == "text"

    def test_mock_encode(self) -> None:
        encoder = MockEncoder()
        result = encoder.encode("/fake/path")
        assert "mock_tensor" in result


# ---------------------------------------------------------------------------
# EncoderRegistry
# ---------------------------------------------------------------------------

class TestEncoderRegistry:
    """Tests for the encoder registry."""

    def test_register_and_get(self) -> None:
        registry = EncoderRegistry()
        encoder = MockEncoder(signal="latent")
        registry.register("latent", encoder)

        retrieved = registry.get("latent")
        assert retrieved is encoder

    def test_has_registered(self) -> None:
        registry = EncoderRegistry()
        registry.register("latent", MockEncoder())
        assert registry.has("latent")
        assert not registry.has("text")

    def test_signal_types(self) -> None:
        registry = EncoderRegistry()
        registry.register("latent", MockEncoder(signal="latent"))
        registry.register("text", MockEncoder(signal="text"))
        assert registry.signal_types == ["latent", "text"]

    def test_get_unregistered_raises(self) -> None:
        registry = EncoderRegistry()
        with pytest.raises(EncoderError, match="No encoder registered"):
            registry.get("latent")

    def test_get_error_shows_available(self) -> None:
        registry = EncoderRegistry()
        registry.register("text", MockEncoder())
        with pytest.raises(EncoderError, match="text"):
            registry.get("latent")

    def test_overwrite_registration(self) -> None:
        registry = EncoderRegistry()
        encoder1 = MockEncoder(eid="v1")
        encoder2 = MockEncoder(eid="v2")
        registry.register("latent", encoder1)
        registry.register("latent", encoder2)
        assert registry.get("latent").encoder_id == "v2"

    def test_cleanup_all(self) -> None:
        registry = EncoderRegistry()
        enc1 = MockEncoder(signal="latent")
        enc2 = MockEncoder(signal="text")
        registry.register("latent", enc1)
        registry.register("text", enc2)

        registry.cleanup_all()
        assert enc1.cleaned_up
        assert enc2.cleaned_up

    def test_cleanup_all_empty(self) -> None:
        """cleanup_all on empty registry doesn't error."""
        registry = EncoderRegistry()
        registry.cleanup_all()  # No error

    def test_register_non_protocol_raises(self) -> None:
        """Registering a non-ControlEncoder object raises."""
        registry = EncoderRegistry()
        with pytest.raises(EncoderError, match="ControlEncoder protocol"):
            registry.register("latent", NotAnEncoder())  # type: ignore

    def test_empty_signal_types(self) -> None:
        registry = EncoderRegistry()
        assert registry.signal_types == []
