"""Encoder protocol and registry for the encoding pipeline.

Defines the ControlEncoder protocol — the interface that all encoders
(VAE, T5, future depth/pose encoders) must implement. The registry
pattern allows adding new encoder types without modifying existing code.

Only VAE and T5 encoders are implemented now (Phase 6). Phase 10 will
add depth, edge, and pose encoders that follow the same protocol.

The protocol is intentionally minimal:
    - encode(): takes a path + target dimensions, returns tensor bytes
    - encoder_id: identifies what model is doing the encoding
    - signal_type: what kind of signal this encoder handles

Actual tensor operations happen in vae_encoder.py and text_encoder.py
(separate modules to isolate heavy dependencies).
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from flimmer.encoding.errors import EncoderError


# ---------------------------------------------------------------------------
# Encoder protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class ControlEncoder(Protocol):
    """Protocol for all signal encoders in the Flimmer pipeline.

    Every encoder converts a raw input (video frames, text, images) into
    a tensor representation suitable for caching and training. The protocol
    is deliberately minimal — implementations handle their own model loading,
    GPU management, and dtype conversion.

    Implementations:
        - VaeEncoder: video/image → latent tensors (via Wan VAE)
        - TextEncoder: caption text → text embeddings (via T5)
        - (Phase 10) DepthEncoder, PoseEncoder, etc.

    Example::

        class MyEncoder:
            @property
            def encoder_id(self) -> str:
                return "my-custom-encoder-v1"

            @property
            def signal_type(self) -> str:
                return "custom_signal"

            def encode(self, input_path: str, **kwargs) -> dict[str, Any]:
                # ... encoding logic ...
                return {"encoded_tensor": tensor_data}

            def cleanup(self) -> None:
                pass
    """

    @property
    def encoder_id(self) -> str:
        """Unique identifier for this encoder configuration.

        Used in the cache manifest to detect when re-encoding is needed
        (different model, different version, etc.). Should include the
        model name/path and any configuration that affects output.

        Examples: 'Wan-AI/Wan2.2-T2V-14B-Diffusers/vae',
                  'google/umt5-xxl'
        """
        ...

    @property
    def signal_type(self) -> str:
        """What kind of signal this encoder handles.

        Maps to SampleRole: 'latent' for VAE (videos/images),
        'text' for T5 (captions), 'reference' for reference images.
        """
        ...

    def encode(self, input_path: str, **kwargs: Any) -> dict[str, Any]:
        """Encode a single input file into tensors.

        Args:
            input_path: Absolute path to the input file.
            **kwargs: Encoder-specific parameters (e.g. target_width,
                target_height, target_frames for VAE).

        Returns:
            Dict of named tensors. Keys are tensor names (e.g. 'latent',
            'text_embedding', 'attention_mask'). Values are tensor-like
            objects that can be saved to safetensors.

        Raises:
            EncoderError: If encoding fails.
        """
        ...

    def cleanup(self) -> None:
        """Release GPU memory and other resources.

        Called after a batch of encoding is complete. Implementations
        should move models off GPU and clear CUDA cache.
        """
        ...


# ---------------------------------------------------------------------------
# Encoder registry
# ---------------------------------------------------------------------------

class EncoderRegistry:
    """Registry of available encoders, keyed by signal type.

    The registry maps signal types ('latent', 'text', 'reference') to
    encoder instances. During caching, the pipeline looks up the right
    encoder for each signal type.

    Usage::

        registry = EncoderRegistry()
        registry.register("latent", VaeEncoder(model_path=...))
        registry.register("text", TextEncoder(model_id=...))

        # Later, during encoding:
        encoder = registry.get("latent")
        result = encoder.encode(video_path, target_frames=81, ...)

    The registry does NOT own the encoder lifecycle — callers are
    responsible for calling cleanup() when done.
    """

    def __init__(self) -> None:
        self._encoders: dict[str, ControlEncoder] = {}

    def register(self, signal_type: str, encoder: ControlEncoder) -> None:
        """Register an encoder for a signal type.

        Overwrites any previously registered encoder for the same type.

        Args:
            signal_type: Signal type key (e.g. 'latent', 'text', 'reference').
            encoder: Encoder instance implementing ControlEncoder protocol.

        Raises:
            EncoderError: If the encoder doesn't implement the protocol.
        """
        if not isinstance(encoder, ControlEncoder):
            raise EncoderError(
                signal_type,
                f"Object does not implement the ControlEncoder protocol. "
                f"It must have encoder_id, signal_type, encode(), and cleanup()."
            )
        self._encoders[signal_type] = encoder

    def get(self, signal_type: str) -> ControlEncoder:
        """Get the encoder for a signal type.

        Args:
            signal_type: Signal type key.

        Returns:
            The registered encoder.

        Raises:
            EncoderError: If no encoder is registered for this type.
        """
        if signal_type not in self._encoders:
            available = ", ".join(sorted(self._encoders.keys())) or "(none)"
            raise EncoderError(
                signal_type,
                f"No encoder registered for signal type '{signal_type}'. "
                f"Available types: {available}. "
                f"Register an encoder with registry.register('{signal_type}', encoder)."
            )
        return self._encoders[signal_type]

    def has(self, signal_type: str) -> bool:
        """Check if an encoder is registered for a signal type."""
        return signal_type in self._encoders

    @property
    def signal_types(self) -> list[str]:
        """List of registered signal types."""
        return sorted(self._encoders.keys())

    def cleanup_all(self) -> None:
        """Call cleanup() on all registered encoders.

        Releases GPU memory across all encoders. Safe to call multiple
        times — idempotent.
        """
        for encoder in self._encoders.values():
            encoder.cleanup()
