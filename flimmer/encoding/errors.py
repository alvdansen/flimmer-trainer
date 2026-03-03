"""Flimmer encoding pipeline errors.

Custom exceptions for the encoding/caching pipeline. Every error message
says what went wrong AND how to fix it.

Hierarchy:
    FlimmerEncodingError          ← base for all encoding errors
    ├── ExpansionError            ← sample expansion failed (bad frame counts, etc.)
    ├── CacheError                ← cache I/O failed (corrupt manifest, write errors)
    └── EncoderError              ← encoder failed (VAE/T5 runtime errors)
"""

from __future__ import annotations


class FlimmerEncodingError(Exception):
    """Base exception for all encoding pipeline errors.

    Covers discovery, expansion, bucketing, encoding, and caching.
    Separate from FlimmerVideoError and FlimmerDatasetError because
    the encoding pipeline operates on validated datasets, not raw files.
    """
    pass


class ExpansionError(FlimmerEncodingError):
    """Raised when sample expansion fails.

    Expansion turns one video into multiple training samples at different
    frame counts and resolutions. This fails when the video is too short
    for any valid frame count, or when configuration is invalid.
    """

    def __init__(self, detail: str) -> None:
        self.detail = detail
        super().__init__(f"Sample expansion failed: {detail}")


class CacheError(FlimmerEncodingError):
    """Raised when cache I/O fails.

    Common causes: corrupt manifest, disk full, permission errors,
    incompatible safetensors files, missing cache directory.
    """

    def __init__(self, detail: str) -> None:
        self.detail = detail
        super().__init__(f"Cache error: {detail}")


class EncoderError(FlimmerEncodingError):
    """Raised when an encoder (VAE or T5) fails at runtime.

    Common causes: out of VRAM, model not found, wrong dtype,
    input dimensions incompatible with the encoder.
    """

    def __init__(self, encoder_name: str, detail: str) -> None:
        self.encoder_name = encoder_name
        self.detail = detail
        super().__init__(
            f"Encoder '{encoder_name}' failed: {detail}"
        )
