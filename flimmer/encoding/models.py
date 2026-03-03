"""Flimmer encoding pipeline data models.

Models for the full encoding pipeline: discovery → expansion → bucketing →
encoding → caching. Each model represents a specific stage:

    DiscoveredSample   — one source file set BEFORE expansion
    ExpandedSample     — one concrete training sample AFTER expansion
    CacheEntry         — one sample's cache state on disk
    CacheManifest      — complete cache state (all entries + metadata)

All models are immutable (frozen) — they represent facts about data state,
not mutable processing state.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class SampleRole(str, Enum):
    """What a file IS in the training sample.

    This drives how the file gets encoded:
    - TARGET_VIDEO: encode through VAE as learning target (the model learns
      to PRODUCE this)
    - TARGET_IMAGE: treat as 1-frame video, encode through VAE as target
    - CAPTION: encode through T5 text encoder
    - REFERENCE_IMAGE: encode through VAE as I2V conditioning (the model
      learns to OBEY this — different from target)

    The target/control separation is Flimmer's core architectural principle:
    what the model produces vs what it responds to.
    """
    TARGET_VIDEO = "target_video"
    TARGET_IMAGE = "target_image"
    CAPTION = "caption"
    REFERENCE_IMAGE = "reference_image"


class FrameExtraction(str, Enum):
    """How to extract frames from a video for this sample.

    - HEAD: take the first N frames from the start of the video.
      Most common — matches how Wan was trained (clips start from frame 0).
    - UNIFORM: take N evenly-spaced frames across the full duration.
      Better temporal coverage for long clips, but breaks temporal coherence.
    """
    HEAD = "head"
    UNIFORM = "uniform"


# ---------------------------------------------------------------------------
# Discovery stage
# ---------------------------------------------------------------------------

class DiscoveredSample(BaseModel):
    """One source file set BEFORE expansion.

    Represents a single stem in an organized dataset — the video target,
    its optional caption, and its optional reference image. This is what
    we find on disk before deciding how many training samples to make from it.

    The dimensions and frame count come from video probe (cached during
    Phase 4 validation or re-probed during discovery).
    """
    model_config = ConfigDict(frozen=True)

    stem: str
    """Filename stem that identifies this sample (e.g. 'clip_001')."""

    target: Path
    """Path to the target file (video or image)."""

    target_role: SampleRole
    """What kind of target this is (TARGET_VIDEO or TARGET_IMAGE)."""

    caption: Path | None = None
    """Path to the caption .txt file, or None if missing."""

    reference: Path | None = None
    """Path to the reference image, or None if missing."""

    width: int = 0
    """Target width in pixels (0 if unknown/image)."""

    height: int = 0
    """Target height in pixels (0 if unknown/image)."""

    frame_count: int = 0
    """Total frames in the source video (0 for images)."""

    fps: float = 0.0
    """Source video FPS (0.0 for images)."""

    duration: float = 0.0
    """Source video duration in seconds (0.0 for images)."""

    repeats: int = 1
    """How many times this sample is repeated per epoch (dataset-level weighting)."""

    loss_multiplier: float = 1.0
    """Per-sample loss weight (1.0 = normal)."""


# ---------------------------------------------------------------------------
# Expansion stage
# ---------------------------------------------------------------------------

class ExpandedSample(BaseModel):
    """One concrete training sample AFTER expansion.

    A single DiscoveredSample can produce multiple ExpandedSamples at
    different frame counts and resolutions. Each ExpandedSample has a
    unique sample_id and knows exactly what bucket it belongs to.

    The sample_id format is "{stem}_{F}x{H}x{W}" for videos, or
    "{stem}_1x{H}x{W}" for images and head-frame extractions.
    """
    model_config = ConfigDict(frozen=True)

    sample_id: str
    """Unique identifier: '{stem}_{F}x{H}x{W}'. Used as cache filename base."""

    source_stem: str
    """Original stem from the DiscoveredSample (for grouping/tracing)."""

    target: Path
    """Path to the source target file."""

    target_role: SampleRole
    """What kind of target this is."""

    caption: Path | None = None
    """Path to the caption file."""

    reference: Path | None = None
    """Path to the reference image."""

    bucket_width: int
    """Target width after bucketing (pixel-aligned)."""

    bucket_height: int
    """Target height after bucketing (pixel-aligned)."""

    bucket_frames: int
    """Target frame count for this sample (4n+1 for video, 1 for image)."""

    frame_extraction: FrameExtraction = FrameExtraction.HEAD
    """How to extract frames from the source video."""

    frame_offset: int = 0
    """Starting frame for extraction (only used with HEAD extraction)."""

    repeats: int = 1
    """Per-epoch repeat count (inherited from DiscoveredSample)."""

    loss_multiplier: float = 1.0
    """Per-sample loss weight (inherited from DiscoveredSample)."""

    @property
    def bucket_key(self) -> str:
        """Bucket key string: '{W}x{H}x{F}'.

        This is the grouping key for batching — all samples in the same
        bucket have identical dimensions and can be batched without padding.
        """
        return f"{self.bucket_width}x{self.bucket_height}x{self.bucket_frames}"

    @property
    def is_image(self) -> bool:
        """True if this is a single-frame sample (image or head-frame)."""
        return self.bucket_frames == 1


# ---------------------------------------------------------------------------
# Cache stage
# ---------------------------------------------------------------------------

class CacheEntry(BaseModel):
    """One sample's cache state on disk.

    Tracks what was cached, when, and from what source — enough information
    to detect staleness (source changed since encoding) and locate cached
    files without re-encoding.
    """
    model_config = ConfigDict(frozen=True)

    sample_id: str
    """Matches ExpandedSample.sample_id — the cache key."""

    source_path: str
    """Absolute path to the source target file (for staleness detection)."""

    source_mtime: float
    """Source file modification time at encoding (os.path.getmtime)."""

    source_size: int
    """Source file size in bytes at encoding (os.path.getsize)."""

    latent_file: str | None = None
    """Relative path to the cached latent .safetensors file (from cache_dir)."""

    text_file: str | None = None
    """Relative path to the cached text encoding .safetensors file."""

    reference_file: str | None = None
    """Relative path to the cached reference encoding .safetensors file."""

    caption_source_path: str | None = None
    """Absolute path to the caption .txt file. Stored explicitly because
    flimmer layout places captions in signals/captions/ — not alongside
    the target file. Without this, cache-text would fail to find captions
    for flimmer-structured datasets."""

    bucket_key: str = ""
    """Bucket key for this sample ('{W}x{H}x{F}')."""

    @property
    def has_latent(self) -> bool:
        """True if latent encoding is cached."""
        return self.latent_file is not None

    @property
    def has_text(self) -> bool:
        """True if text encoding is cached."""
        return self.text_file is not None

    @property
    def has_reference(self) -> bool:
        """True if reference encoding is cached."""
        return self.reference_file is not None

    @property
    def is_complete(self) -> bool:
        """True if all expected encodings are present.

        A sample is complete when its latent is cached. Text and reference
        are optional — not all samples have captions or reference images.
        """
        return self.has_latent


# ---------------------------------------------------------------------------
# Cache manifest
# ---------------------------------------------------------------------------

class CacheManifest(BaseModel):
    """Complete cache state — all entries plus encoding metadata.

    This is the top-level cache manifest written as cache_manifest.json
    in the cache directory. It records what was encoded, with what model,
    at what precision, so we can detect when re-encoding is needed
    (model changed, dtype changed, etc.).

    The manifest is SEPARATE from flimmer_manifest.json (dataset-level).
    Different lifecycle, different directory, different tools update them.
    """
    model_config = ConfigDict(frozen=True)

    format_version: int = 1
    """Manifest format version for forward compatibility."""

    vae_id: str = ""
    """Identifier of the VAE used for latent encoding (e.g. model path or HF ID).
    Empty string means latents haven't been encoded yet."""

    text_encoder_id: str = ""
    """Identifier of the text encoder (e.g. 'google/umt5-xxl').
    Empty string means text hasn't been encoded yet."""

    dtype: str = "bf16"
    """Tensor dtype used for encoding ('bf16', 'fp16', 'fp32')."""

    entries: list[CacheEntry] = Field(default_factory=list)
    """All cached sample entries."""

    @property
    def total_entries(self) -> int:
        """Total number of entries in the manifest."""
        return len(self.entries)

    @property
    def complete_entries(self) -> int:
        """Number of entries with all expected encodings."""
        return sum(1 for e in self.entries if e.is_complete)

    @property
    def latent_count(self) -> int:
        """Number of entries with cached latents."""
        return sum(1 for e in self.entries if e.has_latent)

    @property
    def text_count(self) -> int:
        """Number of entries with cached text encodings."""
        return sum(1 for e in self.entries if e.has_text)

    @property
    def reference_count(self) -> int:
        """Number of entries with cached reference encodings."""
        return sum(1 for e in self.entries if e.has_reference)

    @property
    def bucket_counts(self) -> dict[str, int]:
        """Count of entries per bucket key."""
        counts: dict[str, int] = {}
        for entry in self.entries:
            if entry.bucket_key:
                counts[entry.bucket_key] = counts.get(entry.bucket_key, 0) + 1
        return counts

    def get_entry(self, sample_id: str) -> CacheEntry | None:
        """Look up a cache entry by sample_id. Returns None if not found."""
        for entry in self.entries:
            if entry.sample_id == sample_id:
                return entry
        return None

    def stale_entries(
        self,
        check_fn: None = None,
    ) -> list[CacheEntry]:
        """Find entries whose source files have changed since encoding.

        This is a placeholder — actual staleness checking requires reading
        the filesystem. The cache.py module provides the full implementation
        that compares mtime+size against current file state.
        """
        return []
