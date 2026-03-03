"""Tests for flimmer.encoding.models — data models for the encoding pipeline.

Tests cover:
    - SampleRole enum values and classification
    - FrameExtraction enum values
    - DiscoveredSample creation and defaults
    - ExpandedSample creation, properties (bucket_key, is_image)
    - CacheEntry properties (has_latent, has_text, is_complete)
    - CacheManifest properties (counts, bucket_counts, get_entry)
"""

from __future__ import annotations

from pathlib import Path

import pytest

from flimmer.encoding.models import (
    CacheEntry,
    CacheManifest,
    DiscoveredSample,
    ExpandedSample,
    FrameExtraction,
    SampleRole,
)


# ---------------------------------------------------------------------------
# SampleRole
# ---------------------------------------------------------------------------

class TestSampleRole:
    """Tests for the SampleRole enum."""

    def test_values_exist(self) -> None:
        """All expected role values are defined."""
        assert SampleRole.TARGET_VIDEO == "target_video"
        assert SampleRole.TARGET_IMAGE == "target_image"
        assert SampleRole.CAPTION == "caption"
        assert SampleRole.REFERENCE_IMAGE == "reference_image"

    def test_is_string_enum(self) -> None:
        """SampleRole values are strings (for JSON serialization)."""
        for role in SampleRole:
            assert isinstance(role.value, str)

    def test_four_roles(self) -> None:
        """Exactly four roles exist."""
        assert len(SampleRole) == 4


# ---------------------------------------------------------------------------
# FrameExtraction
# ---------------------------------------------------------------------------

class TestFrameExtraction:
    """Tests for the FrameExtraction enum."""

    def test_values(self) -> None:
        assert FrameExtraction.HEAD == "head"
        assert FrameExtraction.UNIFORM == "uniform"

    def test_from_string(self) -> None:
        """Can construct from string value."""
        assert FrameExtraction("head") == FrameExtraction.HEAD
        assert FrameExtraction("uniform") == FrameExtraction.UNIFORM


# ---------------------------------------------------------------------------
# DiscoveredSample
# ---------------------------------------------------------------------------

class TestDiscoveredSample:
    """Tests for DiscoveredSample model."""

    def test_basic_creation(self) -> None:
        """Create with minimal required fields."""
        sample = DiscoveredSample(
            stem="clip_001",
            target=Path("/data/clip_001.mp4"),
            target_role=SampleRole.TARGET_VIDEO,
        )
        assert sample.stem == "clip_001"
        assert sample.target == Path("/data/clip_001.mp4")
        assert sample.target_role == SampleRole.TARGET_VIDEO

    def test_defaults(self) -> None:
        """Default values for optional fields."""
        sample = DiscoveredSample(
            stem="test",
            target=Path("/test.mp4"),
            target_role=SampleRole.TARGET_VIDEO,
        )
        assert sample.caption is None
        assert sample.reference is None
        assert sample.width == 0
        assert sample.height == 0
        assert sample.frame_count == 0
        assert sample.fps == 0.0
        assert sample.duration == 0.0
        assert sample.repeats == 1
        assert sample.loss_multiplier == 1.0

    def test_full_creation(self) -> None:
        """Create with all fields populated."""
        sample = DiscoveredSample(
            stem="clip_001",
            target=Path("/data/clip_001.mp4"),
            target_role=SampleRole.TARGET_VIDEO,
            caption=Path("/data/clip_001.txt"),
            reference=Path("/data/clip_001.png"),
            width=848,
            height=480,
            frame_count=81,
            fps=16.0,
            duration=5.0625,
            repeats=3,
            loss_multiplier=1.5,
        )
        assert sample.width == 848
        assert sample.frame_count == 81
        assert sample.caption == Path("/data/clip_001.txt")

    def test_frozen(self) -> None:
        """Model is immutable."""
        sample = DiscoveredSample(
            stem="test",
            target=Path("/test.mp4"),
            target_role=SampleRole.TARGET_VIDEO,
        )
        with pytest.raises(Exception):
            sample.stem = "changed"

    def test_image_target(self) -> None:
        """Create a sample with an image target."""
        sample = DiscoveredSample(
            stem="photo_001",
            target=Path("/data/photo_001.png"),
            target_role=SampleRole.TARGET_IMAGE,
            width=1024,
            height=768,
            frame_count=1,
        )
        assert sample.target_role == SampleRole.TARGET_IMAGE
        assert sample.frame_count == 1


# ---------------------------------------------------------------------------
# ExpandedSample
# ---------------------------------------------------------------------------

class TestExpandedSample:
    """Tests for ExpandedSample model."""

    def test_basic_creation(self) -> None:
        """Create with all required fields."""
        sample = ExpandedSample(
            sample_id="clip_001_81x480x848",
            source_stem="clip_001",
            target=Path("/data/clip_001.mp4"),
            target_role=SampleRole.TARGET_VIDEO,
            bucket_width=848,
            bucket_height=480,
            bucket_frames=81,
        )
        assert sample.sample_id == "clip_001_81x480x848"
        assert sample.source_stem == "clip_001"

    def test_bucket_key_property(self) -> None:
        """bucket_key is computed from dimensions."""
        sample = ExpandedSample(
            sample_id="clip_001_81x480x848",
            source_stem="clip_001",
            target=Path("/data/clip_001.mp4"),
            target_role=SampleRole.TARGET_VIDEO,
            bucket_width=848,
            bucket_height=480,
            bucket_frames=81,
        )
        assert sample.bucket_key == "848x480x81"

    def test_is_image_false_for_video(self) -> None:
        """is_image is False when bucket_frames > 1."""
        sample = ExpandedSample(
            sample_id="test_17x480x848",
            source_stem="test",
            target=Path("/test.mp4"),
            target_role=SampleRole.TARGET_VIDEO,
            bucket_width=848,
            bucket_height=480,
            bucket_frames=17,
        )
        assert not sample.is_image

    def test_is_image_true_for_single_frame(self) -> None:
        """is_image is True when bucket_frames == 1."""
        sample = ExpandedSample(
            sample_id="test_1x480x848",
            source_stem="test",
            target=Path("/test.png"),
            target_role=SampleRole.TARGET_IMAGE,
            bucket_width=848,
            bucket_height=480,
            bucket_frames=1,
        )
        assert sample.is_image

    def test_defaults(self) -> None:
        """Default values for optional fields."""
        sample = ExpandedSample(
            sample_id="test",
            source_stem="test",
            target=Path("/test.mp4"),
            target_role=SampleRole.TARGET_VIDEO,
            bucket_width=320,
            bucket_height=240,
            bucket_frames=17,
        )
        assert sample.frame_extraction == FrameExtraction.HEAD
        assert sample.frame_offset == 0
        assert sample.repeats == 1
        assert sample.loss_multiplier == 1.0
        assert sample.caption is None
        assert sample.reference is None

    def test_frozen(self) -> None:
        """Model is immutable."""
        sample = ExpandedSample(
            sample_id="test",
            source_stem="test",
            target=Path("/test.mp4"),
            target_role=SampleRole.TARGET_VIDEO,
            bucket_width=320,
            bucket_height=240,
            bucket_frames=17,
        )
        with pytest.raises(Exception):
            sample.bucket_frames = 33


# ---------------------------------------------------------------------------
# CacheEntry
# ---------------------------------------------------------------------------

class TestCacheEntry:
    """Tests for CacheEntry model."""

    def test_basic_creation(self) -> None:
        """Create with required fields."""
        entry = CacheEntry(
            sample_id="clip_001_81x480x848",
            source_path="/data/clip_001.mp4",
            source_mtime=1700000000.0,
            source_size=50_000_000,
        )
        assert entry.sample_id == "clip_001_81x480x848"
        assert entry.source_mtime == 1700000000.0

    def test_has_latent(self) -> None:
        """has_latent reflects latent_file presence."""
        entry = CacheEntry(
            sample_id="test",
            source_path="/test.mp4",
            source_mtime=0.0,
            source_size=0,
            latent_file="latents/test.safetensors",
        )
        assert entry.has_latent

    def test_has_latent_false(self) -> None:
        entry = CacheEntry(
            sample_id="test",
            source_path="/test.mp4",
            source_mtime=0.0,
            source_size=0,
        )
        assert not entry.has_latent

    def test_has_text(self) -> None:
        entry = CacheEntry(
            sample_id="test",
            source_path="/test.mp4",
            source_mtime=0.0,
            source_size=0,
            text_file="text/test.safetensors",
        )
        assert entry.has_text

    def test_has_reference(self) -> None:
        entry = CacheEntry(
            sample_id="test",
            source_path="/test.mp4",
            source_mtime=0.0,
            source_size=0,
            reference_file="references/test.safetensors",
        )
        assert entry.has_reference

    def test_is_complete(self) -> None:
        """is_complete requires latent_file."""
        complete = CacheEntry(
            sample_id="test",
            source_path="/test.mp4",
            source_mtime=0.0,
            source_size=0,
            latent_file="latents/test.safetensors",
        )
        assert complete.is_complete

    def test_is_complete_false(self) -> None:
        incomplete = CacheEntry(
            sample_id="test",
            source_path="/test.mp4",
            source_mtime=0.0,
            source_size=0,
        )
        assert not incomplete.is_complete


# ---------------------------------------------------------------------------
# CacheManifest
# ---------------------------------------------------------------------------

class TestCacheManifest:
    """Tests for CacheManifest model."""

    def test_empty_manifest(self) -> None:
        """Empty manifest has zero counts."""
        m = CacheManifest()
        assert m.total_entries == 0
        assert m.complete_entries == 0
        assert m.latent_count == 0
        assert m.text_count == 0
        assert m.reference_count == 0
        assert m.bucket_counts == {}

    def test_defaults(self) -> None:
        m = CacheManifest()
        assert m.format_version == 1
        assert m.vae_id == ""
        assert m.text_encoder_id == ""
        assert m.dtype == "bf16"

    def test_with_entries(self) -> None:
        entries = [
            CacheEntry(
                sample_id="a_81x480x848",
                source_path="/a.mp4",
                source_mtime=1.0,
                source_size=100,
                latent_file="latents/a.safetensors",
                text_file="text/a.safetensors",
                bucket_key="848x480x81",
            ),
            CacheEntry(
                sample_id="b_17x480x848",
                source_path="/b.mp4",
                source_mtime=2.0,
                source_size=200,
                latent_file="latents/b.safetensors",
                bucket_key="848x480x17",
            ),
        ]
        m = CacheManifest(entries=entries)
        assert m.total_entries == 2
        assert m.complete_entries == 2
        assert m.latent_count == 2
        assert m.text_count == 1
        assert m.reference_count == 0

    def test_bucket_counts(self) -> None:
        entries = [
            CacheEntry(sample_id="a", source_path="", source_mtime=0, source_size=0, bucket_key="848x480x81"),
            CacheEntry(sample_id="b", source_path="", source_mtime=0, source_size=0, bucket_key="848x480x81"),
            CacheEntry(sample_id="c", source_path="", source_mtime=0, source_size=0, bucket_key="320x240x17"),
        ]
        m = CacheManifest(entries=entries)
        assert m.bucket_counts == {"848x480x81": 2, "320x240x17": 1}

    def test_get_entry_found(self) -> None:
        entries = [
            CacheEntry(sample_id="alpha", source_path="", source_mtime=0, source_size=0),
            CacheEntry(sample_id="beta", source_path="", source_mtime=0, source_size=0),
        ]
        m = CacheManifest(entries=entries)
        assert m.get_entry("alpha") is not None
        assert m.get_entry("alpha").sample_id == "alpha"

    def test_get_entry_not_found(self) -> None:
        m = CacheManifest(entries=[])
        assert m.get_entry("missing") is None
