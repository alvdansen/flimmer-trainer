"""Tests for flimmer.encoding.cache — cache I/O and manifest management.

Tests cover:
    - File naming conventions (latent, text, reference)
    - build_cache_manifest(): from expanded samples
    - save_cache_manifest() / load_cache_manifest(): round-trip I/O
    - find_stale_entries(): staleness detection
    - find_missing_entries(): missing file detection
    - ensure_cache_dirs(): directory structure creation
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from flimmer.encoding.cache import (
    CACHE_MANIFEST_FILENAME,
    LATENTS_SUBDIR,
    REFERENCES_SUBDIR,
    TEXT_SUBDIR,
    build_cache_manifest,
    ensure_cache_dirs,
    find_missing_entries,
    find_stale_entries,
    latent_filename,
    load_cache_manifest,
    reference_filename,
    save_cache_manifest,
    text_filename,
)
from flimmer.encoding.errors import CacheError
from flimmer.encoding.models import (
    CacheEntry,
    CacheManifest,
    ExpandedSample,
    SampleRole,
)


def _make_expanded(
    sample_id: str = "clip_001_81x480x848",
    source_stem: str = "clip_001",
    target_path: str = "/data/clip_001.mp4",
    caption_path: str | None = "/data/clip_001.txt",
    reference_path: str | None = None,
    bucket_key: str = "848x480x81",
) -> ExpandedSample:
    """Helper to create ExpandedSample."""
    return ExpandedSample(
        sample_id=sample_id,
        source_stem=source_stem,
        target=Path(target_path),
        target_role=SampleRole.TARGET_VIDEO,
        caption=Path(caption_path) if caption_path else None,
        reference=Path(reference_path) if reference_path else None,
        bucket_width=848,
        bucket_height=480,
        bucket_frames=81,
    )


# ---------------------------------------------------------------------------
# File naming
# ---------------------------------------------------------------------------

class TestFileNaming:
    """Tests for cache filename conventions."""

    def test_latent_filename(self) -> None:
        result = latent_filename("clip_001_81x480x848")
        assert result == "latents/clip_001_81x480x848.safetensors"

    def test_text_filename(self) -> None:
        result = text_filename("clip_001")
        assert result == "text/clip_001.safetensors"

    def test_reference_filename(self) -> None:
        result = reference_filename("clip_001")
        assert result == "references/clip_001.safetensors"


# ---------------------------------------------------------------------------
# build_cache_manifest
# ---------------------------------------------------------------------------

class TestBuildCacheManifest:
    """Tests for building manifest from expanded samples."""

    def test_basic_build(self, tmp_path: Path) -> None:
        """Builds manifest with correct entry count."""
        # Create a real file so fingerprint works
        target = tmp_path / "clip_001.mp4"
        target.write_bytes(b"\x00" * 100)

        sample = ExpandedSample(
            sample_id="clip_001_81x480x848",
            source_stem="clip_001",
            target=target,
            target_role=SampleRole.TARGET_VIDEO,
            caption=tmp_path / "clip_001.txt",
            bucket_width=848,
            bucket_height=480,
            bucket_frames=81,
        )

        manifest = build_cache_manifest(
            [sample],
            cache_dir=tmp_path / "cache",
            vae_id="test-vae",
            text_encoder_id="test-t5",
            dtype="bf16",
        )

        assert manifest.total_entries == 1
        assert manifest.vae_id == "test-vae"
        assert manifest.text_encoder_id == "test-t5"
        assert manifest.dtype == "bf16"

    def test_entry_has_latent_file(self, tmp_path: Path) -> None:
        target = tmp_path / "clip.mp4"
        target.write_bytes(b"\x00")

        sample = ExpandedSample(
            sample_id="clip_81x480x848",
            source_stem="clip",
            target=target,
            target_role=SampleRole.TARGET_VIDEO,
            bucket_width=848,
            bucket_height=480,
            bucket_frames=81,
        )

        manifest = build_cache_manifest([sample], tmp_path)
        entry = manifest.entries[0]
        assert entry.latent_file is not None
        assert "clip_81x480x848" in entry.latent_file

    def test_text_file_for_caption(self, tmp_path: Path) -> None:
        target = tmp_path / "clip.mp4"
        target.write_bytes(b"\x00")

        sample = ExpandedSample(
            sample_id="clip_81x480x848",
            source_stem="clip",
            target=target,
            target_role=SampleRole.TARGET_VIDEO,
            caption=tmp_path / "clip.txt",
            bucket_width=848,
            bucket_height=480,
            bucket_frames=81,
        )

        manifest = build_cache_manifest([sample], tmp_path)
        entry = manifest.entries[0]
        assert entry.text_file is not None
        assert "clip" in entry.text_file

    def test_no_text_file_without_caption(self, tmp_path: Path) -> None:
        target = tmp_path / "clip.mp4"
        target.write_bytes(b"\x00")

        sample = ExpandedSample(
            sample_id="clip_81x480x848",
            source_stem="clip",
            target=target,
            target_role=SampleRole.TARGET_VIDEO,
            bucket_width=848,
            bucket_height=480,
            bucket_frames=81,
        )

        manifest = build_cache_manifest([sample], tmp_path)
        entry = manifest.entries[0]
        assert entry.text_file is None

    def test_shared_text_across_frame_counts(self, tmp_path: Path) -> None:
        """Same stem gets only one text file across multiple expansions."""
        target = tmp_path / "clip.mp4"
        target.write_bytes(b"\x00")
        caption = tmp_path / "clip.txt"
        caption.write_text("test", encoding="utf-8")

        samples = [
            ExpandedSample(
                sample_id=f"clip_{fc}x480x848",
                source_stem="clip",
                target=target,
                target_role=SampleRole.TARGET_VIDEO,
                caption=caption,
                bucket_width=848,
                bucket_height=480,
                bucket_frames=fc,
            )
            for fc in [17, 33, 81]
        ]

        manifest = build_cache_manifest(samples, tmp_path)

        # Only one entry should have text_file set
        text_entries = [e for e in manifest.entries if e.text_file is not None]
        assert len(text_entries) == 1

    def test_source_fingerprint(self, tmp_path: Path) -> None:
        """Records source file mtime and size."""
        target = tmp_path / "clip.mp4"
        target.write_bytes(b"\x00" * 500)

        sample = ExpandedSample(
            sample_id="clip_81x480x848",
            source_stem="clip",
            target=target,
            target_role=SampleRole.TARGET_VIDEO,
            bucket_width=848,
            bucket_height=480,
            bucket_frames=81,
        )

        manifest = build_cache_manifest([sample], tmp_path)
        entry = manifest.entries[0]
        assert entry.source_size == 500
        assert entry.source_mtime > 0

    def test_empty_samples(self, tmp_path: Path) -> None:
        manifest = build_cache_manifest([], tmp_path)
        assert manifest.total_entries == 0

    def test_bucket_key_recorded(self, tmp_path: Path) -> None:
        target = tmp_path / "clip.mp4"
        target.write_bytes(b"\x00")

        sample = ExpandedSample(
            sample_id="clip_81x480x848",
            source_stem="clip",
            target=target,
            target_role=SampleRole.TARGET_VIDEO,
            bucket_width=848,
            bucket_height=480,
            bucket_frames=81,
        )

        manifest = build_cache_manifest([sample], tmp_path)
        assert manifest.entries[0].bucket_key == "848x480x81"


# ---------------------------------------------------------------------------
# save / load round-trip
# ---------------------------------------------------------------------------

class TestManifestIO:
    """Tests for save_cache_manifest and load_cache_manifest."""

    def test_round_trip(self, tmp_path: Path) -> None:
        """Save and load produces identical manifest."""
        original = CacheManifest(
            vae_id="test-vae",
            text_encoder_id="test-t5",
            dtype="bf16",
            entries=[
                CacheEntry(
                    sample_id="clip_001_81x480x848",
                    source_path="/data/clip.mp4",
                    source_mtime=1700000000.0,
                    source_size=50000,
                    latent_file="latents/clip_001_81x480x848.safetensors",
                    text_file="text/clip_001.safetensors",
                    bucket_key="848x480x81",
                ),
            ],
        )

        cache_dir = tmp_path / "cache"
        save_cache_manifest(original, cache_dir)
        loaded = load_cache_manifest(cache_dir)

        assert loaded.vae_id == original.vae_id
        assert loaded.text_encoder_id == original.text_encoder_id
        assert loaded.dtype == original.dtype
        assert loaded.total_entries == 1
        assert loaded.entries[0].sample_id == "clip_001_81x480x848"
        assert loaded.entries[0].latent_file == "latents/clip_001_81x480x848.safetensors"

    def test_creates_directory(self, tmp_path: Path) -> None:
        """save creates cache directory if needed."""
        cache_dir = tmp_path / "new" / "deep" / "cache"
        assert not cache_dir.exists()

        save_cache_manifest(CacheManifest(), cache_dir)
        assert cache_dir.exists()
        assert (cache_dir / CACHE_MANIFEST_FILENAME).is_file()

    def test_load_missing_manifest(self, tmp_path: Path) -> None:
        with pytest.raises(CacheError, match="No cache manifest"):
            load_cache_manifest(tmp_path)

    def test_load_corrupt_manifest(self, tmp_path: Path) -> None:
        (tmp_path / CACHE_MANIFEST_FILENAME).write_text("{bad", encoding="utf-8")
        with pytest.raises(CacheError, match="parse"):
            load_cache_manifest(tmp_path)

    def test_manifest_filename_correct(self, tmp_path: Path) -> None:
        save_cache_manifest(CacheManifest(), tmp_path)
        assert (tmp_path / "cache_manifest.json").is_file()


# ---------------------------------------------------------------------------
# Staleness detection
# ---------------------------------------------------------------------------

class TestFindStaleEntries:
    """Tests for staleness detection."""

    def test_no_stale_when_unchanged(self, tmp_path: Path) -> None:
        """Entries matching current file state are not stale."""
        target = tmp_path / "clip.mp4"
        target.write_bytes(b"\x00" * 100)
        stat = target.stat()

        manifest = CacheManifest(entries=[
            CacheEntry(
                sample_id="clip",
                source_path=str(target),
                source_mtime=stat.st_mtime,
                source_size=stat.st_size,
            ),
        ])

        stale = find_stale_entries(manifest)
        assert len(stale) == 0

    def test_stale_when_size_changed(self, tmp_path: Path) -> None:
        target = tmp_path / "clip.mp4"
        target.write_bytes(b"\x00" * 100)
        stat = target.stat()

        manifest = CacheManifest(entries=[
            CacheEntry(
                sample_id="clip",
                source_path=str(target),
                source_mtime=stat.st_mtime,
                source_size=50,  # Wrong size
            ),
        ])

        stale = find_stale_entries(manifest)
        assert len(stale) == 1

    def test_stale_when_file_deleted(self, tmp_path: Path) -> None:
        manifest = CacheManifest(entries=[
            CacheEntry(
                sample_id="clip",
                source_path=str(tmp_path / "deleted.mp4"),
                source_mtime=1.0,
                source_size=100,
            ),
        ])

        stale = find_stale_entries(manifest)
        assert len(stale) == 1

    def test_empty_manifest(self) -> None:
        stale = find_stale_entries(CacheManifest())
        assert len(stale) == 0


# ---------------------------------------------------------------------------
# Missing entries
# ---------------------------------------------------------------------------

class TestFindMissingEntries:
    """Tests for missing cache file detection."""

    def test_missing_latent_file(self, tmp_path: Path) -> None:
        """Entry with latent_file that doesn't exist on disk."""
        manifest = CacheManifest(entries=[
            CacheEntry(
                sample_id="clip",
                source_path="/data/clip.mp4",
                source_mtime=0,
                source_size=0,
                latent_file="latents/clip.safetensors",
            ),
        ])

        missing = find_missing_entries(manifest, tmp_path)
        assert len(missing) == 1

    def test_present_latent_file(self, tmp_path: Path) -> None:
        """Entry with latent_file that exists on disk."""
        latent_dir = tmp_path / "latents"
        latent_dir.mkdir()
        (latent_dir / "clip.safetensors").write_bytes(b"\x00")

        manifest = CacheManifest(entries=[
            CacheEntry(
                sample_id="clip",
                source_path="/data/clip.mp4",
                source_mtime=0,
                source_size=0,
                latent_file="latents/clip.safetensors",
            ),
        ])

        missing = find_missing_entries(manifest, tmp_path)
        assert len(missing) == 0

    def test_entry_without_latent_file(self, tmp_path: Path) -> None:
        """Entries without latent_file are not considered missing."""
        manifest = CacheManifest(entries=[
            CacheEntry(
                sample_id="clip",
                source_path="/data/clip.mp4",
                source_mtime=0,
                source_size=0,
            ),
        ])

        missing = find_missing_entries(manifest, tmp_path)
        assert len(missing) == 0


# ---------------------------------------------------------------------------
# ensure_cache_dirs
# ---------------------------------------------------------------------------

class TestEnsureCacheDirs:
    """Tests for cache directory structure creation."""

    def test_creates_structure(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / "cache"
        ensure_cache_dirs(cache_dir)

        assert (cache_dir / LATENTS_SUBDIR).is_dir()
        assert (cache_dir / TEXT_SUBDIR).is_dir()
        assert (cache_dir / REFERENCES_SUBDIR).is_dir()

    def test_idempotent(self, tmp_path: Path) -> None:
        """Can be called multiple times without error."""
        cache_dir = tmp_path / "cache"
        ensure_cache_dirs(cache_dir)
        ensure_cache_dirs(cache_dir)  # No error


# ---------------------------------------------------------------------------
# CacheEntry.reference_source_path
# ---------------------------------------------------------------------------

class TestCacheEntryReferenceSourcePath:
    """Tests for the reference_source_path field on CacheEntry."""

    def test_reference_source_path_stored(self) -> None:
        """reference_source_path is stored when provided."""
        entry = CacheEntry(
            sample_id="clip_001_81x480x848",
            source_path="clips/clip_001.mp4",
            source_stem="clip_001",
            source_mtime=0,
            source_size=0,
            reference_source_path="refs/clip_001.png",
            reference_file="references/clip_001.safetensors",
        )
        assert entry.reference_source_path == "refs/clip_001.png"

    def test_reference_source_path_optional(self) -> None:
        """reference_source_path defaults to None when not provided."""
        entry = CacheEntry(
            sample_id="clip_001_81x480x848",
            source_path="clips/clip_001.mp4",
            source_stem="clip_001",
            source_mtime=0,
            source_size=0,
        )
        assert entry.reference_source_path is None


# ---------------------------------------------------------------------------
# build_cache_manifest: reference_source_path population
# ---------------------------------------------------------------------------

class TestBuildCacheManifestReference:
    """Tests for build_cache_manifest populating reference_source_path."""

    def test_populates_reference_source_path(self, tmp_path: Path) -> None:
        """build_cache_manifest sets reference_source_path from sample.reference."""
        target = tmp_path / "clip.mp4"
        target.write_bytes(b"\x00")
        ref_img = tmp_path / "clip.png"
        ref_img.write_bytes(b"\x00")

        sample = ExpandedSample(
            sample_id="clip_81x480x848",
            source_stem="clip",
            target=target,
            target_role=SampleRole.TARGET_VIDEO,
            reference=ref_img,
            bucket_width=848,
            bucket_height=480,
            bucket_frames=81,
        )

        manifest = build_cache_manifest([sample], tmp_path)
        entry = manifest.entries[0]
        assert entry.reference_source_path == str(ref_img)
        assert entry.reference_file is not None

    def test_no_reference_source_path_without_reference(self, tmp_path: Path) -> None:
        """reference_source_path is None when sample has no reference."""
        target = tmp_path / "clip.mp4"
        target.write_bytes(b"\x00")

        sample = ExpandedSample(
            sample_id="clip_81x480x848",
            source_stem="clip",
            target=target,
            target_role=SampleRole.TARGET_VIDEO,
            bucket_width=848,
            bucket_height=480,
            bucket_frames=81,
        )

        manifest = build_cache_manifest([sample], tmp_path)
        entry = manifest.entries[0]
        assert entry.reference_source_path is None

    def test_shared_reference_across_frame_counts(self, tmp_path: Path) -> None:
        """Same stem gets only one reference file across multiple expansions."""
        target = tmp_path / "clip.mp4"
        target.write_bytes(b"\x00")
        ref_img = tmp_path / "clip.png"
        ref_img.write_bytes(b"\x00")

        samples = [
            ExpandedSample(
                sample_id=f"clip_{fc}x480x848",
                source_stem="clip",
                target=target,
                target_role=SampleRole.TARGET_VIDEO,
                reference=ref_img,
                bucket_width=848,
                bucket_height=480,
                bucket_frames=fc,
            )
            for fc in [17, 33, 81]
        ]

        manifest = build_cache_manifest(samples, tmp_path)

        # Only first entry should have reference_file and reference_source_path
        ref_entries = [e for e in manifest.entries if e.reference_file is not None]
        assert len(ref_entries) == 1
        assert ref_entries[0].reference_source_path == str(ref_img)


# ---------------------------------------------------------------------------
# CacheEntry.repeats field
# ---------------------------------------------------------------------------

class TestCacheEntryRepeats:
    """Tests for the repeats field on CacheEntry."""

    def test_repeats_default_is_1(self) -> None:
        """CacheEntry.repeats defaults to 1 (backwards-compatible)."""
        entry = CacheEntry(
            sample_id="clip_001_81x480x848",
            source_path="/data/clip.mp4",
            source_mtime=0,
            source_size=0,
        )
        assert entry.repeats == 1

    def test_repeats_stored_when_set(self) -> None:
        """CacheEntry stores repeats when explicitly set."""
        entry = CacheEntry(
            sample_id="clip_001_81x480x848",
            source_path="/data/clip.mp4",
            source_mtime=0,
            source_size=0,
            repeats=5,
        )
        assert entry.repeats == 5

    def test_backwards_compatible_manifest_without_repeats(self, tmp_path: Path) -> None:
        """Loading a manifest JSON that lacks repeats field defaults to 1."""
        import json
        manifest_data = {
            "format_version": 1,
            "vae_id": "test",
            "text_encoder_id": "",
            "dtype": "bf16",
            "entries": [{
                "sample_id": "old_entry",
                "source_path": "/old.mp4",
                "source_mtime": 0.0,
                "source_size": 0,
                "latent_file": "latents/old.safetensors",
                "bucket_key": "848x480x81",
            }],
        }
        manifest_path = tmp_path / "cache_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest_data, f)

        loaded = load_cache_manifest(tmp_path)
        assert loaded.entries[0].repeats == 1


# ---------------------------------------------------------------------------
# build_cache_manifest: repeats propagation
# ---------------------------------------------------------------------------

class TestBuildCacheManifestRepeats:
    """Tests for build_cache_manifest populating CacheEntry.repeats."""

    def test_propagates_repeats_from_sample(self, tmp_path: Path) -> None:
        """build_cache_manifest sets CacheEntry.repeats from ExpandedSample.repeats."""
        target = tmp_path / "clip.mp4"
        target.write_bytes(b"\x00")

        sample = ExpandedSample(
            sample_id="clip_81x480x848",
            source_stem="clip",
            target=target,
            target_role=SampleRole.TARGET_VIDEO,
            bucket_width=848,
            bucket_height=480,
            bucket_frames=81,
            repeats=3,
        )

        manifest = build_cache_manifest([sample], tmp_path)
        assert manifest.entries[0].repeats == 3

    def test_default_repeats_1(self, tmp_path: Path) -> None:
        """ExpandedSample with repeats=1 produces CacheEntry.repeats=1."""
        target = tmp_path / "clip.mp4"
        target.write_bytes(b"\x00")

        sample = ExpandedSample(
            sample_id="clip_81x480x848",
            source_stem="clip",
            target=target,
            target_role=SampleRole.TARGET_VIDEO,
            bucket_width=848,
            bucket_height=480,
            bucket_frames=81,
            repeats=1,
        )

        manifest = build_cache_manifest([sample], tmp_path)
        assert manifest.entries[0].repeats == 1

    def test_mixed_repeats(self, tmp_path: Path) -> None:
        """Different samples with different repeats are propagated correctly."""
        target = tmp_path / "clip.mp4"
        target.write_bytes(b"\x00")

        samples = [
            ExpandedSample(
                sample_id="clip_81x480x848",
                source_stem="clip",
                target=target,
                target_role=SampleRole.TARGET_VIDEO,
                bucket_width=848,
                bucket_height=480,
                bucket_frames=81,
                repeats=1,
            ),
            ExpandedSample(
                sample_id="img_1x768x1024",
                source_stem="img",
                target=target,
                target_role=SampleRole.TARGET_IMAGE,
                bucket_width=1024,
                bucket_height=768,
                bucket_frames=1,
                repeats=5,
            ),
        ]

        manifest = build_cache_manifest(samples, tmp_path)
        assert manifest.entries[0].repeats == 1
        assert manifest.entries[1].repeats == 5
