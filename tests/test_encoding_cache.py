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
