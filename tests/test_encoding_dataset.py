"""Tests for flimmer.encoding.dataset — training dataset and batch sampler.

Tests cover:
    - CachedLatentDataset: creation, len, getitem, properties
    - BucketBatchSampler: batching by bucket, shuffle, drop_last
    - collate_cached_batch(): batch collation
"""

from __future__ import annotations

from pathlib import Path

import pytest

from flimmer.encoding.dataset import (
    BucketBatchSampler,
    CachedLatentDataset,
    collate_cached_batch,
)
from flimmer.encoding.models import CacheEntry, CacheManifest


def _make_manifest(n: int = 5, bucket_key: str = "848x480x81") -> CacheManifest:
    """Create a test manifest with n entries."""
    entries = [
        CacheEntry(
            sample_id=f"clip_{i:03d}_{bucket_key.replace('x', 'x')}",
            source_path=f"/data/clip_{i:03d}.mp4",
            source_mtime=float(i),
            source_size=i * 1000,
            latent_file=f"latents/clip_{i:03d}.safetensors",
            text_file=f"text/clip_{i:03d}.safetensors" if i % 2 == 0 else None,
            bucket_key=bucket_key,
        )
        for i in range(n)
    ]
    return CacheManifest(entries=entries)


def _make_multi_bucket_manifest() -> CacheManifest:
    """Create a manifest with samples in different buckets."""
    entries = [
        CacheEntry(
            sample_id="a_81x480x848", source_path="/a.mp4",
            source_mtime=1.0, source_size=100,
            latent_file="latents/a.safetensors", bucket_key="848x480x81",
        ),
        CacheEntry(
            sample_id="b_81x480x848", source_path="/b.mp4",
            source_mtime=2.0, source_size=200,
            latent_file="latents/b.safetensors", bucket_key="848x480x81",
        ),
        CacheEntry(
            sample_id="c_17x240x320", source_path="/c.mp4",
            source_mtime=3.0, source_size=300,
            latent_file="latents/c.safetensors", bucket_key="320x240x17",
        ),
        CacheEntry(
            sample_id="d_17x240x320", source_path="/d.mp4",
            source_mtime=4.0, source_size=400,
            latent_file="latents/d.safetensors", bucket_key="320x240x17",
        ),
        CacheEntry(
            sample_id="e_17x240x320", source_path="/e.mp4",
            source_mtime=5.0, source_size=500,
            latent_file="latents/e.safetensors", bucket_key="320x240x17",
        ),
    ]
    return CacheManifest(entries=entries)


# ---------------------------------------------------------------------------
# CachedLatentDataset
# ---------------------------------------------------------------------------

class TestCachedLatentDataset:
    """Tests for the training dataset class."""

    def test_len(self, tmp_path: Path) -> None:
        manifest = _make_manifest(10)
        dataset = CachedLatentDataset(tmp_path, manifest)
        assert len(dataset) == 10

    def test_empty_manifest(self, tmp_path: Path) -> None:
        manifest = _make_manifest(0)
        dataset = CachedLatentDataset(tmp_path, manifest)
        assert len(dataset) == 0

    def test_getitem_returns_dict(self, tmp_path: Path) -> None:
        manifest = _make_manifest(3)
        dataset = CachedLatentDataset(tmp_path, manifest)
        item = dataset[0]
        assert isinstance(item, dict)
        assert "sample_id" in item
        assert "bucket_key" in item
        assert "latent" in item
        assert "text_emb" in item
        assert "reference" in item

    def test_getitem_missing_files(self, tmp_path: Path) -> None:
        """Returns None for tensors when cache files don't exist."""
        manifest = _make_manifest(1)
        dataset = CachedLatentDataset(tmp_path, manifest)
        item = dataset[0]
        # Files don't exist on disk, so all tensors should be None
        assert item["latent"] is None

    def test_bucket_keys_property(self, tmp_path: Path) -> None:
        manifest = _make_multi_bucket_manifest()
        dataset = CachedLatentDataset(tmp_path, manifest)
        keys = dataset.bucket_keys
        assert len(keys) == 5
        assert keys[0] == "848x480x81"
        assert keys[2] == "320x240x17"

    def test_manifest_property(self, tmp_path: Path) -> None:
        manifest = _make_manifest(3)
        dataset = CachedLatentDataset(tmp_path, manifest)
        assert dataset.manifest is manifest

    def test_sample_id_in_item(self, tmp_path: Path) -> None:
        manifest = _make_manifest(3)
        dataset = CachedLatentDataset(tmp_path, manifest)
        item = dataset[1]
        assert item["sample_id"] == manifest.entries[1].sample_id


# ---------------------------------------------------------------------------
# BucketBatchSampler
# ---------------------------------------------------------------------------

class TestBucketBatchSampler:
    """Tests for the bucket-aware batch sampler."""

    def test_single_bucket_batch_size_1(self, tmp_path: Path) -> None:
        """Batch size 1 yields one sample per batch."""
        manifest = _make_manifest(5, "848x480x81")
        dataset = CachedLatentDataset(tmp_path, manifest)
        sampler = BucketBatchSampler(dataset, batch_size=1, shuffle=False)

        batches = list(sampler)
        assert len(batches) == 5
        assert all(len(b) == 1 for b in batches)

    def test_single_bucket_batch_size_2(self, tmp_path: Path) -> None:
        manifest = _make_manifest(5, "848x480x81")
        dataset = CachedLatentDataset(tmp_path, manifest)
        sampler = BucketBatchSampler(dataset, batch_size=2, shuffle=False)

        batches = list(sampler)
        # 5 samples, batch_size 2: [2, 2, 1]
        assert len(batches) == 3

    def test_drop_last(self, tmp_path: Path) -> None:
        """drop_last removes incomplete final batch."""
        manifest = _make_manifest(5, "848x480x81")
        dataset = CachedLatentDataset(tmp_path, manifest)
        sampler = BucketBatchSampler(dataset, batch_size=2, shuffle=False, drop_last=True)

        batches = list(sampler)
        # 5 samples, batch_size 2, drop_last: [2, 2]
        assert len(batches) == 2
        assert all(len(b) == 2 for b in batches)

    def test_multi_bucket(self, tmp_path: Path) -> None:
        """Batches don't mix samples from different buckets."""
        manifest = _make_multi_bucket_manifest()
        dataset = CachedLatentDataset(tmp_path, manifest)
        sampler = BucketBatchSampler(dataset, batch_size=2, shuffle=False)

        batches = list(sampler)
        # Bucket 848x480x81: 2 samples → 1 batch of 2
        # Bucket 320x240x17: 3 samples → 2 batches (2+1)
        assert len(batches) == 3

        # Verify no cross-bucket mixing
        for batch in batches:
            keys = {dataset.bucket_keys[i] for i in batch}
            assert len(keys) == 1  # All same bucket

    def test_len(self, tmp_path: Path) -> None:
        manifest = _make_manifest(10, "848x480x81")
        dataset = CachedLatentDataset(tmp_path, manifest)
        sampler = BucketBatchSampler(dataset, batch_size=3)
        # 10 / 3 = 4 batches (3, 3, 3, 1)
        assert len(sampler) == 4

    def test_len_drop_last(self, tmp_path: Path) -> None:
        manifest = _make_manifest(10, "848x480x81")
        dataset = CachedLatentDataset(tmp_path, manifest)
        sampler = BucketBatchSampler(dataset, batch_size=3, drop_last=True)
        # 10 / 3 = 3 complete batches
        assert len(sampler) == 3

    def test_bucket_count(self, tmp_path: Path) -> None:
        manifest = _make_multi_bucket_manifest()
        dataset = CachedLatentDataset(tmp_path, manifest)
        sampler = BucketBatchSampler(dataset, batch_size=1)
        assert sampler.bucket_count == 2

    def test_bucket_sizes(self, tmp_path: Path) -> None:
        manifest = _make_multi_bucket_manifest()
        dataset = CachedLatentDataset(tmp_path, manifest)
        sampler = BucketBatchSampler(dataset, batch_size=1)
        sizes = sampler.bucket_sizes
        assert sizes["848x480x81"] == 2
        assert sizes["320x240x17"] == 3

    def test_shuffle_deterministic(self, tmp_path: Path) -> None:
        """Same seed produces same batch order."""
        manifest = _make_manifest(10, "848x480x81")
        dataset = CachedLatentDataset(tmp_path, manifest)

        sampler1 = BucketBatchSampler(dataset, batch_size=2, shuffle=True, seed=42)
        sampler2 = BucketBatchSampler(dataset, batch_size=2, shuffle=True, seed=42)

        batches1 = list(sampler1)
        batches2 = list(sampler2)
        assert batches1 == batches2

    def test_empty_dataset(self, tmp_path: Path) -> None:
        manifest = _make_manifest(0)
        dataset = CachedLatentDataset(tmp_path, manifest)
        sampler = BucketBatchSampler(dataset, batch_size=1)
        assert list(sampler) == []
        assert len(sampler) == 0


# ---------------------------------------------------------------------------
# collate_cached_batch
# ---------------------------------------------------------------------------

class TestCollateCachedBatch:
    """Tests for batch collation."""

    def test_empty_batch(self) -> None:
        assert collate_cached_batch([]) == {}

    def test_single_item(self) -> None:
        batch = [
            {
                "sample_id": "clip_001",
                "bucket_key": "848x480x81",
                "latent": None,
                "text_emb": None,
                "text_mask": None,
                "reference": None,
            }
        ]
        result = collate_cached_batch(batch)
        assert result["sample_ids"] == ["clip_001"]
        assert result["bucket_key"] == "848x480x81"

    def test_multiple_items(self) -> None:
        batch = [
            {
                "sample_id": f"clip_{i}",
                "bucket_key": "848x480x81",
                "latent": None,
                "text_emb": None,
                "text_mask": None,
                "reference": None,
            }
            for i in range(3)
        ]
        result = collate_cached_batch(batch)
        assert len(result["sample_ids"]) == 3
        assert result["bucket_key"] == "848x480x81"

    def test_preserves_none_values(self) -> None:
        """All-None values are collapsed to None (not a list of Nones)."""
        batch = [
            {
                "sample_id": "a",
                "bucket_key": "k",
                "latent": None,
                "text_emb": None,
                "text_mask": None,
                "reference": None,
            }
        ]
        result = collate_cached_batch(batch)
        assert result["latent"] is None
