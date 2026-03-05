"""Training dataset and batch sampler for cached latents.

Provides the PyTorch Dataset and Sampler classes that the training loop
consumes. All data is pre-encoded and cached as safetensors — the dataset
just reads tensors from disk, no encoding at training time.

Two key classes:
    CachedLatentDataset  — map-style Dataset, reads safetensors by index
    BucketBatchSampler   — groups samples by bucket key for uniform batches

Usage (in training loop)::

    dataset = CachedLatentDataset(cache_dir, manifest)
    sampler = BucketBatchSampler(dataset, batch_size=1, shuffle=True)
    dataloader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_cached_batch)

    for batch in dataloader:
        latents = batch["latents"]       # [B, C, F, H, W]
        text_emb = batch["text_emb"]     # [B, seq_len, dim] or None
        ...

This module requires torch but is CPU-only (no GPU needed).
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from flimmer.encoding.errors import CacheError
from flimmer.encoding.models import CacheManifest


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CachedLatentDataset:
    """Map-style dataset that reads pre-encoded tensors from safetensors cache.

    Each item is a dict of tensors for one training sample. The dataset
    is initialized from a cache manifest — it knows where every file is
    without scanning the filesystem.

    Items are dicts with these possible keys:
        - 'latent': the encoded video/image latent tensor
        - 'text_emb': the T5 text embedding (if caption exists)
        - 'text_mask': the T5 attention mask (if caption exists)
        - 'reference': the encoded reference image (if reference exists)
        - 'bucket_key': string identifying the bucket (for sampler)
        - 'sample_id': unique identifier for debugging

    Missing signals (no caption, no reference) are represented as None
    in the dict, not as zero tensors. The training loop handles dropout
    and missing signals.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        manifest: CacheManifest,
    ) -> None:
        """Initialize the dataset from a cache manifest.

        Args:
            cache_dir: Path to the cache directory containing safetensors files.
            manifest: Cache manifest describing all entries.
        """
        self._cache_dir = Path(cache_dir)
        self._manifest = manifest
        self._entries = list(manifest.entries)

        # Pre-compute bucket keys for each index (used by sampler)
        self._bucket_keys: list[str] = [e.bucket_key for e in self._entries]

        # Build stem → text_file lookup so ALL expansions of a stem share
        # the same text encoding. Text is per-stem (same caption regardless
        # of how many frames we take), but the cache manifest may only
        # populate text_file for one expansion per stem.
        self._stem_text_files: dict[str, str] = {}
        for entry in self._entries:
            if entry.text_file:
                # Extract stem from sample_id: "{stem}_{frames}x{H}x{W}"
                stem = _extract_stem(entry.sample_id)
                if stem:
                    self._stem_text_files[stem] = entry.text_file

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Load one sample's cached tensors.

        Reads safetensors files from disk. Tensors are loaded as-is
        (no dtype conversion — that happens during training).

        Args:
            index: Dataset index.

        Returns:
            Dict with 'latent', 'text_emb', 'text_mask', 'reference',
            'bucket_key', 'sample_id' keys.

        Raises:
            CacheError: If required files are missing or corrupt.
        """
        entry = self._entries[index]
        result: dict[str, Any] = {
            "sample_id": entry.sample_id,
            "bucket_key": entry.bucket_key,
            "latent": None,
            "text_emb": None,
            "text_mask": None,
            "reference": None,
        }

        # Load latent
        if entry.latent_file:
            latent_path = self._cache_dir / entry.latent_file
            if latent_path.is_file():
                tensors = _load_safetensors(latent_path)
                latent = tensors.get("latent")
                # VAE encoder saves with batch dim [1, C, F, H, W];
                # squeeze to [C, F, H, W] so DataLoader can add the
                # real batch dimension during collation.
                if latent is not None and latent.ndim == 5 and latent.shape[0] == 1:
                    latent = latent.squeeze(0)
                result["latent"] = latent

        # Load text embedding — try entry's text_file first, then
        # fall back to stem lookup (text is shared across expansions)
        text_file = entry.text_file
        if not text_file:
            stem = _extract_stem(entry.sample_id)
            text_file = self._stem_text_files.get(stem) if stem else None

        if text_file:
            text_path = self._cache_dir / text_file
            if text_path.is_file():
                tensors = _load_safetensors(text_path)
                result["text_emb"] = tensors.get("text_emb")
                result["text_mask"] = tensors.get("text_mask")

        # Load reference
        if entry.reference_file:
            ref_path = self._cache_dir / entry.reference_file
            if ref_path.is_file():
                tensors = _load_safetensors(ref_path)
                result["reference"] = tensors.get("reference")

        return result

    @property
    def bucket_keys(self) -> list[str]:
        """Bucket key for each sample index (for sampler construction)."""
        return self._bucket_keys

    @property
    def manifest(self) -> CacheManifest:
        """The cache manifest this dataset was built from."""
        return self._manifest

    def get_repeats(self, index: int) -> int:
        """Get the repeat count for a sample by index.

        Returns how many times this sample should appear per epoch in the
        sampler. Default is 1. Used by BucketBatchSampler to expand indices.
        """
        return self._entries[index].repeats

    @property
    def repeats(self) -> list[int]:
        """Repeat count for each sample index (for sampler construction)."""
        return [e.repeats for e in self._entries]


# ---------------------------------------------------------------------------
# Batch sampler
# ---------------------------------------------------------------------------

class BucketBatchSampler:
    """Groups samples by bucket key so each batch has uniform dimensions.

    Every batch contains samples from a single bucket — same (W, H, F)
    dimensions. This avoids padding waste and simplifies the training loop
    (no need to handle mixed dimensions within a batch).

    Within each bucket, samples can be shuffled. Buckets are also shuffled
    so the training order varies per epoch.

    Usage::

        sampler = BucketBatchSampler(dataset, batch_size=1, shuffle=True)
        dataloader = DataLoader(dataset, batch_sampler=sampler)
    """

    def __init__(
        self,
        dataset: CachedLatentDataset,
        batch_size: int = 1,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int | None = None,
    ) -> None:
        """Initialize the bucket batch sampler.

        Args:
            dataset: The CachedLatentDataset to sample from.
            batch_size: Number of samples per batch.
            shuffle: Whether to shuffle within and across buckets.
            drop_last: Drop incomplete batches at the end of each bucket.
            seed: Random seed for reproducibility.
        """
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._drop_last = drop_last
        self._rng = random.Random(seed)

        # Group indices by bucket key, expanding by repeats.
        # A sample with repeats=3 appears 3 times in its bucket's index list,
        # so it gets picked ~3x per epoch when the sampler iterates.
        self._bucket_indices: dict[str, list[int]] = {}
        for idx, key in enumerate(dataset.bucket_keys):
            repeat_count = dataset.get_repeats(idx)
            self._bucket_indices.setdefault(key, []).extend([idx] * repeat_count)

    def __iter__(self):
        """Yield batches of indices, one bucket at a time."""
        # Get all bucket keys
        bucket_keys = list(self._bucket_indices.keys())
        if self._shuffle:
            self._rng.shuffle(bucket_keys)

        for key in bucket_keys:
            indices = list(self._bucket_indices[key])
            if self._shuffle:
                self._rng.shuffle(indices)

            # Yield batches from this bucket
            for start in range(0, len(indices), self._batch_size):
                batch = indices[start : start + self._batch_size]
                if self._drop_last and len(batch) < self._batch_size:
                    continue
                yield batch

    def __len__(self) -> int:
        """Total number of batches across all buckets."""
        total = 0
        for indices in self._bucket_indices.values():
            n = len(indices)
            if self._drop_last:
                total += n // self._batch_size
            else:
                total += (n + self._batch_size - 1) // self._batch_size
        return total

    @property
    def bucket_count(self) -> int:
        """Number of distinct buckets."""
        return len(self._bucket_indices)

    @property
    def bucket_sizes(self) -> dict[str, int]:
        """Number of samples in each bucket."""
        return {k: len(v) for k, v in self._bucket_indices.items()}


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

def collate_cached_batch(
    batch: list[dict[str, Any]],
) -> dict[str, Any]:
    """Collate a batch of cached samples into stacked tensors.

    Guarantees every value is either a stacked Tensor or None — never
    a Python list. If all samples have None for a key (e.g. reference
    in T2V), the result is None. If only some are None, those are
    replaced with zero tensors (equivalent to dropout).

    This function requires torch to stack tensors. If torch is not
    available, returns the batch as a list (for testing without GPU).

    Args:
        batch: List of dicts from CachedLatentDataset.__getitem__.

    Returns:
        Dict with stacked tensors and metadata lists.
    """
    if not batch:
        return {}

    result: dict[str, Any] = {
        "sample_ids": [item["sample_id"] for item in batch],
        "bucket_key": batch[0]["bucket_key"],  # All same within a batch
    }

    try:
        import torch

        for key in ("latent", "text_emb", "text_mask", "reference"):
            values = [item[key] for item in batch]

            # All None → store as None (not a list of Nones).
            # This is the normal case for T2V reference, for example.
            if all(v is None for v in values):
                result[key] = None
                continue

            # Mixed None/non-None → replace Nones with zero tensors
            # matching the first non-None value's shape. This acts like
            # dropout (zeroed embeddings = unconditional generation).
            non_none = [v for v in values if v is not None]
            if len(non_none) < len(values):
                template = non_none[0]
                values = [
                    v if v is not None else torch.zeros_like(template)
                    for v in values
                ]

            result[key] = torch.stack(values)

    except ImportError:
        # No torch — return raw values (for testing)
        for key in ("latent", "text_emb", "text_mask", "reference"):
            values = [item[key] for item in batch]
            if all(v is None for v in values):
                result[key] = None
            else:
                result[key] = values

    return result


# ---------------------------------------------------------------------------
# Stem extraction helper
# ---------------------------------------------------------------------------

def _extract_stem(sample_id: str) -> str | None:
    """Extract the stem from a sample_id.

    Sample IDs follow the pattern: "{stem}_{frames}x{H}x{W}"
    e.g. "clip_scene001_33x720x1264" → "clip_scene001"

    Returns None if the pattern doesn't match.
    """
    import re
    match = re.match(r"^(.+)_\d+x\d+x\d+$", sample_id)
    return match.group(1) if match else None


# ---------------------------------------------------------------------------
# safetensors I/O helper
# ---------------------------------------------------------------------------

def _load_safetensors(path: Path) -> dict[str, Any]:
    """Load tensors from a safetensors file.

    Returns a dict mapping tensor names to tensor objects. Uses the
    safetensors library if available, otherwise raises CacheError.
    """
    try:
        from safetensors.torch import load_file
        return load_file(str(path))
    except ImportError:
        raise CacheError(
            f"safetensors library not installed. "
            f"Install it with: pip install safetensors"
        )
    except Exception as e:
        raise CacheError(
            f"Failed to load '{path}': {e}\n"
            f"The file may be corrupted. Delete it and re-run caching."
        ) from e
