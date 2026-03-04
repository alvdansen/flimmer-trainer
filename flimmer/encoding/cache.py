"""Cache I/O for encoded latents.

Handles reading and writing safetensors cache files, building and
maintaining the cache manifest, and detecting stale entries.

Cache directory layout::

    {cache_dir}/
        cache_manifest.json        ← what's cached, with what model/dtype
        latents/
            clip_001_81x480x848.safetensors
            clip_001_49x480x848.safetensors
            clip_001_17x480x848.safetensors
        text/
            clip_001.safetensors    ← one per stem (shared across frame counts)
        references/
            clip_001.safetensors    ← one per stem

File naming conventions:
    - Latents: {sample_id}.safetensors (unique per expansion)
    - Text: {source_stem}.safetensors (shared — same caption regardless of frames)
    - References: {source_stem}.safetensors (shared — same reference)

Staleness detection uses mtime + file size (not content hashing).
Hashing a 500MB video takes seconds; mtime+size is free and catches
all practical changes. Use --force for edge cases.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from flimmer.encoding.errors import CacheError
from flimmer.encoding.models import CacheEntry, CacheManifest, ExpandedSample


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CACHE_MANIFEST_FILENAME = "cache_manifest.json"
"""Name of the cache manifest file within the cache directory."""

LATENTS_SUBDIR = "latents"
TEXT_SUBDIR = "text"
REFERENCES_SUBDIR = "references"


# ---------------------------------------------------------------------------
# File naming
# ---------------------------------------------------------------------------

def latent_filename(sample_id: str) -> str:
    """Generate the latent cache filename for a sample.

    Args:
        sample_id: Unique sample identifier (e.g. 'clip_001_81x480x848').

    Returns:
        Relative path from cache_dir: 'latents/clip_001_81x480x848.safetensors'
    """
    return f"{LATENTS_SUBDIR}/{sample_id}.safetensors"


def text_filename(source_stem: str) -> str:
    """Generate the text encoding cache filename for a stem.

    Text encodings are shared across all frame counts of the same stem —
    the caption doesn't change when you take more or fewer frames.

    Args:
        source_stem: Original filename stem (e.g. 'clip_001').

    Returns:
        Relative path from cache_dir: 'text/clip_001.safetensors'
    """
    return f"{TEXT_SUBDIR}/{source_stem}.safetensors"


def reference_filename(source_stem: str) -> str:
    """Generate the reference encoding cache filename for a stem.

    Reference encodings are shared across all frame counts of the same stem.

    Args:
        source_stem: Original filename stem (e.g. 'clip_001').

    Returns:
        Relative path from cache_dir: 'references/clip_001.safetensors'
    """
    return f"{REFERENCES_SUBDIR}/{source_stem}.safetensors"


# ---------------------------------------------------------------------------
# Source file fingerprint
# ---------------------------------------------------------------------------

def _file_fingerprint(path: Path) -> tuple[float, int]:
    """Get mtime and size for a file. Returns (0.0, 0) if file not found."""
    try:
        stat = path.stat()
        return (stat.st_mtime, stat.st_size)
    except OSError:
        return (0.0, 0)


# ---------------------------------------------------------------------------
# Cache manifest I/O
# ---------------------------------------------------------------------------

def build_cache_manifest(
    samples: list[ExpandedSample],
    cache_dir: str | Path,
    vae_id: str = "",
    text_encoder_id: str = "",
    dtype: str = "bf16",
) -> CacheManifest:
    """Build a cache manifest from expanded samples.

    Creates CacheEntry objects for each sample, recording their source
    file fingerprints and expected cache file locations. Does NOT create
    the actual cache files — that's the encoder's job.

    Entries are populated with:
    - source_path, source_mtime, source_size: for staleness detection
    - latent_file: always set (every sample needs a latent)
    - text_file: set if the sample has a caption
    - reference_file: set if the sample has a reference image

    Args:
        samples: Expanded samples to create entries for.
        cache_dir: Path to the cache directory (for recording context).
        vae_id: Identifier for the VAE model.
        text_encoder_id: Identifier for the text encoder.
        dtype: Tensor dtype string.

    Returns:
        A CacheManifest with entries for all samples.
    """
    # Track which stems we've already assigned text/reference files to
    # (shared across frame counts)
    seen_text_stems: set[str] = set()
    seen_ref_stems: set[str] = set()

    entries: list[CacheEntry] = []

    for sample in samples:
        mtime, size = _file_fingerprint(sample.target)

        # Text file — one per stem, first occurrence claims it
        t_file: str | None = None
        if sample.caption is not None and sample.source_stem not in seen_text_stems:
            t_file = text_filename(sample.source_stem)
            seen_text_stems.add(sample.source_stem)

        # Reference file — one per stem
        r_file: str | None = None
        if sample.reference is not None and sample.source_stem not in seen_ref_stems:
            r_file = reference_filename(sample.source_stem)
            seen_ref_stems.add(sample.source_stem)

        # Preserve caption source path so cache-text can find captions
        # in flimmer layout (where captions are in signals/captions/, not
        # alongside the target file).
        caption_src: str | None = None
        if t_file is not None and sample.caption is not None:
            caption_src = str(sample.caption)

        # Preserve reference source path so cache-latents can find the
        # reference image to VAE-encode for I2V first-frame conditioning.
        ref_src: str | None = None
        if r_file is not None and sample.reference is not None:
            ref_src = str(sample.reference)

        entries.append(CacheEntry(
            sample_id=sample.sample_id,
            source_path=str(sample.target),
            source_mtime=mtime,
            source_size=size,
            latent_file=latent_filename(sample.sample_id),
            text_file=t_file,
            reference_file=r_file,
            caption_source_path=caption_src,
            reference_source_path=ref_src,
            bucket_key=sample.bucket_key,
        ))

    return CacheManifest(
        format_version=1,
        vae_id=vae_id,
        text_encoder_id=text_encoder_id,
        dtype=dtype,
        entries=entries,
    )


def save_cache_manifest(
    manifest: CacheManifest,
    cache_dir: str | Path,
) -> Path:
    """Write a cache manifest to disk as JSON.

    Creates the cache directory if it doesn't exist.

    Args:
        manifest: The manifest to save.
        cache_dir: Path to the cache directory.

    Returns:
        Path to the written manifest file.

    Raises:
        CacheError: If the file can't be written.
    """
    cache_dir = Path(cache_dir)

    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise CacheError(
            f"Failed to create cache directory '{cache_dir}': {e}\n"
            f"Check disk space and permissions."
        ) from e

    manifest_path = cache_dir / CACHE_MANIFEST_FILENAME

    # Serialize to dict (Pydantic v2)
    data = manifest.model_dump(mode="json")

    try:
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except OSError as e:
        raise CacheError(
            f"Failed to write cache manifest to '{manifest_path}': {e}\n"
            f"Check disk space and permissions."
        ) from e

    return manifest_path


def load_cache_manifest(
    cache_dir: str | Path,
) -> CacheManifest:
    """Load a cache manifest from disk.

    Args:
        cache_dir: Path to the cache directory containing cache_manifest.json.

    Returns:
        The loaded CacheManifest.

    Raises:
        CacheError: If the manifest doesn't exist or can't be parsed.
    """
    cache_dir = Path(cache_dir)
    manifest_path = cache_dir / CACHE_MANIFEST_FILENAME

    if not manifest_path.is_file():
        raise CacheError(
            f"No cache manifest found at '{manifest_path}'.\n"
            f"Run 'flimmer cache-latents' first to build the cache."
        )

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise CacheError(
            f"Failed to parse cache manifest '{manifest_path}': {e}\n"
            f"The file may be corrupted. Delete it and re-run caching."
        ) from e

    try:
        return CacheManifest.model_validate(data)
    except Exception as e:
        raise CacheError(
            f"Invalid cache manifest format in '{manifest_path}': {e}\n"
            f"The format may be outdated. Delete it and re-run caching."
        ) from e


# ---------------------------------------------------------------------------
# Staleness detection
# ---------------------------------------------------------------------------

def find_stale_entries(
    manifest: CacheManifest,
) -> list[CacheEntry]:
    """Find cache entries whose source files have changed.

    Compares the recorded mtime+size against current file state.
    An entry is stale if:
    - The source file's mtime has changed
    - The source file's size has changed
    - The source file no longer exists

    Args:
        manifest: The cache manifest to check.

    Returns:
        List of stale CacheEntry objects.
    """
    stale: list[CacheEntry] = []

    for entry in manifest.entries:
        source_path = Path(entry.source_path)
        current_mtime, current_size = _file_fingerprint(source_path)

        if current_mtime == 0.0 and current_size == 0:
            # File doesn't exist anymore
            stale.append(entry)
        elif (
            current_mtime != entry.source_mtime
            or current_size != entry.source_size
        ):
            stale.append(entry)

    return stale


def find_missing_entries(
    manifest: CacheManifest,
    cache_dir: str | Path,
) -> list[CacheEntry]:
    """Find cache entries whose cache files are missing on disk.

    An entry is missing if its latent_file doesn't exist in cache_dir.

    Args:
        manifest: The cache manifest to check.
        cache_dir: Path to the cache directory.

    Returns:
        List of CacheEntry objects with missing cache files.
    """
    cache_dir = Path(cache_dir)
    missing: list[CacheEntry] = []

    for entry in manifest.entries:
        if entry.latent_file:
            latent_path = cache_dir / entry.latent_file
            if not latent_path.is_file():
                missing.append(entry)

    return missing


# ---------------------------------------------------------------------------
# Cache directory setup
# ---------------------------------------------------------------------------

def ensure_cache_dirs(cache_dir: str | Path) -> None:
    """Create the cache directory structure.

    Creates:
        {cache_dir}/
        {cache_dir}/latents/
        {cache_dir}/text/
        {cache_dir}/references/

    Args:
        cache_dir: Root cache directory path.

    Raises:
        CacheError: If directories can't be created.
    """
    cache_dir = Path(cache_dir)

    try:
        for subdir in [LATENTS_SUBDIR, TEXT_SUBDIR, REFERENCES_SUBDIR]:
            (cache_dir / subdir).mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise CacheError(
            f"Failed to create cache directories in '{cache_dir}': {e}\n"
            f"Check disk space and permissions."
        ) from e
