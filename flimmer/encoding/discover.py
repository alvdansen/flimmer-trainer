"""Sample discovery for the encoding pipeline.

Discovers training samples from organized dataset directories or from
a flimmer_manifest.json file. Classifies each file by its SampleRole
and produces DiscoveredSample objects ready for expansion.

This module reuses the existing dataset discovery infrastructure
(flimmer.dataset.discover) and adds the SampleRole classification
that the encoding pipeline needs.

Two discovery modes:
    1. Directory discovery — scans an organized dataset folder, detects
       layout (flat vs flimmer), pairs files by stem, probes video metadata.
    2. Manifest discovery — reads a flimmer_manifest.json for pre-validated
       metadata (faster, no re-probing needed).
"""

from __future__ import annotations

import json
from pathlib import Path

from flimmer.dataset.discover import (
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
    detect_structure,
    discover_files,
    pair_samples,
)
from flimmer.dataset.models import SamplePair, StructureType
from flimmer.encoding.errors import FlimmerEncodingError
from flimmer.encoding.models import DiscoveredSample, SampleRole


# ---------------------------------------------------------------------------
# Role classification
# ---------------------------------------------------------------------------

def _classify_target_role(path: Path) -> SampleRole:
    """Determine whether a target file is a video or image.

    Videos are encoded as multi-frame latents through the VAE.
    Images are treated as 1-frame videos (same VAE path, just F=1).
    """
    ext = path.suffix.lower()
    if ext in VIDEO_EXTENSIONS:
        return SampleRole.TARGET_VIDEO
    if ext in IMAGE_EXTENSIONS:
        return SampleRole.TARGET_IMAGE
    # Default to video — validation will catch truly invalid files
    return SampleRole.TARGET_VIDEO


# ---------------------------------------------------------------------------
# Probe helpers
# ---------------------------------------------------------------------------

def _try_probe_video(path: Path) -> dict:
    """Attempt to probe video metadata. Returns empty dict on failure.

    Uses flimmer.video.probe if available. Gracefully degrades if the
    video module or ffprobe is not installed — discovery still works,
    just without dimensions (expansion will use defaults).
    """
    try:
        from flimmer.video.probe import probe_video
        meta = probe_video(str(path))
        return {
            "width": meta.width,
            "height": meta.height,
            "frame_count": meta.frame_count,
            "fps": meta.fps,
            "duration": meta.duration,
        }
    except Exception:
        return {}


def _try_probe_image(path: Path) -> dict:
    """Attempt to get image dimensions. Returns empty dict on failure.

    Tries PIL first (most accurate), falls back to empty dict.
    Image 'frame_count' is always 1.
    """
    try:
        from PIL import Image
        with Image.open(path) as img:
            w, h = img.size
        return {
            "width": w,
            "height": h,
            "frame_count": 1,
            "fps": 0.0,
            "duration": 0.0,
        }
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Discovery from directory
# ---------------------------------------------------------------------------

def _sample_pair_to_discovered(
    pair: SamplePair,
    probe: bool = True,
) -> DiscoveredSample:
    """Convert a dataset SamplePair to a DiscoveredSample.

    Adds SampleRole classification and optionally probes video metadata.
    """
    role = _classify_target_role(pair.target)

    # Probe for metadata if requested
    meta: dict = {}
    if probe:
        if role == SampleRole.TARGET_VIDEO:
            meta = _try_probe_video(pair.target)
        elif role == SampleRole.TARGET_IMAGE:
            meta = _try_probe_image(pair.target)

    return DiscoveredSample(
        stem=pair.stem,
        target=pair.target,
        target_role=role,
        caption=pair.caption,
        reference=pair.reference,
        width=meta.get("width", 0),
        height=meta.get("height", 0),
        frame_count=meta.get("frame_count", 0),
        fps=meta.get("fps", 0.0),
        duration=meta.get("duration", 0.0),
    )


def discover_from_directory(
    directory: str | Path,
    caption_required: bool = False,
    reference_required: bool = False,
    probe: bool = True,
) -> list[DiscoveredSample]:
    """Discover training samples from an organized dataset directory.

    Detects layout (flat vs flimmer), finds and pairs files by stem,
    classifies each target's role, and optionally probes video metadata.

    Args:
        directory: Path to the organized dataset directory.
        caption_required: If True, samples without captions are skipped.
        reference_required: If True, samples without references are skipped.
        probe: If True, probe video files for dimensions. Slower but
            provides metadata needed for expansion.

    Returns:
        List of DiscoveredSamples, one per valid target file.

    Raises:
        FlimmerEncodingError: If the directory doesn't exist.
    """
    directory = Path(directory).resolve()

    if not directory.is_dir():
        raise FlimmerEncodingError(
            f"Dataset directory not found: {directory}\n"
            f"Check the path in your training config's data_config."
        )

    structure = detect_structure(directory)
    files = discover_files(directory, structure)

    pairs, _orphaned = pair_samples(
        targets=files["targets"],
        captions=files["captions"],
        references=files["references"],
        caption_required=caption_required,
        reference_required=reference_required,
    )

    # Convert to DiscoveredSamples (skip pairs with errors if requirements set)
    samples: list[DiscoveredSample] = []
    for pair in pairs:
        if caption_required and pair.caption is None:
            continue
        if reference_required and pair.reference is None:
            continue
        samples.append(_sample_pair_to_discovered(pair, probe=probe))

    return samples


# ---------------------------------------------------------------------------
# Discovery from manifest
# ---------------------------------------------------------------------------

def discover_from_manifest(
    manifest_path: str | Path,
) -> list[DiscoveredSample]:
    """Discover training samples from a flimmer_manifest.json file.

    Reads pre-validated metadata from the manifest — faster than directory
    discovery because it skips file scanning and video probing.

    The manifest format matches Phase 4's organize output.

    Args:
        manifest_path: Path to the flimmer_manifest.json file.

    Returns:
        List of DiscoveredSamples from the manifest.

    Raises:
        FlimmerEncodingError: If the manifest can't be read or parsed.
    """
    manifest_path = Path(manifest_path).resolve()

    if not manifest_path.is_file():
        raise FlimmerEncodingError(
            f"Manifest not found: {manifest_path}\n"
            f"Run 'flimmer dataset organize' first to generate a manifest."
        )

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise FlimmerEncodingError(
            f"Failed to parse manifest '{manifest_path}': {e}\n"
            f"The file may be corrupted. Re-run 'flimmer dataset organize'."
        ) from e

    if not isinstance(data, dict):
        raise FlimmerEncodingError(
            f"Invalid manifest format in '{manifest_path}'. "
            f"Expected a JSON object, got {type(data).__name__}."
        )

    manifest_dir = manifest_path.parent
    samples_data = data.get("samples", [])
    if not isinstance(samples_data, list):
        raise FlimmerEncodingError(
            f"Invalid manifest: 'samples' must be a list, "
            f"got {type(samples_data).__name__}."
        )

    samples: list[DiscoveredSample] = []
    for entry in samples_data:
        if not isinstance(entry, dict):
            continue

        stem = entry.get("stem", "")
        target_path_str = entry.get("target", "")
        if not stem or not target_path_str:
            continue

        target_path = _resolve_manifest_path(target_path_str, manifest_dir)
        role = _classify_target_role(target_path)

        caption_path = None
        if entry.get("caption"):
            caption_path = _resolve_manifest_path(entry["caption"], manifest_dir)

        reference_path = None
        if entry.get("reference"):
            reference_path = _resolve_manifest_path(entry["reference"], manifest_dir)

        samples.append(DiscoveredSample(
            stem=stem,
            target=target_path,
            target_role=role,
            caption=caption_path,
            reference=reference_path,
            width=entry.get("width", 0),
            height=entry.get("height", 0),
            frame_count=entry.get("frame_count", 0),
            fps=entry.get("fps", 0.0),
            duration=entry.get("duration", 0.0),
            repeats=entry.get("repeats", 1),
            loss_multiplier=entry.get("loss_multiplier", 1.0),
        ))

    return samples


def _resolve_manifest_path(path_str: str, base_dir: Path) -> Path:
    """Resolve a path from a manifest relative to the manifest's directory."""
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (base_dir / p).resolve()


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

def discover_samples(
    source: str | Path,
    caption_required: bool = False,
    reference_required: bool = False,
    probe: bool = True,
) -> list[DiscoveredSample]:
    """Discover training samples from a directory or manifest.

    Auto-detects the source type:
    - If source is a .json file, reads it as a manifest
    - Otherwise, treats it as a directory and scans for files

    This is the main entry point for the encoding pipeline's discovery step.

    Args:
        source: Path to dataset directory or manifest.json file.
        caption_required: Skip samples without captions.
        reference_required: Skip samples without references.
        probe: Probe video files for metadata (ignored for manifests).

    Returns:
        List of DiscoveredSamples.
    """
    source = Path(source)

    if source.suffix.lower() == ".json" and source.is_file():
        return discover_from_manifest(source)

    return discover_from_directory(
        source,
        caption_required=caption_required,
        reference_required=reference_required,
        probe=probe,
    )
