"""CLI for the Flimmer encoding pipeline.

Commands:
    info           Show cache status for a training config
    cache-latents  Encode video/image targets through VAE (requires GPU)
    cache-text     Encode captions through T5 text encoder (requires GPU)

Usage::

    # Show what would be cached (no GPU needed)
    python -m flimmer.encoding info --config path/to/train.yaml

    # Encode latents (VAE — needs GPU)
    python -m flimmer.encoding cache-latents --config path/to/train.yaml

    # Encode text (T5 — needs GPU, run separately from latents)
    python -m flimmer.encoding cache-text --config path/to/train.yaml

The two-step caching design ensures VAE and T5 never compete for VRAM.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_dataset_dirs(data_config_path: str | Path) -> list[Path]:
    """Get the actual dataset directories from a data config.

    Loads the data config and returns the resolved paths from the
    datasets[].path entries. Falls back to the data config file's
    parent directory if loading fails.
    """
    data_config_path = Path(data_config_path)
    try:
        from flimmer.config.loader import load_data_config
        data_config = load_data_config(data_config_path)
        return [Path(ds.path) for ds in data_config.datasets]
    except Exception:
        # Fallback: use the data config file's parent directory
        if data_config_path.is_file():
            return [data_config_path.parent]
        return [data_config_path]


def _auto_extract_first_frames(
    data_config_path: str | Path,
    dataset_dirs: list[Path],
) -> int:
    """Auto-extract first frames as reference images for I2V training.

    Only called when data config has source: first_frame. Writes PNGs
    to the references location so discovery finds them.

    Returns number of frames extracted.
    """
    from flimmer.dataset.discover import detect_structure, discover_files, StructureType
    from flimmer.video.extract import extract_first_frame

    extracted = 0
    for dataset_dir in dataset_dirs:
        structure = detect_structure(dataset_dir)
        files = discover_files(dataset_dir, structure)

        # Determine output directory for references
        if structure == StructureType.FLIMMER:
            ref_dir = dataset_dir / "training" / "signals" / "first_frame"
        else:
            ref_dir = dataset_dir  # flat: same directory as videos

        for video_path in files["targets"]:
            # Only extract for videos, not images
            if video_path.suffix.lower() not in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
                continue

            ref_path = ref_dir / f"{video_path.stem}.png"
            if ref_path.exists():
                continue

            ref_path.parent.mkdir(parents=True, exist_ok=True)
            extract_first_frame(video_path, ref_path)
            extracted += 1

    return extracted


def _discover_from_config(data_config_path: str | Path, probe: bool = True):
    """Discover samples from all dataset directories in a data config."""
    from flimmer.encoding.discover import discover_samples

    dataset_dirs = _get_dataset_dirs(data_config_path)
    all_samples = []
    for dataset_dir in dataset_dirs:
        samples = discover_samples(str(dataset_dir), probe=probe)
        all_samples.extend(samples)
    return all_samples, dataset_dirs


# ---------------------------------------------------------------------------
# Info command
# ---------------------------------------------------------------------------

def cmd_info(args: argparse.Namespace) -> None:
    """Show cache status for a training config.

    Discovers samples, expands them, and reports what would be cached.
    If a cache already exists, shows how many entries are complete vs stale.
    """
    from flimmer.config.training_loader import load_training_config
    from flimmer.encoding.cache import (
        CACHE_MANIFEST_FILENAME,
        find_missing_entries,
        find_stale_entries,
        load_cache_manifest,
    )
    from flimmer.encoding.expand import expand_samples
    from flimmer.encoding.bucket import bucket_groups

    config = load_training_config(args.config)

    samples, dataset_dirs = _discover_from_config(config.data_config)

    print(f"Training config: {args.config}")
    for d in dataset_dirs:
        print(f"Dataset: {d}")
    print(f"Cache dir: {config.cache.cache_dir}")
    print()

    print(f"Discovered: {len(samples)} source samples")

    if not samples:
        print("No samples found. Check your data_config path.")
        return

    # Expand
    expanded = expand_samples(
        samples,
        target_frames=config.cache.target_frames,
        frame_extraction=config.cache.frame_extraction,
        include_head_frame=config.cache.include_head_frame,
        step=config.cache.reso_step,
    )
    print(f"Expanded: {len(expanded)} training samples")

    # Bucket distribution
    groups = bucket_groups(expanded)
    print(f"Buckets: {len(groups)}")
    for key in sorted(groups.keys()):
        print(f"  {key}: {len(groups[key])} samples")

    # Check existing cache
    cache_dir = Path(config.cache.cache_dir)
    manifest_path = cache_dir / CACHE_MANIFEST_FILENAME

    if manifest_path.is_file():
        print()
        try:
            manifest = load_cache_manifest(cache_dir)
            print(f"Cache manifest: {manifest.total_entries} entries")
            print(f"  Latents: {manifest.latent_count}")
            print(f"  Text: {manifest.text_count}")
            print(f"  References: {manifest.reference_count}")

            stale = find_stale_entries(manifest)
            if stale:
                print(f"  Stale: {len(stale)} (source files changed)")

            missing = find_missing_entries(manifest, cache_dir)
            if missing:
                print(f"  Missing: {len(missing)} (cache files not on disk)")

            if not stale and not missing:
                print("  Status: all entries up to date")
        except Exception as e:
            print(f"  Error reading cache: {e}")
    else:
        print()
        print("No cache found. Run 'cache-latents' and 'cache-text' to build it.")


# ---------------------------------------------------------------------------
# Cache-latents command (placeholder for GPU encoding)
# ---------------------------------------------------------------------------

def cmd_cache_latents(args: argparse.Namespace) -> None:
    """Encode video/image targets through VAE and cache to disk."""
    from flimmer.config.training_loader import load_training_config
    from flimmer.encoding.cache import (
        build_cache_manifest,
        ensure_cache_dirs,
        save_cache_manifest,
    )
    from flimmer.encoding.expand import expand_samples

    config = load_training_config(args.config)

    cache_dir = Path(config.cache.cache_dir)

    print(f"Cache directory: {cache_dir}")
    print(f"Dtype: {config.cache.dtype}")
    print(f"Frame counts: {config.cache.target_frames}")
    print()

    # Auto-extract first-frame references for I2V if configured
    from flimmer.config.loader import load_data_config
    data_config = load_data_config(config.data_config)
    if data_config.controls.images.reference.source == "first_frame" and not args.dry_run:
        dataset_dirs = _get_dataset_dirs(config.data_config)
        count = _auto_extract_first_frames(config.data_config, dataset_dirs)
        if count:
            print(f"Auto-extracted {count} first-frame references for I2V")
            print()

    # Discover from actual dataset paths in data config
    samples, _ = _discover_from_config(config.data_config)
    print(f"Discovered: {len(samples)} source samples")

    if not samples:
        print("No samples found. Nothing to cache.")
        return

    # Expand
    expanded = expand_samples(
        samples,
        target_frames=config.cache.target_frames,
        frame_extraction=config.cache.frame_extraction,
        include_head_frame=config.cache.include_head_frame,
        step=config.cache.reso_step,
    )
    print(f"Expanded: {len(expanded)} training samples")

    # Create cache dirs
    ensure_cache_dirs(cache_dir)

    # Build manifest
    vae_id = config.model.vae or config.model.path or "unknown"
    manifest = build_cache_manifest(
        expanded,
        cache_dir=cache_dir,
        vae_id=vae_id,
        dtype=config.cache.dtype,
    )

    # Save manifest
    manifest_path = save_cache_manifest(manifest, cache_dir)
    print(f"\nManifest written: {manifest_path}")
    print(f"Entries: {manifest.total_entries}")
    print()

    if args.dry_run:
        print("Dry run — no encoding performed.")
        return

    # --- Real VAE encoding ---
    from flimmer.encoding.vae_encoder import WanVaeEncoder

    encoder = WanVaeEncoder(
        model_path=config.model.path or "",
        vae_path=getattr(config.model, "vae", None),
        dtype=config.cache.dtype,
    )

    encoded_count = 0
    skip_count = 0
    error_count = 0

    for i, (sample, entry) in enumerate(zip(expanded, manifest.entries)):
        latent_path = cache_dir / entry.latent_file
        if latent_path.is_file() and not args.force:
            skip_count += 1
            continue

        print(f"  [{i+1}/{len(expanded)}] Encoding: {sample.source_stem} "
              f"({sample.bucket_frames}f {sample.bucket_width}x{sample.bucket_height})")

        try:
            result = encoder.encode(
                str(sample.target),
                target_width=sample.bucket_width,
                target_height=sample.bucket_height,
                target_frames=sample.bucket_frames,
                frame_extraction=sample.frame_extraction.value,
            )

            from safetensors.torch import save_file
            latent_path.parent.mkdir(parents=True, exist_ok=True)
            save_file({"latent": result["latent"]}, str(latent_path))
            encoded_count += 1

        except Exception as e:
            print(f"    ERROR: {e}")
            error_count += 1

    # --- Reference image encoding (I2V first-frame conditioning) ---
    # Deduplicate: one reference per source path (multiple entries may
    # share the same reference image across frame-count expansions).
    ref_entries = [
        e for e in manifest.entries
        if e.reference_source_path and e.reference_file
    ]
    seen_ref_paths: set[str] = set()
    unique_refs: list[tuple] = []
    for entry in ref_entries:
        if entry.reference_source_path not in seen_ref_paths:
            unique_refs.append(entry)
            seen_ref_paths.add(entry.reference_source_path)

    ref_encoded = 0
    ref_skipped = 0
    ref_errors = 0

    if unique_refs:
        print(f"\nEncoding {len(unique_refs)} reference images...")
        for i, entry in enumerate(unique_refs):
            ref_out_path = cache_dir / entry.reference_file
            if ref_out_path.is_file() and not args.force:
                ref_skipped += 1
                continue

            stem = Path(entry.reference_source_path).stem
            print(f"  [{i+1}/{len(unique_refs)}] Reference: {stem}")

            try:
                # Get target resolution from the entry's bucket key
                # bucket_key format: "{W}x{H}x{F}"
                parts = entry.bucket_key.split("x")
                target_w = int(parts[0])
                target_h = int(parts[1])

                result = encoder.encode_reference_image(
                    Path(entry.reference_source_path),
                    target_width=target_w,
                    target_height=target_h,
                )

                from safetensors.torch import save_file
                ref_out_path.parent.mkdir(parents=True, exist_ok=True)
                save_file(result, str(ref_out_path))
                ref_encoded += 1

            except Exception as e:
                print(f"    ERROR: {e}")
                ref_errors += 1

    encoder.cleanup()
    save_cache_manifest(manifest, cache_dir)

    print(f"\nVAE encoding complete:")
    print(f"  Latents encoded: {encoded_count}")
    print(f"  Latents skipped (cached): {skip_count}")
    if ref_encoded or ref_skipped:
        print(f"  References encoded: {ref_encoded}")
        print(f"  References skipped (cached): {ref_skipped}")
    if error_count or ref_errors:
        print(f"  Errors: {error_count + ref_errors}")


# ---------------------------------------------------------------------------
# Cache-text command (placeholder for GPU encoding)
# ---------------------------------------------------------------------------

def cmd_cache_text(args: argparse.Namespace) -> None:
    """Encode captions through T5 and cache to disk."""
    from flimmer.config.training_loader import load_training_config
    from flimmer.encoding.cache import (
        ensure_cache_dirs,
        load_cache_manifest,
        save_cache_manifest,
    )

    config = load_training_config(args.config)
    cache_dir = Path(config.cache.cache_dir)

    print(f"Cache directory: {cache_dir}")
    print()

    # Load existing manifest (cache-latents must run first)
    try:
        manifest = load_cache_manifest(cache_dir)
    except Exception as e:
        print(f"Error: {e}")
        print("Run 'cache-latents' first to build the cache manifest.")
        sys.exit(1)

    # Count entries that need text encoding
    needs_text = [e for e in manifest.entries if e.text_file and not e.has_text]
    stems_with_text = {
        e.sample_id.rsplit("_", 1)[0]
        for e in manifest.entries
        if e.text_file
    }
    print(f"Stems with captions: {len(stems_with_text)}")

    if args.dry_run:
        print("Dry run — no encoding performed.")
        return

    # --- Real T5 encoding ---
    from flimmer.encoding.text_encoder import T5TextEncoder

    encoder = T5TextEncoder(
        model_id=config.model.path or "google/umt5-xxl",
        t5_path=getattr(config.model, "t5", None),
        dtype=config.cache.dtype,
    )

    encoded_count = 0
    skip_count = 0
    error_count = 0

    # Find unique text entries (one per stem).
    # Use caption_source_path from the manifest when available (flimmer layout
    # stores captions in signals/captions/, not alongside the target file).
    # Fall back to source.with_suffix(".txt") for flat-layout compatibility.
    text_entries: dict[str, str] = {}
    for entry in manifest.entries:
        if entry.text_file and entry.text_file not in text_entries:
            # Prefer the explicit caption path stored in the manifest
            if entry.caption_source_path:
                caption_path = Path(entry.caption_source_path)
            else:
                # Flat layout fallback: caption alongside target
                caption_path = Path(entry.source_path).with_suffix(".txt")
            if caption_path.is_file():
                text_entries[entry.text_file] = str(caption_path)

    print(f"Unique captions to encode: {len(text_entries)}")

    for i, (text_file, caption_path) in enumerate(text_entries.items()):
        text_out_path = cache_dir / text_file
        if text_out_path.is_file() and not getattr(args, "force", False):
            skip_count += 1
            continue

        stem = Path(text_file).stem
        print(f"  [{i+1}/{len(text_entries)}] Encoding: {stem}")

        try:
            result = encoder.encode(caption_path)

            from safetensors.torch import save_file
            text_out_path.parent.mkdir(parents=True, exist_ok=True)
            save_file(
                {"text_emb": result["text_emb"], "text_mask": result["text_mask"]},
                str(text_out_path),
            )
            encoded_count += 1

        except Exception as e:
            print(f"    ERROR: {e}")
            error_count += 1

    encoder.cleanup()
    save_cache_manifest(manifest, cache_dir)

    print(f"\nT5 encoding complete:")
    print(f"  Encoded: {encoded_count}")
    print(f"  Skipped (cached): {skip_count}")
    if error_count:
        print(f"  Errors: {error_count}")


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the encoding CLI."""
    parser = argparse.ArgumentParser(
        prog="python -m flimmer.encoding",
        description="Flimmer latent pre-encoding and caching pipeline.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ─── info ───
    info_parser = subparsers.add_parser(
        "info",
        help="Show cache status for a training config.",
    )
    info_parser.add_argument(
        "--config", "-c",
        required=True,
        help="Path to the Flimmer training config YAML.",
    )

    # ─── cache-latents ───
    latents_parser = subparsers.add_parser(
        "cache-latents",
        help="Encode video/image targets through VAE.",
    )
    latents_parser.add_argument(
        "--config", "-c",
        required=True,
        help="Path to the Flimmer training config YAML.",
    )
    latents_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build manifest without encoding (preview what would be cached).",
    )
    latents_parser.add_argument(
        "--force",
        action="store_true",
        help="Re-encode all entries, even if cache is up to date.",
    )

    # ─── cache-text ───
    text_parser = subparsers.add_parser(
        "cache-text",
        help="Encode captions through T5 text encoder.",
    )
    text_parser.add_argument(
        "--config", "-c",
        required=True,
        help="Path to the Flimmer training config YAML.",
    )
    text_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be encoded without actually encoding.",
    )

    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point for the encoding CLI."""
    parser = build_parser()
    args = parser.parse_args()

    commands = {
        "info": cmd_info,
        "cache-latents": cmd_cache_latents,
        "cache-text": cmd_cache_text,
    }

    try:
        commands[args.command](args)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
    except SystemExit:
        raise
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
