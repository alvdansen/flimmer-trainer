"""Area-based bucketing for latent pre-encoding.

Production bucketing that assigns training samples to (W, H, F) buckets.
Unlike Phase 4's preview bucketing (which just snaps to grid), this module
generates the actual bucket set and assigns samples to the closest match.

Bucketing ensures every sample in a batch has identical dimensions —
no padding waste, efficient GPU utilization. The bucket set is generated
from configurable constraints:

    - Target area (total pixels): controls VRAM usage per sample
    - Aspect ratio range: prevents extreme distortion
    - Resolution step: alignment for VAE compatibility (usually 16px)
    - Frame counts: which temporal lengths to support (4n+1 for Wan)

Two entry points:
    - generate_buckets(): creates the set of valid (W, H) spatial buckets
    - assign_buckets(): maps ExpandedSamples to their nearest bucket
"""

from __future__ import annotations

import math

from flimmer.encoding.models import ExpandedSample


# ---------------------------------------------------------------------------
# Bucket generation
# ---------------------------------------------------------------------------

def generate_buckets(
    target_area: int = 512 * 512,
    min_dim: int = 256,
    max_dim: int = 1024,
    step: int = 16,
    min_aspect: float = 0.5,
    max_aspect: float = 2.0,
) -> list[tuple[int, int]]:
    """Generate all valid (width, height) spatial bucket dimensions.

    Produces every (W, H) pair where:
    - Both W and H are multiples of `step`
    - Both W and H are within [min_dim, max_dim]
    - The aspect ratio W/H is within [min_aspect, max_aspect]
    - W * H is closest to target_area (within step tolerance)

    The result is sorted by area descending, then by width descending.

    Args:
        target_area: Target pixel area per frame. 512*512 = 262144 for
            480p-ish training. Controls VRAM per sample.
        min_dim: Minimum dimension in pixels. Below this, quality degrades.
        max_dim: Maximum dimension in pixels. Above this, VRAM explodes.
        step: Pixel alignment step. 16 for Wan VAE compatibility.
        min_aspect: Minimum aspect ratio (W/H). 0.5 = 1:2 portrait.
        max_aspect: Maximum aspect ratio (W/H). 2.0 = 2:1 landscape.

    Returns:
        List of (width, height) tuples, sorted by area desc then width desc.

    Raises:
        ValueError: If constraints produce no valid buckets.
    """
    if step <= 0:
        raise ValueError(
            f"step must be positive, got {step}. "
            f"Use 16 for standard VAE alignment."
        )
    if min_dim > max_dim:
        raise ValueError(
            f"min_dim ({min_dim}) must be <= max_dim ({max_dim})."
        )
    if min_aspect > max_aspect:
        raise ValueError(
            f"min_aspect ({min_aspect}) must be <= max_aspect ({max_aspect})."
        )

    buckets: list[tuple[int, int]] = []

    # Enumerate all valid width values
    w = min_dim
    while w <= max_dim:
        # For this width, find the height closest to target_area
        ideal_h = target_area / w
        # Snap to step grid (round to nearest, not just floor)
        snapped_h = round(ideal_h / step) * step
        # Clamp to valid range
        snapped_h = max(min_dim, min(max_dim, snapped_h))

        # Check aspect ratio
        if w > 0 and snapped_h > 0:
            aspect = w / snapped_h
            if min_aspect <= aspect <= max_aspect:
                pair = (w, snapped_h)
                if pair not in buckets:
                    buckets.append(pair)

        w += step

    if not buckets:
        raise ValueError(
            f"No valid buckets for target_area={target_area}, "
            f"min_dim={min_dim}, max_dim={max_dim}, step={step}, "
            f"aspect=[{min_aspect}, {max_aspect}]. "
            f"Try relaxing the constraints."
        )

    # Sort by area descending, then width descending for deterministic order
    buckets.sort(key=lambda b: (-b[0] * b[1], -b[0]))
    return buckets


def _closest_bucket(
    width: int,
    height: int,
    buckets: list[tuple[int, int]],
) -> tuple[int, int]:
    """Find the bucket with the smallest area difference from (width, height).

    When multiple buckets tie on area difference, prefer the one with
    the closest aspect ratio to the source.

    Args:
        width: Source width in pixels.
        height: Source height in pixels.
        buckets: Available (W, H) bucket dimensions.

    Returns:
        The best-matching (W, H) bucket.
    """
    source_area = width * height
    source_aspect = width / height if height > 0 else 1.0

    best = buckets[0]
    best_area_diff = abs(best[0] * best[1] - source_area)
    best_aspect_diff = abs(best[0] / best[1] - source_aspect) if best[1] > 0 else float("inf")

    for bw, bh in buckets[1:]:
        area_diff = abs(bw * bh - source_area)
        aspect_diff = abs(bw / bh - source_aspect) if bh > 0 else float("inf")

        if area_diff < best_area_diff or (
            area_diff == best_area_diff and aspect_diff < best_aspect_diff
        ):
            best = (bw, bh)
            best_area_diff = area_diff
            best_aspect_diff = aspect_diff

    return best


# ---------------------------------------------------------------------------
# Bucket assignment
# ---------------------------------------------------------------------------

def assign_buckets(
    samples: list[ExpandedSample],
    buckets: list[tuple[int, int]] | None = None,
    target_area: int = 512 * 512,
    step: int = 16,
) -> list[ExpandedSample]:
    """Assign bucket dimensions to a list of ExpandedSamples.

    If samples already have bucket dimensions set (from expansion), they're
    kept as-is. If buckets are provided, each sample is matched to the
    closest bucket by area and aspect ratio.

    This returns NEW ExpandedSample objects with updated bucket dimensions.
    The originals are not modified (frozen models).

    Args:
        samples: Samples to assign buckets to.
        buckets: Pre-generated bucket set. If None, uses simple snap-to-grid.
        target_area: Target area for snap-to-grid fallback.
        step: Pixel alignment step.

    Returns:
        New list of ExpandedSamples with bucket dimensions set.
    """
    if not samples:
        return []

    result: list[ExpandedSample] = []

    for sample in samples:
        if sample.bucket_width > 0 and sample.bucket_height > 0:
            # Already has bucket dimensions (set during expansion)
            result.append(sample)
            continue

        # Need to assign a bucket
        w = sample.bucket_width if sample.bucket_width > 0 else 512
        h = sample.bucket_height if sample.bucket_height > 0 else 512

        if buckets:
            bw, bh = _closest_bucket(w, h, buckets)
        else:
            # Simple snap-to-grid
            bw = (w // step) * step
            bh = (h // step) * step
            bw = max(step, bw)
            bh = max(step, bh)

        # Create new sample with updated bucket dims
        result.append(sample.model_copy(update={
            "bucket_width": bw,
            "bucket_height": bh,
        }))

    return result


def bucket_groups(
    samples: list[ExpandedSample],
) -> dict[str, list[ExpandedSample]]:
    """Group samples by bucket key.

    Returns a dict mapping bucket keys ('{W}x{H}x{F}') to the list
    of samples in that bucket. Useful for batch construction and
    cache statistics.

    Args:
        samples: List of ExpandedSamples with bucket dimensions set.

    Returns:
        Dict mapping bucket_key → list of samples.
    """
    groups: dict[str, list[ExpandedSample]] = {}
    for sample in samples:
        key = sample.bucket_key
        groups.setdefault(key, []).append(sample)
    return groups
