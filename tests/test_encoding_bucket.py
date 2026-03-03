"""Tests for flimmer.encoding.bucket — area-based bucketing.

Tests cover:
    - generate_buckets(): valid bucket generation, edge cases, errors
    - _closest_bucket(): matching by area and aspect ratio
    - assign_buckets(): bucket assignment on ExpandedSamples
    - bucket_groups(): grouping by bucket key
"""

from __future__ import annotations

from pathlib import Path

import pytest

from flimmer.encoding.bucket import (
    _closest_bucket,
    assign_buckets,
    bucket_groups,
    generate_buckets,
)
from flimmer.encoding.models import ExpandedSample, SampleRole


def _make_expanded(
    sample_id: str = "test",
    bucket_width: int = 0,
    bucket_height: int = 0,
    bucket_frames: int = 17,
) -> ExpandedSample:
    """Helper to create ExpandedSample for testing."""
    return ExpandedSample(
        sample_id=sample_id,
        source_stem="test",
        target=Path("/test.mp4"),
        target_role=SampleRole.TARGET_VIDEO,
        bucket_width=bucket_width,
        bucket_height=bucket_height,
        bucket_frames=bucket_frames,
    )


# ---------------------------------------------------------------------------
# generate_buckets
# ---------------------------------------------------------------------------

class TestGenerateBuckets:
    """Tests for bucket generation."""

    def test_default_params(self) -> None:
        """Generates buckets with default parameters."""
        buckets = generate_buckets()
        assert len(buckets) > 0
        # All dimensions should be multiples of 16
        for w, h in buckets:
            assert w % 16 == 0
            assert h % 16 == 0

    def test_512x512_in_buckets(self) -> None:
        """512x512 should be a bucket at target_area=512*512."""
        buckets = generate_buckets(target_area=512 * 512)
        assert (512, 512) in buckets

    def test_all_within_range(self) -> None:
        """All generated buckets are within min/max dim."""
        min_dim, max_dim = 256, 1024
        buckets = generate_buckets(min_dim=min_dim, max_dim=max_dim)
        for w, h in buckets:
            assert w >= min_dim
            assert w <= max_dim
            assert h >= min_dim
            assert h <= max_dim

    def test_aspect_ratio_bounds(self) -> None:
        """All buckets respect aspect ratio limits."""
        buckets = generate_buckets(min_aspect=0.5, max_aspect=2.0)
        for w, h in buckets:
            aspect = w / h
            assert 0.5 <= aspect <= 2.0

    def test_sorted_by_area_desc(self) -> None:
        """Buckets are sorted by area descending."""
        buckets = generate_buckets()
        areas = [w * h for w, h in buckets]
        for i in range(len(areas) - 1):
            assert areas[i] >= areas[i + 1]

    def test_step_alignment(self) -> None:
        """Custom step size is respected."""
        buckets = generate_buckets(step=32, target_area=512 * 512)
        for w, h in buckets:
            assert w % 32 == 0
            assert h % 32 == 0

    def test_small_target_area(self) -> None:
        """Works with small target area."""
        buckets = generate_buckets(
            target_area=256 * 256,
            min_dim=128,
            max_dim=512,
            step=16,
        )
        assert len(buckets) > 0

    def test_no_duplicates(self) -> None:
        """No duplicate buckets."""
        buckets = generate_buckets()
        assert len(buckets) == len(set(buckets))

    def test_error_on_zero_step(self) -> None:
        with pytest.raises(ValueError, match="step must be positive"):
            generate_buckets(step=0)

    def test_error_on_negative_step(self) -> None:
        with pytest.raises(ValueError, match="step must be positive"):
            generate_buckets(step=-1)

    def test_error_on_min_gt_max(self) -> None:
        with pytest.raises(ValueError, match="min_dim.*max_dim"):
            generate_buckets(min_dim=1024, max_dim=256)

    def test_error_on_bad_aspect_range(self) -> None:
        with pytest.raises(ValueError, match="min_aspect.*max_aspect"):
            generate_buckets(min_aspect=3.0, max_aspect=0.5)


# ---------------------------------------------------------------------------
# _closest_bucket
# ---------------------------------------------------------------------------

class TestClosestBucket:
    """Tests for bucket matching."""

    def test_exact_match(self) -> None:
        """Returns exact match when available."""
        buckets = [(320, 240), (512, 512), (848, 480)]
        assert _closest_bucket(512, 512, buckets) == (512, 512)

    def test_closest_by_area(self) -> None:
        """Matches closest by area when no exact match."""
        buckets = [(256, 256), (512, 512), (1024, 1024)]
        # 400x400 = 160000, closest to 256*256=65536? No, 512*512=262144
        # Actually closest: abs(65536-160000)=94464, abs(262144-160000)=102144
        result = _closest_bucket(400, 400, buckets)
        assert result == (256, 256)  # Closest by area

    def test_aspect_ratio_tiebreak(self) -> None:
        """When areas tie, prefer closer aspect ratio."""
        # Two buckets with same area: 640x400 and 400x640
        buckets = [(640, 400), (400, 640)]
        # Source is landscape 800x500 → aspect 1.6
        result = _closest_bucket(800, 500, buckets)
        assert result == (640, 400)  # Landscape matches landscape

    def test_single_bucket(self) -> None:
        """Works with a single bucket."""
        buckets = [(512, 512)]
        assert _closest_bucket(1920, 1080, buckets) == (512, 512)


# ---------------------------------------------------------------------------
# assign_buckets
# ---------------------------------------------------------------------------

class TestAssignBuckets:
    """Tests for bucket assignment on ExpandedSamples."""

    def test_empty_list(self) -> None:
        assert assign_buckets([]) == []

    def test_already_assigned(self) -> None:
        """Samples with bucket dims set are kept as-is."""
        sample = _make_expanded(bucket_width=848, bucket_height=480)
        result = assign_buckets([sample])
        assert result[0].bucket_width == 848
        assert result[0].bucket_height == 480

    def test_snap_to_grid_fallback(self) -> None:
        """Without bucket list, snaps to grid."""
        sample = _make_expanded(bucket_width=0, bucket_height=0)
        result = assign_buckets([sample], step=16)
        # Default fallback is 512x512 snapped to 16
        assert result[0].bucket_width % 16 == 0
        assert result[0].bucket_height % 16 == 0

    def test_assign_from_bucket_list(self) -> None:
        """Assigns to closest bucket from provided list."""
        buckets = [(320, 240), (640, 480), (848, 480)]
        sample = _make_expanded(bucket_width=0, bucket_height=0)
        result = assign_buckets([sample], buckets=buckets)
        # Default 512x512, closest to (640, 480) by area
        assert result[0].bucket_width in [b[0] for b in buckets]
        assert result[0].bucket_height in [b[1] for b in buckets]


# ---------------------------------------------------------------------------
# bucket_groups
# ---------------------------------------------------------------------------

class TestBucketGroups:
    """Tests for grouping samples by bucket key."""

    def test_empty(self) -> None:
        assert bucket_groups([]) == {}

    def test_single_bucket(self) -> None:
        samples = [
            _make_expanded("a", 848, 480, 81),
            _make_expanded("b", 848, 480, 81),
        ]
        groups = bucket_groups(samples)
        assert len(groups) == 1
        assert "848x480x81" in groups
        assert len(groups["848x480x81"]) == 2

    def test_multiple_buckets(self) -> None:
        samples = [
            _make_expanded("a", 848, 480, 81),
            _make_expanded("b", 320, 240, 17),
            _make_expanded("c", 848, 480, 81),
        ]
        groups = bucket_groups(samples)
        assert len(groups) == 2
        assert len(groups["848x480x81"]) == 2
        assert len(groups["320x240x17"]) == 1
