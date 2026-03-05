"""Tests for flimmer.encoding.expand — multi-sample expansion.

Tests cover:
    - validate_frame_count(): 4n+1 constraint
    - validate_target_frames(): list validation
    - snap_resolution(): pixel alignment
    - _expand_image_sample(): single-frame expansion
    - _expand_video_sample(): multi-frame expansion
    - expand_samples(): batch expansion
"""

from __future__ import annotations

from pathlib import Path

import pytest

from flimmer.encoding.errors import ExpansionError
from flimmer.encoding.expand import (
    DEFAULT_TARGET_FRAMES,
    _expand_image_sample,
    _expand_video_sample,
    expand_samples,
    snap_resolution,
    validate_frame_count,
    validate_target_frames,
)
from flimmer.encoding.models import (
    DiscoveredSample,
    ExpandedSample,
    FrameExtraction,
    SampleRole,
)


def _make_video_sample(
    stem: str = "clip_001",
    width: int = 848,
    height: int = 480,
    frame_count: int = 81,
    fps: float = 16.0,
    caption: Path | None = Path("/data/clip_001.txt"),
    reference: Path | None = None,
) -> DiscoveredSample:
    """Helper to create a video DiscoveredSample."""
    return DiscoveredSample(
        stem=stem,
        target=Path(f"/data/{stem}.mp4"),
        target_role=SampleRole.TARGET_VIDEO,
        caption=caption,
        reference=reference,
        width=width,
        height=height,
        frame_count=frame_count,
        fps=fps,
        duration=frame_count / fps if fps > 0 else 0,
    )


def _make_image_sample(
    stem: str = "photo_001",
    width: int = 1024,
    height: int = 768,
) -> DiscoveredSample:
    """Helper to create an image DiscoveredSample."""
    return DiscoveredSample(
        stem=stem,
        target=Path(f"/data/{stem}.png"),
        target_role=SampleRole.TARGET_IMAGE,
        width=width,
        height=height,
        frame_count=1,
    )


# ---------------------------------------------------------------------------
# validate_frame_count
# ---------------------------------------------------------------------------

class TestValidateFrameCount:
    """Tests for the 4n+1 constraint."""

    @pytest.mark.parametrize("n", [1, 5, 9, 13, 17, 33, 49, 81])
    def test_valid_counts(self, n: int) -> None:
        assert validate_frame_count(n) is True

    @pytest.mark.parametrize("n", [0, 2, 3, 4, 6, 7, 8, 10, 16, 18, 50, 80])
    def test_invalid_counts(self, n: int) -> None:
        assert validate_frame_count(n) is False

    def test_negative(self) -> None:
        assert validate_frame_count(-1) is False


# ---------------------------------------------------------------------------
# validate_target_frames
# ---------------------------------------------------------------------------

class TestValidateTargetFrames:
    """Tests for target frame list validation."""

    def test_default_frames_valid(self) -> None:
        """DEFAULT_TARGET_FRAMES passes validation."""
        validate_target_frames(DEFAULT_TARGET_FRAMES)  # No exception

    def test_empty_list_raises(self) -> None:
        with pytest.raises(ExpansionError, match="cannot be empty"):
            validate_target_frames([])

    def test_invalid_frame_count(self) -> None:
        with pytest.raises(ExpansionError, match="4n\\+1"):
            validate_target_frames([16])  # 16 is not 4n+1

    def test_negative_frame_count(self) -> None:
        with pytest.raises(ExpansionError, match="invalid value"):
            validate_target_frames([-1])

    def test_zero_frame_count(self) -> None:
        with pytest.raises(ExpansionError, match="invalid value"):
            validate_target_frames([0])

    def test_helpful_error_message(self) -> None:
        """Error suggests nearest valid values."""
        with pytest.raises(ExpansionError, match="13 or 17"):
            validate_target_frames([15])


# ---------------------------------------------------------------------------
# snap_resolution
# ---------------------------------------------------------------------------

class TestSnapResolution:
    """Tests for resolution snapping to grid."""

    def test_already_aligned(self) -> None:
        assert snap_resolution(848, 480) == (848, 480)

    def test_snap_down(self) -> None:
        assert snap_resolution(850, 485) == (848, 480)

    def test_minimum_value(self) -> None:
        """Never returns 0 — minimum is step."""
        assert snap_resolution(10, 10, step=16) == (16, 16)

    def test_custom_step(self) -> None:
        assert snap_resolution(100, 100, step=32) == (96, 96)

    def test_large_values(self) -> None:
        w, h = snap_resolution(1920, 1080)
        assert w == 1920
        assert h == 1072  # 1080 // 16 * 16


# ---------------------------------------------------------------------------
# _expand_image_sample
# ---------------------------------------------------------------------------

class TestExpandImageSample:
    """Tests for single-image expansion."""

    def test_single_sample(self) -> None:
        """Image produces exactly one sample."""
        sample = _make_image_sample(width=1024, height=768)
        expanded = _expand_image_sample(sample)
        assert len(expanded) == 1

    def test_frame_count_is_1(self) -> None:
        sample = _make_image_sample()
        expanded = _expand_image_sample(sample)[0]
        assert expanded.bucket_frames == 1
        assert expanded.is_image

    def test_sample_id_format(self) -> None:
        sample = _make_image_sample(stem="photo", width=1024, height=768)
        expanded = _expand_image_sample(sample)[0]
        assert expanded.sample_id == "photo_1x768x1024"

    def test_resolution_snapped(self) -> None:
        sample = _make_image_sample(width=1000, height=750)
        expanded = _expand_image_sample(sample, step=16)[0]
        assert expanded.bucket_width % 16 == 0
        assert expanded.bucket_height % 16 == 0

    def test_preserves_caption(self) -> None:
        sample = DiscoveredSample(
            stem="test",
            target=Path("/test.png"),
            target_role=SampleRole.TARGET_IMAGE,
            caption=Path("/test.txt"),
            width=512,
            height=512,
            frame_count=1,
        )
        expanded = _expand_image_sample(sample)[0]
        assert expanded.caption == Path("/test.txt")


# ---------------------------------------------------------------------------
# _expand_video_sample
# ---------------------------------------------------------------------------

class TestExpandVideoSample:
    """Tests for multi-frame video expansion."""

    def test_81_frame_video(self) -> None:
        """81-frame video produces samples at 81, 49, 33, 17."""
        sample = _make_video_sample(frame_count=81)
        expanded = _expand_video_sample(sample, target_frames=[17, 33, 49, 81])
        assert len(expanded) == 4

    def test_short_video_fewer_samples(self) -> None:
        """20-frame video can only produce 17-frame sample."""
        sample = _make_video_sample(frame_count=20)
        expanded = _expand_video_sample(sample, target_frames=[17, 33, 49, 81])
        assert len(expanded) == 1
        assert expanded[0].bucket_frames == 17

    def test_too_short_video_empty(self) -> None:
        """10-frame video produces nothing at [17, 33, 49, 81]."""
        sample = _make_video_sample(frame_count=10)
        expanded = _expand_video_sample(sample, target_frames=[17, 33, 49, 81])
        assert len(expanded) == 0

    def test_frame_1_excluded_from_video(self) -> None:
        """Frame count 1 in target_frames is excluded from video expansion."""
        sample = _make_video_sample(frame_count=81)
        expanded = _expand_video_sample(sample, target_frames=[1, 17])
        # Only 17-frame sample, not 1-frame (that's include_head_frame's job)
        assert len(expanded) == 1
        assert expanded[0].bucket_frames == 17

    def test_include_head_frame(self) -> None:
        """include_head_frame adds a 1-frame sample."""
        sample = _make_video_sample(frame_count=81)
        expanded = _expand_video_sample(
            sample,
            target_frames=[17, 33],
            include_head_frame=True,
        )
        frame_counts = [e.bucket_frames for e in expanded]
        assert 1 in frame_counts
        assert 17 in frame_counts
        assert 33 in frame_counts

    def test_head_frame_role_is_target_image(self) -> None:
        """Head frame extraction has TARGET_IMAGE role."""
        sample = _make_video_sample(frame_count=81)
        expanded = _expand_video_sample(
            sample,
            target_frames=[17],
            include_head_frame=True,
        )
        head = [e for e in expanded if e.bucket_frames == 1][0]
        assert head.target_role == SampleRole.TARGET_IMAGE

    def test_sample_id_format(self) -> None:
        sample = _make_video_sample(stem="clip_001", width=848, height=480)
        expanded = _expand_video_sample(sample, target_frames=[81])
        assert expanded[0].sample_id == "clip_001_81x480x848"

    def test_preserves_repeats(self) -> None:
        sample = DiscoveredSample(
            stem="test",
            target=Path("/test.mp4"),
            target_role=SampleRole.TARGET_VIDEO,
            width=848,
            height=480,
            frame_count=81,
            fps=16.0,
            repeats=3,
        )
        expanded = _expand_video_sample(sample, target_frames=[17])
        assert expanded[0].repeats == 3

    def test_sorted_descending_frames(self) -> None:
        """Longer samples come first in output."""
        sample = _make_video_sample(frame_count=81)
        expanded = _expand_video_sample(sample, target_frames=[17, 81, 33, 49])
        frames = [e.bucket_frames for e in expanded]
        assert frames == sorted(frames, reverse=True)

    def test_uniform_extraction(self) -> None:
        sample = _make_video_sample(frame_count=81)
        expanded = _expand_video_sample(
            sample,
            target_frames=[17],
            frame_extraction=FrameExtraction.UNIFORM,
        )
        assert expanded[0].frame_extraction == FrameExtraction.UNIFORM


# ---------------------------------------------------------------------------
# expand_samples (batch)
# ---------------------------------------------------------------------------

class TestExpandSamples:
    """Tests for batch expansion."""

    def test_empty_input(self) -> None:
        assert expand_samples([]) == []

    def test_mixed_video_and_image(self) -> None:
        """Videos and images are expanded correctly together."""
        video = _make_video_sample(frame_count=81)
        image = _make_image_sample()

        expanded = expand_samples(
            [video, image],
            target_frames=[17, 81],
        )

        video_samples = [e for e in expanded if e.source_stem == "clip_001"]
        image_samples = [e for e in expanded if e.source_stem == "photo_001"]

        assert len(video_samples) == 2  # 17 and 81 frames
        assert len(image_samples) == 1  # single frame

    def test_default_target_frames(self) -> None:
        """Default target_frames are used when not specified."""
        video = _make_video_sample(frame_count=100)
        expanded = expand_samples([video])
        # Should use DEFAULT_TARGET_FRAMES: [17, 33, 49, 81]
        frame_counts = {e.bucket_frames for e in expanded}
        assert 17 in frame_counts
        assert 81 in frame_counts

    def test_invalid_target_frames_raises(self) -> None:
        """Invalid frame counts raise ExpansionError."""
        video = _make_video_sample()
        with pytest.raises(ExpansionError, match="4n\\+1"):
            expand_samples([video], target_frames=[16])

    def test_include_head_frame(self) -> None:
        video = _make_video_sample(frame_count=81)
        expanded = expand_samples(
            [video],
            target_frames=[17],
            include_head_frame=True,
        )
        frame_counts = {e.bucket_frames for e in expanded}
        assert 1 in frame_counts
        assert 17 in frame_counts


# ---------------------------------------------------------------------------
# image_repeat in expand_samples
# ---------------------------------------------------------------------------

class TestImageRepeatExpansion:
    """Tests for image_repeat parameter in expand_samples."""

    def test_image_repeat_multiplies_image_repeats(self) -> None:
        """image_repeat=3 triples repeats for TARGET_IMAGE samples."""
        image = _make_image_sample()
        expanded = expand_samples([image], image_repeat=3)
        assert len(expanded) == 1
        assert expanded[0].repeats == 3

    def test_image_repeat_does_not_affect_video(self) -> None:
        """image_repeat does NOT change repeats for TARGET_VIDEO samples."""
        video = _make_video_sample(frame_count=81)
        expanded = expand_samples(
            [video],
            target_frames=[17],
            image_repeat=5,
        )
        for sample in expanded:
            assert sample.repeats == 1  # unchanged

    def test_image_repeat_default_leaves_unchanged(self) -> None:
        """image_repeat=1 (default) leaves image repeats at 1."""
        image = _make_image_sample()
        expanded = expand_samples([image], image_repeat=1)
        assert expanded[0].repeats == 1

    def test_image_repeat_multiplies_with_existing_repeats(self) -> None:
        """image_repeat stacks with dataset-level repeats.

        sample.repeats=2, image_repeat=3 -> expanded.repeats=6
        """
        image = DiscoveredSample(
            stem="photo_rep",
            target=Path("/data/photo_rep.png"),
            target_role=SampleRole.TARGET_IMAGE,
            width=1024,
            height=768,
            frame_count=1,
            repeats=2,
        )
        expanded = expand_samples([image], image_repeat=3)
        assert expanded[0].repeats == 6

    def test_image_repeat_mixed_dataset(self) -> None:
        """In a mixed dataset, only image samples get multiplied."""
        video = _make_video_sample(frame_count=81)
        image = _make_image_sample()

        expanded = expand_samples(
            [video, image],
            target_frames=[17],
            image_repeat=4,
        )

        video_samples = [e for e in expanded if e.source_stem == "clip_001"]
        image_samples = [e for e in expanded if e.source_stem == "photo_001"]

        assert all(s.repeats == 1 for s in video_samples)
        assert all(s.repeats == 4 for s in image_samples)
