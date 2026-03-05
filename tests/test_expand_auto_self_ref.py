"""Tests for auto_self_reference in expand.py.

Tests cover:
    - _expand_image_sample with auto_self_reference=True/False
    - expand_samples passes auto_self_reference through
"""

from __future__ import annotations

from pathlib import Path

import pytest

from flimmer.encoding.expand import (
    _expand_image_sample,
    expand_samples,
)
from flimmer.encoding.models import (
    DiscoveredSample,
    SampleRole,
)


def _make_image_sample(
    stem: str = "photo_001",
    width: int = 1024,
    height: int = 768,
    reference: Path | None = None,
) -> DiscoveredSample:
    """Helper to create an image DiscoveredSample."""
    return DiscoveredSample(
        stem=stem,
        target=Path(f"/data/{stem}.png"),
        target_role=SampleRole.TARGET_IMAGE,
        width=width,
        height=height,
        frame_count=1,
        reference=reference,
    )


def _make_video_sample(
    stem: str = "clip_001",
    width: int = 848,
    height: int = 480,
    frame_count: int = 81,
    reference: Path | None = None,
) -> DiscoveredSample:
    """Helper to create a video DiscoveredSample."""
    return DiscoveredSample(
        stem=stem,
        target=Path(f"/data/{stem}.mp4"),
        target_role=SampleRole.TARGET_VIDEO,
        width=width,
        height=height,
        frame_count=frame_count,
        fps=16.0,
        duration=frame_count / 16.0,
        reference=reference,
    )


# ---------------------------------------------------------------------------
# _expand_image_sample auto_self_reference
# ---------------------------------------------------------------------------

class TestExpandImageSampleAutoSelfRef:
    """Tests for auto_self_reference on _expand_image_sample."""

    def test_auto_self_ref_true_no_existing_reference(self) -> None:
        """When auto_self_reference=True and reference is None, sets reference to target."""
        sample = _make_image_sample(reference=None)
        expanded = _expand_image_sample(sample, auto_self_reference=True)

        assert len(expanded) == 1
        assert expanded[0].reference == sample.target

    def test_auto_self_ref_true_keeps_existing_reference(self) -> None:
        """When auto_self_reference=True and reference already set, keeps existing."""
        existing_ref = Path("/data/custom_ref.png")
        sample = _make_image_sample(reference=existing_ref)
        expanded = _expand_image_sample(sample, auto_self_reference=True)

        assert expanded[0].reference == existing_ref

    def test_auto_self_ref_false_no_reference(self) -> None:
        """When auto_self_reference=False and reference is None, keeps None."""
        sample = _make_image_sample(reference=None)
        expanded = _expand_image_sample(sample, auto_self_reference=False)

        assert expanded[0].reference is None

    def test_auto_self_ref_default_false(self) -> None:
        """Default auto_self_reference is False — existing behavior preserved."""
        sample = _make_image_sample(reference=None)
        expanded = _expand_image_sample(sample)

        assert expanded[0].reference is None


# ---------------------------------------------------------------------------
# expand_samples passes auto_self_reference
# ---------------------------------------------------------------------------

class TestExpandSamplesAutoSelfRef:
    """Tests for auto_self_reference threading in expand_samples."""

    def test_passes_auto_self_ref_to_image_samples(self) -> None:
        """expand_samples with auto_self_reference=True sets reference on image samples."""
        image = _make_image_sample(reference=None)
        expanded = expand_samples(
            [image],
            target_frames=[17],
            auto_self_reference=True,
        )

        image_samples = [e for e in expanded if e.is_image]
        assert len(image_samples) == 1
        assert image_samples[0].reference == image.target

    def test_auto_self_ref_false_preserves_none(self) -> None:
        """expand_samples with auto_self_reference=False keeps None references."""
        image = _make_image_sample(reference=None)
        expanded = expand_samples(
            [image],
            target_frames=[17],
            auto_self_reference=False,
        )

        image_samples = [e for e in expanded if e.is_image]
        assert len(image_samples) == 1
        assert image_samples[0].reference is None

    def test_mixed_video_image_only_affects_images(self) -> None:
        """auto_self_reference only affects image samples, not video samples."""
        video = _make_video_sample(reference=None)
        image = _make_image_sample(reference=None)

        expanded = expand_samples(
            [video, image],
            target_frames=[17],
            auto_self_reference=True,
        )

        video_samples = [e for e in expanded if e.source_stem == "clip_001"]
        image_samples = [e for e in expanded if e.source_stem == "photo_001"]

        # Video samples should NOT get self-reference
        for vs in video_samples:
            assert vs.reference is None

        # Image sample should get self-reference
        assert image_samples[0].reference == image.target
