"""Multi-sample expansion for training.

Takes DiscoveredSamples and expands each into one or more ExpandedSamples
at different frame counts and resolutions. This is how Flimmer gets more
training value from each video:

    One 81-frame video clip → samples at 81, 49, 33, 17 frames
                             → optionally a 1-frame head image too

Key rules:
    - Frame counts must be 4n+1 (Wan VAE temporal compression requirement)
    - A video can only produce samples at frame counts <= its total frames
    - Resolution is snapped to a step grid (default 16px for VAE alignment)
    - Image targets always expand to exactly one 1-frame sample

The expansion step runs BEFORE encoding — it decides what the encoder
will need to produce. Each ExpandedSample carries its bucket dimensions
so the encoder knows exactly what size to output.
"""

from __future__ import annotations

from flimmer.encoding.errors import ExpansionError
from flimmer.encoding.models import (
    DiscoveredSample,
    ExpandedSample,
    FrameExtraction,
    SampleRole,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default target frame counts for Wan models (must be 4n+1)
DEFAULT_TARGET_FRAMES: list[int] = [17, 33, 49, 81]
"""Standard frame counts for Wan training samples.
81 = ~5s at 16fps, 49 = ~3s, 33 = ~2s, 17 = ~1s.
All satisfy the 4n+1 constraint required by Wan's 3D causal VAE.
Single-frame image samples are opt-in via include_head_frame."""


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_frame_count(n: int) -> bool:
    """Check if a frame count satisfies the 4n+1 constraint.

    Wan's 3D causal VAE compresses temporally with a factor of ~4.
    Only frame counts of the form 4n+1 (1, 5, 9, 13, 17, ...) produce
    clean latent dimensions without padding artifacts.

    Args:
        n: Frame count to validate.

    Returns:
        True if n is a valid 4n+1 frame count.
    """
    return n >= 1 and (n - 1) % 4 == 0


def validate_target_frames(target_frames: list[int]) -> None:
    """Validate a list of target frame counts.

    All values must be positive and satisfy 4n+1. The list must not
    be empty.

    Args:
        target_frames: List of frame counts to validate.

    Raises:
        ExpansionError: If any value is invalid.
    """
    if not target_frames:
        raise ExpansionError(
            "target_frames cannot be empty. "
            "Provide at least one frame count (e.g. [17, 33, 49, 81])."
        )

    for fc in target_frames:
        if fc < 1:
            raise ExpansionError(
                f"target_frames contains invalid value {fc}. "
                f"All frame counts must be >= 1."
            )
        if not validate_frame_count(fc):
            # Find the nearest valid values for a helpful message
            lower = ((fc - 1) // 4) * 4 + 1
            upper = lower + 4
            raise ExpansionError(
                f"Frame count {fc} does not satisfy the 4n+1 constraint "
                f"(required by Wan's 3D causal VAE). "
                f"Nearest valid values: {lower} or {upper}."
            )


# ---------------------------------------------------------------------------
# Resolution snapping
# ---------------------------------------------------------------------------

def snap_resolution(
    width: int,
    height: int,
    step: int = 16,
) -> tuple[int, int]:
    """Snap width and height down to the nearest multiple of step.

    VAE encoders require dimensions aligned to their compression factor.
    For Wan VAE, this is 16 pixels in each spatial dimension.

    Args:
        width: Source width in pixels.
        height: Source height in pixels.
        step: Alignment step (default 16).

    Returns:
        (snapped_width, snapped_height), both multiples of step.
        Minimum value is step itself (never returns 0).
    """
    snapped_w = max(step, (width // step) * step)
    snapped_h = max(step, (height // step) * step)
    return snapped_w, snapped_h


# ---------------------------------------------------------------------------
# Single-sample expansion
# ---------------------------------------------------------------------------

def _expand_image_sample(
    sample: DiscoveredSample,
    step: int = 16,
    auto_self_reference: bool = False,
) -> list[ExpandedSample]:
    """Expand an image target into a single 1-frame sample.

    When auto_self_reference is True and the sample has no explicit
    reference image, use the target image as its own first-frame
    conditioning. This enables I2V training on still images without
    requiring users to manually set up reference images.
    """
    w, h = snap_resolution(
        sample.width if sample.width > 0 else 512,
        sample.height if sample.height > 0 else 512,
        step,
    )

    # Auto self-reference: image IS its own first frame for I2V
    reference = sample.reference
    if auto_self_reference and reference is None:
        reference = sample.target

    sample_id = f"{sample.stem}_1x{h}x{w}"

    return [ExpandedSample(
        sample_id=sample_id,
        source_stem=sample.stem,
        target=sample.target,
        target_role=sample.target_role,
        caption=sample.caption,
        reference=reference,
        bucket_width=w,
        bucket_height=h,
        bucket_frames=1,
        frame_extraction=FrameExtraction.HEAD,
        frame_offset=0,
        repeats=sample.repeats,
        loss_multiplier=sample.loss_multiplier,
    )]


def _expand_video_sample(
    sample: DiscoveredSample,
    target_frames: list[int],
    frame_extraction: FrameExtraction = FrameExtraction.HEAD,
    include_head_frame: bool = False,
    step: int = 16,
) -> list[ExpandedSample]:
    """Expand a video target into multiple samples at different frame counts.

    Only produces samples for frame counts that fit within the source video's
    total frames. If no frame counts fit, returns an empty list.
    """
    w, h = snap_resolution(
        sample.width if sample.width > 0 else 512,
        sample.height if sample.height > 0 else 512,
        step,
    )

    source_frames = sample.frame_count
    expanded: list[ExpandedSample] = []

    # Sort frame counts descending so longer samples come first
    valid_fcs = sorted(
        [fc for fc in target_frames if fc > 1 and fc <= source_frames],
        reverse=True,
    )

    for fc in valid_fcs:
        sample_id = f"{sample.stem}_{fc}x{h}x{w}"
        expanded.append(ExpandedSample(
            sample_id=sample_id,
            source_stem=sample.stem,
            target=sample.target,
            target_role=SampleRole.TARGET_VIDEO,
            caption=sample.caption,
            reference=sample.reference,
            bucket_width=w,
            bucket_height=h,
            bucket_frames=fc,
            frame_extraction=frame_extraction,
            frame_offset=0,
            repeats=sample.repeats,
            loss_multiplier=sample.loss_multiplier,
        ))

    # Optionally add a 1-frame head image sample
    if include_head_frame and source_frames >= 1:
        head_id = f"{sample.stem}_1x{h}x{w}"
        expanded.append(ExpandedSample(
            sample_id=head_id,
            source_stem=sample.stem,
            target=sample.target,
            target_role=SampleRole.TARGET_IMAGE,
            caption=sample.caption,
            reference=sample.reference,
            bucket_width=w,
            bucket_height=h,
            bucket_frames=1,
            frame_extraction=FrameExtraction.HEAD,
            frame_offset=0,
            repeats=sample.repeats,
            loss_multiplier=sample.loss_multiplier,
        ))

    return expanded


# ---------------------------------------------------------------------------
# Batch expansion
# ---------------------------------------------------------------------------

def expand_samples(
    samples: list[DiscoveredSample],
    target_frames: list[int] | None = None,
    frame_extraction: str = "head",
    include_head_frame: bool = False,
    step: int = 16,
    image_repeat: int = 1,
    auto_self_reference: bool = False,
) -> list[ExpandedSample]:
    """Expand a list of DiscoveredSamples into ExpandedSamples.

    Each video sample can produce multiple training samples at different
    frame counts. Image samples always produce exactly one 1-frame sample.

    Args:
        samples: Discovered samples to expand.
        target_frames: Valid frame counts to produce. Default: [1, 17, 33, 49, 81].
            Must all satisfy the 4n+1 constraint.
        frame_extraction: How to extract frames: 'head' or 'uniform'.
        include_head_frame: Also extract a 1-frame image sample from each video.
        step: Resolution alignment step in pixels.
        image_repeat: Repeat multiplier for image samples. Images train faster
            than multi-frame videos, so this lets users increase their presence
            in training batches (e.g., image_repeat=5 means each image appears
            5x per epoch). Multiplies with dataset-level repeats. Default 1.
        auto_self_reference: When True, image samples with no explicit reference
            use themselves as first-frame conditioning. Enables zero-config I2V
            training on still images. Default False.

    Returns:
        List of ExpandedSamples, potentially many per input sample.

    Raises:
        ExpansionError: If target_frames contains invalid values.
    """
    if target_frames is None:
        target_frames = DEFAULT_TARGET_FRAMES.copy()

    validate_target_frames(target_frames)

    extraction = FrameExtraction(frame_extraction)

    expanded: list[ExpandedSample] = []

    for sample in samples:
        if sample.target_role == SampleRole.TARGET_IMAGE:
            img_expanded = _expand_image_sample(
                sample,
                step=step,
                auto_self_reference=auto_self_reference,
            )
            if image_repeat > 1:
                img_expanded = [
                    s.model_copy(update={"repeats": s.repeats * image_repeat})
                    for s in img_expanded
                ]
            expanded.extend(img_expanded)
        else:
            expanded.extend(_expand_video_sample(
                sample,
                target_frames=target_frames,
                frame_extraction=extraction,
                include_head_frame=include_head_frame,
                step=step,
            ))

    return expanded
