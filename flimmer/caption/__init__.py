"""Flimmer VLM captioning pipeline.

Standalone caption generation, auditing, and scoring tools for video
training datasets. Supports Gemini (direct API) and Replicate backends.

Quick start:
    from flimmer.caption import caption_clips
    from flimmer.caption.models import CaptionConfig

    config = CaptionConfig(provider="gemini")
    results = caption_clips("./clips", config)

Scoring (no API needed):
    from flimmer.caption import score_caption, score_directory

    score = score_caption("A person walks through a sunlit forest...")
    print(score.overall, score.issues)
"""

from flimmer.caption.captioner import audit_captions, caption_clips
from flimmer.caption.models import AuditResult, CaptionConfig, CaptionResult
from flimmer.caption.scoring import (
    CaptionScore,
    ScoringConfig,
    format_score_report,
    score_caption,
    score_directory,
)

__all__ = [
    "caption_clips",
    "audit_captions",
    "CaptionConfig",
    "CaptionResult",
    "AuditResult",
    "score_caption",
    "score_directory",
    "format_score_report",
    "CaptionScore",
    "ScoringConfig",
]
