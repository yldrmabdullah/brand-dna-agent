"""Resolution / size / format quality filter.

This is the cheapest filter — runs without any model loaded. We apply it
*first* to avoid burning CLIP forward passes on rejects.

The 512px floor is justified in detail in BrandConfig.filter.min_shorter_side.
TL;DR: it's the floor for SDXL-class generative pipelines downstream — the
case is about feeding the Brand DNA into a model that *generates* designs.
"""

from __future__ import annotations

from brand_dna.core.models import ImageRecord
from brand_dna.core.observability import get_logger

logger = get_logger(__name__)


def filter_by_quality(
    images: list[ImageRecord],
    *,
    min_shorter_side: int = 512,
    min_bytes: int = 10_000,
    max_bytes: int = 15_000_000,
    allowed_formats: tuple[str, ...] = ("jpeg", "png", "webp", "avif"),
) -> tuple[list[ImageRecord], list[ImageRecord]]:
    """Returns (kept, rejected). Rejection reason is set on each rejected
    record so the run log shows *why* we dropped images, not just that we did."""
    kept: list[ImageRecord] = []
    rejected: list[ImageRecord] = []

    for img in images:
        reasons: list[str] = []
        if img.shorter_side < min_shorter_side:
            reasons.append(
                f"shorter_side={img.shorter_side}<{min_shorter_side}"
            )
        if img.bytes_size < min_bytes:
            reasons.append(f"bytes_size={img.bytes_size}<{min_bytes}")
        if img.bytes_size > max_bytes:
            reasons.append(f"bytes_size={img.bytes_size}>{max_bytes}")
        if img.format not in allowed_formats:
            reasons.append(f"format={img.format}∉{allowed_formats}")

        # Sanity: aspect ratio. Anything more extreme than 4:1 is almost
        # always banner / header / decorative — not lookbook content.
        if img.width and img.height:
            ratio = max(img.width, img.height) / max(1, min(img.width, img.height))
            if ratio > 4.0:
                reasons.append(f"aspect_ratio={ratio:.1f}>4.0")

        if reasons:
            img.quality_passed = False
            img.rejection_reason = "; ".join(reasons)
            rejected.append(img)
        else:
            img.quality_passed = True
            kept.append(img)

    logger.info(
        "quality.filter",
        input=len(images),
        kept=len(kept),
        rejected=len(rejected),
    )
    return kept, rejected
