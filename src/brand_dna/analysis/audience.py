"""Audience signal extraction. Multi-modal: text corpus + representative images.

We hand the LLM:
- The same text corpus the voice analyser used (cheap to reuse — already
  in context if called sequentially, and we pay for it once if not).
- 6-8 representative images, sampled across aesthetic clusters so the model
  isn't biased by one cluster's dominance.

The model returns demographic + psychographic cues, an estimated price
positioning band, and 3-5 evidence snippets it relied on.
"""

from __future__ import annotations

import random
from pathlib import Path

from brand_dna.core.models import AestheticCluster, AudienceProfile, ImageRecord, Page
from brand_dna.core.observability import get_logger
from brand_dna.llm.client import LLMClient
from brand_dna.llm.prompts import render

logger = get_logger(__name__)


def _sample_images(
    images: list[ImageRecord],
    clusters: list[AestheticCluster],
    n: int = 6,
) -> list[str]:
    """Pick image paths across clusters for visual diversity."""
    by_id = {img.image_id: img for img in images}
    sampled_paths: list[str] = []
    seen: set[str] = set()

    # 1 representative per cluster, up to n
    for c in clusters:
        for img_id in c.representative_image_ids:
            if img_id in by_id and img_id not in seen:
                sampled_paths.append(by_id[img_id].local_path)
                seen.add(img_id)
                break
        if len(sampled_paths) >= n:
            break

    # Top up with random unseen images if needed
    remaining = [img for img in images if img.image_id not in seen]
    random.Random(42).shuffle(remaining)
    for img in remaining:
        if len(sampled_paths) >= n:
            break
        sampled_paths.append(img.local_path)
    return sampled_paths[:n]


def _build_text_snippet(pages: list[Page], max_chars: int = 12_000) -> str:
    """Compact corpus for the audience prompt — about/blog only."""
    relevant = [
        p
        for p in pages
        if p.page_type.value in {"about", "homepage", "blog", "editorial", "press"}
    ]
    parts: list[str] = []
    total = 0
    for p in relevant:
        text = (p.body_text or p.meta_description or "").strip()
        if not text:
            continue
        snippet = text[:3000]
        block = f"\n--- {p.page_type.value.upper()}\n{snippet}\n"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    return "".join(parts)


async def extract_audience_profile(
    pages: list[Page],
    images: list[ImageRecord],
    clusters: list[AestheticCluster],
    *,
    llm: LLMClient,
    model: str,
    brand_name: str,
    n_image_samples: int = 6,
) -> tuple[AudienceProfile, dict]:
    """Returns (profile, telemetry). Telemetry includes 'n_images_sampled'
    and 'corpus_chars' for confidence scoring upstream."""
    text = _build_text_snippet(pages)
    image_paths = _sample_images(images, clusters, n=n_image_samples)

    if not text and not image_paths:
        logger.warning("audience.no_signals", brand=brand_name)
        return AudienceProfile(), {"corpus_chars": 0, "n_images_sampled": 0}

    prompt = render("audience", brand_name=brand_name, corpus=text or "(no text)")
    data, _ = await llm.chat_json(
        prompt,
        model=model,
        system=(
            "You are a senior fashion brand strategist. Read both the brand's "
            "own copy and the supplied images to infer who they speak to. "
            "Be specific but only state what's supported."
        ),
        images=[Path(p) for p in image_paths] if image_paths else None,
        temperature=0.2,
        max_tokens=1500,
    )

    profile = AudienceProfile(
        demographic_cues=_as_list(data.get("demographic_cues")),
        psychographic_cues=_as_list(data.get("psychographic_cues")),
        aspirational_signals=_as_list(data.get("aspirational_signals")),
        price_positioning=data.get("price_positioning") or None,
        evidence_snippets=_as_list(data.get("evidence_snippets")),
    )
    telemetry = {
        "corpus_chars": len(text),
        "n_images_sampled": len(image_paths),
    }
    logger.info(
        "audience.complete",
        brand=brand_name,
        demographic_n=len(profile.demographic_cues),
        psychographic_n=len(profile.psychographic_cues),
        **telemetry,
    )
    return profile, telemetry


def _as_list(v: object) -> list[str]:
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    if isinstance(v, str) and v.strip():
        return [v.strip()]
    return []
