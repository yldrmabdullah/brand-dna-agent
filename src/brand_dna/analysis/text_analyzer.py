"""Brand voice extraction from collected page text.

Strategy:
- Aggregate text from high-signal page types only: ABOUT, BLOG, EDITORIAL,
  HOMEPAGE, PRESS. Product pages have transactional copy that pollutes the
  voice signal — they're useful for vocabulary but not for tone.
- Cap total tokens so we stay within model context (and cost).
- Single LLM call with structured JSON output.

Why LLM vs traditional NLP (tf-idf, etc.): "tone" is a soft, gestalt concept.
A statistician's "most frequent adjectives" misses that "Worn by those who
know their own mind." is positioning even though no adjective in it is
unusual. LLMs are the right tool for tonal extraction.
"""

from __future__ import annotations

from brand_dna.core.models import BrandVoice, Page, PageType
from brand_dna.core.observability import get_logger
from brand_dna.llm.client import LLMClient
from brand_dna.llm.prompts import render

logger = get_logger(__name__)


HIGH_SIGNAL_PAGES = {
    PageType.ABOUT,
    PageType.HOMEPAGE,
    PageType.BLOG,
    PageType.EDITORIAL,
    PageType.PRESS,
    PageType.LOOKBOOK,
}


def _aggregate_text(
    pages: list[Page],
    *,
    max_chars_per_page: int = 4000,
    max_total_chars: int = 30_000,
) -> str:
    """Build a bounded corpus from high-signal pages."""
    blocks: list[str] = []
    total = 0
    # Prioritise: ABOUT first (most signal), then HOMEPAGE, then editorials.
    priority = [
        PageType.ABOUT,
        PageType.HOMEPAGE,
        PageType.EDITORIAL,
        PageType.LOOKBOOK,
        PageType.BLOG,
        PageType.PRESS,
    ]
    pages_by_type: dict[PageType, list[Page]] = {pt: [] for pt in priority}
    for p in pages:
        if p.page_type in pages_by_type:
            pages_by_type[p.page_type].append(p)

    for pt in priority:
        for p in pages_by_type[pt]:
            text = p.body_text.strip()
            if not text:
                # Fall back to meta description if body is empty (JS-rendered site)
                text = (p.meta_description or "").strip()
            if not text:
                continue
            snippet = text[:max_chars_per_page]
            header = f"\n\n--- {pt.value.upper()} | {p.url}\n"
            block = header + snippet
            if total + len(block) > max_total_chars:
                break
            blocks.append(block)
            total += len(block)
        if total >= max_total_chars:
            break

    return "".join(blocks)


async def analyse_brand_voice(
    pages: list[Page],
    *,
    llm: LLMClient,
    model: str,
    brand_name: str,
) -> tuple[BrandVoice, int]:
    """Returns (BrandVoice, source_text_chars). The char count feeds confidence
    scoring downstream."""
    corpus = _aggregate_text(pages)
    if not corpus:
        logger.warning("text_analyzer.empty_corpus", brand=brand_name)
        return (
            BrandVoice(
                tone_descriptors=[],
                recurring_vocabulary=[],
                stated_values=[],
                positioning_statement=None,
                representative_quotes=[],
            ),
            0,
        )

    target_lang = getattr(llm, "target_language", "English") # Fallback to English

    prompt = render("brand_voice", brand_name=brand_name, corpus=corpus)
    data, _ = await llm.chat_json(
        prompt,
        model=model,
        system=(
            f"You are a senior brand strategist. The following corpus might be in "
            f"any language. Analyze it in its original context, but provide "
            f"ALL output fields in {target_lang}. Extract tonal "
            f"and positioning signals with precision."
        ),
        temperature=0.15,
        max_tokens=1800,
    )

    voice = BrandVoice(
        tone_descriptors=_as_list(data.get("tone_descriptors")),
        recurring_vocabulary=_as_list(data.get("recurring_vocabulary")),
        stated_values=_as_list(data.get("stated_values")),
        positioning_statement=data.get("positioning_statement") or None,
        representative_quotes=_as_list(data.get("representative_quotes")),
    )
    logger.info(
        "text_analyzer.complete",
        brand=brand_name,
        tone_count=len(voice.tone_descriptors),
        vocab_count=len(voice.recurring_vocabulary),
        values_count=len(voice.stated_values),
        corpus_chars=len(corpus),
    )
    return voice, len(corpus)


def _as_list(v: object) -> list[str]:
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    if isinstance(v, str) and v.strip():
        return [v.strip()]
    return []
