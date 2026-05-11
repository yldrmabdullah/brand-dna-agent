"""Heuristic page-type classifier.

Three signals (descending strength):
1. JSON-LD @type — when present, dispositive
2. OpenGraph og:type — strong (article vs product vs website)
3. URL path patterns — weak but universal

The classifier never *requires* any signal. Missing all three → UNKNOWN.

We deliberately avoid LLM classification here: this runs on every page, so
keeping it deterministic and cheap matters. The LLM gets called once per
*brand*, not once per page.
"""

from __future__ import annotations

from urllib.parse import urlparse

from brand_dna.core.models import PageType

# URL path → page_type. Order matters; first match wins.
# These are *hints*, not hardcoded selectors — every brand uses some subset
# of these conventions because they're SEO-canonical.
URL_PATTERNS: list[tuple[PageType, tuple[str, ...]]] = [
    (PageType.PRODUCT, ("/products/", "/product/", "/p/", "/shop/", "/item/")),
    (
        PageType.COLLECTION,
        (
            "/collections/",
            "/collection/",
            "/category/",
            "/c/",
            "/women",
            "/men",
            "/shop-all",
        ),
    ),
    (
        PageType.LOOKBOOK,
        ("/lookbook", "/look-book", "/editorials", "/editorial/", "/look/"),
    ),
    (PageType.ABOUT, ("/about", "/our-story", "/who-we-are", "/heritage", "/brand")),
    (PageType.BLOG, ("/blog", "/journal", "/stories", "/the-edit", "/news")),
    (PageType.PRESS, ("/press", "/media", "/in-the-press")),
    (PageType.FAQ, ("/faq", "/help", "/customer-care", "/support")),
    (PageType.EDITORIAL, ("/campaign", "/feature/", "/world-of")),
]


OG_TYPE_MAP: dict[str, PageType] = {
    "product": PageType.PRODUCT,
    "product.item": PageType.PRODUCT,
    "article": PageType.BLOG,
    "blog": PageType.BLOG,
    "website": PageType.UNKNOWN,
}


JSONLD_TYPE_MAP: dict[str, PageType] = {
    "Product": PageType.PRODUCT,
    "ItemList": PageType.COLLECTION,
    "Article": PageType.BLOG,
    "BlogPosting": PageType.BLOG,
    "NewsArticle": PageType.BLOG,
    "AboutPage": PageType.ABOUT,
    "FAQPage": PageType.FAQ,
}


def _types_from_jsonld(nodes: list[dict]) -> set[str]:
    out: set[str] = set()
    for n in nodes:
        t = n.get("@type")
        if isinstance(t, str):
            out.add(t)
        elif isinstance(t, list):
            out.update(x for x in t if isinstance(x, str))
    return out


def classify_page(
    url: str,
    *,
    structured_data: list[dict] | None = None,
    opengraph: dict[str, str] | None = None,
    url_hints: dict[str, list[str]] | None = None,
) -> PageType:
    """Classify a page using all available signals.

    Args:
        url: The page URL.
        structured_data: JSON-LD nodes already extracted (optional).
        opengraph: OG meta dict (optional).
        url_hints: Per-brand URL pattern overrides from BrandConfig.crawl.
            Adds *additional* patterns; never replaces the defaults.
    """
    # 1. JSON-LD wins if it speaks.
    if structured_data:
        types = _types_from_jsonld(structured_data)
        for jt, pt in JSONLD_TYPE_MAP.items():
            if jt in types:
                return pt

    # 2. OpenGraph.
    og_type = (opengraph or {}).get("og:type", "").lower()
    if og_type and og_type in OG_TYPE_MAP and OG_TYPE_MAP[og_type] != PageType.UNKNOWN:
        return OG_TYPE_MAP[og_type]

    # 3. URL path.
    path = urlparse(url).path.lower()

    # Brand-supplied hints first — they know their site quirks.
    if url_hints:
        for pt_name, patterns in url_hints.items():
            for pat in patterns:
                if pat.lower() in path:
                    try:
                        return PageType(pt_name)
                    except ValueError:
                        continue

    # Default patterns.
    for pt, patterns in URL_PATTERNS:
        if any(pat in path for pat in patterns):
            return pt

    # Homepage heuristic.
    if path in ("", "/", "/home", "/index", "/index.html"):
        return PageType.HOMEPAGE

    return PageType.UNKNOWN
