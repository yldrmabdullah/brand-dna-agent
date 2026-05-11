"""OpenGraph + Twitter Card meta extraction.

Most brands curate these tags for social sharing — they're remarkably consistent
across sites and give us:
- og:image (the brand's chosen hero image for a page)
- og:title, og:description (curated copy)
- og:type (article, product, website)
- og:site_name (canonical brand name)
"""

from __future__ import annotations

from selectolax.parser import HTMLParser

OG_KEYS = {
    "og:image",
    "og:image:secure_url",
    "og:title",
    "og:description",
    "og:site_name",
    "og:type",
    "og:url",
    "og:locale",
    "twitter:title",
    "twitter:description",
    "twitter:image",
    "twitter:site",
    "twitter:creator",
    "article:published_time",
    "article:author",
}


def extract_opengraph(html: str) -> dict[str, str]:
    """Returns a flat dict of og:* and twitter:* meta values."""
    parser = HTMLParser(html)
    out: dict[str, str] = {}
    for meta in parser.css("meta"):
        prop = meta.attributes.get("property") or meta.attributes.get("name") or ""
        prop = prop.strip().lower()
        if prop in OG_KEYS:
            content = meta.attributes.get("content", "") or ""
            if content:
                out[prop] = content.strip()
    return out


def extract_canonical_url(html: str) -> str | None:
    """Returns <link rel=canonical>'s href, if present."""
    parser = HTMLParser(html)
    link = parser.css_first('link[rel="canonical"]')
    if link:
        return link.attributes.get("href")
    return None


def extract_meta_description(html: str) -> str | None:
    """Returns the standard meta description, if present."""
    parser = HTMLParser(html)
    el = parser.css_first('meta[name="description"]')
    if el:
        return el.attributes.get("content")
    return None


def extract_title(html: str) -> str | None:
    parser = HTMLParser(html)
    if parser.tags("title"):
        return parser.tags("title")[0].text(strip=True) or None
    return None
