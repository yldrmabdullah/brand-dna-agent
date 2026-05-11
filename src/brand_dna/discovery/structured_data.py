"""Schema.org / JSON-LD extraction.

Brands embed structured data because Google asks them to. We harvest the
canonical types we care about — site-agnostic by design:

- `Product`: name, image, description, brand, offers
- `ImageObject`: caption, contentUrl
- `BreadcrumbList`: category hierarchy
- `Article` / `BlogPosting`: editorial text
- `WebPage`: description, name
- `Organization`: brand-level claims (values, founding, locations)
"""

from __future__ import annotations

import json
from typing import Any, Iterable

from selectolax.parser import HTMLParser

from brand_dna.core.observability import get_logger

logger = get_logger(__name__)

CARE_TYPES = {
    "Product",
    "ImageObject",
    "BreadcrumbList",
    "Article",
    "BlogPosting",
    "NewsArticle",
    "WebPage",
    "AboutPage",
    "Organization",
    "ItemList",
    "FAQPage",
    "Question",
}


def _flatten_graph(obj: Any) -> Iterable[dict[str, Any]]:
    """JSON-LD can be a node, a list, or a `@graph` wrapper. Yield nodes."""
    if isinstance(obj, list):
        for item in obj:
            yield from _flatten_graph(item)
    elif isinstance(obj, dict):
        if "@graph" in obj and isinstance(obj["@graph"], list):
            yield from _flatten_graph(obj["@graph"])
        yield obj


def _node_type(node: dict[str, Any]) -> set[str]:
    raw = node.get("@type")
    if isinstance(raw, str):
        return {raw}
    if isinstance(raw, list):
        return {t for t in raw if isinstance(t, str)}
    return set()


def extract_structured_data(html: str) -> list[dict[str, Any]]:
    """Extract relevant JSON-LD nodes from an HTML page.

    Returns *only* nodes whose @type intersects CARE_TYPES — saves downstream
    consumers from sifting through site furniture (WebSite, SiteNavigationElement).
    """
    try:
        parser = HTMLParser(html)
    except Exception as exc:
        logger.warning("structured_data.parse_failed", error=str(exc))
        return []

    out: list[dict[str, Any]] = []
    for script in parser.css('script[type="application/ld+json"]'):
        raw = (script.text() or "").strip()
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Some sites embed multiple JSON objects concatenated. Best-effort.
            continue
        for node in _flatten_graph(data):
            if _node_type(node) & CARE_TYPES:
                out.append(node)
    return out


def collect_product_images(nodes: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    """Pull (image_url, product_metadata) pairs from Product nodes.

    Returns tuples so callers can persist per-image provenance.
    """
    results: list[tuple[str, dict[str, Any]]] = []
    for node in nodes:
        if "Product" not in _node_type(node):
            continue
        product_meta = {
            "product_name": node.get("name"),
            "description": node.get("description"),
            "brand": _extract_brand(node.get("brand")),
            "sku": node.get("sku"),
            "color": node.get("color"),
            "material": node.get("material"),
        }
        for img in _normalise_image_field(node.get("image")):
            results.append((img, product_meta))
    return results


def _extract_brand(b: Any) -> str | None:
    if isinstance(b, str):
        return b
    if isinstance(b, dict):
        return b.get("name")
    return None


def _normalise_image_field(field: Any) -> list[str]:
    """Schema.org image field can be a string, list, or ImageObject."""
    out: list[str] = []
    if not field:
        return out
    items = field if isinstance(field, list) else [field]
    for item in items:
        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, dict):
            url = item.get("contentUrl") or item.get("url")
            if url:
                out.append(url)
    return out


def extract_text_from_article_nodes(nodes: list[dict[str, Any]]) -> list[str]:
    """Pull editorial / about / FAQ text bodies from structured data."""
    bodies: list[str] = []
    for node in nodes:
        types = _node_type(node)
        if types & {"Article", "BlogPosting", "NewsArticle", "AboutPage", "WebPage"}:
            text = node.get("articleBody") or node.get("description") or ""
            if isinstance(text, str) and len(text) > 100:
                bodies.append(text)
        elif "Question" in types:
            q = node.get("name") or ""
            a = node.get("acceptedAnswer", {})
            if isinstance(a, dict):
                a_text = a.get("text") or ""
                if q and a_text:
                    bodies.append(f"Q: {q}\nA: {a_text}")
        elif "Organization" in types:
            desc = node.get("description") or ""
            if isinstance(desc, str) and desc:
                bodies.append(desc)
    return bodies
