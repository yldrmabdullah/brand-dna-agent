"""XML sitemap discovery. The single highest-leverage signal for site-agnostic
crawling — every ecommerce platform that wants to be indexed publishes one.

Handles:
- sitemap index files (sitemap-of-sitemaps)
- gzipped sitemaps (.xml.gz)
- standard urlset / image sitemap extension (image:image)
"""

from __future__ import annotations

import gzip
import io
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlparse

import httpx
from lxml import etree

from brand_dna.core.observability import get_logger

logger = get_logger(__name__)

SITEMAP_NAMESPACES = {
    "sm": "http://www.sitemaps.org/schemas/sitemap/0.9",
    "image": "http://www.google.com/schemas/sitemap-image/1.1",
}


@dataclass
class SitemapEntry:
    url: str
    image_urls: list[str] = field(default_factory=list)
    last_modified: str | None = None


async def _fetch_sitemap_bytes(url: str, client: httpx.AsyncClient) -> bytes | None:
    try:
        resp = await client.get(url, timeout=20.0, follow_redirects=True)
    except httpx.HTTPError as exc:
        logger.warning("sitemap.fetch_failed", url=url, error=str(exc))
        return None
    if resp.status_code != 200:
        logger.info("sitemap.bad_status", url=url, status=resp.status_code)
        return None

    content = resp.content
    if url.endswith(".gz") or resp.headers.get("content-type", "").startswith(
        "application/x-gzip"
    ):
        try:
            content = gzip.decompress(content)
        except OSError as exc:
            logger.warning("sitemap.gunzip_failed", url=url, error=str(exc))
            return None
    return content


def _parse_sitemap(content: bytes, base_url: str) -> tuple[list[str], list[SitemapEntry]]:
    """Returns (sub_sitemaps, entries). Either may be empty."""
    try:
        tree = etree.fromstring(content)
    except etree.XMLSyntaxError as exc:
        logger.warning("sitemap.parse_failed", error=str(exc), sample=content[:200])
        return [], []

    tag = etree.QName(tree.tag).localname
    sub_sitemaps: list[str] = []
    entries: list[SitemapEntry] = []

    if tag == "sitemapindex":
        for sm in tree.findall("sm:sitemap", SITEMAP_NAMESPACES):
            loc = sm.findtext("sm:loc", default="", namespaces=SITEMAP_NAMESPACES).strip()
            if loc:
                sub_sitemaps.append(urljoin(base_url, loc))
    elif tag == "urlset":
        for u in tree.findall("sm:url", SITEMAP_NAMESPACES):
            loc = u.findtext("sm:loc", default="", namespaces=SITEMAP_NAMESPACES).strip()
            if not loc:
                continue
            lastmod = u.findtext("sm:lastmod", default="", namespaces=SITEMAP_NAMESPACES) or None
            images = [
                img.findtext("image:loc", default="", namespaces=SITEMAP_NAMESPACES).strip()
                for img in u.findall("image:image", SITEMAP_NAMESPACES)
            ]
            entries.append(
                SitemapEntry(
                    url=urljoin(base_url, loc),
                    image_urls=[i for i in images if i],
                    last_modified=lastmod,
                )
            )
    return sub_sitemaps, entries


async def discover_sitemap_urls(
    seed_urls: list[str],
    client: httpx.AsyncClient,
    *,
    max_total_entries: int = 5000,
    max_sub_sitemaps: int = 50,
) -> list[SitemapEntry]:
    """BFS through one or more sitemap URLs. Caps total entries to avoid pulling
    a 200k URL retail sitemap into memory.

    `seed_urls` is typically the sitemap URLs from robots.txt, plus
    `${base}/sitemap.xml` as a fallback guess.
    """
    visited: set[str] = set()
    queue: list[str] = list(dict.fromkeys(seed_urls))  # de-dup, preserve order
    entries: list[SitemapEntry] = []
    sub_count = 0

    while queue and len(entries) < max_total_entries and sub_count < max_sub_sitemaps:
        sm_url = queue.pop(0)
        if sm_url in visited:
            continue
        visited.add(sm_url)
        sub_count += 1

        content = await _fetch_sitemap_bytes(sm_url, client)
        if content is None:
            continue

        sub_sitemaps, found = _parse_sitemap(content, sm_url)
        logger.info(
            "sitemap.parsed",
            url=sm_url,
            sub_sitemaps=len(sub_sitemaps),
            entries=len(found),
        )
        queue.extend(s for s in sub_sitemaps if s not in visited)
        entries.extend(found)

    return entries[:max_total_entries]


def fallback_sitemap_urls(base_url: str) -> list[str]:
    """Common locations to try when robots.txt doesn't list a Sitemap directive."""
    parsed = urlparse(base_url)
    root = f"{parsed.scheme}://{parsed.netloc}"
    return [
        urljoin(root, "/sitemap.xml"),
        urljoin(root, "/sitemap_index.xml"),
        urljoin(root, "/sitemap-index.xml"),
        urljoin(root, "/sitemaps.xml"),
    ]
