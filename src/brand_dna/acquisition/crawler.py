"""Brand-agnostic web crawler.

Pipeline per page:
  fetch → parse → extract structured data + OG + canonical → classify → collect
  image URLs → enqueue same-domain links (BFS, capped).

Choices that matter:
- **httpx async** with HTTP/2 → most retail sites serve from a CDN, h2 multiplexing
  pulls down 4-5x the throughput of h1 with the same concurrency.
- **selectolax** (lexbor) for HTML parsing — ~30x faster than BeautifulSoup,
  same selector API.
- **Playwright** only when `crawl.render_js=true`. Most fashion sites these days
  pre-render meta tags server-side for SEO; the JS-rendered DOM rarely adds
  extra image URLs that the static parse doesn't already see. We keep it as
  a config flag rather than the default to keep the Docker image lean.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx
import tldextract
from selectolax.parser import HTMLParser
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from brand_dna.acquisition.rate_limiter import HostRateLimiter
from brand_dna.core.config import BrandConfig
from brand_dna.core.exceptions import AcquisitionError
from brand_dna.core.models import Page, PageType
from brand_dna.core.observability import get_logger
from brand_dna.discovery.opengraph import (
    extract_canonical_url,
    extract_meta_description,
    extract_opengraph,
    extract_title,
)
from brand_dna.discovery.page_classifier import classify_page
from brand_dna.discovery.robots import RobotsPolicy, load_robots_policy
from brand_dna.discovery.sitemap import (
    SitemapEntry,
    discover_sitemap_urls,
    fallback_sitemap_urls,
)
from brand_dna.discovery.structured_data import (
    collect_product_images,
    extract_structured_data,
)

logger = get_logger(__name__)


@dataclass
class CrawlResult:
    """Aggregate output of a crawl pass."""

    pages: list[Page] = field(default_factory=list)
    image_candidates: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    """List of (image_url, page_metadata) where page_metadata carries the
    provenance context (page url, alt text, product name)."""

    robots_blocked_count: int = 0
    fetch_errors: list[str] = field(default_factory=list)


class BrandCrawler:
    """Crawls a single brand's web presence. Stateful — one instance per run."""

    def __init__(
        self,
        config: BrandConfig,
        user_agent: str,
        rate_limiter: HostRateLimiter | None = None,
    ) -> None:
        self.config = config
        self.user_agent = user_agent
        self.rate_limiter = rate_limiter or HostRateLimiter(
            default_delay_ms=config.crawl.delay_ms
        )
        self._semaphore = asyncio.Semaphore(config.crawl.max_concurrency)
        self._client: httpx.AsyncClient | None = None
        self._robots: RobotsPolicy | None = None

        ext = tldextract.extract(config.url)
        self._registered_domain = f"{ext.domain}.{ext.suffix}"
        self._allowed_hosts: set[str] = {urlparse(config.url).netloc}
        for sub in config.crawl.allowed_subdomains:
            self._allowed_hosts.add(sub)

    async def __aenter__(self) -> "BrandCrawler":
        headers = {
            "User-Agent": self.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }
        self._client = httpx.AsyncClient(
            headers=headers,
            timeout=self.config.crawl.request_timeout_s,
            follow_redirects=True,
            http2=True,
            limits=httpx.Limits(max_connections=self.config.crawl.max_concurrency * 2),
        )
        self._robots = await load_robots_policy(
            self.config.url, self.user_agent, self._client
        )
        if self._robots.crawl_delay_s:
            host = urlparse(self.config.url).netloc
            self.rate_limiter.set_host_delay(host, self._robots.crawl_delay_s)
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self._client:
            await self._client.aclose()

    # ─── Crawl entry point ────────────────────────────────────────────────

    async def crawl(self) -> CrawlResult:
        if not self._client or not self._robots:
            raise AcquisitionError("Crawler not entered as context manager")

        result = CrawlResult()

        # 1. Seed URLs: brand homepage, configured seeds, plus sitemap entries.
        seed_pages = [self.config.url] + list(self.config.seed_pages)
        sitemap_seeds = list(self._robots.sitemap_urls) or fallback_sitemap_urls(
            self.config.url
        )
        sitemap_entries = await discover_sitemap_urls(
            sitemap_seeds, self._client, max_total_entries=self.config.crawl.max_pages * 3
        )
        logger.info("sitemap.discovered", count=len(sitemap_entries))

        # Merge: prefer sitemap entries (richer — they carry image hints), then
        # the explicit seeds for anything the sitemap missed.
        sitemap_by_url = {e.url: e for e in sitemap_entries}
        candidate_urls = list(sitemap_by_url.keys())
        for s in seed_pages:
            if s not in sitemap_by_url:
                candidate_urls.append(s)

        # 2. Prioritise: collection > product > about > everything else.
        # Reasoning: collection/lookbook pages are the densest source of
        # representative imagery. Product detail pages give us per-image
        # metadata (alt text, product name). About/blog give us text identity.
        prioritised = self._prioritise(candidate_urls)
        to_crawl = prioritised[: self.config.crawl.max_pages]

        logger.info("crawl.start", pages=len(to_crawl))

        # 3. Concurrent fetch + parse.
        tasks = [self._process_page(url, sitemap_by_url.get(url), result) for url in to_crawl]
        await asyncio.gather(*tasks, return_exceptions=False)

        # 4. Dedup image candidates by URL.
        seen_imgs: set[str] = set()
        deduped: list[tuple[str, dict[str, Any]]] = []
        for url, meta in result.image_candidates:
            if url in seen_imgs:
                continue
            seen_imgs.add(url)
            deduped.append((url, meta))
        result.image_candidates = deduped[: self.config.crawl.max_images]

        logger.info(
            "crawl.finish",
            pages_crawled=len(result.pages),
            image_candidates=len(result.image_candidates),
            robots_blocked=result.robots_blocked_count,
            errors=len(result.fetch_errors),
        )
        return result

    # ─── Per-page work ────────────────────────────────────────────────────

    async def _process_page(
        self,
        url: str,
        sitemap_entry: SitemapEntry | None,
        result: CrawlResult,
    ) -> None:
        if not self._url_in_scope(url):
            return
        if not self._robots.can_fetch(url):  # type: ignore[union-attr]
            result.robots_blocked_count += 1
            logger.debug("crawl.robots_blocked", url=url)
            return

        async with self._semaphore:
            await self.rate_limiter.acquire(url)
            try:
                page = await self._fetch_and_parse(url)
            except (httpx.HTTPError, AcquisitionError) as exc:
                result.fetch_errors.append(f"{url}: {exc}")
                logger.warning("crawl.fetch_error", url=url, error=str(exc))
                return

            if page is None:
                return
            result.pages.append(page)

            # Image candidates from this page
            page_meta_base = {
                "page_url": page.url,
                "page_type": page.page_type.value,
                "page_title": page.title,
            }

            # 1. Sitemap-listed images (high-signal — brand explicitly indexed them)
            if sitemap_entry:
                for img_url in sitemap_entry.image_urls:
                    result.image_candidates.append(
                        (img_url, {**page_meta_base, "source": "sitemap_image_ext"})
                    )

            # 2. JSON-LD Product images (very high-signal — these are canonical)
            for img_url, prod_meta in collect_product_images(page.structured_data):
                result.image_candidates.append(
                    (
                        img_url,
                        {
                            **page_meta_base,
                            "product_name": prod_meta.get("product_name"),
                            "alt_text": prod_meta.get("description"),
                            "source": "jsonld_product",
                        },
                    )
                )

            # 3. OpenGraph hero image
            og_img = page.opengraph.get("og:image") or page.opengraph.get(
                "og:image:secure_url"
            )
            if og_img:
                result.image_candidates.append(
                    (
                        urljoin(page.url, og_img),
                        {
                            **page_meta_base,
                            "alt_text": page.opengraph.get("og:description"),
                            "source": "opengraph",
                        },
                    )
                )

            # 4. Inline <img> tags with alt — only if we still need more
            #    candidates. This is the noisiest source.
            for img_url, alt in page.image_urls_with_alt:  # type: ignore[attr-defined]
                result.image_candidates.append(
                    (
                        urljoin(page.url, img_url),
                        {
                            **page_meta_base,
                            "alt_text": alt,
                            "source": "inline_img",
                        },
                    )
                )

    async def _fetch_and_parse(self, url: str) -> Page | None:
        retry = AsyncRetrying(
            reraise=True,
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=8),
            retry=retry_if_exception_type((httpx.TimeoutException, httpx.RemoteProtocolError)),
        )
        async for attempt in retry:
            with attempt:
                resp = await self._client.get(url)  # type: ignore[union-attr]

        if resp.status_code >= 400:
            logger.debug("crawl.bad_status", url=url, status=resp.status_code)
            return None
        ct = resp.headers.get("content-type", "")
        if "text/html" not in ct and "application/xhtml" not in ct:
            return None

        html = resp.text
        return self._parse_html(url, html, resp.status_code)

    def _parse_html(self, url: str, html: str, status: int) -> Page:
        parser = HTMLParser(html)
        structured = extract_structured_data(html)
        og = extract_opengraph(html)
        title = extract_title(html)
        meta_desc = extract_meta_description(html)
        page_type = classify_page(
            url,
            structured_data=structured,
            opengraph=og,
            url_hints=self.config.crawl.page_type_hints,
        )

        # Inline images — we extract URL+alt pairs and keep them as
        # an attribute on the Page model (not the Pydantic field — kept ephemeral).
        img_pairs: list[tuple[str, str | None]] = []
        for img in parser.css("img"):
            src = (
                img.attributes.get("src")
                or img.attributes.get("data-src")
                or img.attributes.get("data-lazy-src")
                or ""
            )
            if not src:
                continue
            if src.startswith("data:"):
                continue
            alt = img.attributes.get("alt")
            img_pairs.append((src, alt))

        # Discovered links — used for fallback BFS if sitemap is sparse.
        links: list[str] = []
        for a in parser.css("a[href]"):
            href = a.attributes.get("href", "") or ""
            if href and not href.startswith(("javascript:", "mailto:", "tel:", "#")):
                links.append(urljoin(url, href))

        # Body text — extract from main content tags, fall back to <body>
        body = (
            parser.css_first("main")
            or parser.css_first("article")
            or parser.css_first("body")
        )
        body_text = body.text(separator="\n", strip=True) if body else ""

        page = Page(
            url=url,
            page_type=page_type,
            title=title,
            body_text=body_text[:50_000],  # cap to keep memory bounded
            meta_description=meta_desc,
            structured_data=structured,
            opengraph=og,
            discovered_links=links[:200],
            http_status=status,
        )
        # Ephemeral attribute for image URL extraction — not in Pydantic schema.
        page.image_urls = [src for src, _ in img_pairs]  # type: ignore[attr-defined]
        page.image_urls_with_alt = img_pairs  # type: ignore[attr-defined]
        return page

    # ─── URL scoping & prioritisation ────────────────────────────────────

    def _url_in_scope(self, url: str) -> bool:
        try:
            parsed = urlparse(url)
        except ValueError:
            return False
        if parsed.scheme not in ("http", "https"):
            return False
        host = parsed.netloc
        if host in self._allowed_hosts:
            return True
        ext = tldextract.extract(url)
        if f"{ext.domain}.{ext.suffix}" == self._registered_domain:
            return True
        return False

    def _prioritise(self, urls: list[str]) -> list[str]:
        """Order pages by likely value-density for the dossier."""
        priority_order = [
            PageType.COLLECTION,
            PageType.LOOKBOOK,
            PageType.PRODUCT,
            PageType.EDITORIAL,
            PageType.ABOUT,
            PageType.BLOG,
            PageType.HOMEPAGE,
            PageType.PRESS,
            PageType.FAQ,
            PageType.UNKNOWN,
        ]
        priority_idx = {pt: i for i, pt in enumerate(priority_order)}

        def sort_key(u: str) -> tuple[int, int]:
            pt = classify_page(u, url_hints=self.config.crawl.page_type_hints)
            return (priority_idx.get(pt, 99), len(u))

        return sorted(urls, key=sort_key)
