"""robots.txt parsing — politeness first.

We always check robots.txt before crawling. If a path is disallowed for our
user-agent, we skip it. We also extract Sitemap: directives — these are the
canonical entrypoint for site-wide URL discovery.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import httpx

from brand_dna.core.observability import get_logger

logger = get_logger(__name__)


@dataclass
class RobotsPolicy:
    """Encapsulates robots.txt rules for a host."""

    host: str
    user_agent: str
    parser: RobotFileParser
    sitemap_urls: list[str] = field(default_factory=list)
    crawl_delay_s: float | None = None
    fetched: bool = False

    def can_fetch(self, url: str) -> bool:
        """If robots.txt couldn't be fetched, we default to *allowed* — that's
        the conventional interpretation (absence = no restriction). Sites that
        actively want to exclude us state it explicitly."""
        if not self.fetched:
            return True
        try:
            return self.parser.can_fetch(self.user_agent, url)
        except Exception:
            return True


async def load_robots_policy(
    base_url: str,
    user_agent: str,
    client: httpx.AsyncClient,
) -> RobotsPolicy:
    """Fetch and parse robots.txt for a given base URL.

    Returns a permissive policy if fetching fails — we still log the failure.
    """
    parsed = urlparse(base_url)
    robots_url = urljoin(f"{parsed.scheme}://{parsed.netloc}", "/robots.txt")

    parser = RobotFileParser()
    parser.set_url(robots_url)

    policy = RobotsPolicy(host=parsed.netloc, user_agent=user_agent, parser=parser)

    try:
        resp = await client.get(robots_url, timeout=10.0)
    except httpx.HTTPError as exc:
        logger.warning("robots.fetch_failed", url=robots_url, error=str(exc))
        return policy

    if resp.status_code != 200:
        logger.info("robots.not_found", url=robots_url, status=resp.status_code)
        return policy

    text = resp.text
    parser.parse(text.splitlines())
    policy.fetched = True

    # Hand-parse the Sitemap: and Crawl-delay: directives — RobotFileParser
    # doesn't expose them cleanly.
    for line in text.splitlines():
        line = line.strip()
        lower = line.lower()
        if lower.startswith("sitemap:"):
            sm = line.split(":", 1)[1].strip()
            if sm:
                policy.sitemap_urls.append(sm)
        elif lower.startswith("crawl-delay:"):
            try:
                policy.crawl_delay_s = float(line.split(":", 1)[1].strip())
            except (ValueError, IndexError):
                pass

    logger.info(
        "robots.loaded",
        url=robots_url,
        sitemaps=len(policy.sitemap_urls),
        crawl_delay=policy.crawl_delay_s,
    )
    return policy
