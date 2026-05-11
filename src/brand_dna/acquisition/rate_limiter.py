"""Async per-host rate limiter.

Why per-host (and not just global): we may follow a CDN link mid-crawl (e.g.,
images on `cdn.shopify.com`) — we don't want those to count against the brand's
own host budget. Each host gets its own minimum interval.
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from urllib.parse import urlparse


class HostRateLimiter:
    """Per-host minimum interval between requests, awaited cooperatively."""

    def __init__(self, default_delay_ms: int = 400) -> None:
        self._default_delay_s = default_delay_ms / 1000.0
        self._last: dict[str, float] = defaultdict(float)
        self._locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._host_overrides: dict[str, float] = {}

    def set_host_delay(self, host: str, delay_s: float) -> None:
        """Used after parsing robots.txt Crawl-delay."""
        self._host_overrides[host] = delay_s

    async def acquire(self, url: str) -> None:
        host = urlparse(url).netloc
        delay = self._host_overrides.get(host, self._default_delay_s)
        async with self._locks[host]:
            now = time.monotonic()
            wait = (self._last[host] + delay) - now
            if wait > 0:
                await asyncio.sleep(wait)
            self._last[host] = time.monotonic()
