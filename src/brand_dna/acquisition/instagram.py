"""Instagram public-profile scraper. Best-effort.

ToS context (important): Meta restricts automated access; we deliberately use
only publicly-served HTML and *never* authenticate. We extract:
- OpenGraph image (profile or pinned post hero)
- Meta description (bio + post counts)

What we do NOT do:
- Login. No auth flows, no cookies bartered.
- Headless browser sessions emulating a logged-in user.
- High-volume scraping. One profile fetch per run, that's it.

If Instagram returns the login wall (very common in 2024+), we degrade
gracefully — no IG content in the dossier, logged as a known limitation.
The case explicitly accepts "one social channel done well" — we prioritise
ToS hygiene over comprehensive coverage.

For brands where IG is the critical signal, the right answer at production
scale is the official Graph API with Business Discovery scope (requires
brand consent + Meta app review). That's out of scope here but documented
in ARCHITECTURE.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import httpx
from selectolax.parser import HTMLParser

from brand_dna.core.observability import get_logger
from brand_dna.discovery.opengraph import extract_opengraph

logger = get_logger(__name__)


@dataclass
class InstagramSnapshot:
    handle: str
    profile_url: str
    bio: str | None = None
    profile_image_url: str | None = None
    follower_signal: str | None = None  # "1.2M followers" — text, not a number
    image_candidates: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    """Same shape as crawler output: (image_url, provenance_meta)."""
    blocked: bool = False
    note: str = ""


class InstagramScraper:
    """Public, unauthenticated. Single-shot per handle."""

    BASE_URL = "https://www.instagram.com"

    def __init__(self, user_agent: str, timeout_s: int = 20) -> None:
        # IG is finicky about UA. Use a plausible desktop browser UA — this is
        # *not* impersonation for ToS-circumvention, it's compatibility with
        # the publicly-served HTML that gets gated when UA looks bot-like.
        self.user_agent = user_agent
        self.timeout_s = timeout_s

    async def fetch_profile(self, handle: str) -> InstagramSnapshot:
        handle = handle.lstrip("@").strip()
        url = f"{self.BASE_URL}/{handle}/"
        snap = InstagramSnapshot(handle=handle, profile_url=url)

        headers = {
            "User-Agent": self.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }

        try:
            async with httpx.AsyncClient(
                headers=headers,
                timeout=self.timeout_s,
                follow_redirects=True,
            ) as client:
                resp = await client.get(url)
        except httpx.HTTPError as exc:
            snap.blocked = True
            snap.note = f"Network error: {exc}"
            logger.warning("instagram.fetch_error", handle=handle, error=str(exc))
            return snap

        if resp.status_code != 200:
            snap.blocked = True
            snap.note = f"HTTP {resp.status_code} — likely login wall or rate limit"
            logger.warning("instagram.bad_status", handle=handle, status=resp.status_code)
            return snap

        html = resp.text
        return self._parse_profile(snap, html)

    def _parse_profile(self, snap: InstagramSnapshot, html: str) -> InstagramSnapshot:
        og = extract_opengraph(html)

        # Login wall heuristic — IG serves a tiny "log in" page when blocked.
        if "Login" in (og.get("og:title", "") or "") and not og.get("og:image"):
            snap.blocked = True
            snap.note = "Login wall — public HTML not served to anonymous UA"
            logger.info("instagram.login_wall", handle=snap.handle)
            return snap

        snap.bio = og.get("og:description")
        snap.profile_image_url = og.get("og:image")

        if snap.profile_image_url:
            snap.image_candidates.append(
                (
                    snap.profile_image_url,
                    {
                        "page_url": snap.profile_url,
                        "page_type": "social_post",
                        "alt_text": snap.bio,
                        "page_title": og.get("og:title"),
                        "source": "instagram_profile_og",
                    },
                )
            )

        # Try to pull preloaded images from JSON in <script> tags — IG sometimes
        # ships SharedData / __NEXT_DATA__ even unauth.
        parser = HTMLParser(html)
        for script in parser.css("script"):
            text = script.text() or ""
            if "display_url" in text or "thumbnail_src" in text:
                # We don't fully parse this — extracting display URLs from
                # the obfuscated blob is brittle and changes monthly. For
                # production we'd swap to the Graph API.
                snap.note = (
                    "Public profile HTML available; deep post images require "
                    "Graph API. Using profile hero only."
                )
                break

        logger.info(
            "instagram.parsed",
            handle=snap.handle,
            has_bio=bool(snap.bio),
            has_image=bool(snap.profile_image_url),
        )
        return snap
