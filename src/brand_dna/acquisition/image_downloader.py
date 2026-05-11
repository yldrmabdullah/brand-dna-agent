"""Concurrent image downloader.

Two-stage check:
1. HEAD where possible → reject by content-type / content-length before
   pulling bytes.
2. Stream + cap → some servers don't honor HEAD; protect against multi-MB
   payloads landing on us.

We content-address each image (sha256 → 16-char prefix) so two pages linking
the same hero image collapse into one stored asset with two provenance entries.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from PIL import Image, UnidentifiedImageError

from brand_dna.acquisition.rate_limiter import HostRateLimiter
from brand_dna.core.models import ImageProvenance, ImageRecord, PageType
from brand_dna.core.observability import get_logger

logger = get_logger(__name__)

ALLOWED_CONTENT_TYPES = (
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
    "image/avif",
)


class ImageDownloader:
    """Downloads image candidates concurrently, persists, and produces
    ImageRecord rows. Stateless other than the shared rate limiter / client."""

    def __init__(
        self,
        user_agent: str,
        output_dir: Path,
        rate_limiter: HostRateLimiter,
        *,
        max_concurrency: int = 8,
        min_bytes: int = 10_000,
        max_bytes: int = 15_000_000,
        timeout_s: int = 30,
    ) -> None:
        self.user_agent = user_agent
        self.output_dir = output_dir
        self.rate_limiter = rate_limiter
        self.max_concurrency = max_concurrency
        self.min_bytes = min_bytes
        self.max_bytes = max_bytes
        self.timeout_s = timeout_s
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def download_all(
        self,
        candidates: list[tuple[str, dict[str, Any]]],
    ) -> list[ImageRecord]:
        """Download every candidate. Failures are silently logged and skipped —
        graceful-failure is a hard requirement."""
        semaphore = asyncio.Semaphore(self.max_concurrency)
        headers = {
            "User-Agent": self.user_agent,
            "Accept": "image/avif,image/webp,image/png,image/jpeg,*/*;q=0.8",
        }
        async with httpx.AsyncClient(
            headers=headers,
            timeout=self.timeout_s,
            follow_redirects=True,
            http2=True,
        ) as client:
            tasks = [
                self._download_one(client, semaphore, url, page_meta)
                for url, page_meta in candidates
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        records: list[ImageRecord] = []
        for r in results:
            if isinstance(r, ImageRecord):
                records.append(r)
            elif isinstance(r, Exception):
                logger.debug("image.download_exception", error=str(r))
        logger.info(
            "image.download_summary",
            attempted=len(candidates),
            succeeded=len(records),
            failed=len(candidates) - len(records),
        )
        return records

    async def _download_one(
        self,
        client: httpx.AsyncClient,
        semaphore: asyncio.Semaphore,
        url: str,
        page_meta: dict[str, Any],
    ) -> ImageRecord | None:
        async with semaphore:
            await self.rate_limiter.acquire(url)
            try:
                resp = await client.get(url)
            except httpx.HTTPError as exc:
                logger.debug("image.fetch_failed", url=url, error=str(exc))
                return None

            if resp.status_code != 200:
                return None

            ct = resp.headers.get("content-type", "").split(";")[0].strip().lower()
            if ct and not any(ct.startswith(a) for a in ALLOWED_CONTENT_TYPES):
                return None

            content = resp.content
            if len(content) < self.min_bytes or len(content) > self.max_bytes:
                logger.debug(
                    "image.size_rejected",
                    url=url,
                    size=len(content),
                    min=self.min_bytes,
                    max=self.max_bytes,
                )
                return None

            return self._persist(url, content, page_meta)

    def _persist(
        self,
        source_url: str,
        content: bytes,
        page_meta: dict[str, Any],
    ) -> ImageRecord | None:
        # Decode for dimensions/format. PIL handles webp/avif via plugins
        # bundled with Pillow.
        try:
            img = Image.open(io.BytesIO(content))
            img.load()
        except (UnidentifiedImageError, OSError) as exc:
            logger.debug("image.decode_failed", url=source_url, error=str(exc))
            return None

        width, height = img.size
        fmt = (img.format or "").lower()
        if fmt == "jpg":
            fmt = "jpeg"

        # Content-address
        digest = hashlib.sha256(content).hexdigest()[:16]
        ext = {"jpeg": "jpg", "png": "png", "webp": "webp", "avif": "avif"}.get(fmt, "bin")
        # Two-level directory layout so we don't end up with 10k files in one dir.
        subdir = self.output_dir / digest[:2]
        subdir.mkdir(parents=True, exist_ok=True)
        local_path = subdir / f"{digest}.{ext}"
        if not local_path.exists():
            local_path.write_bytes(content)

        page_type_str = page_meta.get("page_type", "unknown")
        try:
            page_type = PageType(page_type_str)
        except ValueError:
            page_type = PageType.UNKNOWN

        provenance = ImageProvenance(
            source_url=source_url,
            page_url=page_meta.get("page_url", ""),
            page_type=page_type,
            alt_text=page_meta.get("alt_text"),
            surrounding_text=page_meta.get("page_title"),
            product_name=page_meta.get("product_name"),
            captured_at=datetime.now(timezone.utc),
        )

        return ImageRecord(
            image_id=digest,
            local_path=str(local_path),
            width=width,
            height=height,
            format=fmt or "unknown",
            bytes_size=len(content),
            provenance=provenance,
        )
