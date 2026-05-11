"""Filtering layer tests.

We don't load CLIP in unit tests (too slow + downloads weights). Instead we
test the quality filter (pure logic) and the dedup algorithm with synthetic
records.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from brand_dna.core.models import ImageProvenance, ImageRecord, PageType
from brand_dna.filtering.deduplicator import VisualDeduplicator
from brand_dna.filtering.quality import filter_by_quality


def _mk_record(
    image_id: str,
    width: int,
    height: int,
    bytes_size: int,
    format: str = "jpeg",
    path: str = "/tmp/x.jpg",
) -> ImageRecord:
    return ImageRecord(
        image_id=image_id,
        local_path=path,
        width=width,
        height=height,
        format=format,
        bytes_size=bytes_size,
        provenance=ImageProvenance(
            source_url=f"https://example.com/{image_id}.jpg",
            page_url="https://example.com/",
            page_type=PageType.PRODUCT,
            captured_at=datetime.now(timezone.utc),
        ),
    )


class TestQualityFilter:
    def test_keeps_good_image(self) -> None:
        rec = _mk_record("a", 1024, 1280, 200_000)
        kept, rej = filter_by_quality([rec], min_shorter_side=512)
        assert len(kept) == 1
        assert len(rej) == 0

    def test_rejects_small_image(self) -> None:
        rec = _mk_record("a", 200, 200, 50_000)
        kept, rej = filter_by_quality([rec], min_shorter_side=512)
        assert kept == []
        assert "shorter_side" in (rej[0].rejection_reason or "")

    def test_rejects_tiny_bytes(self) -> None:
        rec = _mk_record("a", 1024, 1280, 100)
        kept, rej = filter_by_quality([rec], min_bytes=1000)
        assert kept == []
        assert "bytes_size" in (rej[0].rejection_reason or "")

    def test_rejects_extreme_aspect(self) -> None:
        # 5:1 banner — typical decorative header, not lookbook content
        rec = _mk_record("a", 2000, 320, 200_000)
        kept, rej = filter_by_quality([rec])
        assert kept == []
        assert "aspect_ratio" in (rej[0].rejection_reason or "")

    def test_rejects_disallowed_format(self) -> None:
        rec = _mk_record("a", 1024, 1280, 100_000, format="svg")
        kept, rej = filter_by_quality([rec])
        assert kept == []
        assert "format" in (rej[0].rejection_reason or "")


class TestDeduplicator:
    def _save_img(self, tmp_path: Path, name: str, color: tuple[int, int, int]) -> Path:
        img = Image.new("RGB", (600, 800), color=color)
        p = tmp_path / f"{name}.jpg"
        img.save(p, format="JPEG", quality=85)
        return p

    def test_removes_identical_dupes(self, tmp_path: Path) -> None:
        # Two identical solid-color images — should collapse to one
        p1 = self._save_img(tmp_path, "a", (120, 80, 60))
        p2 = self._save_img(tmp_path, "b", (120, 80, 60))
        recs = [
            _mk_record("a", 600, 800, p1.stat().st_size, path=str(p1)),
            _mk_record("b", 600, 800, p2.stat().st_size, path=str(p2)),
        ]
        dd = VisualDeduplicator()
        kept, dropped = dd.dedup(recs)
        assert len(kept) == 1
        assert len(dropped) == 1

    def test_keeps_distinct_images(self, tmp_path: Path) -> None:
        p1 = self._save_img(tmp_path, "a", (10, 10, 10))
        p2 = self._save_img(tmp_path, "b", (240, 240, 240))
        recs = [
            _mk_record("a", 600, 800, p1.stat().st_size, path=str(p1)),
            _mk_record("b", 600, 800, p2.stat().st_size, path=str(p2)),
        ]
        dd = VisualDeduplicator()
        kept, dropped = dd.dedup(recs)
        assert len(kept) == 2
        assert dropped == []

    def test_phash_threshold_strictness(self, tmp_path: Path) -> None:
        # Very strict threshold (0) — should reject only exact dupes
        # Reasonable threshold (10) — should reject more
        p1 = self._save_img(tmp_path, "a", (100, 100, 100))
        p2 = self._save_img(tmp_path, "b", (100, 100, 100))
        recs = [
            _mk_record("a", 600, 800, p1.stat().st_size, path=str(p1)),
            _mk_record("b", 600, 800, p2.stat().st_size, path=str(p2)),
        ]
        # Threshold 0 — identical hashes still collapse (hamming = 0)
        kept, _ = VisualDeduplicator(phash_hamming_threshold=0).dedup(recs)
        assert len(kept) == 1
