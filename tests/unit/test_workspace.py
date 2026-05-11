"""Run workspace + SQLite metadata store tests."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from brand_dna.core.models import ImageProvenance, ImageRecord, Page, PageType
from brand_dna.storage.image_store import RunWorkspace, slugify
from brand_dna.storage.metadata_store import MetadataStore


class TestSlugify:
    def test_basic(self) -> None:
        assert slugify("COS") == "cos"

    def test_spaces(self) -> None:
        assert slugify("Les Benjamins") == "les-benjamins"

    def test_special_chars(self) -> None:
        assert slugify("Acmé & Co.") == "acm-co"

    def test_empty_fallback(self) -> None:
        assert slugify("") == "brand"
        assert slugify("///") == "brand"


class TestRunWorkspace:
    def test_directory_layout(self, tmp_path: Path) -> None:
        ws = RunWorkspace(output_root=tmp_path, brand_name="Acme Co")
        ws.init()
        assert ws.brand_slug == "acme-co"
        assert ws.root.exists()
        assert ws.images_dir.exists()
        assert (ws.root / ".run-id").read_text() == ws.run_id

    def test_run_id_format(self, tmp_path: Path) -> None:
        ws = RunWorkspace(output_root=tmp_path, brand_name="X")
        # YYYYMMDDTHHMMSSZ-xxxxxx (UTC timestamp + 6 hex)
        assert "T" in ws.run_id and "Z-" in ws.run_id
        assert len(ws.run_id.split("-")[-1]) == 6


class TestMetadataStore:
    def _mk_record(self, idx: int) -> ImageRecord:
        return ImageRecord(
            image_id=f"img{idx:03d}",
            local_path=f"/tmp/img{idx:03d}.jpg",
            width=1024,
            height=1280,
            format="jpeg",
            bytes_size=200_000,
            phash="abc123",
            fashion_score=0.85,
            garment_labels=["Tops"],
            cluster_id=1,
            provenance=ImageProvenance(
                source_url=f"https://x.com/{idx}.jpg",
                page_url="https://x.com/",
                page_type=PageType.PRODUCT,
                captured_at=datetime.now(timezone.utc),
            ),
        )

    def test_round_trip_images(self, tmp_path: Path) -> None:
        store = MetadataStore(tmp_path / "meta.sqlite")
        store.connect()
        records = [self._mk_record(i) for i in range(5)]
        store.insert_images(records)
        assert store.count_images(only_kept=True) == 5
        phashes = store.fetch_phashes()
        assert phashes == {"abc123"}
        store.close()

    def test_round_trip_pages(self, tmp_path: Path) -> None:
        store = MetadataStore(tmp_path / "meta.sqlite")
        store.connect()
        pages = [
            Page(
                url=f"https://x.com/p/{i}",
                page_type=PageType.PRODUCT,
                title=f"Page {i}",
                body_text="Sample body",
                http_status=200,
            )
            for i in range(3)
        ]
        store.insert_pages(pages)
        # Upsert behavior — inserting same URL again replaces
        store.insert_pages(pages)
        store.close()

    def test_set_meta(self, tmp_path: Path) -> None:
        store = MetadataStore(tmp_path / "meta.sqlite")
        store.connect()
        store.set_meta("run_id", "test-001")
        store.set_meta("config", {"a": 1, "b": [1, 2]})
        store.close()
