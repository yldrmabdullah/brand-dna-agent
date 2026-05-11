"""SQLite-backed metadata persistence.

Why SQLite (and not DuckDB or Postgres):
- Zero infra. The case asks for `docker run` and one PDF out — SQLite ships
  in stdlib, zero migrations.
- Per-run database = self-contained, shippable. Strategists get an
  inspectable .sqlite file for ad-hoc queries.
- Schema is small (3 tables). The overhead of an OLAP engine isn't worth it.

At production scale (thousands of brands, diff mode, cross-brand analytics)
the right answer is Postgres + S3 for images. That migration is straightforward
because this layer is the only place that touches storage — everywhere else
sees Pydantic models.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from brand_dna.core.models import ImageRecord, Page


class MetadataStore:
    """Owns the SQLite connection for one run."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._conn: sqlite3.Connection | None = None

    def connect(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA foreign_keys=ON;")
        self._create_schema()

    def close(self) -> None:
        if self._conn:
            self._conn.commit()
            self._conn.close()
            self._conn = None

    def _create_schema(self) -> None:
        assert self._conn
        cur = self._conn.cursor()
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS pages (
                url TEXT PRIMARY KEY,
                page_type TEXT NOT NULL,
                title TEXT,
                meta_description TEXT,
                body_text TEXT,
                http_status INTEGER,
                fetched_at TEXT NOT NULL,
                opengraph_json TEXT,
                structured_data_json TEXT
            );

            CREATE TABLE IF NOT EXISTS images (
                image_id TEXT PRIMARY KEY,
                local_path TEXT NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                format TEXT,
                bytes_size INTEGER,
                phash TEXT,
                fashion_score REAL,
                quality_passed INTEGER,
                rejection_reason TEXT,
                garment_labels_json TEXT,
                silhouette_tags_json TEXT,
                cluster_id INTEGER,
                source_url TEXT NOT NULL,
                page_url TEXT,
                page_type TEXT,
                alt_text TEXT,
                product_name TEXT,
                captured_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_images_cluster ON images(cluster_id);
            CREATE INDEX IF NOT EXISTS idx_images_quality ON images(quality_passed);
            CREATE INDEX IF NOT EXISTS idx_images_phash ON images(phash);
            CREATE INDEX IF NOT EXISTS idx_pages_type ON pages(page_type);

            CREATE TABLE IF NOT EXISTS run_meta (
                key TEXT PRIMARY KEY,
                value TEXT
            );
            """
        )
        self._conn.commit()

    @contextmanager
    def _cursor(self) -> Iterator[sqlite3.Cursor]:
        assert self._conn, "MetadataStore not connected"
        cur = self._conn.cursor()
        try:
            yield cur
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        finally:
            cur.close()

    # ─── Writes ───────────────────────────────────────────────────────────

    def insert_pages(self, pages: list[Page]) -> None:
        with self._cursor() as cur:
            cur.executemany(
                """
                INSERT OR REPLACE INTO pages
                (url, page_type, title, meta_description, body_text,
                 http_status, fetched_at, opengraph_json, structured_data_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        p.url,
                        p.page_type.value,
                        p.title,
                        p.meta_description,
                        p.body_text,
                        p.http_status,
                        p.fetched_at.isoformat(),
                        json.dumps(p.opengraph),
                        json.dumps(p.structured_data),
                    )
                    for p in pages
                ],
            )

    def insert_images(self, images: list[ImageRecord]) -> None:
        with self._cursor() as cur:
            cur.executemany(
                """
                INSERT OR REPLACE INTO images
                (image_id, local_path, width, height, format, bytes_size,
                 phash, fashion_score, quality_passed, rejection_reason,
                 garment_labels_json, silhouette_tags_json, cluster_id,
                 source_url, page_url, page_type, alt_text, product_name,
                 captured_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        img.image_id,
                        img.local_path,
                        img.width,
                        img.height,
                        img.format,
                        img.bytes_size,
                        img.phash,
                        img.fashion_score,
                        1 if img.quality_passed else 0,
                        img.rejection_reason,
                        json.dumps(img.garment_labels),
                        json.dumps(img.silhouette_tags),
                        img.cluster_id,
                        img.provenance.source_url,
                        img.provenance.page_url,
                        img.provenance.page_type.value,
                        img.provenance.alt_text,
                        img.provenance.product_name,
                        img.provenance.captured_at.isoformat(),
                    )
                    for img in images
                ],
            )

    def set_meta(self, key: str, value: Any) -> None:
        with self._cursor() as cur:
            cur.execute(
                "INSERT OR REPLACE INTO run_meta(key, value) VALUES (?, ?)",
                (key, json.dumps(value)),
            )

    # ─── Reads (useful for diff mode + tests) ────────────────────────────

    def count_images(self, only_kept: bool = True) -> int:
        with self._cursor() as cur:
            if only_kept:
                cur.execute("SELECT COUNT(*) FROM images WHERE quality_passed = 1")
            else:
                cur.execute("SELECT COUNT(*) FROM images")
            return int(cur.fetchone()[0])

    def fetch_phashes(self) -> set[str]:
        """For dedup across runs (diff mode in v2)."""
        with self._cursor() as cur:
            cur.execute("SELECT phash FROM images WHERE phash IS NOT NULL")
            return {row[0] for row in cur.fetchall()}
