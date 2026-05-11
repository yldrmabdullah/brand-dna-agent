"""Run workspace layout.

Each run lives in its own directory so we can ship reproducible bundles to
stakeholders without disambiguation:

    outputs/{brand_slug}/{run_id}/
    ├── images/                  ← content-addressed, two-level fanout (00/, 01/, ...)
    │   └── ab/abcd1234....jpg
    ├── metadata.sqlite          ← image + page metadata
    ├── run_log.jsonl            ← structured log stream
    ├── brand_dna.json           ← Pydantic dossier dump
    └── brand_dna.pdf            ← human-readable dossier

This layout is shippable: zip the run dir, hand it to a strategist, done.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from pathlib import Path


def slugify(value: str) -> str:
    """Brand name → filesystem-safe slug."""
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = re.sub(r"-+", "-", value).strip("-")
    return value or "brand"


class RunWorkspace:
    """Owns paths for a single agent run. Created at the orchestrator boundary."""

    def __init__(self, output_root: Path, brand_name: str, run_id: str | None = None) -> None:
        self.brand_slug = slugify(brand_name)
        self.run_id = run_id or self._mint_run_id()
        self.root = output_root / self.brand_slug / self.run_id
        self.images_dir = self.root / "images"
        self.metadata_db_path = self.root / "metadata.sqlite"
        self.log_path = self.root / "run_log.jsonl"
        self.dossier_json_path = self.root / "brand_dna.json"
        self.dossier_pdf_path = self.root / "brand_dna.pdf"
        self.train_manifest_path = self.root / "train_modules.json"
        self.report_path = self.root / "run_report.json"

    def init(self) -> None:
        """Materialise the directory tree."""
        self.images_dir.mkdir(parents=True, exist_ok=True)
        # Marker file so `outputs/{brand}/latest` symlinks can point here later.
        (self.root / ".run-id").write_text(self.run_id)

    @staticmethod
    def _mint_run_id() -> str:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        suffix = uuid.uuid4().hex[:6]
        return f"{ts}-{suffix}"
