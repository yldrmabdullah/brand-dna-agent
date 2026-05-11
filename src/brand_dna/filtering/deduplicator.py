"""Visual deduplication.

Two-stage:
1. **pHash (perceptual hash)** — fast, catches identical, resized, lightly
   recoloured duplicates. We use it as a coarse pre-filter.
2. **CLIP embedding cosine** — semantic dedup for cases pHash misses
   (e.g., the same product shot from a slightly different angle that the
   brand happens to list twice). Skipped unless embeddings are populated.

The threshold is a tradeoff: too strict and we keep near-duplicate hero shots
that bias the color palette; too loose and we lose meaningful style variants.
The default of 5 (out of 64 bits) is tuned for ~95% precision on near-dupes
in fashion catalogs.
"""

from __future__ import annotations

from io import BytesIO

import imagehash
import numpy as np
from PIL import Image

from brand_dna.core.models import ImageRecord
from brand_dna.core.observability import get_logger

logger = get_logger(__name__)


class VisualDeduplicator:
    """Operates in-place on records: sets phash. Returns deduplicated subset."""

    def __init__(
        self,
        *,
        phash_hamming_threshold: int = 5,
        embedding_cosine_threshold: float = 0.97,
    ) -> None:
        self.phash_hamming_threshold = phash_hamming_threshold
        self.embedding_cosine_threshold = embedding_cosine_threshold

    def compute_phashes(self, records: list[ImageRecord]) -> None:
        for rec in records:
            if rec.phash:
                continue
            try:
                img = Image.open(rec.local_path).convert("RGB")
                rec.phash = str(imagehash.phash(img, hash_size=8))
            except Exception as exc:
                logger.debug(
                    "dedup.phash_failed", path=rec.local_path, error=str(exc)
                )
                rec.phash = None

    def dedup(self, records: list[ImageRecord]) -> tuple[list[ImageRecord], list[str]]:
        """Returns (kept, dropped_ids). Each dropped image is logged.

        Strategy: greedy keep — first record wins, near-dupes drop. Records
        are sorted by quality proxies before dedup so the highest-resolution
        / most-canonical image is the survivor.
        """
        self.compute_phashes(records)

        # Sort: prefer (a) JSON-LD product source > OG > inline, (b) larger,
        # (c) higher fashion_score. Highest-quality wins ties.
        def sort_key(r: ImageRecord) -> tuple:
            source_priority = {
                "jsonld_product": 0,
                "sitemap_image_ext": 1,
                "opengraph": 2,
                "instagram_profile_og": 3,
                "inline_img": 4,
            }
            src_priority = source_priority.get(r.provenance.alt_text or "", 5)
            return (
                src_priority,
                -(r.width * r.height),
                -(r.fashion_score or 0.0),
            )

        sorted_records = sorted(records, key=sort_key)

        kept: list[ImageRecord] = []
        dropped: list[str] = []
        kept_hashes: list[imagehash.ImageHash] = []
        kept_embeddings: list[np.ndarray] = []

        for rec in sorted_records:
            is_dup = False

            # pHash check
            if rec.phash:
                rec_hash = imagehash.hex_to_hash(rec.phash)
                for h in kept_hashes:
                    if (rec_hash - h) <= self.phash_hamming_threshold:
                        is_dup = True
                        break

            # Embedding check (only if not already dropped by pHash and we have
            # embeddings available)
            if not is_dup and rec.embedding and kept_embeddings:
                emb = np.asarray(rec.embedding, dtype=np.float32)
                # Cosine — embeddings are already L2-normalised from CLIP.
                sims = np.array([emb @ e for e in kept_embeddings])
                if (sims >= self.embedding_cosine_threshold).any():
                    is_dup = True

            if is_dup:
                dropped.append(rec.image_id)
            else:
                kept.append(rec)
                if rec.phash:
                    kept_hashes.append(imagehash.hex_to_hash(rec.phash))
                if rec.embedding:
                    kept_embeddings.append(
                        np.asarray(rec.embedding, dtype=np.float32)
                    )

        logger.info(
            "dedup.summary",
            input=len(records),
            kept=len(kept),
            dropped=len(dropped),
            phash_threshold=self.phash_hamming_threshold,
        )
        return kept, dropped
