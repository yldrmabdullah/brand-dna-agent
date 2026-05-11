"""Aggregates per-image garment labels into a brand-level distribution.

The FashionCLIP classifier already emitted top-K labels per image during the
filtering stage. Here we count, normalise, and surface the categorical
distribution that goes into the dossier.

We also derive a 'silhouette summary' as a separate concept — silhouettes are
zero-shot labels that describe *shape* (fitted, oversized, draped) rather
than *garment type*. These overlap but aren't identical; the dossier shows
them in distinct sections.
"""

from __future__ import annotations

from collections import Counter
from functools import lru_cache

import numpy as np
import torch

from brand_dna.core.models import GarmentDistribution, ImageRecord
from brand_dna.core.observability import get_logger
from brand_dna.filtering.fashion_classifier import _load_clip

logger = get_logger(__name__)


SILHOUETTE_PROMPTS = [
    "a fitted, body-conscious silhouette",
    "an oversized, voluminous silhouette",
    "a draped, flowing silhouette",
    "a structured, tailored silhouette",
    "a minimal, clean silhouette",
    "a layered, complex silhouette",
    "a relaxed, easy silhouette",
    "an elongated, columnar silhouette",
]

SILHOUETTE_LABELS = [
    "Fitted",
    "Oversized",
    "Draped",
    "Tailored",
    "Minimal",
    "Layered",
    "Relaxed",
    "Elongated",
]


def aggregate_garments(images: list[ImageRecord]) -> GarmentDistribution:
    """Counts garment_labels across images. Each image contributes its top-K labels."""
    counter: Counter[str] = Counter()
    counted = 0
    for img in images:
        if not img.garment_labels:
            continue
        counted += 1
        for label in img.garment_labels:
            counter[label] += 1

    total = sum(counter.values()) or 1
    return GarmentDistribution(
        counts=dict(counter),
        percentages={k: round(v / total * 100.0, 1) for k, v in counter.items()},
        sample_size=counted,
    )


def derive_silhouette_summary(
    images: list[ImageRecord],
    *,
    model_id: str = "patrickjohncyh/fashion-clip",
    top_k: int = 4,
    threshold: float = 0.15,
    sample_size: int = 60,
) -> list[str]:
    """Zero-shot silhouette tags ranked across a sample of images.

    Reuses the CLIP model loaded by the fashion classifier. Subsamples to
    keep cost bounded on large catalogs.
    """
    if not images:
        return []

    model, processor = _load_clip(model_id)
    device = next(model.parameters()).device

    # Encode silhouette prompts once
    text_inputs = processor(text=SILHOUETTE_PROMPTS, return_tensors="pt", padding=True).to(
        device
    )
    with torch.no_grad():
        text_feats = model.get_text_features(**text_inputs)
    text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

    # Sample images uniformly
    rng = np.random.default_rng(42)
    sample = images
    if len(images) > sample_size:
        idx = rng.choice(len(images), sample_size, replace=False)
        sample = [images[i] for i in idx]

    # We already have embeddings from the classifier stage — reuse them.
    embeddings = []
    for img in sample:
        if img.embedding:
            embeddings.append(np.asarray(img.embedding, dtype=np.float32))
    if not embeddings:
        logger.warning("silhouette.no_embeddings")
        return []

    emb_matrix = np.vstack(embeddings)
    emb_tensor = torch.from_numpy(emb_matrix).to(device)
    # Embeddings from FashionClassifier are already L2-normalised.
    sims = (emb_tensor @ text_feats.T).cpu().numpy()  # N × K
    # Average similarity per silhouette prompt
    avg = sims.mean(axis=0)

    # Top-K above threshold
    order = avg.argsort()[::-1]
    out: list[str] = []
    for i in order[:top_k]:
        if avg[i] >= threshold:
            out.append(SILHOUETTE_LABELS[i])
    return out
