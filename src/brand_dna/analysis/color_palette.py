"""Dominant color palette extraction in LAB space.

Why LAB and not RGB:
- LAB is perceptually uniform: Euclidean distance ≈ human-perceived color
  difference. KMeans in RGB clusters by *encoding* similarity, not perception,
  and ends up grouping warm beige with mid-grey because their RGB centroids
  happen to be close.
- The case explicitly asks for a brand's "color logic" — that's a perceptual
  concept, so we should compute it in perceptual space.

For each cluster we also compute the nearest Pantone PMS for designer-friendly
output. The Pantone matching is approximate (we use a small built-in reference
table, not the licensed Pantone library) but good enough for the dossier's
"warm beige ≈ PMS 7501" cues.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.color import lab2rgb, rgb2lab
from sklearn.cluster import KMeans

from brand_dna.analysis._pantone import nearest_pantone
from brand_dna.core.models import ColorEntry, ColorPalette, ImageRecord
from brand_dna.core.observability import get_logger

logger = get_logger(__name__)

_DOWNSAMPLE_SIZE = 128  # px per side — color stats are robust at low res
_PIXELS_PER_IMAGE = 800  # subsample stride
_MAX_TOTAL_PIXELS = 200_000  # cap to keep memory bounded


def _load_pixels_lab(image_path: str | Path) -> np.ndarray:
    """Returns N×3 LAB pixel array, downsampled."""
    img = Image.open(image_path).convert("RGB")
    img.thumbnail((_DOWNSAMPLE_SIZE, _DOWNSAMPLE_SIZE), Image.Resampling.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    lab = rgb2lab(arr).reshape(-1, 3)
    if lab.shape[0] > _PIXELS_PER_IMAGE:
        idx = np.random.default_rng(42).choice(
            lab.shape[0], _PIXELS_PER_IMAGE, replace=False
        )
        lab = lab[idx]
    return lab.astype(np.float32)


def _lab_to_hex_rgb(lab: np.ndarray) -> tuple[str, tuple[int, int, int]]:
    """LAB centroid → hex + (r,g,b) ints."""
    rgb = lab2rgb(lab.reshape(1, 1, 3)).reshape(3)
    rgb = (np.clip(rgb, 0.0, 1.0) * 255.0).round().astype(int)
    r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
    return f"#{r:02X}{g:02X}{b:02X}", (r, g, b)


def extract_palette(
    images: list[ImageRecord],
    *,
    k: int = 8,
    drop_extreme_lightness: bool = True,
) -> ColorPalette:
    """KMeans palette over pooled pixels from all images."""
    if not images:
        return ColorPalette(entries=[], sample_size=0)

    all_pixels: list[np.ndarray] = []
    for rec in images:
        try:
            all_pixels.append(_load_pixels_lab(rec.local_path))
        except Exception as exc:
            logger.debug("palette.image_skip", path=rec.local_path, error=str(exc))
            continue

    if not all_pixels:
        return ColorPalette(entries=[], sample_size=0)

    pixels = np.vstack(all_pixels)
    if pixels.shape[0] > _MAX_TOTAL_PIXELS:
        idx = np.random.default_rng(42).choice(
            pixels.shape[0], _MAX_TOTAL_PIXELS, replace=False
        )
        pixels = pixels[idx]

    # Optionally drop near-pure-white (studio backdrop) and near-black to surface
    # the brand's *actual* color story rather than the photography style.
    if drop_extreme_lightness:
        mask = (pixels[:, 0] > 5.0) & (pixels[:, 0] < 96.0)
        if mask.sum() > 1000:  # keep filter only if it leaves enough data
            pixels = pixels[mask]

    n_clusters = min(k, max(2, pixels.shape[0] // 1000))
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    km.fit(pixels)

    # Sort clusters by size, descending — biggest cluster = dominant color
    counts = np.bincount(km.labels_, minlength=n_clusters)
    order = counts.argsort()[::-1]
    centroids = km.cluster_centers_[order]
    counts = counts[order]
    total = counts.sum()

    entries: list[ColorEntry] = []
    for i, centroid in enumerate(centroids):
        hex_, rgb = _lab_to_hex_rgb(centroid)
        pantone = nearest_pantone(rgb)
        entries.append(
            ColorEntry(
                hex=hex_,
                rgb=rgb,
                lab=(float(centroid[0]), float(centroid[1]), float(centroid[2])),
                percentage=float(counts[i]) / float(total) * 100.0,
                nearest_pantone=pantone,
                descriptor=None,  # LLM fills in later (in composer)
            )
        )

    logger.info(
        "palette.extracted",
        n_clusters=n_clusters,
        sample_pixels=pixels.shape[0],
        n_images=len(images),
    )
    return ColorPalette(
        entries=entries,
        extraction_method=f"kmeans_lab_k={n_clusters}",
        sample_size=len(images),
    )
