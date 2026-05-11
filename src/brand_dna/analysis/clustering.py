"""Aesthetic clustering on CLIP embeddings.

We constrain to 3-6 clusters (the dossier spec) rather than letting HDBSCAN
choose freely. Reasoning: a brand strategist reading the report doesn't want
17 micro-clusters; they want a digestible 3-6 aesthetic territories. We pick
the best k inside that range by silhouette score.

For each cluster we surface:
- 3-5 representative images (closest to centroid)
- An LLM-assigned label + description (filled in by the composer, not here)
- Cluster cohesion (centroid distance stddev) as a quality signal
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from brand_dna.core.models import AestheticCluster, ImageRecord
from brand_dna.core.observability import get_logger

logger = get_logger(__name__)


@dataclass
class _ClusterCandidate:
    k: int
    labels: np.ndarray
    centroids: np.ndarray
    silhouette: float


class AestheticClusterer:
    """KMeans with k selected by silhouette score in [k_min, k_max]."""

    def __init__(self, k_min: int = 3, k_max: int = 6, random_state: int = 42) -> None:
        if k_min < 2 or k_max < k_min:
            raise ValueError(f"Invalid k range: [{k_min}, {k_max}]")
        self.k_min = k_min
        self.k_max = k_max
        self.random_state = random_state

    def cluster(self, images: list[ImageRecord]) -> list[AestheticCluster]:
        # Filter to images with embeddings
        embedded = [img for img in images if img.embedding]
        if len(embedded) < self.k_min * 2:
            logger.warning(
                "clustering.too_few_embeddings",
                count=len(embedded),
                min_needed=self.k_min * 2,
            )
            return []

        X = np.vstack([np.asarray(img.embedding, dtype=np.float32) for img in embedded])
        # Normalise (defensive — should already be normalised from CLIP)
        norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
        X = X / norms

        # Try each k, pick the one with highest silhouette
        best: _ClusterCandidate | None = None
        # Bound k_max by number of samples
        effective_k_max = min(self.k_max, max(self.k_min, len(embedded) // 5))
        for k in range(self.k_min, effective_k_max + 1):
            if k >= len(embedded):
                break
            km = KMeans(n_clusters=k, n_init=10, random_state=self.random_state)
            labels = km.fit_predict(X)
            try:
                sil = float(silhouette_score(X, labels, metric="cosine", sample_size=min(2000, len(X))))
            except ValueError:
                sil = -1.0
            cand = _ClusterCandidate(
                k=k, labels=labels, centroids=km.cluster_centers_, silhouette=sil
            )
            logger.debug("clustering.k_eval", k=k, silhouette=sil)
            if best is None or cand.silhouette > best.silhouette:
                best = cand

        if best is None:
            return []

        logger.info("clustering.chose_k", k=best.k, silhouette=best.silhouette)

        # Build AestheticCluster entries
        clusters: list[AestheticCluster] = []
        for cid in range(best.k):
            mask = best.labels == cid
            cluster_indices = np.where(mask)[0]
            if len(cluster_indices) == 0:
                continue

            # Distance to centroid (cosine since we normalised)
            cluster_emb = X[cluster_indices]
            centroid = best.centroids[cid]
            centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-9)
            # Higher sim = closer
            sims = cluster_emb @ centroid_norm
            order = np.argsort(sims)[::-1]
            top = order[:5]
            reps = [embedded[cluster_indices[i]].image_id for i in top]

            # Cohesion = stddev of distance-to-centroid
            distances = 1.0 - sims
            cohesion = float(np.std(distances))

            # Update in-place: tag each image with its cluster_id
            for idx in cluster_indices:
                embedded[idx].cluster_id = cid

            clusters.append(
                AestheticCluster(
                    cluster_id=cid,
                    label=f"Cluster {cid + 1}",  # composer overrides with LLM label
                    description="",  # composer overrides
                    size=int(mask.sum()),
                    representative_image_ids=reps,
                    centroid_distance_stddev=cohesion,
                )
            )
        # Sort by size, largest first — biggest aesthetic territory leads.
        clusters.sort(key=lambda c: -c.size)
        return clusters
