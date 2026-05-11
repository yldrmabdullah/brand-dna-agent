"""Pydantic data models — the contract between every layer of the agent.

These models are deliberately verbose: each carries provenance (where it came
from) and confidence (how sure we are). The PDF rendering layer consumes these
directly, and the JSON manifest is just `.model_dump(mode="json")`.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ─── Page-level models ────────────────────────────────────────────────────


class PageType(str, Enum):
    """Heuristic classification of a discovered page. Used to weight extraction."""

    PRODUCT = "product"
    COLLECTION = "collection"
    LOOKBOOK = "lookbook"
    EDITORIAL = "editorial"
    ABOUT = "about"
    BLOG = "blog"
    PRESS = "press"
    FAQ = "faq"
    HOMEPAGE = "homepage"
    SOCIAL_POST = "social_post"
    UNKNOWN = "unknown"


class Page(BaseModel):
    """A crawled page with extracted content."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    url: str
    page_type: PageType = PageType.UNKNOWN
    title: str | None = None
    body_text: str = ""
    meta_description: str | None = None
    structured_data: list[dict[str, Any]] = Field(default_factory=list)
    opengraph: dict[str, str] = Field(default_factory=dict)
    image_urls: list[str] = Field(default_factory=list)
    image_urls_with_alt: list[tuple[str, str | None]] = Field(default_factory=list)
    discovered_links: list[str] = Field(default_factory=list)
    fetched_at: datetime = Field(default_factory=_utcnow)
    http_status: int = 0


# ─── Image models ─────────────────────────────────────────────────────────


class ImageProvenance(BaseModel):
    """Where this image came from — surfaced in the PDF and JSON manifest."""

    source_url: str
    page_url: str
    page_type: PageType = PageType.UNKNOWN
    alt_text: str | None = None
    surrounding_text: str | None = None
    product_name: str | None = None
    captured_at: datetime = Field(default_factory=_utcnow)


class ImageRecord(BaseModel):
    """A single image after acquisition. Filtering/analysis enrich this in place."""

    image_id: str
    """Stable content-addressed ID — sha256 of bytes, first 16 chars."""

    local_path: str
    """Filesystem path under data store. Relative to the run output dir."""

    width: int
    height: int
    format: str
    bytes_size: int
    phash: str | None = None
    """Perceptual hash — for visual dedup."""

    provenance: ImageProvenance

    # Filtering signals (populated by filtering layer)
    fashion_score: float | None = None
    """0..1 — confidence this image is fashion/clothing content."""
    quality_passed: bool = True
    rejection_reason: str | None = None

    # Analysis signals (populated by analysis layer)
    garment_labels: list[str] = Field(default_factory=list)
    silhouette_tags: list[str] = Field(default_factory=list)
    embedding: list[float] | None = None
    """CLIP embedding — only kept in-memory typically, not in manifest."""
    cluster_id: int | None = None

    @property
    def shorter_side(self) -> int:
        return min(self.width, self.height)


# ─── Color models ─────────────────────────────────────────────────────────


class ColorEntry(BaseModel):
    """One palette entry, with the multi-space representation a designer expects."""

    hex: str
    rgb: tuple[int, int, int]
    lab: tuple[float, float, float]
    percentage: float
    nearest_pantone: str | None = None
    descriptor: str | None = None  # "warm beige", "deep navy" — LLM-assigned


class ColorPalette(BaseModel):
    entries: list[ColorEntry]
    extraction_method: str = "kmeans_lab_k=8"
    sample_size: int


# ─── Garment / silhouette / aesthetic ─────────────────────────────────────


class GarmentDistribution(BaseModel):
    counts: dict[str, int] = Field(default_factory=dict)
    percentages: dict[str, float] = Field(default_factory=dict)
    sample_size: int = 0


class AestheticCluster(BaseModel):
    cluster_id: int
    label: str  # "Tailored Minimalism", "Streetwear Heritage" — LLM-assigned
    description: str
    size: int
    representative_image_ids: list[str]  # 3-6 best representatives
    centroid_distance_stddev: float | None = None  # cluster cohesion signal


# ─── Text identity ────────────────────────────────────────────────────────


class BrandVoice(BaseModel):
    tone_descriptors: list[str] = Field(default_factory=list)
    """e.g., ["confident", "understated", "playful"]"""
    recurring_vocabulary: list[str] = Field(default_factory=list)
    stated_values: list[str] = Field(default_factory=list)
    positioning_statement: str | None = None
    representative_quotes: list[str] = Field(default_factory=list)


class AudienceProfile(BaseModel):
    demographic_cues: list[str] = Field(default_factory=list)
    psychographic_cues: list[str] = Field(default_factory=list)
    aspirational_signals: list[str] = Field(default_factory=list)
    price_positioning: str | None = None  # "accessible luxury", "mass market", ...
    evidence_snippets: list[str] = Field(default_factory=list)


# ─── Confidence + Provenance (cross-cutting) ──────────────────────────────


class Confidence(BaseModel):
    """Attaches to any analysis output. Surfaced in PDF and JSON.

    `score` is a heuristic 0..1 combining sample size, signal consistency,
    and (for LLM outputs) self-eval. Calibration is approximate — these are
    for prioritisation, not statistical claims.
    """

    score: float = Field(ge=0.0, le=1.0)
    sample_size: int
    method: str
    notes: str | None = None


class SourceRef(BaseModel):
    """One piece of evidence supporting a claim."""

    kind: str  # "image" | "page" | "social_post"
    ref: str  # image_id, page URL, etc.
    snippet: str | None = None


class ProvenanceTrail(BaseModel):
    """Maps any claim or finding back to its sources."""

    claim: str
    sources: list[SourceRef]


# ─── Refabric Train Module mapping (value-add) ────────────────────────────


class TrainLookModule(BaseModel):
    """Maps onto Refabric's `Train Look`: past collections + best-sellers + trends."""

    representative_image_ids: list[str]
    aesthetic_summary: str
    seasonal_signals: list[str] = Field(default_factory=list)


class TrainMoodModule(BaseModel):
    """Maps onto Refabric's `Train Mood`: structures, objects, color schemes."""

    mood_descriptors: list[str]
    emotional_atmosphere: str
    color_palette: ColorPalette


class TrainAttributeModule(BaseModel):
    """Maps onto Refabric's `Train Attribute`: stitching, embellishments, silhouettes."""

    silhouettes: list[str]
    construction_details: list[str]
    embellishments: list[str]


class TrainFabricModule(BaseModel):
    """Maps onto Refabric's `Train Fabric`: texture, durability, comfort."""

    detected_materials: list[str]
    texture_descriptors: list[str]
    sustainability_signals: list[str] = Field(default_factory=list)


class TrainPatternModule(BaseModel):
    """Maps onto Refabric's `Train Pattern`: geometric/floral motifs."""

    pattern_types: list[str]
    motif_descriptors: list[str]


class TrainModuleManifest(BaseModel):
    """The full mapping. Ingest-ready for Refabric's training pipelines.

    Note: this is our *value-add* layer — it doesn't replace their Train inputs,
    it primes them with extracted defaults so a brand strategist starts at 70%
    rather than 0%.
    """

    look: TrainLookModule
    mood: TrainMoodModule
    attribute: TrainAttributeModule
    fabric: TrainFabricModule
    pattern: TrainPatternModule


# ─── Run metadata + final dossier ─────────────────────────────────────────


class StageTiming(BaseModel):
    stage: str
    duration_s: float
    items_processed: int = 0
    notes: str | None = None


class RunMetadata(BaseModel):
    run_id: str
    brand_name: str
    started_at: datetime
    finished_at: datetime | None = None
    total_duration_s: float | None = None
    stages: list[StageTiming] = Field(default_factory=list)
    images_acquired: int = 0
    images_after_filter: int = 0
    pages_crawled: int = 0
    llm_tokens_in: int = 0
    llm_tokens_out: int = 0
    estimated_cost_usd: float = 0.0
    errors: list[str] = Field(default_factory=list)


class BrandDNADossier(BaseModel):
    """The complete output. Renders to both PDF (humans) and JSON (machines)."""

    # Identity
    brand_name: str
    brand_url: str
    social_handles: dict[str, str] = Field(default_factory=dict)

    # Visual identity
    color_palette: ColorPalette
    garment_distribution: GarmentDistribution
    aesthetic_clusters: list[AestheticCluster]
    silhouette_summary: list[str] = Field(default_factory=list)
    styling_cues: list[str] = Field(default_factory=list)

    # Textual identity
    brand_voice: BrandVoice

    # Audience
    audience: AudienceProfile

    # Refabric-aligned ingest manifest (value-add)
    train_modules: TrainModuleManifest

    # Cross-cutting
    confidences: dict[str, Confidence] = Field(default_factory=dict)
    """Keyed by section name: 'color_palette', 'brand_voice', etc."""
    provenance: list[ProvenanceTrail] = Field(default_factory=list)
    """Each major claim mapped back to its evidence."""

    # Top-level summary (LLM-composed)
    executive_summary: str
    one_line_positioning: str

    # Run telemetry
    run_metadata: RunMetadata

    def to_manifest(self) -> dict[str, Any]:
        """JSON-serialisable dump. Suitable for downstream ingestion."""
        return self.model_dump(mode="json", exclude_none=False)
