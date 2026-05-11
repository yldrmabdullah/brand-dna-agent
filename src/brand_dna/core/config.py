"""Configuration: environment-level settings + per-brand YAML configs.

Two layers:
- `AppSettings`: shared, env-driven (API keys, defaults).
- `BrandConfig`: per-brand YAML — what makes onboarding a new brand a YAML edit,
  not a code change.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from brand_dna.core.exceptions import ConfigurationError


# ─── App-level settings (env-driven) ──────────────────────────────────────


class AppSettings(BaseSettings):
    """Process-wide settings. Read from environment + .env file."""

    model_config = SettingsConfigDict(
        env_prefix="BRAND_DNA_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # OpenRouter
    openrouter_api_key: str = Field(default="", validation_alias="OPENROUTER_API_KEY")
    openrouter_referer: str = Field(default="", validation_alias="OPENROUTER_REFERER")
    openrouter_app_title: str = Field(
        default="Brand DNA Agent", validation_alias="OPENROUTER_APP_TITLE"
    )

    # Models — overridable per-brand in YAML
    model_primary: str = "anthropic/claude-sonnet-4.5"
    model_fast: str = "google/gemini-2.5-flash"
    model_synthesis: str = "anthropic/claude-opus-4.1"

    # Runtime
    log_level: str = "INFO"
    log_format: str = "json"  # "json" | "console"
    output_dir: Path = Path("./outputs")
    cache_dir: Path = Path("./data/cache")

    # Crawling politeness
    user_agent: str = "BrandDNA-Agent/0.1 (+contact@example.com)"
    max_concurrency: int = 4
    request_timeout_s: int = 20
    delay_between_requests_ms: int = 400

    # Playwright
    enable_playwright: bool = False


_settings: AppSettings | None = None


def get_settings() -> AppSettings:
    """Lazy global. Test code can call `_settings = None` to reset."""
    global _settings
    if _settings is None:
        _settings = AppSettings()
    return _settings


# Public read-only alias used elsewhere
settings = get_settings()


# ─── Per-brand configuration (YAML-driven) ────────────────────────────────


class CrawlConfig(BaseModel):
    """How politely to crawl this specific brand."""

    max_pages: int = 200
    max_images: int = 400
    max_concurrency: int = 4
    delay_ms: int = 400
    request_timeout_s: int = 20
    render_js: bool = False
    """Use Playwright. Default off — most fashion sites work with static HTML
    once you parse JSON-LD + OpenGraph."""

    respect_robots_txt: bool = True
    follow_pagination: bool = True
    allowed_subdomains: list[str] = Field(default_factory=list)
    """Empty = same registered domain. Add subdomains like 'shop.brand.com'."""

    page_type_hints: dict[str, list[str]] = Field(default_factory=dict)
    """Optional URL pattern hints per page_type. Used to bias the classifier;
    NOT a hardcoded selector. Example:
        product: ["/products/", "/p/"]
        about: ["/about", "/our-story"]
    These are *hints*, the classifier still works without them."""


class FilterConfig(BaseModel):
    """Image filtering thresholds. Defaults are production-tuned."""

    min_shorter_side: int = 512
    """Justification: 512 is the floor for CLIP ViT-L/14 (224 input is too lossy
    for downstream gen models like SDXL which expect ≥512). Going lower would
    cap the dossier's downstream usefulness."""

    min_bytes: int = 10_000
    """Reject tiny tracking pixels / placeholders."""

    max_bytes: int = 15_000_000
    """Reject 4K hero images that bloat storage with no aesthetic gain."""

    fashion_score_threshold: float = 0.55
    """FashionCLIP confidence floor. Tuned empirically on COS+Les Benjamins."""

    phash_hamming_threshold: int = 5
    """Lower = stricter dedup. 5/64 catches resized + lightly recolored dupes."""

    target_image_count: int = 100
    """The case requires ≥100 — we ask for ≥120 from acquisition to leave
    headroom for filtering rejections."""


class AnalysisConfig(BaseModel):
    palette_k: int = 8
    """Number of color clusters. 8 is a sweet spot: enough granularity to
    distinguish a brand's accent colors from its core neutrals."""
    palette_color_space: str = "lab"
    """LAB is perceptually uniform — Euclidean distance ≈ perceived difference."""

    n_aesthetic_clusters_min: int = 3
    n_aesthetic_clusters_max: int = 6
    """Refabric's brand strategist spec: 3-6 clusters max. Forced range
    even if HDBSCAN would prefer more — keeps the dossier readable."""

    embedding_model: str = "openai/clip-vit-base-patch32"
    """Vision encoder. Default is OpenAI's base CLIP (stable, broadly available).
    Can be set to `patrickjohncyh/fashion-clip` per-brand for the fashion-tuned
    variant — sanity-check fallback kicks in if those weights are unavailable."""

    fashion_classifier_model: str = "openai/clip-vit-base-patch32"


class ModelOverrides(BaseModel):
    """Per-brand model overrides (rare — usually defaults are fine)."""

    primary: str | None = None
    fast: str | None = None
    synthesis: str | None = None


class BrandConfig(BaseModel):
    """A single brand's onboarding config. This is *the* file that gets edited
    to onboard a new brand. No code change required."""

    # Identity
    name: str
    url: str
    social: dict[str, str] = Field(default_factory=dict)
    """E.g., {'instagram': 'cosstores', 'pinterest': 'cosstores'}."""

    # Optional context the strategist already has
    known_categories: list[str] = Field(default_factory=list)
    """Optional. If provided, biases the garment classifier toward these.
    E.g., ['outerwear', 'knitwear', 'denim'] for a denim-led brand."""

    seed_pages: list[str] = Field(default_factory=list)
    """Optional explicit pages to crawl in addition to sitemap discovery.
    Useful when the lookbook lives at a non-standard URL."""

    notes: str = ""
    """Free-form notes. Surfaced in the run log only, not the dossier."""

    # Behavior knobs
    crawl: CrawlConfig = Field(default_factory=CrawlConfig)
    filter: FilterConfig = Field(default_factory=FilterConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    models: ModelOverrides = Field(default_factory=ModelOverrides)

    @field_validator("url")
    @classmethod
    def _validate_url(cls, v: str) -> str:
        if not v.startswith(("http://", "https://")):
            raise ValueError("brand.url must include scheme (https://...)")
        return v.rstrip("/")

    def model_for(self, role: str) -> str:
        """Returns the chosen model id for a given role, honoring overrides."""
        defaults = {
            "primary": settings.model_primary,
            "fast": settings.model_fast,
            "synthesis": settings.model_synthesis,
        }
        override = getattr(self.models, role, None)
        return override or defaults[role]


def load_brand_config(path: str | Path) -> BrandConfig:
    """Load a brand YAML config from disk."""
    p = Path(path)
    if not p.exists():
        raise ConfigurationError(f"Brand config not found: {p}")
    try:
        raw: dict[str, Any] = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Invalid YAML in {p}: {exc}") from exc

    try:
        return BrandConfig(**raw)
    except Exception as exc:
        raise ConfigurationError(f"Invalid brand config {p}: {exc}") from exc
