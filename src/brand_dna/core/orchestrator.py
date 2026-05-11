"""Pipeline orchestrator — the single entry point for an end-to-end run.

Stage flow:
    discovery → acquisition → image download → quality filter → fashion filter
    → dedup → analysis (visual + text + audience + clustering) → synthesis
    → PDF render → manifest write

Every stage is wrapped in `time_stage()` so the run report contains
per-stage duration + item counts. Exceptions inside a stage are logged and
the stage is marked degraded — we *always* try to produce a dossier, even
a partial one, rather than crashing.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from brand_dna.acquisition.crawler import BrandCrawler
from brand_dna.acquisition.image_downloader import ImageDownloader
from brand_dna.acquisition.instagram import InstagramScraper
from brand_dna.acquisition.rate_limiter import HostRateLimiter
from brand_dna.analysis.audience import extract_audience_profile
from brand_dna.analysis.clustering import AestheticClusterer
from brand_dna.analysis.color_palette import extract_palette
from brand_dna.analysis.garment_aggregator import (
    aggregate_garments,
    derive_silhouette_summary,
)
from brand_dna.analysis.text_analyzer import analyse_brand_voice
from brand_dna.core.config import AppSettings, BrandConfig, settings
from brand_dna.core.exceptions import BrandDNAError
from brand_dna.core.models import (
    BrandDNADossier,
    ImageRecord,
    Page,
    RunMetadata,
    StageTiming,
)
from brand_dna.core.observability import bind_brand, get_logger, time_stage
from brand_dna.filtering.deduplicator import VisualDeduplicator
from brand_dna.filtering.fashion_classifier import FashionClassifier
from brand_dna.filtering.quality import filter_by_quality
from brand_dna.llm.client import LLMClient, get_llm_client
from brand_dna.storage.image_store import RunWorkspace
from brand_dna.storage.metadata_store import MetadataStore
from brand_dna.synthesis.composer import DossierComposer
from brand_dna.synthesis.pdf_renderer import PDFRenderer

logger = get_logger(__name__)


class Orchestrator:
    """Single-shot orchestrator for one brand run."""

    def __init__(
        self,
        brand_config: BrandConfig,
        app_settings: AppSettings | None = None,
        llm: LLMClient | None = None,
    ) -> None:
        self.brand_config = brand_config
        self.settings = app_settings or settings
        self.llm = llm or get_llm_client()
        self.workspace = RunWorkspace(
            output_root=self.settings.output_dir,
            brand_name=brand_config.name,
        )
        self.run_id = self.workspace.run_id
        self.metadata_store = MetadataStore(self.workspace.metadata_db_path)
        self.started_at = datetime.now(timezone.utc)
        self._stage_timings: list[StageTiming] = []

    async def run(self) -> BrandDNADossier:
        bind_brand(self.brand_config.name, self.run_id)
        self.workspace.init()
        self.metadata_store.connect()

        logger.info(
            "run.start",
            brand=self.brand_config.name,
            url=self.brand_config.url,
            workspace=str(self.workspace.root),
        )

        try:
            return await self._run_stages()
        finally:
            self.metadata_store.close()
            logger.info(
                "run.finish",
                brand=self.brand_config.name,
                run_id=self.run_id,
                workspace=str(self.workspace.root),
            )

    # ─── Stage flow ───────────────────────────────────────────────────────

    async def _run_stages(self) -> BrandDNADossier:
        pages: list[Page] = []
        image_candidates: list[tuple[str, dict[str, Any]]] = []

        # ── 1. Web crawl ─────────────────────────────────────────────────
        rate_limiter = HostRateLimiter(default_delay_ms=self.brand_config.crawl.delay_ms)
        with self._stage("crawl") as timing:
            try:
                async with BrandCrawler(
                    self.brand_config, self.settings.user_agent, rate_limiter
                ) as crawler:
                    crawl_result = await crawler.crawl()
                pages = crawl_result.pages
                image_candidates.extend(crawl_result.image_candidates)
                timing["items"] = len(pages)
                logger.info(
                    "crawl.summary",
                    pages=len(pages),
                    image_candidates=len(crawl_result.image_candidates),
                    fetch_errors=len(crawl_result.fetch_errors),
                )
            except Exception as exc:
                logger.error("crawl.failed", error=str(exc))

        # ── 2. Social (Instagram, best-effort) ──────────────────────────
        with self._stage("social_instagram") as timing:
            ig_handle = self.brand_config.social.get("instagram")
            if ig_handle:
                try:
                    scraper = InstagramScraper(user_agent=self.settings.user_agent)
                    snap = await scraper.fetch_profile(ig_handle)
                    if snap.image_candidates:
                        image_candidates.extend(snap.image_candidates)
                        timing["items"] = len(snap.image_candidates)
                    if snap.blocked:
                        logger.info("instagram.degraded", reason=snap.note)
                except Exception as exc:
                    logger.warning("instagram.error", error=str(exc))
            else:
                logger.info("instagram.skipped", reason="no handle configured")

        # ── 3. Image download ──────────────────────────────────────────
        images: list[ImageRecord] = []
        with self._stage("image_download") as timing:
            downloader = ImageDownloader(
                user_agent=self.settings.user_agent,
                output_dir=self.workspace.images_dir,
                rate_limiter=rate_limiter,
                max_concurrency=max(2, self.brand_config.crawl.max_concurrency * 2),
                min_bytes=self.brand_config.filter.min_bytes,
                max_bytes=self.brand_config.filter.max_bytes,
            )
            images = await downloader.download_all(image_candidates)
            timing["items"] = len(images)

        # ── 4. Quality filter (cheap, runs first) ──────────────────────
        with self._stage("quality_filter") as timing:
            kept, _ = filter_by_quality(
                images,
                min_shorter_side=self.brand_config.filter.min_shorter_side,
                min_bytes=self.brand_config.filter.min_bytes,
                max_bytes=self.brand_config.filter.max_bytes,
            )
            images = kept
            timing["items"] = len(images)

        # ── 5. Fashion classification + embeddings ─────────────────────
        classification_signals: dict[str, Any] = {}
        with self._stage("fashion_classifier") as timing:
            try:
                clf = FashionClassifier(
                    model_id=self.brand_config.analysis.fashion_classifier_model
                )
                images, _ = clf.apply(
                    images,
                    fashion_threshold=self.brand_config.filter.fashion_score_threshold,
                )
                timing["items"] = len(images)
            except BrandDNAError as exc:
                logger.error("fashion_classifier.failed", error=str(exc))

        # ── 6. Visual dedup ────────────────────────────────────────────
        with self._stage("dedup") as timing:
            dedup = VisualDeduplicator(
                phash_hamming_threshold=self.brand_config.filter.phash_hamming_threshold
            )
            images, _ = dedup.dedup(images)
            timing["items"] = len(images)

        # ── 7. Persist pages + images to SQLite ────────────────────────
        with self._stage("persist") as timing:
            try:
                self.metadata_store.insert_pages(pages)
                self.metadata_store.insert_images(images)
                timing["items"] = len(images) + len(pages)
            except Exception as exc:
                logger.warning("persist.failed", error=str(exc))

        # ── 8. Visual analysis (color, garments, silhouettes) ──────────
        with self._stage("color_palette") as timing:
            palette = extract_palette(images, k=self.brand_config.analysis.palette_k)
            timing["items"] = len(palette.entries)

        with self._stage("garment_aggregate") as timing:
            garments = aggregate_garments(images)
            timing["items"] = garments.sample_size

        with self._stage("silhouette") as timing:
            try:
                silhouettes = derive_silhouette_summary(
                    images,
                    model_id=self.brand_config.analysis.fashion_classifier_model,
                )
            except Exception as exc:
                logger.warning("silhouette.failed", error=str(exc))
                silhouettes = []
            timing["items"] = len(silhouettes)

        # ── 9. Aesthetic clustering ────────────────────────────────────
        with self._stage("clustering") as timing:
            clusterer = AestheticClusterer(
                k_min=self.brand_config.analysis.n_aesthetic_clusters_min,
                k_max=self.brand_config.analysis.n_aesthetic_clusters_max,
            )
            clusters = clusterer.cluster(images)
            timing["items"] = len(clusters)
            # Re-persist images now that cluster_ids are set.
            try:
                self.metadata_store.insert_images(images)
            except Exception:
                pass

        # ── 10. Text analysis (brand voice) ────────────────────────────
        with self._stage("brand_voice") as timing:
            try:
                voice, voice_chars = await analyse_brand_voice(
                    pages,
                    llm=self.llm,
                    model=self.brand_config.model_for("primary"),
                    brand_name=self.brand_config.name,
                )
            except Exception as exc:
                logger.error("brand_voice.failed", error=str(exc))
                from brand_dna.core.models import BrandVoice
                voice = BrandVoice()
                voice_chars = 0
            timing["items"] = voice_chars

        # ── 11. Audience profile (multi-modal) ─────────────────────────
        with self._stage("audience") as timing:
            try:
                audience, audience_telemetry = await extract_audience_profile(
                    pages,
                    images,
                    clusters,
                    llm=self.llm,
                    model=self.brand_config.model_for("primary"),
                    brand_name=self.brand_config.name,
                )
            except Exception as exc:
                logger.error("audience.failed", error=str(exc))
                from brand_dna.core.models import AudienceProfile
                audience = AudienceProfile()
                audience_telemetry = {"corpus_chars": 0, "n_images_sampled": 0}
            timing["items"] = audience_telemetry.get("corpus_chars", 0)

        # ── 12. Compose dossier ────────────────────────────────────────
        composer = DossierComposer(
            llm=self.llm,
            model_synthesis=self.brand_config.model_for("synthesis"),
            model_primary=self.brand_config.model_for("primary"),
        )

        with self._stage("cluster_labels") as timing:
            try:
                clusters = await composer.label_clusters(
                    clusters, images, self.brand_config.name
                )
            except Exception as exc:
                logger.error("cluster_labels.failed", error=str(exc))
            timing["items"] = len(clusters)

        # Build run metadata so the composer can include it in the dossier.
        run_metadata = self._build_run_metadata(
            images_acquired=len([i for i in images if i.quality_passed]),
            images_after_filter=len(images),
            pages_crawled=len(pages),
        )

        signal_strengths = {
            "color_palette": {"sample_size": palette.sample_size},
            "garments": {"sample_size": garments.sample_size},
            "clusters": {
                "sample_size": sum(c.size for c in clusters),
                "chosen_k": len(clusters),
            },
            "brand_voice": {"corpus_chars": voice_chars},
            "audience": audience_telemetry,
        }

        with self._stage("synthesis") as timing:
            try:
                dossier = await composer.synthesise(
                    brand_name=self.brand_config.name,
                    brand_url=self.brand_config.url,
                    social_handles=self.brand_config.social,
                    palette=palette,
                    garments=garments,
                    silhouettes=silhouettes,
                    clusters=clusters,
                    voice=voice,
                    audience=audience,
                    images=images,
                    signal_strengths=signal_strengths,
                    run_metadata=run_metadata,
                )
            except Exception as exc:
                logger.error("synthesis.failed", error=str(exc))
                raise
            timing["items"] = 1

        # Finalise run metadata after synthesis (composer needed it earlier, but
        # the final tokens-in/out aren't known until now).
        dossier.run_metadata = self._build_run_metadata(
            images_acquired=run_metadata.images_acquired,
            images_after_filter=run_metadata.images_after_filter,
            pages_crawled=run_metadata.pages_crawled,
        )

        # ── 13. Write JSON manifests ───────────────────────────────────
        with self._stage("write_manifests") as timing:
            self.workspace.dossier_json_path.write_text(
                json.dumps(dossier.to_manifest(), indent=2, default=str),
                encoding="utf-8",
            )
            self.workspace.train_manifest_path.write_text(
                json.dumps(
                    dossier.train_modules.model_dump(mode="json"), indent=2, default=str
                ),
                encoding="utf-8",
            )
            timing["items"] = 2

        # ── 14. Render PDF ─────────────────────────────────────────────
        with self._stage("pdf_render") as timing:
            renderer = PDFRenderer()
            try:
                renderer.render_pdf(
                    dossier, images, self.workspace.dossier_pdf_path
                )
                timing["items"] = 1
            except Exception as exc:
                logger.error("pdf_render.failed", error=str(exc))

        # ── 15. Write run report ──────────────────────────────────────
        report = {
            "run_id": self.run_id,
            "brand": self.brand_config.name,
            "started_at": self.started_at.isoformat(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "stages": [s.model_dump(mode="json") for s in self._stage_timings],
            "llm_usage": {
                "calls": self.llm.ledger.calls,
                "tokens_in": self.llm.ledger.tokens_in,
                "tokens_out": self.llm.ledger.tokens_out,
                "cost_usd": self.llm.ledger.cost_usd,
            },
            "images_acquired": dossier.run_metadata.images_acquired,
            "images_after_filter": dossier.run_metadata.images_after_filter,
            "pages_crawled": dossier.run_metadata.pages_crawled,
            "workspace": str(self.workspace.root),
        }
        self.workspace.report_path.write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        logger.info("run.report_written", path=str(self.workspace.report_path))

        return dossier

    # ─── Helpers ──────────────────────────────────────────────────────────

    def _build_run_metadata(
        self,
        *,
        images_acquired: int,
        images_after_filter: int,
        pages_crawled: int,
    ) -> RunMetadata:
        now = datetime.now(timezone.utc)
        total = (now - self.started_at).total_seconds()
        return RunMetadata(
            run_id=self.run_id,
            brand_name=self.brand_config.name,
            started_at=self.started_at,
            finished_at=now,
            total_duration_s=total,
            stages=list(self._stage_timings),
            images_acquired=images_acquired,
            images_after_filter=images_after_filter,
            pages_crawled=pages_crawled,
            llm_tokens_in=self.llm.ledger.tokens_in,
            llm_tokens_out=self.llm.ledger.tokens_out,
            estimated_cost_usd=self.llm.ledger.cost_usd,
        )

    def _stage(self, name: str):
        """Wraps time_stage() and appends a StageTiming to the run record."""
        outer = self
        ctx = time_stage(name, logger)

        class _StageRecorder:
            def __enter__(self) -> dict:
                self._payload = ctx.__enter__()
                self._start = datetime.now(timezone.utc)
                return self._payload

            def __exit__(self, exc_type, exc, tb) -> None:
                ctx.__exit__(exc_type, exc, tb)
                duration = (datetime.now(timezone.utc) - self._start).total_seconds()
                outer._stage_timings.append(
                    StageTiming(
                        stage=name,
                        duration_s=duration,
                        items_processed=self._payload.get("items", 0),
                    )
                )

        return _StageRecorder()


async def run_brand(brand_config: BrandConfig) -> BrandDNADossier:
    """Convenience wrapper used by the CLI."""
    orchestrator = Orchestrator(brand_config)
    return await orchestrator.run()
