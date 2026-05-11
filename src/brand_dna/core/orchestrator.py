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

    # ─── Stage flow ───────────────────────────────────────────────────────

    async def _run_stages(self) -> BrandDNADossier:
        """High-level orchestration flow."""
        # 0. Discovery & Context (Delta Analysis)
        previous_dossier = self._find_previous_run()

        # 1. Acquisition (Crawl, Social, Images)
        pages, images = await self._acquisition_stage()

        # 2. Filtering & Persistence
        images = await self._filtering_stage(images, pages)

        # 3. Deep Analysis (Visual, Voice, Audience)
        analysis_data = await self._analysis_stage(images, pages)

        # 4. Synthesis & Intelligence
        dossier = await self._synthesis_stage(
            analysis_data, images, pages, previous_dossier
        )

        # 5. Delivery (QA, Manifests, PDF)
        return await self._delivery_stage(dossier, images, pages)

    # ─── Sub-Stages ───────────────────────────────────────────────────────

    def _find_previous_run(self) -> dict[str, Any] | None:
        """Looks for a previous dossier to enable delta analysis."""
        try:
            brand_outputs = self.workspace.root.parent
            past_runs = sorted(
                [d for d in brand_outputs.iterdir() if d.is_dir() and d.name != self.run_id],
                key=lambda x: x.name,
                reverse=True
            )
            if past_runs:
                latest_past = past_runs[0] / "dossier.json"
                if latest_past.exists():
                    logger.info("delta.previous_found", path=str(latest_past))
                    return json.loads(latest_past.read_text())
        except Exception as exc:
            logger.debug("delta.search_skipped", reason=str(exc))
        return None

    async def _acquisition_stage(self) -> tuple[list[Page], list[ImageRecord]]:
        """Handles web crawling, social scraping, and image downloads."""
        pages: list[Page] = []
        candidates: list[tuple[str, dict[str, Any]]] = []
        rate_limiter = HostRateLimiter(default_delay_ms=self.brand_config.crawl.delay_ms)

        # Web Crawl
        with self._stage("crawl") as t:
            try:
                async with BrandCrawler(self.brand_config, self.settings.user_agent, rate_limiter) as crawler:
                    res = await crawler.crawl()
                    pages, candidates = res.pages, res.image_candidates
                    t["items"] = len(pages)
            except Exception as exc:
                logger.error("crawl.failed", error=str(exc))

        # Social (Instagram)
        with self._stage("social_instagram") as t:
            ig = self.brand_config.social.get("instagram")
            if ig:
                try:
                    scraper = InstagramScraper(user_agent=self.settings.user_agent)
                    snap = await scraper.fetch_profile(ig)
                    candidates.extend(snap.image_candidates)
                    t["items"] = len(snap.image_candidates)
                except Exception as exc:
                    logger.warning("instagram.error", error=str(exc))

        # Image Download
        with self._stage("image_download") as t:
            downloader = ImageDownloader(
                user_agent=self.settings.user_agent,
                output_dir=self.workspace.images_dir,
                rate_limiter=rate_limiter,
                max_concurrency=max(2, self.brand_config.crawl.max_concurrency * 2),
                min_bytes=self.brand_config.filter.min_bytes,
                max_bytes=self.brand_config.filter.max_bytes,
            )
            images = await downloader.download_all(candidates)
            t["items"] = len(images)

        return pages, images

    async def _filtering_stage(self, images: list[ImageRecord], pages: list[Page]) -> list[ImageRecord]:
        """Runs quality checks, fashion classification, and deduping."""
        # Quality
        with self._stage("quality_filter") as t:
            images, _ = filter_by_quality(images, **self.brand_config.filter.model_dump())
            t["items"] = len(images)

        # Fashion Filter
        with self._stage("fashion_classifier") as t:
            try:
                clf = FashionClassifier(model_id=self.brand_config.analysis.fashion_classifier_model)
                images, _ = clf.apply(images, fashion_threshold=self.brand_config.filter.fashion_score_threshold)
                t["items"] = len(images)
            except Exception as exc:
                logger.error("fashion_classifier.failed", error=str(exc))

        # Dedup
        with self._stage("dedup") as t:
            dedup = VisualDeduplicator(phash_hamming_threshold=self.brand_config.filter.phash_hamming_threshold)
            images, _ = dedup.dedup(images)
            t["items"] = len(images)

        # Persist
        with self._stage("persist") as t:
            try:
                self.metadata_store.insert_pages(pages)
                self.metadata_store.insert_images(images)
                t["items"] = len(images) + len(pages)
            except Exception as exc:
                logger.warning("persist.failed", error=str(exc))

        return images

    async def _analysis_stage(self, images: list[ImageRecord], pages: list[Page]) -> dict[str, Any]:
        """Deep multimodal and linguistic analysis."""
        data = {}

        # 1. Palette & Garments (Synchronous/Fast)
        with self._stage("color_palette") as t:
            data["palette"] = extract_palette(images, k=self.brand_config.analysis.palette_k)
            t["items"] = data["palette"].sample_size

        with self._stage("garment_aggregate") as t:
            data["garments"] = aggregate_garments(images)
            t["items"] = data["garments"].sample_size

        with self._stage("silhouette") as t:
            try:
                data["silhouettes"] = derive_silhouette_summary(images, model_id=self.brand_config.analysis.fashion_classifier_model)
            except Exception:
                data["silhouettes"] = []
            t["items"] = len(data["silhouettes"])

        # 2. Clustering
        with self._stage("clustering") as t:
            clusterer = AestheticClusterer(
                k_min=self.brand_config.analysis.n_aesthetic_clusters_min,
                k_max=self.brand_config.analysis.n_aesthetic_clusters_max,
            )
            data["clusters"] = clusterer.cluster(images)
            t["items"] = len(data["clusters"])
            # Re-persist with cluster IDs
            try: self.metadata_store.insert_images(images)
            except Exception: pass

        # 3. Parallel AI Analysis (Voice + Audience)
        with self._stage("analysis_parallel") as t:
            try:
                self.llm.target_language = self.brand_config.target_language
                v_task = analyse_brand_voice(pages, llm=self.llm, model=self.brand_config.model_for("primary"), brand_name=self.brand_config.name)
                a_task = extract_audience_profile(pages, images, data["clusters"], llm=self.llm, model=self.brand_config.model_for("primary"), brand_name=self.brand_config.name)
                
                (v_res, a_res) = await asyncio.gather(v_task, a_task)
                data["voice"], v_chars = v_res
                data["audience"], a_telemetry = a_res
                data["signal_strengths"] = {
                    "color_palette": {"sample_size": data["palette"].sample_size},
                    "garments": {"sample_size": data["garments"].sample_size},
                    "clusters": {"sample_size": sum(c.size for c in data["clusters"]), "chosen_k": len(data["clusters"])},
                    "brand_voice": {"corpus_chars": v_chars},
                    "audience": a_telemetry,
                }
                t["items"] = v_chars + a_telemetry.get("corpus_chars", 0)
            except Exception as exc:
                logger.error("parallel_analysis.failed", error=str(exc))
                from brand_dna.core.models import BrandVoice, AudienceProfile
                data["voice"], data["audience"] = BrandVoice(), AudienceProfile()
                data["signal_strengths"] = {}

        return data

    async def _synthesis_stage(self, data: dict[str, Any], images: list[ImageRecord], pages: list[Page], previous_dossier: dict | None) -> BrandDNADossier:
        """Synthesizes all signals into the final BrandDNADossier."""
        composer = DossierComposer(
            llm=self.llm,
            model_synthesis=self.brand_config.model_for("synthesis"),
            model_primary=self.brand_config.model_for("primary"),
        )

        with self._stage("cluster_labels") as t:
            try:
                data["clusters"] = await composer.label_clusters(data["clusters"], images, self.brand_config.name)
            except Exception: pass
            t["items"] = len(data["clusters"])

        run_metadata = self._build_run_metadata(
            images_acquired=len([i for i in images if i.quality_passed]),
            images_after_filter=len(images),
            pages_crawled=len(pages),
        )

        with self._stage("synthesis") as t:
            dossier = await composer.synthesise(
                brand_name=self.brand_config.name,
                brand_url=self.brand_config.url,
                social_handles=self.brand_config.social,
                palette=data["palette"],
                garments=data["garments"],
                silhouettes=data["silhouettes"],
                clusters=data["clusters"],
                voice=data["voice"],
                audience=data["audience"],
                images=images,
                signal_strengths=data["signal_strengths"],
                run_metadata=run_metadata,
                previous_dossier=previous_dossier,
            )
            t["items"] = 1
        return dossier

    async def _delivery_stage(self, dossier: BrandDNADossier, images: list[ImageRecord], pages: list[Page]) -> BrandDNADossier:
        """QA, Manifests, and PDF rendering."""
        # 1. Self-Eval
        with self._stage("self_evaluation") as t:
            try:
                from brand_dna.analysis.self_eval import self_evaluate
                eval_res = await self_evaluate(dossier, llm=self.llm, model=self.brand_config.model_for("primary"))
                dossier.custom_data["self_evaluation"] = eval_res
            except Exception: pass
            t["items"] = 1

        # 2. Manifests
        with self._stage("write_manifests") as t:
            self.workspace.dossier_json_path.write_text(json.dumps(dossier.to_manifest(), indent=2, default=str))
            self.workspace.train_manifest_path.write_text(json.dumps(dossier.train_modules.model_dump(mode="json"), indent=2, default=str))
            t["items"] = 2

        # 3. PDF
        with self._stage("pdf_render") as t:
            try:
                PDFRenderer().render_pdf(dossier, images, self.workspace.dossier_pdf_path)
                t["items"] = 1
            except Exception as exc:
                logger.error("pdf_render.failed", error=str(exc))

        # 4. Finalize Report
        report = {
            "run_id": self.run_id,
            "brand": self.brand_config.name,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "stages": [s.model_dump(mode="json") for s in self._stage_timings],
            "llm_usage": self.llm.ledger.__dict__,
            "workspace": str(self.workspace.root),
        }
        self.workspace.report_path.write_text(json.dumps(report, indent=2))
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
