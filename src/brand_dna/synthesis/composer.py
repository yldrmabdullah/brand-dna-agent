"""Dossier composer — wires LLM calls into the final Pydantic BrandDNADossier.

This is where the value-add layers (Train modules, provenance, confidence)
get materialised. The composer is the only place that has the full picture
across every analytical signal, so we centralise:
- Cluster labelling (LLM vision over each cluster's representative images)
- Executive summary + positioning composition
- Color descriptor naming
- Train Module manifest synthesis (Refabric-aligned)
- Confidence scoring across sections
- Provenance trail assembly
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from brand_dna.core.models import (
    AestheticCluster,
    AudienceProfile,
    BrandDNADossier,
    BrandVoice,
    ColorEntry,
    ColorPalette,
    Confidence,
    GarmentDistribution,
    ImageRecord,
    ProvenanceTrail,
    RunMetadata,
    SourceRef,
    TrainAttributeModule,
    TrainFabricModule,
    TrainLookModule,
    TrainModuleManifest,
    TrainMoodModule,
    TrainPatternModule,
)
from brand_dna.core.observability import get_logger
from brand_dna.llm.client import LLMClient
from brand_dna.llm.prompts import render
from brand_dna.synthesis import utils

logger = get_logger(__name__)


class DossierComposer:
    """Stateful composer. Holds shared resources (LLM, model id, image index)."""

    def __init__(
        self,
        llm: LLMClient,
        model_synthesis: str,
        model_primary: str,
    ) -> None:
        self.llm = llm
        self.model_synthesis = model_synthesis
        self.model_primary = model_primary

    # ─── Cluster labelling (vision-multi-modal) ──────────────────────────

    async def label_clusters(
        self,
        clusters: list[AestheticCluster],
        images: list[ImageRecord],
        brand_name: str,
    ) -> list[AestheticCluster]:
        """Side-effects: sets label + description on each cluster."""
        by_id = {img.image_id: img for img in images}
        for cluster in clusters:
            sample_paths = [
                by_id[i].local_path
                for i in cluster.representative_image_ids
                if i in by_id
            ]
            if not sample_paths:
                cluster.label = f"Cluster {cluster.cluster_id + 1}"
                cluster.description = "Insufficient images to characterise."
                continue

            prompt = render(
                "cluster_labels",
                brand_name=brand_name,
                n_images=len(sample_paths),
            )
            try:
                data, _ = await self.llm.chat_json(
                    prompt,
                    model=self.model_primary,
                    system=(
                        "You are a senior fashion brand strategist. Be specific "
                        "and confident; avoid filler language."
                    ),
                    images=[Path(p) for p in sample_paths[:5]],
                    temperature=0.25,
                    max_tokens=600,
                )
            except Exception as exc:
                logger.warning(
                    "compose.cluster_label_failed",
                    cluster_id=cluster.cluster_id,
                    error=str(exc),
                )
                cluster.label = f"Cluster {cluster.cluster_id + 1}"
                cluster.description = "Could not label automatically."
                continue

            cluster.label = (data.get("label") or f"Cluster {cluster.cluster_id + 1}").strip()
            cluster.description = (data.get("description") or "").strip()
        return clusters

    # ─── Top-level synthesis ──────────────────────────────────────────────

    async def synthesise(
        self,
        *,
        brand_name: str,
        brand_url: str,
        social_handles: dict[str, str],
        palette: ColorPalette,
        garments: GarmentDistribution,
        silhouettes: list[str],
        clusters: list[AestheticCluster],
        voice: BrandVoice,
        audience: AudienceProfile,
        images: list[ImageRecord],
        signal_strengths: dict[str, dict[str, Any]],
        run_metadata: RunMetadata,
        previous_dossier: dict[str, Any] | None = None,
    ) -> BrandDNADossier:
        # Delta Analysis context
        delta_context = ""
        if previous_dossier:
            prev_pos = previous_dossier.get("one_line_positioning", "Unknown")
            prev_summary = previous_dossier.get("executive_summary", "")[:500]
            delta_context = (
                f"\nPREVIOUS ANALYSIS DETECTED:\n"
                f"Past Positioning: {prev_pos}\n"
                f"Past Summary Snippet: {prev_summary}\n"
                "Please highlight any shifts in strategy, aesthetic, or audience focus since this previous snapshot."
            )
        # Prepare summary strings to feed the synthesis prompt — bounded length
        # so we stay in context cheaply.
        palette_top5 = palette.entries[:5]
        palette_hexes = ", ".join(c.hex for c in palette_top5)
        palette_summary = "; ".join(
            f"{c.hex} ({c.percentage:.0f}%, ~{c.nearest_pantone})" for c in palette_top5
        )
        garment_summary = ", ".join(
            f"{k} {v:.0f}%"
            for k, v in sorted(garments.percentages.items(), key=lambda x: -x[1])[:6]
        ) or "none detected"
        silhouette_summary = ", ".join(silhouettes) or "no strong signal"
        voice_summary = (
            f"tone={voice.tone_descriptors}; values={voice.stated_values}; "
            f"positioning={voice.positioning_statement!r}"
        )
        audience_summary = (
            f"demographic={audience.demographic_cues}; "
            f"psycho={audience.psychographic_cues}; "
            f"price={audience.price_positioning}"
        )
        clusters_summary = "; ".join(
            f"#{c.cluster_id+1} '{c.label}' ({c.size} items)" for c in clusters
        ) or "no clusters extracted"

        prompt = render(
            "synthesis",
            brand_name=brand_name,
            voice_summary=voice_summary,
            audience_summary=audience_summary,
            garment_summary=garment_summary,
            silhouette_summary=silhouette_summary,
            palette_summary=palette_summary,
            palette_hexes=palette_hexes,
            clusters_summary=clusters_summary,
        )

        try:
            data, _ = await self.llm.chat_json(
                prompt,
                model=self.model_synthesis,
                system=(
                    "You are a senior fashion brand strategist composing a "
                    "brand DNA dossier for a luxury client. Premium, restrained, "
                    "specific. No marketing-speak."
                ),
                temperature=0.2,
                max_tokens=2200,
            )
        except Exception as exc:
            logger.error("compose.synthesis_failed", error=str(exc))
            # Hard fallback so the run still produces a dossier
            data = self._fallback_synthesis(
                brand_name, voice, audience, palette_top5, garments
            )

        # Backfill color descriptors
        descriptors = data.get("color_descriptors") or []
        for entry, descr in zip(palette.entries, descriptors, strict=False):
            if isinstance(descr, str) and descr.strip():
                entry.descriptor = descr.strip()

        # Build Train Module manifest
        seeds = data.get("train_module_seeds") or {}
        train_modules = TrainModuleManifest(
            look=TrainLookModule(
                representative_image_ids=utils.collect_reps(clusters, n=8),
                aesthetic_summary=utils.coerce_str(seeds.get("train_look")),
                seasonal_signals=[],
            ),
            mood=TrainMoodModule(
                mood_descriptors=utils.coerce_list(seeds.get("train_mood")),
                emotional_atmosphere=f"{brand_name} atmosphere",
                color_palette=palette,
            ),
            attribute=TrainAttributeModule(
                silhouettes=silhouettes,
                construction_details=utils.coerce_list(seeds.get("train_attribute")),
                embellishments=[],
            ),
            fabric=TrainFabricModule(
                detected_materials=utils.coerce_list(seeds.get("train_fabric")),
                texture_descriptors=utils.coerce_list(seeds.get("train_fabric")),
                sustainability_signals=[],
            ),
            pattern=TrainPatternModule(
                pattern_types=utils.coerce_list(seeds.get("train_pattern")),
                motif_descriptors=utils.coerce_list(seeds.get("train_pattern")),
            ),
        )

        confidences = self._confidence_scores(signal_strengths)
        provenance = self._build_provenance(
            voice=voice,
            audience=audience,
            clusters=clusters,
            images=images,
        )

        dossier = BrandDNADossier(
            brand_name=brand_name,
            brand_url=brand_url,
            social_handles=social_handles,
            color_palette=palette,
            garment_distribution=garments,
            aesthetic_clusters=clusters,
            silhouette_summary=silhouettes,
            styling_cues=[],
            brand_voice=voice,
            audience=audience,
            train_modules=train_modules,
            confidences=confidences,
            provenance=provenance,
            executive_summary=utils.coerce_str(data.get("executive_summary")),
            one_line_positioning=utils.coerce_str(data.get("one_line_positioning")),
            run_metadata=run_metadata,
        )
        
        # ─── New: Generative DNA Prompt (GenAI Ready) ───────────────
        dossier.custom_data["genai_prompts"] = {
            "stable_diffusion": utils.build_sd_prompt(dossier),
            "midjourney": utils.build_mj_prompt(dossier)
        }
        
        return dossier

def _build_sd_prompt(dossier: BrandDNADossier) -> str:
    palette = ", ".join(c.descriptor or c.hex for c in dossier.color_palette.entries[:3])
    silhouette = ", ".join(dossier.silhouette_summary[:2])
    return (
        f"Professional fashion editorial photography for {dossier.brand_name}, "
        f"{dossier.one_line_positioning}. Colors: {palette}. "
        f"Features {silhouette} silhouettes, high-end {dossier.train_modules.fabric.texture_descriptors[0] if dossier.train_modules.fabric.texture_descriptors else 'fashion'} textures, "
        "8k resolution, cinematic lighting, vogue style."
    )

def _build_mj_prompt(dossier: BrandDNADossier) -> str:
    tone = ", ".join(dossier.brand_voice.tone_descriptors[:3])
    return f"{dossier.one_line_positioning} fashion lookbook, {tone} atmosphere, {dossier.brand_name} aesthetic --ar 4:5 --v 6.0"

    # ─── Helpers ──────────────────────────────────────────────────────────

    def _confidence_scores(
        self, signals: dict[str, dict[str, Any]]
    ) -> dict[str, Confidence]:
        """Heuristic confidence: each section's score reflects sample size."""
        out: dict[str, Confidence] = {}

        # Color palette
        cp = signals.get("color_palette", {})
        n_palette_images = cp.get("sample_size", 0)
        out["color_palette"] = Confidence(
            score=_score_by_count(n_palette_images, low=20, high=80),
            sample_size=n_palette_images,
            method="kmeans_lab",
            notes="LAB-space KMeans across pooled pixels.",
        )

        # Garments
        g = signals.get("garments", {})
        n_garment = g.get("sample_size", 0)
        out["garment_distribution"] = Confidence(
            score=utils.score_by_count(n_garment, low=30, high=120),
            sample_size=n_garment,
            method="fashion_clip_zero_shot",
        )

        # Clusters
        c = signals.get("clusters", {})
        out["aesthetic_clusters"] = Confidence(
            score=_score_by_count(c.get("sample_size", 0), low=30, high=120),
            sample_size=c.get("sample_size", 0),
            method=f"kmeans_silhouette_k={c.get('chosen_k', '?')}",
            notes=f"Silhouette score: {c.get('silhouette', 'n/a')}",
        )

        # Brand voice
        v = signals.get("brand_voice", {})
        v_chars = v.get("corpus_chars", 0)
        out["brand_voice"] = Confidence(
            score=_score_by_count(v_chars, low=2000, high=15000),
            sample_size=v_chars,
            method="llm_extraction",
            notes="Sample size in characters of brand-published text.",
        )

        # Audience
        a = signals.get("audience", {})
        out["audience"] = Confidence(
            score=_score_by_count(
                a.get("corpus_chars", 0) + a.get("n_images_sampled", 0) * 500,
                low=2000,
                high=10000,
            ),
            sample_size=a.get("corpus_chars", 0) + a.get("n_images_sampled", 0),
            method="multimodal_llm",
            notes=(
                f"{a.get('corpus_chars', 0)} text chars + "
                f"{a.get('n_images_sampled', 0)} visual samples."
            ),
        )

        return out

    def _build_provenance(
        self,
        *,
        voice: BrandVoice,
        audience: AudienceProfile,
        clusters: list[AestheticCluster],
        images: list[ImageRecord],
    ) -> list[ProvenanceTrail]:
        """Maps claims back to evidence. We don't trace *every* claim — focus
        on the ones a reviewer is most likely to challenge."""
        by_id = {img.image_id: img for img in images}
        trails: list[ProvenanceTrail] = []

        # Brand voice quotes ↔ corpus
        for quote in voice.representative_quotes:
            trails.append(
                ProvenanceTrail(
                    claim=f"Voice sample: “{quote}”",
                    sources=[SourceRef(kind="page", ref="brand_corpus", snippet=quote)],
                )
            )

        # Audience evidence snippets ↔ corpus/images
        for snippet in audience.evidence_snippets:
            trails.append(
                ProvenanceTrail(
                    claim=f"Audience signal: {snippet}",
                    sources=[SourceRef(kind="page", ref="brand_corpus", snippet=snippet)],
                )
            )

        # Cluster labels ↔ representative images
        for cluster in clusters:
            sources = []
            for img_id in cluster.representative_image_ids[:3]:
                img = by_id.get(img_id)
                if not img:
                    continue
                sources.append(
                    SourceRef(
                        kind="image",
                        ref=img.image_id,
                        snippet=img.provenance.product_name
                        or img.provenance.alt_text
                        or img.provenance.source_url,
                    )
                )
            if sources:
                trails.append(
                    ProvenanceTrail(
                        claim=f"Cluster: {cluster.label}",
                        sources=sources,
                    )
                )

        return trails

    def _fallback_synthesis(
        self,
        brand_name: str,
        voice: BrandVoice,
        audience: AudienceProfile,
        palette_top: list[ColorEntry],
        garments: GarmentDistribution,
    ) -> dict[str, Any]:
        """Used when synthesis LLM call fails. Produces a usable shell so we
        still ship a dossier."""
        top_categories = sorted(
            garments.percentages.items(), key=lambda x: -x[1]
        )[:3]
        return {
            "executive_summary": (
                f"{brand_name} presents primarily {', '.join(c for c, _ in top_categories) or 'apparel'} "
                f"with a palette anchored on {', '.join(c.hex for c in palette_top[:3])}. "
                f"Voice cues include {', '.join(voice.tone_descriptors[:3]) or 'restrained, declarative'}. "
                f"Synthesis fallback used — full LLM pass not available."
            ),
            "one_line_positioning": voice.positioning_statement
            or f"{brand_name} — synthesis unavailable.",
            "color_descriptors": [None] * len(palette_top),
            "train_module_seeds": {
                "train_look": "",
                "train_mood": voice.tone_descriptors[:3],
                "train_attribute": [],
                "train_fabric": [],
                "train_pattern": [],
            },
        }
