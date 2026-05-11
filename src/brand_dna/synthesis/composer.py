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
    ) -> BrandDNADossier:
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
                representative_image_ids=_collect_reps(clusters, n=8),
                aesthetic_summary=_coerce_str(seeds.get("train_look")),
                seasonal_signals=[],  # filled if future seasonal analyser added
            ),
            mood=TrainMoodModule(
                mood_descriptors=_coerce_list(seeds.get("train_mood")),
                emotional_atmosphere=_emotional_atmosphere(voice, audience),
                color_palette=palette,
            ),
            attribute=TrainAttributeModule(
                silhouettes=silhouettes,
                construction_details=_coerce_list(seeds.get("train_attribute")),
                embellishments=_extract_embellishments(seeds.get("train_attribute")),
            ),
            fabric=TrainFabricModule(
                detected_materials=_coerce_list(seeds.get("train_fabric")),
                texture_descriptors=_coerce_list(seeds.get("train_fabric")),
                sustainability_signals=_extract_sustainability(voice.stated_values),
            ),
            pattern=TrainPatternModule(
                pattern_types=_coerce_list(seeds.get("train_pattern")),
                motif_descriptors=_coerce_list(seeds.get("train_pattern")),
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
            styling_cues=_derive_styling_cues(voice, audience),
            brand_voice=voice,
            audience=audience,
            train_modules=train_modules,
            confidences=confidences,
            provenance=provenance,
            executive_summary=_coerce_str(data.get("executive_summary")),
            one_line_positioning=_coerce_str(data.get("one_line_positioning")),
            run_metadata=run_metadata,
        )
        return dossier

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
            score=_score_by_count(n_garment, low=30, high=120),
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


# ─── Local helpers ────────────────────────────────────────────────────────


def _score_by_count(n: int, *, low: int, high: int) -> float:
    """Map a sample count to a 0..1 confidence via a smooth ramp."""
    if n <= low:
        return 0.3 * (n / max(1, low))  # 0..0.3 below "low"
    if n >= high:
        return 0.95
    # Linear ramp 0.3 → 0.95 between low and high
    span = high - low
    return 0.3 + 0.65 * ((n - low) / span)


def _coerce_str(x: Any) -> str:
    if isinstance(x, str):
        return x.strip()
    return ""


def _coerce_list(x: Any) -> list[str]:
    if isinstance(x, list):
        return [str(i).strip() for i in x if str(i).strip()]
    if isinstance(x, str) and x.strip():
        return [x.strip()]
    return []


def _collect_reps(clusters: list[AestheticCluster], n: int = 8) -> list[str]:
    out: list[str] = []
    for c in clusters:
        for rid in c.representative_image_ids[:2]:
            out.append(rid)
            if len(out) >= n:
                return out
    return out


def _extract_embellishments(seed: Any) -> list[str]:
    """Best-effort: pull obviously-embellishment-like terms from train_attribute seed."""
    terms = _coerce_list(seed)
    keywords = (
        "embroidery", "beading", "sequin", "metallic", "embellish",
        "stud", "appliqué", "lace", "ruffle", "fringe",
    )
    return [t for t in terms if any(kw in t.lower() for kw in keywords)]


def _extract_sustainability(values: list[str]) -> list[str]:
    keywords = (
        "sustain", "organic", "recycle", "circular", "regenerat",
        "responsib", "ethical", "low-impact", "fair-trade", "gots", "oeko",
    )
    return [v for v in values if any(kw in v.lower() for kw in keywords)]


def _emotional_atmosphere(voice: BrandVoice, audience: AudienceProfile) -> str:
    parts: list[str] = []
    parts.extend(voice.tone_descriptors[:3])
    parts.extend(audience.aspirational_signals[:2])
    return ", ".join(dict.fromkeys(parts)) or "neutral, undefined"


def _derive_styling_cues(voice: BrandVoice, audience: AudienceProfile) -> list[str]:
    """A lightweight derivation from voice + audience signals."""
    cues = []
    tone = " ".join(voice.tone_descriptors).lower()
    if any(w in tone for w in ("minimal", "understat", "restrain")):
        cues.append("minimal styling")
    if any(w in tone for w in ("layer", "complex", "rich")):
        cues.append("layered styling")
    if any(w in tone for w in ("formal", "tailored", "polished")):
        cues.append("formal-leaning")
    if any(w in tone for w in ("casual", "easy", "relaxed")):
        cues.append("casual-leaning")
    psycho = " ".join(audience.psychographic_cues).lower()
    if "monochrome" in psycho or "tonal" in psycho:
        cues.append("monochrome / tonal palettes")
    return cues or ["styling cues inconclusive"]
