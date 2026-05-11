"""Utilities for dossier synthesis — formatting, prompting, and data coercion."""

from __future__ import annotations
from typing import Any
from brand_dna.core.models import (
    BrandDNADossier, AestheticCluster, BrandVoice, AudienceProfile, ColorEntry
)

def build_sd_prompt(dossier: BrandDNADossier) -> str:
    palette = ", ".join(c.descriptor or c.hex for c in dossier.color_palette.entries[:3])
    silhouette = ", ".join(dossier.silhouette_summary[:2])
    return (
        f"Professional fashion editorial photography for {dossier.brand_name}, "
        f"{dossier.one_line_positioning}. Colors: {palette}. "
        f"Features {silhouette} silhouettes, vogue style."
    )

def build_mj_prompt(dossier: BrandDNADossier) -> str:
    tone = ", ".join(dossier.brand_voice.tone_descriptors[:3])
    return f"{dossier.one_line_positioning} fashion lookbook, {tone} atmosphere --ar 4:5 --v 6.0"

def coerce_str(x: Any) -> str:
    return str(x).strip() if x else ""

def coerce_list(x: Any) -> list[str]:
    if isinstance(x, list):
        return [str(i).strip() for i in x if i]
    if isinstance(x, str) and x.strip():
        return [x.strip()]
    return []

def collect_reps(clusters: list[AestheticCluster], n: int = 8) -> list[str]:
    out: list[str] = []
    for c in clusters:
        for rid in c.representative_image_ids[:2]:
            out.append(rid)
            if len(out) >= n: return out
    return out

def score_by_count(n: int, low: int, high: int) -> float:
    if n <= low: return 0.3 * (n / max(1, low))
    if n >= high: return 0.95
    return 0.3 + 0.65 * ((n - low) / (high - low))
