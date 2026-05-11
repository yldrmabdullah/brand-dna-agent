"""FashionCLIP-based image filtering + embedding generation.

Why FashionCLIP:
- Fine-tuned on ~700k fashion image-caption pairs from Farfetch.
- Markedly better than vanilla CLIP for garment/style discrimination — vanilla
  CLIP is biased toward common ImageNet concepts and underweights fashion-
  specific nuance (silhouette, fabric drape, era cues).
- Free, open-weight, runs on CPU at ~5-10 imgs/sec (acceptable for 100-400
  images per brand).

We use it for three jobs in one forward pass:
1. **Is this fashion?** — fashion-vs-non-fashion prompt scoring.
2. **What garment?** — zero-shot category prompts.
3. **Embedding** — kept for clustering downstream (saves a second pass).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from brand_dna.core.exceptions import AnalysisError
from brand_dna.core.models import ImageRecord
from brand_dna.core.observability import get_logger

logger = get_logger(__name__)


FASHION_POSITIVE_PROMPTS = [
    "a photo of a person wearing clothing",
    "a fashion product photograph",
    "a model wearing an outfit",
    "a flat lay of a garment",
    "a fashion editorial photo",
    "a clothing item on a hanger",
    "a runway photo",
]

FASHION_NEGATIVE_PROMPTS = [
    "a photo of a logo",
    "a screenshot of text",
    "a corporate office building",
    "an icon or graphic illustration",
    "a stock photo of a chart",
    "a map or diagram",
    "a photo of a smartphone screen",
    "a photo of food",
]


# Coarse garment taxonomy. Intentionally not exhaustive — we want stable
# top-level buckets that compose into the dossier's "garment mix" without
# overwhelming the strategist.
GARMENT_CATEGORIES = [
    "dress",
    "top or shirt or blouse",
    "knitwear or sweater",
    "outerwear or jacket or coat",
    "trousers or pants",
    "jeans or denim",
    "skirt",
    "shorts",
    "suit or tailored set",
    "lingerie or underwear",
    "swimwear",
    "activewear",
    "footwear or shoes",
    "bag or handbag",
    "accessory or jewelry or hat",
]

# Maps a verbose prompt back to a clean label for the dossier.
_CATEGORY_LABELS = {
    "dress": "Dresses",
    "top or shirt or blouse": "Tops",
    "knitwear or sweater": "Knitwear",
    "outerwear or jacket or coat": "Outerwear",
    "trousers or pants": "Trousers",
    "jeans or denim": "Denim",
    "skirt": "Skirts",
    "shorts": "Shorts",
    "suit or tailored set": "Tailoring",
    "lingerie or underwear": "Intimates",
    "swimwear": "Swim",
    "activewear": "Activewear",
    "footwear or shoes": "Footwear",
    "bag or handbag": "Bags",
    "accessory or jewelry or hat": "Accessories",
}


@dataclass
class ClassificationResult:
    fashion_score: float
    garment_labels: list[str]  # top-K labels above threshold
    embedding: np.ndarray  # raw image embedding for downstream clustering


def _unwrap_features(output) -> torch.Tensor:
    """transformers 4.x returned raw tensors from get_*_features().
    transformers 5.x wraps them in a BaseModelOutputWithPooling-style object.
    Either way, we want the projected feature tensor."""
    if isinstance(output, torch.Tensor):
        return output
    # Try common attribute names across transformers versions
    for attr in ("pooler_output", "pooled_output", "last_hidden_state", "image_embeds", "text_embeds"):
        feat = getattr(output, attr, None)
        if isinstance(feat, torch.Tensor):
            return feat
    raise AnalysisError(
        f"Unknown CLIP feature output type: {type(output).__name__}. "
        f"Available attrs: {list(vars(output).keys()) if hasattr(output, '__dict__') else 'n/a'}"
    )


FALLBACK_CLIP_MODEL = "openai/clip-vit-base-patch32"


def _clip_sanity_check(model: CLIPModel, processor: CLIPProcessor) -> bool:
    """Verify the loaded checkpoint produces valid (non-NaN) features.
    Some FashionCLIP HF cached weights can be corrupt / partially-downloaded;
    we'd rather catch it here than NaN-pollute the entire run."""
    try:
        inputs = processor(text=["sanity check"], return_tensors="pt", padding=True)
        with torch.no_grad():
            out = model.get_text_features(**inputs)
            feats = _unwrap_features(out)
        return bool(torch.isfinite(feats).all().item())
    except Exception:
        return False


@lru_cache(maxsize=2)
def _load_clip(model_id: str) -> tuple[CLIPModel, CLIPProcessor]:
    """Load + cache CLIP. Singleton per model_id within a process.
    Falls back to OpenAI base CLIP if the requested checkpoint is corrupt
    (a real-world failure mode we've hit with FashionCLIP weights)."""
    logger.info("clip.loading", model=model_id)
    try:
        model = CLIPModel.from_pretrained(model_id)
        processor = CLIPProcessor.from_pretrained(model_id)
    except Exception as exc:
        raise AnalysisError(f"Failed to load CLIP model '{model_id}': {exc}") from exc
    model.eval()
    if torch.cuda.is_available():
        model = model.to("cuda")
        logger.info("clip.gpu_enabled")
    if not _clip_sanity_check(model, processor):
        if model_id == FALLBACK_CLIP_MODEL:
            raise AnalysisError(
                f"CLIP model '{model_id}' produced non-finite features; "
                "fallback unavailable. Try clearing $HF_HOME."
            )
        logger.warning("clip.sanity_failed_fallback", model=model_id, fallback=FALLBACK_CLIP_MODEL)
        return _load_clip(FALLBACK_CLIP_MODEL)
    return model, processor


class FashionClassifier:
    """Stateful wrapper. Encodes prompt sets once, then reuses for every image."""

    def __init__(self, model_id: str = "patrickjohncyh/fashion-clip") -> None:
        self.model_id = model_id
        self.model, self.processor = _load_clip(model_id)
        self.device = next(self.model.parameters()).device
        # Pre-encode prompt text features so per-image cost is image-only.
        self._fashion_pos_feats = self._encode_text(FASHION_POSITIVE_PROMPTS)
        self._fashion_neg_feats = self._encode_text(FASHION_NEGATIVE_PROMPTS)
        self._category_feats = self._encode_text(GARMENT_CATEGORIES)

    def _encode_text(self, prompts: list[str]) -> torch.Tensor:
        inputs = self.processor(text=prompts, return_tensors="pt", padding=True).to(
            self.device
        )
        with torch.no_grad():
            output = self.model.get_text_features(**inputs)
            feats = _unwrap_features(output)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats

    def classify_batch(
        self,
        records: list[ImageRecord],
        *,
        batch_size: int = 16,
        category_top_k: int = 2,
        category_threshold: float = 0.20,
    ) -> list[ClassificationResult]:
        """Returns one result per input record, in the same order."""
        results: list[ClassificationResult] = []
        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]
            images = []
            for rec in batch:
                try:
                    img = Image.open(rec.local_path).convert("RGB")
                except Exception as exc:
                    logger.warning(
                        "clip.image_open_failed", path=rec.local_path, error=str(exc)
                    )
                    images.append(None)
                    continue
                images.append(img)

            # Filter out failed loads but track positions for output ordering.
            valid_indices = [j for j, im in enumerate(images) if im is not None]
            valid_images = [images[j] for j in valid_indices]
            if not valid_images:
                results.extend(
                    [
                        ClassificationResult(
                            fashion_score=0.0,
                            garment_labels=[],
                            embedding=np.zeros(512, dtype=np.float32),
                        )
                        for _ in batch
                    ]
                )
                continue

            inputs = self.processor(images=valid_images, return_tensors="pt").to(
                self.device
            )
            with torch.no_grad():
                output = self.model.get_image_features(**inputs)
                img_feats = _unwrap_features(output)
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)

            # Fashion score = max(positive sim) - max(negative sim), squashed.
            pos_sim = (img_feats @ self._fashion_pos_feats.T).max(dim=-1).values
            neg_sim = (img_feats @ self._fashion_neg_feats.T).max(dim=-1).values
            diff = (pos_sim - neg_sim).cpu().numpy()
            fashion_scores = 1.0 / (1.0 + np.exp(-8.0 * diff))  # sigmoid-ish

            # Category zero-shot: softmax over category prompts; keep top-K above
            # threshold.
            cat_sim = (img_feats @ self._category_feats.T).cpu().numpy()
            cat_sim = _softmax(cat_sim, axis=-1)

            valid_outputs: list[ClassificationResult] = []
            for k, vi in enumerate(valid_indices):
                top_idx = cat_sim[k].argsort()[::-1][:category_top_k]
                labels = [
                    _CATEGORY_LABELS[GARMENT_CATEGORIES[i]]
                    for i in top_idx
                    if cat_sim[k][i] >= category_threshold
                ]
                valid_outputs.append(
                    ClassificationResult(
                        fashion_score=float(fashion_scores[k]),
                        garment_labels=labels,
                        embedding=img_feats[k].cpu().numpy().astype(np.float32),
                    )
                )

            # Stitch back, leaving zero-results for failed loads.
            out_iter = iter(valid_outputs)
            for j in range(len(batch)):
                if j in valid_indices:
                    results.append(next(out_iter))
                else:
                    results.append(
                        ClassificationResult(
                            fashion_score=0.0,
                            garment_labels=[],
                            embedding=np.zeros(img_feats.shape[-1], dtype=np.float32),
                        )
                    )

        return results

    def apply(
        self,
        records: list[ImageRecord],
        *,
        fashion_threshold: float = 0.55,
    ) -> tuple[list[ImageRecord], list[ImageRecord]]:
        """Side-effects records: sets fashion_score, garment_labels, embedding.
        Returns (kept, rejected)."""
        results = self.classify_batch(records)
        kept: list[ImageRecord] = []
        rejected: list[ImageRecord] = []
        for rec, res in zip(records, results, strict=True):
            rec.fashion_score = res.fashion_score
            rec.garment_labels = res.garment_labels
            rec.embedding = res.embedding.tolist()
            if res.fashion_score >= fashion_threshold:
                kept.append(rec)
            else:
                rec.quality_passed = False
                rec.rejection_reason = (
                    f"fashion_score={res.fashion_score:.2f}<{fashion_threshold:.2f}"
                )
                rejected.append(rec)
        logger.info(
            "fashion_classifier.filter",
            input=len(records),
            kept=len(kept),
            rejected=len(rejected),
            threshold=fashion_threshold,
        )
        return kept, rejected


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)
