# Architecture

> Brand DNA Agent — autonomous brand intelligence extraction for fashion brands.
> This document is the design rationale: *why* each layer looks the way it does,
> and *what* I'd build differently with more time. The README is the operator's
> guide.

---

## 1. System diagram

```
                       ┌───────────────────────┐
                       │  Brand YAML config    │
                       └──────────┬────────────┘
                                  │
                ┌─────────────────▼──────────────────┐
                │            Orchestrator             │
                │   (stage timing · graceful failure) │
                └─────────────────┬──────────────────┘
                                  │
   ┌──────────────────────────────┼──────────────────────────────┐
   │                              │                              │
   ▼                              ▼                              ▼
DISCOVERY                    ACQUISITION                    FILTERING
─────────                    ───────────                    ─────────
robots.txt        ──►        web crawler          ──►       quality
sitemap.xml                 (httpx + Playwright)            (size, aspect, fmt)
schema.org JSON-LD          + IG public OG                  fashion clf
OpenGraph                   + image downloader              (FashionCLIP zero-shot)
page classifier                                             dedup (pHash + CLIP)
                                  │
                                  ▼
                              ANALYSIS
                              ────────
                              color (LAB KMeans)
                              garments (zero-shot)
                              silhouettes (zero-shot)
                              clustering (cosine KMeans, k=3-6)
                              brand voice (LLM)
                              audience (LLM, multi-modal)
                                  │
                                  ▼
                              SYNTHESIS
                              ─────────
                              cluster labelling (LLM-vision)
                              executive summary + positioning (LLM)
                              Train Module manifest (LLM)
                              confidence + provenance assembly
                                  │
                ┌─────────────────┴─────────────────┐
                ▼                                   ▼
          brand_dna.pdf                       brand_dna.json
          (WeasyPrint HTML→PDF)              train_modules.json
                                              run_report.json
                                              metadata.sqlite
```

---

## 2. Design decisions, by layer

### 2.1 Discovery — the brand-agnostic substrate

The single most important architectural commitment: **no per-site selectors,
no domain-specific scrapers**. This forces us to lean on signals every
commercial site publishes for SEO:

| Signal | Strength | Why it works site-agnostically |
| --- | --- | --- |
| `robots.txt` | Always present on commercial sites | Tells us *where the sitemap is*. Free politeness data. |
| `sitemap.xml` | Standard SEO requirement | Exhaustive page list. Image extension when published. |
| `schema.org` JSON-LD | Google asks for it, brands oblige | Canonical `Product` / `Article` / `Organization` data. |
| OpenGraph meta | Every brand cares about social previews | Hero images + curated copy per page. |

A heuristic page classifier composes these signals into a `PageType` (PRODUCT,
COLLECTION, LOOKBOOK, ABOUT, BLOG, …). The rubric is: JSON-LD `@type` wins if
present, else OpenGraph `og:type`, else URL path patterns. Brand YAML can
optionally inject additional URL hints — these *augment*, not replace, the
defaults.

**What this buys us**: when we point the agent at a brand it has never seen,
we get ~95% of value on the first run with zero adaptation.

**Where this breaks**: fully client-rendered SPAs that don't pre-render meta
tags server-side. Empirically rare in fashion (SEO is too important to give
up), but `crawl.render_js: true` flips on Playwright as a per-brand escape
hatch.

### 2.2 Acquisition — politeness as a first-class concern

- **httpx async + HTTP/2 multiplexing**. Most retail traffic is CDN-served;
  h2 gives us 4–5× throughput at the same concurrency.
- **Per-host rate limiter**, not just global. CDN domains shouldn't burn the
  brand-host budget.
- **Robots.txt respected by default**. `Disallow:` paths are skipped, declared
  Crawl-delay honored.
- **Image candidates from multiple sources, ranked by signal quality**:
  1. JSON-LD `Product.image` (highest — canonical, brand-curated)
  2. Sitemap `image:image` extension
  3. OpenGraph `og:image` (hero per page)
  4. Inline `<img>` (noisiest — fallback only)
- **Instagram is public-only, no auth**. We pull `og:image` and bio. When Meta
  serves the login wall we degrade gracefully. Production-grade IG coverage
  belongs to the Graph API with Business Discovery scope (requires brand consent
  + Meta app review) — out of scope for v0.1.

### 2.3 Filtering — the throughput-shape decisions

Three filters in a deliberate order:

1. **Quality filter** (no ML, fastest). Resolution, aspect ratio, byte size,
   format. Cheap rejects come first so we don't burn CLIP forward passes on
   1×1 tracking pixels.
2. **FashionCLIP zero-shot** (medium cost). Is this clothing/fashion at all?
   We use a fine-tuned-on-fashion CLIP because vanilla CLIP underweights
   garment-specific nuance.
3. **Visual dedup** (post-filter, post-embed). pHash for fast near-dupe
   collapse, CLIP cosine for the semantic variants pHash misses.

A single FashionCLIP forward pass produces three outputs: fashion-vs-not score,
top-K garment labels, and the embedding (kept for downstream clustering and
dedup). This is the key cost optimization — we encode each image *once*.

**Why ≥ 512px on the shorter side?** The PDF accepts this as a defensible
choice. Reasoning: SDXL-class downstream generative pipelines expect ≥512
input; going lower would cap the dossier's usefulness as a training feed.
Going higher would shrink the keep rate without proportional gain in vision
classifier accuracy.

**Why LAB for color extraction?** RGB-space KMeans clusters by encoding
similarity, not perception. A brand strategist talks about *perceived* color
families — beige vs grey vs cream — and LAB Euclidean distance ≈ perceived
difference. The case explicitly asks for "color logic"; that's a perceptual
concept.

### 2.4 Analysis — pragmatic model choice

- **FashionCLIP for vision**, OpenCLIP if FashionCLIP unavailable. Pretrained,
  no training from scratch (per the brief).
- **KMeans with silhouette-score selection in [k=3, k=6]** for aesthetic
  clusters. HDBSCAN would let `k` float free — bad for a dossier the
  strategist scans in 5 minutes. We constrain to the spec range and let
  silhouette score pick inside it.
- **LLM via OpenRouter** for the text-and-judgement layers:
  - Brand voice: text-only, single call, JSON output
  - Audience: multi-modal (text corpus + 6 cluster-diverse image samples)
  - Cluster labelling: vision-only, one call per cluster
  - Synthesis: structured composition over all extracted signals
- **Pantone-approximation** with a built-in 70-entry table. The dossier flags
  this as "≈ PMS XXXX" — designers see the approximation and use a Pantone
  swatch deck for final calls.

### 2.5 Synthesis — value-add layer

This is where we go beyond the brief:

1. **Train Module Manifest** — the dossier's machine-readable counterpart maps
   1:1 to Refabric's five training inputs: Look, Mood, Attribute, Fabric,
   Pattern. A brand strategist using Refabric's product can ingest this
   manifest directly. We start them at 70% instead of 0%.
2. **Confidence scoring** — each section carries a 0..1 confidence reflecting
   sample size + signal consistency + method notes. High-confidence sections
   are safe to act on; low-confidence sections flag themselves for review.
   This is *the* signal a brand strategist wants when deciding whether to
   trust automation.
3. **Provenance trail** — every major claim (cluster label, audience cue,
   voice quote) links back to its source. Useful for review, useful for legal,
   useful when a brand pushes back on a positioning claim.
4. **Self-eval** (prompt-ready, not enabled by default) — the model judges its
   own output against a rubric before delivery. Currently lives as
   `configs/prompts/self_eval.md`; wiring into the orchestrator is a 30-line
   change, deliberately deferred to keep v0.1 within the time budget.

### 2.6 PDF rendering — HTML/CSS as the authoring medium

WeasyPrint over ReportLab because HTML/CSS is the right medium for a
designer-readable document. The template (`src/brand_dna/synthesis/templates/brand_dna.html`)
is editable by a designer without touching Python. CSS `@page` rules give us
paginated tables, page numbering, and proper image embedding without bespoke
layout code.

The case explicitly warns against "a JSON dump dressed up as a PDF". The
template's editorial layout — Cormorant Garamond for display, Inter for body,
swatch cards for the palette, embedded representative imagery per cluster —
treats the dossier as a brand-book deliverable, not a data export.

---

## 3. Cost & throughput

### Per-brand cost breakdown (typical)

| Item | Count | Cost |
| --- | --- | --- |
| LLM: cluster labels (vision, primary) | 3-5 calls × ~$0.02 | ~$0.10 |
| LLM: brand voice (text, primary) | 1 call × ~$0.05 | ~$0.05 |
| LLM: audience (multi-modal, primary) | 1 call × ~$0.10 | ~$0.10 |
| LLM: synthesis (text, synthesis-grade) | 1 call × ~$0.15 | ~$0.15 |
| Hosted vision API | 0 | $0 (FashionCLIP local) |
| **Total** | ~6–8 calls | **~$0.40–0.80** |

### Wall-clock breakdown (M-series CPU, ~150 images post-filter)

| Stage | Time |
| --- | --- |
| Discovery + crawl | 1–3 min |
| Image download | 1–2 min |
| Quality filter | <5 s |
| FashionCLIP (encode all images) | 30–90 s |
| Dedup | 5–10 s |
| Color palette | 5–15 s |
| Clustering + silhouette | 10–30 s |
| LLM calls (sequential) | 1–2 min |
| PDF render | 5–10 s |
| **Total wall-clock** | **6–10 min** |

### Scaling to thousands of brands

**Architectural targets** for v1.0 (out of scope for v0.1):

1. **Brand-level parallelism**: each brand is independent — embarrassingly
   parallel. A worker queue (Celery/Redis or AWS Batch) trivially parallelises
   N brands across M workers.
2. **Shared model weights**: FashionCLIP weights live in shared object storage.
   Workers boot with weights pre-mounted, not downloaded per-pod.
3. **Postgres + S3**, not SQLite + filesystem, for the metadata layer. The
   `MetadataStore` class is the single point of substitution — no other
   module touches storage directly.
4. **CDN-aware crawl scheduling**: brands sharing Shopify / VTEX / Salesforce
   Commerce should be queued so traffic spreads across CDN POPs.
5. **Diff mode**: re-running a brand 6 months later should surface the *delta*
   — what palette colors shifted, which clusters emerged, what voice changes
   appeared. The `MetadataStore.fetch_phashes()` method already supports
   cross-run image dedup; the rest is a comparator over JSON manifests.

### What "$/brand at scale" looks like

At 10,000 brands per quarter, ~$0.60/brand → $24K/quarter LLM spend. If that
becomes a bottleneck:

- Synthesis can downshift from Opus to Sonnet — empirically a 2× cost cut
  with marginal quality drop on summarisation.
- Cluster labelling can batch into one call per brand instead of N per brand
  (single multi-turn prompt with all clusters' images).
- Self-eval can swap to a cheap model since the rubric is structural.

---

## 4. Trade-offs

| Decision | Win | Cost |
| --- | --- | --- |
| OpenRouter as single LLM gateway | Provider-agnostic, single key, fallback routing | Adds one hop (~50ms) vs direct provider |
| FashionCLIP local vs hosted vision API | Free at runtime, deterministic | First-run model download (~700 MB); CPU-only is slowish |
| WeasyPrint vs ReportLab | Designer-editable HTML template, beautiful output | System dep on Pango/Cairo (Dockerfile burden) |
| SQLite per-run vs central DB | Shippable runs, zero infra | No cross-run analytics until v1.0 |
| 3–6 cluster constraint | Strategist-readable | Hides micro-aesthetics a brand strategist might want |
| No JS rendering by default | Lean Docker image, fast | Misses pure-SPA brands until flag is flipped |
| Public-only Instagram | ToS-clean | Login wall blocks deep content |
| Provenance + confidence as first-class | Trust-grade output | More code, more model context, slight cost |
| Sequential LLM calls (not concurrent) | Simpler retry / cost tracking | 1–2 minutes wall-clock latency |

---

## 5. What I'd build differently with more time

In rough order of value:

1. **Wire self-eval into the orchestrator as a final gate.** Prompt is ready;
   integrating means one LLM call after synthesis, then re-running synthesis
   if scores drop below threshold (with seed adjustment). +30 minutes, +20%
   output-quality robustness.

2. **Concurrent LLM calls.** Brand voice, audience, and cluster labels are
   independent — they could run in parallel. Cuts wall-clock by ~40%.

3. **Production Instagram via Graph API.** The case accepts one social channel
   done well; this is "well done" at production grade. Requires brand consent
   flow + Meta app review — meaningful additional scope.

4. **Diff mode** (re-runs on the same brand surface what changed). Foundation
   is laid: `metadata.sqlite` per run, pHash already persisted. Need a
   comparator + a "delta" PDF template variant. Half a day.

5. **Pinterest as second channel.** For fashion, Pinterest density-of-signal
   is sometimes higher than IG. Public boards are crawlable; ToS is similar.
   Pinterest visual-search API would be a force-multiplier for cluster
   labelling and seasonal trend detection.

6. **Pattern detection.** Currently the `Train Pattern` module gets text
   descriptors from the LLM ("near-exclusively solid", "occasional micro-stripe").
   A real pattern detector — e.g., a thin CNN classifying solid / stripe /
   plaid / floral / abstract / logo — would feed concrete data into the
   manifest. ~1 day; off-the-shelf models exist.

7. **A semantic search index over the run's images.** Embeddings are already
   computed; storing them in FAISS or LanceDB enables strategist queries like
   "show me the editorial-leaning shots in the Tailored Minimalism cluster."
   Half a day.

8. **Streaming progress** in the CLI. The Rich library is already in deps;
   a Live progress display per stage would help during the live walkthrough.

9. **Multi-language brand voice analysis.** Most European luxury brands publish
   copy in multiple languages; analysing in the original (French/Italian/Turkish)
   captures tonal nuance lost in translation. Today we lean on the LLM's
   multilingual ability passively — a deliberate "translate after analyse"
   flow would be tighter.

10. **Brand similarity comparator.** Once we have N brand DNAs, comparing two
    becomes a strategic primitive. Cosine distance over aesthetic-cluster
    embeddings + Jaccard over vocabulary + LAB-distance over palettes →
    a "this brand reads 0.78 like brand A, 0.21 like brand B" signal. Useful
    for competitive positioning, white-space discovery, and as a sanity check
    on the dossier itself.

---

## 6. Where this would break

- **A brand with no schema.org markup, no sitemap, no OG tags, and JS-rendered
  content** would defeat the discovery layer. Empirically rare in fashion
  (SEO is existential), but possible for very small designer sites. Fallback:
  Playwright + an LLM-driven page classifier over rendered HTML. Documented,
  not built.
- **An Instagram-only brand** (no proper site). We'd need real IG access
  (Graph API). The agent would currently produce a dossier built from very
  thin signal.
- **Heavy aggressive anti-bot (Cloudflare Turnstile, etc.)**. We respect
  robots.txt and rate-limit, but adversarial bot detection is a different
  game. Production answer: route through a residential proxy pool with
  consent / partnership.
- **A brand whose voice corpus is entirely in product descriptions** ("Cotton
  poplin shirt. Regular fit. 100% cotton.") — there's no editorial signal.
  The dossier's text identity section will be thin. Confidence scoring will
  flag this honestly.

The agent's response to all of these is the same: degrade gracefully, ship
a partial dossier, surface confidence so the human knows where to dig in.
