You are composing the executive summary and positioning line for the **Brand DNA dossier** of **{brand_name}**.

You have already extracted:
- Brand voice: {voice_summary}
- Audience: {audience_summary}
- Dominant garment categories: {garment_summary}
- Silhouette signals: {silhouette_summary}
- Color palette (top 5): {palette_summary}
- Aesthetic clusters: {clusters_summary}

Write copy that a senior brand strategist would feel comfortable presenting in a kickoff meeting. No filler, no marketing-speak, no superlatives. Specific over general.

Return strict JSON:

**executive_summary** (120–180 words, 2–3 short paragraphs): synthesise the brand's identity. Lead with positioning, then layer in visual identity, then audience. The reader should finish knowing *what makes this brand recognisable* and *who it talks to*.

**one_line_positioning** (≤22 words): a single sentence that captures the brand's market position. Avoid "for those who…" clichés. Be assertive.

**color_descriptors**: for the top 5 colors in this palette ({palette_hexes}), give a tasteful one-line descriptor for each (e.g., "warm bone — unbleached cotton", "ink — late-night navy"). Return as a list of 5 strings, in the same order as the input hexes.

**train_module_seeds**: short ingest-ready strings that prime each Refabric training module. Concise, declarative, no markdown.
  - **train_look**: 1–2 sentences describing what a "look" from this brand canonically looks like (e.g., "Floor-length wool coat, neutral palette, paired with leather loafers and minimal jewellery.").
  - **train_mood**: 2–4 mood descriptors (e.g., ["contemplative", "architectural", "northern light"]).
  - **train_attribute**: 3–5 construction or finishing details unique to this brand (e.g., ["dropped shoulder seams", "raw hems", "tonal stitching"]).
  - **train_fabric**: 3–5 material/texture cues (e.g., ["organic cotton", "merino wool", "matte technical nylon"]).
  - **train_pattern**: 2–4 pattern observations (e.g., ["near-exclusively solid", "occasional micro-stripe", "no logo motifs"]). If the brand is essentially solid-color, say so explicitly.

Output JSON only:
{{
  "executive_summary": "...",
  "one_line_positioning": "...",
  "color_descriptors": [...],
  "train_module_seeds": {{
    "train_look": "...",
    "train_mood": [...],
    "train_attribute": [...],
    "train_fabric": [...],
    "train_pattern": [...]
  }}
}}
