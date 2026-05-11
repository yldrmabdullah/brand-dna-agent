You are labelling aesthetic clusters for the brand **{brand_name}**.

I will show you {n_images} representative images from a single aesthetic cluster (one visual territory inside this brand's catalog). Other clusters exist with different aesthetics — your job here is to characterise *this* cluster precisely.

Return strict JSON:

**label** (2–4 words): a sharp name for this aesthetic. Use language a brand strategist would use, not a stylist. Examples: "Tailored Minimalism", "Streetwear Heritage", "Romantic Drape", "Athleisure Polish", "Workwear Revival". Avoid bland labels like "Casual" or "Modern".

**description** (1–2 sentences, ≤60 words): describe what unifies these images — silhouettes, fabrics, palette, styling cues. Be concrete. Example: "Loose, fluid silhouettes in matte fabrics with strong vertical proportion. Palette stays in cream / camel / charcoal. Styling is restrained, single-tone, often unaccessorised."

**key_signifiers** (3–5 items): the visual elements that identify this cluster at a glance. Examples: "drop-shoulder coats", "raw denim", "metallic embellishment", "matte black", "studio backdrops".

Output JSON only:
{{
  "label": "...",
  "description": "...",
  "key_signifiers": [...]
}}
