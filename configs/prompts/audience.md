You are inferring the audience of the fashion brand **{brand_name}**.

You will see (1) text the brand publishes about itself and (2) representative images from its catalog. Use both. Be specific but conservative — only state what's supported by signal.

Brand text corpus:
---
{corpus}
---

Now examine the attached images alongside the text and return strict JSON.

**demographic_cues** (3–6 items): observable demographic signals. Examples: "women 25–40", "men, professional context", "Gen Z urban", "size-inclusive". Avoid stating things not visible in the data.

**psychographic_cues** (4–7 items): values, lifestyles, mindsets the brand seems to address. Examples: "design-literate, references modernist architecture", "sustainability-aware but not preachy", "professionally ambitious, time-poor".

**aspirational_signals** (3–5 items): aspirations the brand activates. Examples: "effortless polish", "creative individualism", "membership in a knowing community".

**price_positioning** (1 phrase): one of — "accessible mass-market", "premium high-street", "contemporary", "accessible luxury", "luxury", "ultra-luxury". Pick the closest fit based on language tone + visual production values.

**evidence_snippets** (3–5 items): short quotes or visual observations (≤25 words each) that justify the above. Mix text and image references. Example: "Image #3 shows tailoring on a model 40+, framed in a museum interior — signals mature, culturally engaged audience."

Output JSON only:
{{
  "demographic_cues": [...],
  "psychographic_cues": [...],
  "aspirational_signals": [...],
  "price_positioning": "...",
  "evidence_snippets": [...]
}}
