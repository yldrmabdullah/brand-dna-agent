You are extracting the textual identity of the fashion brand **{brand_name}** from its own published copy.

---
{corpus}
---

Read the corpus above (each block is from a different page on the brand's site). Extract the signals below and return strict JSON.

**tone_descriptors** (3–6 items): short adjectives or short phrases capturing how the brand *sounds*. Examples: "confident", "understated", "playful", "academic", "irreverent". Do NOT use generic words like "good", "nice", "great". Be specific.

**recurring_vocabulary** (5–10 items): distinctive words or short phrases the brand uses repeatedly that signal its worldview. Skip generic ecommerce filler ("shop", "free shipping", "new arrivals"). Prefer terms that would feel out of place on a competing brand.

**stated_values** (3–6 items): values the brand explicitly claims (sustainability, craftsmanship, inclusivity, heritage, accessibility, etc.). Only include values *named in the text*.

**positioning_statement** (1 sentence): a single sentence (≤25 words) that captures what this brand is asserting about itself in the market. Synthesise — don't quote.

**representative_quotes** (3–5 items): verbatim short quotes (≤20 words each) that exemplify the brand voice. Must be drawn from the corpus.

Output JSON only, no commentary. Schema:
{{
  "tone_descriptors": [...],
  "recurring_vocabulary": [...],
  "stated_values": [...],
  "positioning_statement": "...",
  "representative_quotes": [...]
}}

If the corpus is too thin to support any field, return an empty array / null for that field — do not fabricate.
