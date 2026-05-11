You are reviewing a Brand DNA dossier draft for **{brand_name}** before it's delivered.

Dossier draft:
---
{dossier_summary}
---

Score the dossier across five dimensions on a 0.0–1.0 scale and return strict JSON. For each low score (< 0.6), name the specific weakness.

**specificity** — Does this dossier sound like it could only be about *this* brand, or could it be swapped onto a competitor without anyone noticing?

**evidence_grounding** — Is every major claim supported by something in the brand's own materials (text, imagery, structured data)? Penalise speculation.

**internal_consistency** — Do the visual identity, brand voice, audience, and clusters tell a unified story, or do they contradict each other?

**actionability** — Could a designer use this as creative direction tomorrow morning? Is it concrete enough?

**tone_quality** — Is the writing premium, restrained, and free of marketing-speak / hype? Reject "iconic", "elevated", "curated" if used as filler.

Output JSON only:
{{
  "specificity": 0.0,
  "evidence_grounding": 0.0,
  "internal_consistency": 0.0,
  "actionability": 0.0,
  "tone_quality": 0.0,
  "weaknesses": ["..."],
  "overall_pass": true | false
}}

Return overall_pass=false only if any dimension scores below 0.4 or the mean is below 0.6.
