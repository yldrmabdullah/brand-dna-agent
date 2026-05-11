"""Approximate nearest-Pantone matching.

A licensed Pantone library is out of scope. This small reference table covers
the ~70 most-used PMS Solid Coated colors in fashion (neutrals, classic brand
palettes, common accents). It's not authoritative — we surface it as
"nearest PMS approximation" in the dossier, not as a substitute for a Pantone
swatch deck.
"""

from __future__ import annotations

import math

# (PMS code, R, G, B) — sRGB approximations
_PANTONE_REFERENCE: list[tuple[str, int, int, int]] = [
    # Neutrals & whites
    ("11-0601 TPG (Bright White)", 248, 246, 240),
    ("Cool Gray 1", 217, 217, 214),
    ("Cool Gray 3", 200, 201, 199),
    ("Cool Gray 5", 177, 179, 179),
    ("Cool Gray 7", 151, 153, 155),
    ("Cool Gray 9", 117, 120, 123),
    ("Cool Gray 11", 83, 86, 90),
    ("Black 6 C", 16, 24, 32),
    # Beiges & creams
    ("11-0507 TPG (Vanilla Ice)", 240, 234, 214),
    ("13-1106 TPG (Cream)", 234, 224, 200),
    ("13-1010 TPG (Almond Buff)", 218, 192, 161),
    ("14-1116 TPG (Sand)", 210, 184, 144),
    ("PMS 7501", 220, 211, 184),
    ("PMS 7502", 200, 178, 138),
    # Browns
    ("PMS 4715 (Mocha)", 158, 119, 96),
    ("PMS 4625 (Chocolate)", 76, 47, 39),
    ("PMS 477 (Espresso)", 84, 56, 47),
    # Reds
    ("PMS 186 (Red)", 200, 16, 46),
    ("PMS 199", 213, 0, 50),
    ("PMS 1797 (Lipstick)", 203, 51, 59),
    ("PMS 7421 (Wine)", 124, 35, 51),
    # Pinks
    ("PMS 182 (Powder)", 246, 187, 200),
    ("PMS 197 (Rose)", 230, 144, 162),
    ("PMS 198 (Hot Pink)", 222, 73, 104),
    # Oranges
    ("PMS 1505 (Tangerine)", 255, 130, 0),
    ("PMS 165 (Orange)", 255, 103, 31),
    ("PMS 1525 (Burnt Orange)", 191, 87, 0),
    # Yellows
    ("PMS 1235 (Honey)", 255, 184, 28),
    ("PMS 109 (Yellow)", 255, 209, 0),
    ("PMS 7406 (Mustard)", 241, 196, 0),
    # Greens
    ("PMS 7491 (Olive)", 121, 129, 60),
    ("PMS 5535 (Forest)", 27, 53, 47),
    ("PMS 348 (Emerald)", 0, 132, 61),
    ("PMS 7717 (Sea Green)", 0, 119, 122),
    ("PMS 5635 (Sage)", 192, 199, 173),
    # Blues
    ("PMS 282 (Navy)", 4, 30, 66),
    ("PMS 295 (Deep Navy)", 0, 47, 87),
    ("PMS 540 (Royal)", 0, 60, 113),
    ("PMS 2945 (Cobalt)", 0, 95, 169),
    ("PMS 285 (Sky)", 0, 114, 206),
    ("PMS 297 (Light Blue)", 113, 197, 232),
    ("PMS 5415 (Slate Blue)", 92, 124, 144),
    # Purples
    ("PMS 2685 (Royal Purple)", 47, 0, 124),
    ("PMS 267 (Violet)", 89, 23, 168),
    ("PMS 7445 (Lavender)", 173, 169, 209),
    # Metallics (approximate)
    ("PMS 871 (Gold)", 132, 113, 68),
    ("PMS 877 (Silver)", 138, 141, 143),
]


def _euclidean_rgb(a: tuple[int, int, int], b: tuple[int, int, int]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def nearest_pantone(rgb: tuple[int, int, int]) -> str:
    """Returns 'PMS … (≈ E_rgb=N)' string — the distance signals confidence."""
    best = min(
        _PANTONE_REFERENCE,
        key=lambda ref: _euclidean_rgb(rgb, (ref[1], ref[2], ref[3])),
    )
    dist = _euclidean_rgb(rgb, (best[1], best[2], best[3]))
    return f"{best[0]} (~Δ{dist:.0f})"
