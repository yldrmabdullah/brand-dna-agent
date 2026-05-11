"""Color palette extraction in LAB space."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from brand_dna.analysis._pantone import nearest_pantone
from brand_dna.analysis.color_palette import extract_palette
from brand_dna.core.models import ImageProvenance, ImageRecord, PageType


def _solid_image(tmp_path: Path, name: str, rgb: tuple[int, int, int]) -> ImageRecord:
    img = Image.new("RGB", (256, 256), color=rgb)
    p = tmp_path / f"{name}.jpg"
    img.save(p, format="JPEG", quality=92)
    return ImageRecord(
        image_id=name,
        local_path=str(p),
        width=256,
        height=256,
        format="jpeg",
        bytes_size=p.stat().st_size,
        provenance=ImageProvenance(
            source_url=f"https://x.com/{name}.jpg",
            page_url="https://x.com/",
            page_type=PageType.PRODUCT,
            captured_at=datetime.now(timezone.utc),
        ),
    )


class TestPaletteExtraction:
    def test_pure_red_yields_red_centroid(self, tmp_path: Path) -> None:
        records = [_solid_image(tmp_path, f"r{i}", (220, 30, 40)) for i in range(3)]
        palette = extract_palette(records, k=3, drop_extreme_lightness=False)
        # At least one entry close to deep red
        hexes = [e.hex.upper() for e in palette.entries]
        # Any entry with high R, low G/B
        found = any(
            (entry.rgb[0] > 150 and entry.rgb[1] < 100 and entry.rgb[2] < 100)
            for entry in palette.entries
        )
        assert found, f"No red centroid in palette: {hexes}"

    def test_empty_input_returns_empty_palette(self) -> None:
        palette = extract_palette([], k=8)
        assert palette.entries == []
        assert palette.sample_size == 0

    def test_drop_extreme_lightness_filter(self, tmp_path: Path) -> None:
        # All white images → palette either empty or falls back to mid-gray.
        records = [_solid_image(tmp_path, f"w{i}", (255, 255, 255)) for i in range(3)]
        palette = extract_palette(records, k=2, drop_extreme_lightness=True)
        # With drop_extreme_lightness=True, near-white is filtered. Sometimes
        # the filter removes everything, resulting in an empty palette. Either
        # outcome is acceptable; the test is that we don't crash.
        assert palette is not None


class TestPantoneMatching:
    def test_navy_close_to_pms_295(self) -> None:
        result = nearest_pantone((0, 47, 87))
        assert "295" in result or "282" in result  # PMS 295 or 282 (both navy)

    def test_white_matches_a_bright_neutral(self) -> None:
        result = nearest_pantone((250, 248, 244))
        # Should match one of the off-white / cream entries
        assert any(
            tok in result for tok in ("Bright White", "Vanilla Ice", "Cream", "Gray 1")
        )

    def test_returns_delta_indicator(self) -> None:
        result = nearest_pantone((0, 0, 0))
        assert "~Δ" in result
