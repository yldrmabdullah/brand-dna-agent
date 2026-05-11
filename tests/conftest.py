"""Shared pytest fixtures."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture
def tmp_image_path(tmp_path: Path) -> Path:
    """A 600x800 deterministic test image (gradient)."""
    img = Image.new("RGB", (600, 800))
    pixels = img.load()
    for y in range(800):
        for x in range(600):
            pixels[x, y] = (x % 256, y % 256, (x + y) % 256)
    p = tmp_path / "sample.jpg"
    img.save(p, format="JPEG", quality=90)
    return p


@pytest.fixture
def tmp_image_path_small(tmp_path: Path) -> Path:
    """A 200x200 test image — too small to pass quality filter."""
    img = Image.new("RGB", (200, 200), color=(120, 80, 60))
    p = tmp_path / "small.jpg"
    img.save(p, format="JPEG", quality=85)
    return p


@pytest.fixture
def sample_product_html() -> str:
    return """
    <!doctype html><html><head>
    <title>The Linen Shirt — Acme</title>
    <meta name="description" content="A relaxed linen shirt for warm days.">
    <meta property="og:title" content="The Linen Shirt">
    <meta property="og:type" content="product">
    <meta property="og:image" content="https://cdn.example.com/p/linen.jpg">
    <link rel="canonical" href="https://example.com/products/linen-shirt">
    <script type="application/ld+json">
    {
      "@context": "https://schema.org",
      "@type": "Product",
      "name": "The Linen Shirt",
      "image": ["https://cdn.example.com/p/linen-1.jpg", "https://cdn.example.com/p/linen-2.jpg"],
      "description": "A relaxed linen shirt with mother-of-pearl buttons.",
      "brand": {"@type": "Brand", "name": "Acme"},
      "color": "stone",
      "material": "linen"
    }
    </script>
    </head><body>
    <main>
      <h1>The Linen Shirt</h1>
      <p>Cut from European-grown linen, finished by hand in Porto.</p>
      <img src="/cdn/linen-detail.jpg" alt="Detail of mother-of-pearl buttons" />
    </main>
    </body></html>
    """


@pytest.fixture
def sample_about_html() -> str:
    return """
    <!doctype html><html><head>
    <title>About — Acme</title>
    <meta name="description" content="Acme makes considered everyday clothing.">
    <meta property="og:title" content="About Acme">
    <meta property="og:type" content="website">
    <script type="application/ld+json">
    {
      "@context": "https://schema.org",
      "@type": "AboutPage",
      "description": "Acme was founded in 1998 around a single idea: that clothing should outlast the season that made it. We are based in Lisbon, work with European mills, and prefer slow over loud."
    }
    </script>
    </head><body>
    <main>
      <h1>About Acme</h1>
      <p>Founded in 1998. Lisbon. We design garments meant to outlast their season.</p>
    </main>
    </body></html>
    """


@pytest.fixture
def sample_sitemap_xml() -> bytes:
    return b"""<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"
        xmlns:image="http://www.google.com/schemas/sitemap-image/1.1">
  <url>
    <loc>https://example.com/products/linen-shirt</loc>
    <lastmod>2024-09-01</lastmod>
    <image:image>
      <image:loc>https://cdn.example.com/p/linen-1.jpg</image:loc>
    </image:image>
  </url>
  <url>
    <loc>https://example.com/products/wool-coat</loc>
    <image:image>
      <image:loc>https://cdn.example.com/p/wool-1.jpg</image:loc>
    </image:image>
  </url>
  <url>
    <loc>https://example.com/about</loc>
  </url>
</urlset>
"""


@pytest.fixture
def sample_sitemap_index() -> bytes:
    return b"""<?xml version="1.0" encoding="UTF-8"?>
<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <sitemap><loc>https://example.com/sitemap-products.xml</loc></sitemap>
  <sitemap><loc>https://example.com/sitemap-pages.xml</loc></sitemap>
</sitemapindex>
"""
