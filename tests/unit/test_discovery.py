"""Discovery layer tests — JSON-LD, OpenGraph, sitemap, page classifier.

These are the *most* important tests in the project. Discovery is the heart
of brand-agnostic crawling: if these parsers break, every downstream stage
fails silently because there's no signal to chew on.
"""

from __future__ import annotations

from brand_dna.core.models import PageType
from brand_dna.discovery.opengraph import (
    extract_canonical_url,
    extract_meta_description,
    extract_opengraph,
    extract_title,
)
from brand_dna.discovery.page_classifier import classify_page
from brand_dna.discovery.sitemap import _parse_sitemap
from brand_dna.discovery.structured_data import (
    collect_product_images,
    extract_structured_data,
    extract_text_from_article_nodes,
)


class TestStructuredData:
    def test_extracts_product(self, sample_product_html: str) -> None:
        nodes = extract_structured_data(sample_product_html)
        assert len(nodes) == 1
        assert nodes[0]["@type"] == "Product"
        assert nodes[0]["name"] == "The Linen Shirt"

    def test_collects_product_images(self, sample_product_html: str) -> None:
        nodes = extract_structured_data(sample_product_html)
        pairs = collect_product_images(nodes)
        urls = [u for u, _ in pairs]
        assert "https://cdn.example.com/p/linen-1.jpg" in urls
        assert "https://cdn.example.com/p/linen-2.jpg" in urls

    def test_extracts_about_text(self, sample_about_html: str) -> None:
        nodes = extract_structured_data(sample_about_html)
        texts = extract_text_from_article_nodes(nodes)
        assert any("considered" in t.lower() or "lisbon" in t.lower() for t in texts)

    def test_ignores_irrelevant_types(self) -> None:
        html = """
        <html><head>
        <script type="application/ld+json">
        {"@type": "WebSite", "name": "Acme"}
        </script>
        <script type="application/ld+json">
        {"@type": "SiteNavigationElement", "name": "Nav"}
        </script>
        </head><body></body></html>
        """
        # WebSite is not in CARE_TYPES (we only care about WebPage / AboutPage)
        nodes = extract_structured_data(html)
        for n in nodes:
            assert n.get("@type") != "SiteNavigationElement"

    def test_handles_graph_wrapper(self) -> None:
        html = """
        <script type="application/ld+json">
        {"@context": "https://schema.org", "@graph": [
            {"@type": "Product", "name": "A", "image": "https://x.com/a.jpg"},
            {"@type": "ItemList", "name": "List"}
        ]}
        </script>
        """
        nodes = extract_structured_data(html)
        types = {n.get("@type") for n in nodes}
        assert "Product" in types
        assert "ItemList" in types


class TestOpenGraph:
    def test_extracts_og_tags(self, sample_product_html: str) -> None:
        og = extract_opengraph(sample_product_html)
        assert og["og:title"] == "The Linen Shirt"
        assert og["og:type"] == "product"
        assert og["og:image"] == "https://cdn.example.com/p/linen.jpg"

    def test_canonical_url(self, sample_product_html: str) -> None:
        assert (
            extract_canonical_url(sample_product_html)
            == "https://example.com/products/linen-shirt"
        )

    def test_meta_description(self, sample_product_html: str) -> None:
        d = extract_meta_description(sample_product_html)
        assert d and "linen" in d.lower()

    def test_title(self, sample_product_html: str) -> None:
        assert extract_title(sample_product_html) == "The Linen Shirt — Acme"


class TestSitemap:
    def test_parses_urlset(self, sample_sitemap_xml: bytes) -> None:
        subs, entries = _parse_sitemap(sample_sitemap_xml, "https://example.com/sitemap.xml")
        assert subs == []
        urls = [e.url for e in entries]
        assert "https://example.com/products/linen-shirt" in urls
        assert "https://example.com/about" in urls
        # Image extension picked up
        shirt = next(e for e in entries if "linen-shirt" in e.url)
        assert "https://cdn.example.com/p/linen-1.jpg" in shirt.image_urls

    def test_parses_sitemap_index(self, sample_sitemap_index: bytes) -> None:
        subs, entries = _parse_sitemap(sample_sitemap_index, "https://example.com/sitemap.xml")
        assert "https://example.com/sitemap-products.xml" in subs
        assert "https://example.com/sitemap-pages.xml" in subs
        assert entries == []

    def test_malformed_xml_returns_empty(self) -> None:
        subs, entries = _parse_sitemap(b"not xml", "https://example.com/")
        assert subs == []
        assert entries == []


class TestPageClassifier:
    def test_jsonld_wins(self) -> None:
        pt = classify_page(
            "https://x.com/random-path",
            structured_data=[{"@type": "Product", "name": "x"}],
        )
        assert pt == PageType.PRODUCT

    def test_og_type_when_no_jsonld(self) -> None:
        pt = classify_page(
            "https://x.com/random-path",
            opengraph={"og:type": "article"},
        )
        assert pt == PageType.BLOG

    def test_url_path_fallback_product(self) -> None:
        assert classify_page("https://x.com/products/sku-123") == PageType.PRODUCT

    def test_url_path_collection(self) -> None:
        assert classify_page("https://x.com/collections/dresses") == PageType.COLLECTION

    def test_url_path_about(self) -> None:
        assert classify_page("https://x.com/about-us") == PageType.ABOUT

    def test_homepage(self) -> None:
        assert classify_page("https://x.com/") == PageType.HOMEPAGE
        assert classify_page("https://x.com") == PageType.HOMEPAGE

    def test_unknown_when_no_signal(self) -> None:
        assert classify_page("https://x.com/some/weird/path") == PageType.UNKNOWN

    def test_url_hints_override(self) -> None:
        pt = classify_page(
            "https://x.com/weird-path",
            url_hints={"lookbook": ["/weird-path"]},
        )
        assert pt == PageType.LOOKBOOK
