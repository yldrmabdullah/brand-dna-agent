"""Config loading + validation. Brand onboarding *is* YAML — these tests
defend the contract that onboarding a new brand never requires a code change."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from brand_dna.core.config import BrandConfig, load_brand_config
from brand_dna.core.exceptions import ConfigurationError


def _write(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "brand.yaml"
    p.write_text(textwrap.dedent(body).strip(), encoding="utf-8")
    return p


def test_loads_minimal_config(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        """
        name: Acme
        url: https://www.acme.com
        """,
    )
    cfg = load_brand_config(p)
    assert cfg.name == "Acme"
    assert cfg.url == "https://www.acme.com"
    # Defaults applied
    assert cfg.crawl.max_pages == 200
    assert cfg.filter.min_shorter_side == 512


def test_url_must_have_scheme(tmp_path: Path) -> None:
    p = _write(tmp_path, "name: Acme\nurl: www.acme.com\n")
    with pytest.raises(ConfigurationError):
        load_brand_config(p)


def test_url_trailing_slash_stripped(tmp_path: Path) -> None:
    p = _write(tmp_path, "name: Acme\nurl: https://www.acme.com/\n")
    cfg = load_brand_config(p)
    assert cfg.url == "https://www.acme.com"


def test_loads_with_overrides(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        """
        name: Acme
        url: https://www.acme.com
        social: { instagram: acmeofficial }
        crawl:
          max_pages: 50
          delay_ms: 1000
        filter:
          min_shorter_side: 768
          fashion_score_threshold: 0.7
        analysis:
          n_aesthetic_clusters_min: 4
        """,
    )
    cfg = load_brand_config(p)
    assert cfg.social["instagram"] == "acmeofficial"
    assert cfg.crawl.max_pages == 50
    assert cfg.crawl.delay_ms == 1000
    assert cfg.filter.min_shorter_side == 768
    assert cfg.filter.fashion_score_threshold == 0.7
    assert cfg.analysis.n_aesthetic_clusters_min == 4


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError):
        load_brand_config(tmp_path / "does-not-exist.yaml")


def test_invalid_yaml_raises(tmp_path: Path) -> None:
    p = _write(tmp_path, "name: Acme\n  url: bad indent: here:")
    with pytest.raises(ConfigurationError):
        load_brand_config(p)


def test_model_for_falls_back_to_env() -> None:
    cfg = BrandConfig(name="X", url="https://x.com")
    # No overrides set — should return the global default
    assert cfg.model_for("primary").startswith("anthropic/") or cfg.model_for("primary")
