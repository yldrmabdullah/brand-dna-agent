"""DossierComposer private helpers — pure logic, no LLM."""

from __future__ import annotations

from brand_dna.synthesis.composer import (
    _coerce_list,
    _coerce_str,
    _extract_embellishments,
    _extract_sustainability,
    _score_by_count,
)


class TestScoreByCount:
    def test_zero_returns_zero(self) -> None:
        assert _score_by_count(0, low=10, high=100) == 0.0

    def test_at_low_threshold(self) -> None:
        # At exactly `low`, score should be 0.3 (transition point)
        assert _score_by_count(10, low=10, high=100) == 0.3

    def test_above_high_caps_at_95(self) -> None:
        assert _score_by_count(1000, low=10, high=100) == 0.95
        assert _score_by_count(100, low=10, high=100) == 0.95

    def test_monotonic(self) -> None:
        prev = -1.0
        for n in [0, 5, 10, 50, 100, 500]:
            score = _score_by_count(n, low=10, high=100)
            assert score >= prev
            prev = score


class TestCoercion:
    def test_coerce_str_strips(self) -> None:
        assert _coerce_str("  hello  ") == "hello"

    def test_coerce_str_non_string(self) -> None:
        assert _coerce_str(None) == ""
        assert _coerce_str(123) == ""

    def test_coerce_list_from_list(self) -> None:
        assert _coerce_list(["a", "b", "", "c"]) == ["a", "b", "c"]

    def test_coerce_list_from_string(self) -> None:
        assert _coerce_list("single") == ["single"]

    def test_coerce_list_empty(self) -> None:
        assert _coerce_list(None) == []
        assert _coerce_list("") == []
        assert _coerce_list([]) == []


class TestEmbellishmentExtraction:
    def test_picks_relevant_terms(self) -> None:
        terms = ["dropped shoulder", "metallic embroidery", "raw hem", "sequin trim"]
        out = _extract_embellishments(terms)
        assert "metallic embroidery" in out
        assert "sequin trim" in out
        assert "dropped shoulder" not in out

    def test_empty_input(self) -> None:
        assert _extract_embellishments(None) == []
        assert _extract_embellishments([]) == []


class TestSustainabilitySignals:
    def test_recognises_keywords(self) -> None:
        values = [
            "Sustainability",
            "Heritage",
            "Organic cotton",
            "Inclusivity",
            "Recycled polyester",
        ]
        out = _extract_sustainability(values)
        assert "Sustainability" in out
        assert "Organic cotton" in out
        assert "Recycled polyester" in out
        assert "Heritage" not in out
