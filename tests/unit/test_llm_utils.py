"""LLM client utilities: JSON coercion, image encoding.

We don't make real API calls in unit tests. The retry / cost-tracking
integration is tested via mocks in a future integration suite.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from brand_dna.core.exceptions import LLMError
from brand_dna.llm.client import _coerce_json, _image_to_data_uri


class TestCoerceJson:
    def test_clean_json(self) -> None:
        assert _coerce_json('{"a": 1, "b": "two"}') == {"a": 1, "b": "two"}

    def test_array_root(self) -> None:
        assert _coerce_json("[1, 2, 3]") == [1, 2, 3]

    def test_strips_code_fence(self) -> None:
        text = '```json\n{"a": 1}\n```'
        assert _coerce_json(text) == {"a": 1}

    def test_strips_unlabelled_fence(self) -> None:
        text = '```\n{"a": 1}\n```'
        assert _coerce_json(text) == {"a": 1}

    def test_extracts_object_from_chatty_response(self) -> None:
        text = 'Sure! Here you go: {"a": 1, "b": [2, 3]} — let me know.'
        assert _coerce_json(text) == {"a": 1, "b": [2, 3]}

    def test_raises_on_empty(self) -> None:
        with pytest.raises(LLMError):
            _coerce_json("")

    def test_raises_on_unparseable(self) -> None:
        with pytest.raises(LLMError):
            _coerce_json("just some prose with no JSON anywhere")


class TestImageEncoding:
    def test_encodes_to_data_uri(self, tmp_path: Path) -> None:
        from PIL import Image
        p = tmp_path / "x.png"
        Image.new("RGB", (10, 10), color=(255, 0, 0)).save(p, format="PNG")
        uri = _image_to_data_uri(p)
        assert uri.startswith("data:image/png;base64,")
        assert len(uri) > 60

    def test_jpg_extension_normalised(self, tmp_path: Path) -> None:
        from PIL import Image
        p = tmp_path / "x.jpg"
        Image.new("RGB", (10, 10)).save(p, format="JPEG")
        uri = _image_to_data_uri(p)
        assert uri.startswith("data:image/jpeg;base64,")
