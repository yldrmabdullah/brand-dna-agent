"""Prompt templates. Loaded as text files so non-engineers can iterate on them
without touching Python.

Each template uses Python format-string syntax for variable substitution. Keep
templates declarative — no logic.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

_PROMPTS_DIR = Path(__file__).parent.parent.parent.parent / "configs" / "prompts"


@lru_cache(maxsize=32)
def load_prompt(name: str) -> str:
    """Load a prompt template by name (without .md extension)."""
    path = _PROMPTS_DIR / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    return path.read_text(encoding="utf-8")


def render(name: str, **kwargs: object) -> str:
    """Load + format a prompt template. KeyErrors → ValueError with context."""
    template = load_prompt(name)
    try:
        return template.format(**kwargs)
    except KeyError as exc:
        raise ValueError(
            f"Missing variable {exc} when rendering prompt '{name}'"
        ) from exc
