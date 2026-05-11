"""LLM client over OpenRouter (OpenAI-compatible endpoint).

Three roles, one client:
- `primary`: vision + text reasoning for nuanced analysis (default: Claude Sonnet 4.6)
- `fast`: high-volume light tasks like image captioning (default: Gemini Flash)
- `synthesis`: final dossier composition (default: Claude Opus)

Per-call cost tracking via OpenRouter's response metadata so we can answer
"$/brand" precisely in the architecture doc.

Why OpenRouter:
- Single key, swap providers via config — model-agnostic stance matches the
  consumer's posture (Refabric doesn't name-drop models either).
- Built-in fallback routing if a provider has an outage.
"""

from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import httpx
from openai import APIError, APIStatusError, AsyncOpenAI, RateLimitError
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from brand_dna.core.config import AppSettings, settings
from brand_dna.core.exceptions import LLMError
from brand_dna.core.observability import get_logger

logger = get_logger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


@dataclass
class LLMResponse:
    """Normalised response. Cost is OpenRouter's reported figure when present."""

    text: str
    model: str
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class UsageLedger:
    """Aggregates LLM cost across a single run. Surfaced in run metadata."""

    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    calls: int = 0

    def add(self, resp: LLMResponse) -> None:
        self.tokens_in += resp.tokens_in
        self.tokens_out += resp.tokens_out
        self.cost_usd += resp.cost_usd
        self.calls += 1


def _image_to_data_uri(image_path: str | Path) -> str:
    """Encode a local image as a data URI for vision messages."""
    p = Path(image_path)
    ext = p.suffix.lstrip(".").lower() or "jpeg"
    if ext == "jpg":
        ext = "jpeg"
    raw = p.read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:image/{ext};base64,{b64}"


_JSON_FENCE = re.compile(r"```(?:json)?\s*([\s\S]+?)\s*```", re.MULTILINE)


def _coerce_json(text: str) -> Any:
    """Best-effort JSON extraction. Strips code fences if the model emitted them."""
    text = text.strip()
    if not text:
        raise LLMError("Empty LLM response when JSON was expected")
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try fenced
    match = _JSON_FENCE.search(text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    # Last resort: locate the outermost {...} or [...]
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise LLMError(f"Could not parse JSON from model output: {text[:200]}...")


class LLMClient:
    """Async client around OpenRouter. Owns retries, cost ledger, and JSON coercion."""

    def __init__(self, app_settings: AppSettings | None = None) -> None:
        self._settings = app_settings or settings
        if not self._settings.openrouter_api_key:
            logger.warning(
                "llm.no_api_key",
                hint="Set OPENROUTER_API_KEY in .env. LLM calls will fail.",
            )
        self.ledger = UsageLedger()
        self._client = AsyncOpenAI(
            api_key=self._settings.openrouter_api_key or "missing",
            base_url=OPENROUTER_BASE_URL,
            default_headers={
                # OpenRouter convention — used for their analytics + attribution.
                "HTTP-Referer": self._settings.openrouter_referer or "",
                "X-Title": self._settings.openrouter_app_title,
            },
            timeout=httpx.Timeout(60.0, connect=10.0),
        )

    async def aclose(self) -> None:
        await self._client.close()

    # ─── Core call ────────────────────────────────────────────────────────

    async def chat(
        self,
        prompt: str,
        *,
        model: str,
        system: str | None = None,
        images: Iterable[str | Path] | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1500,
        json_mode: bool = False,
    ) -> LLMResponse:
        """One-shot chat. Supports vision via `images` and JSON output via `json_mode`."""

        # Build message content
        user_content: list[dict[str, Any]] | str
        if images:
            parts: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
            for img in images:
                parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": _image_to_data_uri(img)},
                    }
                )
            user_content = parts
        else:
            user_content = prompt

        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user_content})

        extra: dict[str, Any] = {}
        if json_mode:
            # Not every model on OpenRouter honors this — we still defensively
            # coerce in `chat_json`. But sending it helps the ones that do.
            extra["response_format"] = {"type": "json_object"}

        retry = AsyncRetrying(
            reraise=True,
            stop=stop_after_attempt(4),
            wait=wait_exponential(multiplier=1.5, min=1, max=20),
            retry=retry_if_exception_type((RateLimitError, APIStatusError)),
        )

        async for attempt in retry:
            with attempt:
                logger.debug("llm.request", model=model, json_mode=json_mode)
                try:
                    completion = await self._client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        **extra,
                    )
                except APIError as exc:
                    logger.error("llm.api_error", model=model, error=str(exc))
                    raise LLMError(f"OpenRouter call failed: {exc}") from exc

        # ── Extract + ledger ────────────────────────────────────────────
        raw = completion.model_dump() if hasattr(completion, "model_dump") else {}
        text = (completion.choices[0].message.content or "").strip()
        usage = raw.get("usage") or {}
        tokens_in = int(usage.get("prompt_tokens", 0) or 0)
        tokens_out = int(usage.get("completion_tokens", 0) or 0)

        # OpenRouter sometimes returns cost in usage or in a top-level field.
        cost = 0.0
        for path in (("usage", "cost"), ("cost",)):
            val: Any = raw
            for key in path:
                val = val.get(key) if isinstance(val, dict) else None
                if val is None:
                    break
            if isinstance(val, (int, float)):
                cost = float(val)
                break

        resp = LLMResponse(
            text=text,
            model=model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost,
            raw=raw,
        )
        self.ledger.add(resp)
        logger.debug(
            "llm.response",
            model=model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost,
        )
        return resp

    async def chat_json(
        self,
        prompt: str,
        *,
        model: str,
        system: str | None = None,
        images: Iterable[str | Path] | None = None,
        temperature: float = 0.1,
        max_tokens: int = 2000,
    ) -> tuple[Any, LLMResponse]:
        """Chat that expects JSON. Returns (parsed_json, raw_response)."""
        resp = await self.chat(
            prompt,
            model=model,
            system=system,
            images=images,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=True,
        )
        try:
            data = _coerce_json(resp.text)
        except LLMError:
            logger.warning(
                "llm.json_parse_failed",
                model=model,
                sample=resp.text[:200],
            )
            raise
        return data, resp


_global_client: LLMClient | None = None


def get_llm_client() -> LLMClient:
    global _global_client
    if _global_client is None:
        _global_client = LLMClient()
    return _global_client
