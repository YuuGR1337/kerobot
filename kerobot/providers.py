"""LLM providers.

The engine only depends on the small `LLMProvider` protocol, so you can plug in
any backend. Two are shipped:

- `MockProvider`   — no network, deterministic. Used in tests and as the default
                     so the project runs offline immediately.
- `OpenAIProvider` — calls the OpenAI Chat Completions API if `openai` is
                     installed and an API key is configured.
"""

from __future__ import annotations

import json
import os
import re
from typing import Optional, Protocol


class LLMProvider(Protocol):
    """Minimal interface the NLU layer needs."""

    def classify(self, prompt: str) -> dict: ...


class MockProvider:
    """Offline provider.

    Returns an empty result so the NLU layer transparently falls back to its
    deterministic keyword rules. This keeps the whole pipeline runnable with
    zero configuration and zero cost — useful for local dev, CI, and demos.
    """

    def classify(self, prompt: str) -> dict:  # noqa: ARG002 - signature parity
        return {}


class OpenAIProvider:
    """Thin wrapper over the OpenAI chat API for intent + slot extraction.

    Uses a cheap model by default; NLU classification is a tiny task and does
    not need a frontier model. The model is asked to return strict JSON.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._client = None

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI  # imported lazily so it stays optional
        except ImportError as exc:  # pragma: no cover - depends on env
            raise RuntimeError(
                "OpenAIProvider requires the 'openai' package. "
                "Install with: pip install openai"
            ) from exc
        if not self._api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        self._client = OpenAI(api_key=self._api_key)
        return self._client

    def classify(self, prompt: str) -> dict:
        client = self._ensure_client()
        resp = client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an intent + slot classifier for a customer "
                        "support bot. Reply with strict JSON only."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )
        content = resp.choices[0].message.content or "{}"
        return _safe_json(content)


def _safe_json(text: str) -> dict:
    """Best-effort JSON parse — tolerates code fences and surrounding prose."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
    return {}
