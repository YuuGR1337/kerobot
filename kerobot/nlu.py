"""Natural-language understanding: intent classification + slot extraction.

Two-tier design, mirroring what works in production:

1. An LLM provider classifies intent and extracts slots in a single call.
2. If the provider is unavailable or unsure, deterministic keyword rules take
   over. Rules are also the offline default, so the bot is never dead.

Light normalization (lowercasing, collapsing repeated characters, typo lookup)
runs first — it cheaply rescues informal, misspelled real-world input before it
ever reaches the model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from .providers import LLMProvider, MockProvider
from .types import NluResult


@dataclass
class IntentSpec:
    """Declarative definition of one intent the bot understands."""

    name: str
    keywords: list[str] = field(default_factory=list)
    slots: list[str] = field(default_factory=list)
    description: str = ""


# Collapse 3+ repeated *letters*: "heeelllooo" -> "heelloo" (keep doubles).
# Digits are deliberately excluded so order ids / phone numbers like "1111"
# or "0000" are never mangled.
_REPEAT_RE = re.compile(r"([^\W\d_])\1{2,}")


def normalize(text: str, typo_map: Optional[dict[str, str]] = None) -> str:
    """Lowercase, trim, squeeze repeated characters, apply typo corrections."""
    text = text.lower().strip()
    text = _REPEAT_RE.sub(r"\1\1", text)
    if typo_map:
        words = [typo_map.get(w, w) for w in text.split()]
        text = " ".join(words)
    return text


class Nlu:
    def __init__(
        self,
        intents: list[IntentSpec],
        provider: Optional[LLMProvider] = None,
        typo_map: Optional[dict[str, str]] = None,
        default_intent: str = "general",
    ) -> None:
        self.intents = {i.name: i for i in intents}
        self.provider = provider or MockProvider()
        self.typo_map = typo_map or {}
        self.default_intent = default_intent

    def analyze(self, text: str) -> NluResult:
        normalized = normalize(text, self.typo_map)

        # Tier 1: ask the model (no-op for MockProvider -> empty dict).
        llm = self._analyze_llm(normalized)
        if llm and llm.intent in self.intents and llm.is_confident:
            return llm

        # Tier 2: deterministic keyword fallback.
        kw = self._analyze_keywords(normalized)
        # Prefer the model's slots if it returned any but lost on confidence.
        if llm and llm.slots:
            kw.slots = {**llm.slots, **kw.slots}
        return kw

    # -- tier 1 -----------------------------------------------------------
    def _analyze_llm(self, normalized: str) -> Optional[NluResult]:
        prompt = self._build_prompt(normalized)
        try:
            data = self.provider.classify(prompt)
        except Exception:
            return None
        if not data:
            return None
        intent = str(data.get("intent", "")).strip()
        if not intent:
            return None
        try:
            confidence = float(data.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        slots = data.get("slots") or {}
        if not isinstance(slots, dict):
            slots = {}
        return NluResult(
            intent=intent,
            confidence=max(0.0, min(1.0, confidence)),
            slots={k: v for k, v in slots.items() if v not in (None, "")},
            normalized=normalized,
        )

    def _build_prompt(self, normalized: str) -> str:
        lines = ["Classify the user message into one intent and extract slots.", ""]
        lines.append("Intents:")
        for spec in self.intents.values():
            slot_str = f" slots={spec.slots}" if spec.slots else ""
            desc = f" — {spec.description}" if spec.description else ""
            lines.append(f"- {spec.name}{slot_str}{desc}")
        lines += [
            "",
            f'User message: "{normalized}"',
            "",
            'Reply as JSON: {"intent": "...", "confidence": 0.0-1.0, '
            '"slots": {"name": "value"}}',
        ]
        return "\n".join(lines)

    # -- tier 2 -----------------------------------------------------------
    def _analyze_keywords(self, normalized: str) -> NluResult:
        best_intent = self.default_intent
        best_hits = 0
        for spec in self.intents.values():
            hits = sum(1 for kw in spec.keywords if kw in normalized)
            if hits > best_hits:
                best_hits, best_intent = hits, spec.name

        if best_hits == 0:
            return NluResult(self.default_intent, 0.3, {}, normalized)

        # More keyword hits -> higher confidence, capped below LLM certainty.
        confidence = min(0.5 + 0.15 * best_hits, 0.85)
        slots = self._extract_slots(normalized, self.intents[best_intent])
        return NluResult(best_intent, confidence, slots, normalized)

    def _extract_slots(self, normalized: str, spec: IntentSpec) -> dict:
        """Pull common, well-shaped entities by regex.

        Intentionally conservative — only extracts patterns that are unambiguous
        (emails, order ids, long numbers). Free-text slots are left to the LLM
        or to flow-level prompting.
        """
        slots: dict[str, str] = {}
        for slot in spec.slots:
            if slot in ("email",):
                m = re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", normalized)
                if m:
                    slots["email"] = m.group(0)
            elif slot in ("order_id", "order", "ticket", "reference"):
                m = re.search(r"\b[a-z]{0,3}[-#]?\d{4,}\b", normalized)
                if m:
                    slots[slot] = m.group(0).lstrip("#")
        return slots
