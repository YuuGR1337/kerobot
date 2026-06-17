"""Escalation: know when to hand off to a human.

A support bot that loops forever is worse than no bot. This service watches for
the signals that mean "stop trying, get a person":

- explicit requests ("talk to a human", "real agent")
- frustration keywords repeated across turns
- the bot repeating itself — detected with *fuzzy* matching, not exact equality,
  because near-identical replies ("Please send your order id." vs "Please send
  the order id.") are the real-world loop, and exact matching misses them
- an over-long unresolved conversation

The fuzzy threshold + emoji/punctuation stripping come straight from production
loop-detection fixes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Optional

from .types import Conversation

_DEFAULT_HUMAN_REQUESTS = [
    "human", "real agent", "real person", "speak to someone",
    "talk to a person", "talk to an agent", "live agent", "customer service rep",
]
_DEFAULT_FRUSTRATION = [
    "useless", "not helping", "doesn't help", "stupid bot", "this is ridiculous",
    "frustrated", "angry", "terrible", "worst", "fed up",
]


def _strip(text: str) -> str:
    """Normalize for similarity: lowercase, drop emoji/punctuation, squeeze spaces."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


@dataclass
class EscalationConfig:
    human_requests: list[str] = field(default_factory=lambda: list(_DEFAULT_HUMAN_REQUESTS))
    frustration_words: list[str] = field(default_factory=lambda: list(_DEFAULT_FRUSTRATION))
    loop_similarity: float = 0.8       # 0–1; ≥ this between two bot replies = loop
    max_bot_turns: int = 12            # unresolved length ceiling
    frustration_limit: int = 2         # frustrated user turns before handoff


@dataclass
class EscalationDecision:
    escalate: bool
    reason: Optional[str] = None


class EscalationService:
    def __init__(self, config: Optional[EscalationConfig] = None) -> None:
        self.cfg = config or EscalationConfig()

    def evaluate(self, convo: Conversation, user_text: str) -> EscalationDecision:
        norm = _strip(user_text)

        # 1. Explicit human request.
        if any(p in norm for p in self.cfg.human_requests):
            return EscalationDecision(True, "user_requested_human")

        # 2. Repeated frustration across the conversation.
        frustrated_turns = sum(
            1
            for m in convo.recent(role="user", limit=6)
            if any(w in _strip(m.text) for w in self.cfg.frustration_words)
        )
        if any(w in norm for w in self.cfg.frustration_words):
            frustrated_turns += 1
        if frustrated_turns >= self.cfg.frustration_limit:
            return EscalationDecision(True, "repeated_frustration")

        # 3. Bot talking in circles (fuzzy, not exact).
        if self._is_looping(convo):
            return EscalationDecision(True, "bot_loop_detected")

        # 4. Conversation dragging on without resolution.
        bot_turns = len(convo.recent(role="bot", limit=self.cfg.max_bot_turns + 1))
        if bot_turns > self.cfg.max_bot_turns:
            return EscalationDecision(True, "max_turns_exceeded")

        return EscalationDecision(False)

    def _is_looping(self, convo: Conversation) -> bool:
        bot_msgs = convo.recent(role="bot", limit=3)
        if len(bot_msgs) < 2:
            return False
        a, b = _strip(bot_msgs[-1].text), _strip(bot_msgs[-2].text)
        if not a or not b:
            return False
        ratio = SequenceMatcher(None, a, b).ratio()
        return ratio >= self.cfg.loop_similarity
