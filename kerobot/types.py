"""Core data types shared across the pipeline."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Message:
    """A single chat message in a conversation."""

    role: str  # "user" | "bot" | "agent"
    text: str
    ts: float = field(default_factory=time.time)


@dataclass
class NluResult:
    """Output of the NLU layer for one user message.

    `intent` is an application-defined string (e.g. "track_order").
    `confidence` is 0.0–1.0. `slots` holds extracted entities.
    """

    intent: str
    confidence: float
    slots: dict[str, Any] = field(default_factory=dict)
    normalized: str = ""

    @property
    def is_confident(self) -> bool:
        return self.confidence >= 0.6


@dataclass
class BotReply:
    """What the engine returns for a turn."""

    text: str
    intent: str
    escalated: bool = False
    handoff_reason: Optional[str] = None
    slots: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Conversation:
    """Mutable per-conversation state."""

    id: str
    messages: list[Message] = field(default_factory=list)
    active_intent: Optional[str] = None
    active_step: Optional[str] = None
    slots: dict[str, Any] = field(default_factory=dict)
    escalated: bool = False
    escalated_reason: Optional[str] = None
    updated_at: float = field(default_factory=time.time)

    def add(self, role: str, text: str) -> Message:
        msg = Message(role=role, text=text)
        self.messages.append(msg)
        self.updated_at = time.time()
        return msg

    def recent(self, role: Optional[str] = None, limit: int = 10) -> list[Message]:
        msgs = [m for m in self.messages if role is None or m.role == role]
        return msgs[-limit:]
