"""In-memory conversation store with TTL.

Holds `Conversation` state (slots, active flow, history) keyed by conversation
id, and expires idle conversations so long-running processes don't leak. Swap
this class for a Redis/DB-backed one with the same three methods in production.
"""

from __future__ import annotations

import time
from typing import Optional

from .types import Conversation


class ConversationMemory:
    def __init__(self, ttl_seconds: int = 7200) -> None:
        self.ttl = ttl_seconds
        self._store: dict[str, Conversation] = {}

    def get(self, conversation_id: str) -> Conversation:
        self._sweep()
        convo = self._store.get(conversation_id)
        if convo is None:
            convo = Conversation(id=conversation_id)
            self._store[conversation_id] = convo
        return convo

    def save(self, convo: Conversation) -> None:
        convo.updated_at = time.time()
        self._store[convo.id] = convo

    def drop(self, conversation_id: str) -> None:
        self._store.pop(conversation_id, None)

    def _sweep(self) -> None:
        cutoff = time.time() - self.ttl
        stale = [cid for cid, c in self._store.items() if c.updated_at < cutoff]
        for cid in stale:
            del self._store[cid]
