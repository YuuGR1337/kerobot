"""Multi-step flows with slot-filling.

A flow is an ordered list of steps; each step needs one slot. The key behaviour
(learned the hard way in production) is that a flow must *skip* any step whose
slot is already known — so a user who volunteers everything up front
("track order ABC1234 for jane@x.com") reaches the end in one turn instead of
being re-asked for what they already said.

Each step can validate its input; a rejected value re-prompts instead of
advancing. The default validator guards against a common failure: accepting a
question or a whole sentence as if it were a short entity (a name, an id).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from .types import Conversation

# (raw_value) -> (is_valid, cleaned_value_or_error)
Validator = Callable[[str], tuple[bool, str]]
# (conversation) -> reply text when the flow completes
Completion = Callable[[Conversation], str]


def looks_like_sentence_not_entity(text: str) -> bool:
    """True if `text` reads like a question/sentence rather than a short entity.

    Mirrors a real bug fix: free-text capture steps were swallowing things like
    "what is my order status?" as if it were a name/id.
    """
    t = text.strip()
    if "?" in t:
        return True
    if len(t.split()) > 5:
        return True
    return False


def default_entity_validator(value: str) -> tuple[bool, str]:
    value = value.strip()
    if not value:
        return False, "That looks empty — could you send it again?"
    if looks_like_sentence_not_entity(value):
        return False, "Sorry, that doesn't look right — could you send just that detail?"
    return True, value


@dataclass
class FlowStep:
    slot: str
    prompt: str
    validator: Validator = default_entity_validator


@dataclass
class Flow:
    intent: str
    steps: list[FlowStep]
    on_complete: Completion
    # Optional fast path when all slots are already present.
    complete_prompt: Optional[str] = None

    def next_step(self, convo: Conversation) -> Optional[FlowStep]:
        """Return the first step whose slot is still missing, or None if done."""
        for step in self.steps:
            if not convo.slots.get(step.slot):
                return step
        return None


class FlowRegistry:
    def __init__(self, flows: Optional[list[Flow]] = None) -> None:
        self._flows: dict[str, Flow] = {}
        for f in flows or []:
            self.register(f)

    def register(self, flow: Flow) -> None:
        self._flows[flow.intent] = flow

    def get(self, intent: str) -> Optional[Flow]:
        return self._flows.get(intent)

    def has(self, intent: str) -> bool:
        return intent in self._flows
