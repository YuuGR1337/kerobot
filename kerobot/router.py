"""Flow routing with context-switch detection.

Decides whether an incoming message continues the active flow or switches to a
new one. The rule that matters in practice: if the user is mid-flow but clearly
raises a *different*, confidently-classified intent, abandon the old flow rather
than trapping them in it. Low-confidence shifts keep the current flow so a
mumbled aside doesn't derail a checkout.
"""

from __future__ import annotations

from dataclasses import dataclass

from .flows import FlowRegistry
from .types import Conversation, NluResult


@dataclass
class RouteDecision:
    intent: str
    switched: bool
    reason: str


class FlowRouter:
    # Default switch bar = the "confident" floor (NluResult.is_confident). A
    # confident, handleable, *different* intent is enough to switch; raise this
    # toward 0.9 for LLM deployments that want stickier flows.
    def __init__(self, registry: FlowRegistry, switch_threshold: float = 0.6) -> None:
        self.registry = registry
        self.switch_threshold = switch_threshold

    def route(self, convo: Conversation, nlu: NluResult) -> RouteDecision:
        active = convo.active_intent

        # No active flow: take the new intent if we have a flow for it.
        if not active:
            return RouteDecision(nlu.intent, switched=False, reason="no_active_flow")

        # Same intent: continue.
        if nlu.intent == active:
            return RouteDecision(active, switched=False, reason="continue")

        # Different, confidently-classified intent that we can actually handle:
        # switch and clear the old flow's progress.
        if (
            nlu.intent != active
            and nlu.is_confident
            and nlu.confidence >= self.switch_threshold
            and self.registry.has(nlu.intent)
        ):
            return RouteDecision(nlu.intent, switched=True, reason="context_switch")

        # Otherwise stay put — likely an aside or low-confidence noise.
        return RouteDecision(active, switched=False, reason="stay_low_confidence")
