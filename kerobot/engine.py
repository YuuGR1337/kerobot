"""ReplyEngine — orchestrates one conversation turn end to end.

Pipeline per user message:

    escalation check ─► NLU ─► route (continue / switch) ─► flow slot-filling
                                                          └► intent answer

The engine is deliberately small: it wires the pieces together and owns the
turn lifecycle. Each piece (NLU, router, flows, escalation, memory) is injected
so it can be replaced or tested in isolation.
"""

from __future__ import annotations

from typing import Callable, Optional

from .escalation import EscalationService
from .flows import FlowRegistry
from .memory import ConversationMemory
from .nlu import Nlu
from .router import FlowRouter
from .types import BotReply, Conversation, NluResult

# (intent, conversation, nlu) -> reply text, for non-flow ("answer") intents.
AnswerFn = Callable[[str, Conversation, NluResult], str]


class ReplyEngine:
    def __init__(
        self,
        nlu: Nlu,
        flows: Optional[FlowRegistry] = None,
        answer_fn: Optional[AnswerFn] = None,
        escalation: Optional[EscalationService] = None,
        memory: Optional[ConversationMemory] = None,
        handoff_message: str = "Let me connect you with a member of our team.",
    ) -> None:
        self.nlu = nlu
        self.flows = flows or FlowRegistry()
        self.answer_fn = answer_fn or (lambda intent, convo, nlu: "")
        self.escalation = escalation or EscalationService()
        self.memory = memory or ConversationMemory()
        self.handoff_message = handoff_message

    def handle(self, conversation_id: str, text: str) -> BotReply:
        convo = self.memory.get(conversation_id)
        convo.add("user", text)

        # Already handed off — stay quiet so a human owns the thread.
        if convo.escalated:
            return self._finish(convo, BotReply(
                text="", intent="escalated", escalated=True,
                handoff_reason=convo.escalated_reason,
            ))

        # 1. Escalation has priority over everything else.
        esc = self.escalation.evaluate(convo, text)
        if esc.escalate:
            convo.escalated = True
            convo.escalated_reason = esc.reason
            reply = BotReply(
                text=self.handoff_message, intent="escalate",
                escalated=True, handoff_reason=esc.reason,
            )
            convo.add("bot", reply.text)
            return self._finish(convo, reply)

        # 2. Understand the message.
        nlu = self.nlu.analyze(text)

        # 3. Merge any freshly extracted slots into conversation state.
        if nlu.slots:
            convo.slots.update(nlu.slots)

        # 4. Decide which flow/intent this turn belongs to.
        decision = self.router.route(convo, nlu)
        if decision.switched:
            # New topic: drop the old flow's half-filled slots AND its pending
            # step, so the switch-triggering message is never mistaken for the
            # answer to a step that belonged to the abandoned flow.
            convo.slots = dict(nlu.slots)
            convo.active_step = None
        convo.active_intent = decision.intent

        # 5. Flow intent -> slot-filling; otherwise -> answer.
        flow = self.flows.get(decision.intent)
        if flow is not None:
            reply = self._advance_flow(convo, nlu)
        else:
            text_out = self.answer_fn(decision.intent, convo, nlu) or (
                "I'm not sure I understood — could you rephrase that?"
            )
            reply = BotReply(text=text_out, intent=decision.intent, slots=dict(convo.slots))

        reply.meta["route"] = decision.reason
        reply.meta["confidence"] = nlu.confidence
        convo.add("bot", reply.text)
        return self._finish(convo, reply)

    # -- internals --------------------------------------------------------
    @property
    def router(self) -> FlowRouter:
        # Lazily bind a router to the active registry.
        if not hasattr(self, "_router"):
            self._router = FlowRouter(self.flows)
        return self._router

    def _advance_flow(self, convo: Conversation, nlu: NluResult) -> BotReply:
        flow = self.flows.get(convo.active_intent)
        assert flow is not None

        # If we're waiting on a specific step, treat this message as its answer.
        if convo.active_step:
            step = next((s for s in flow.steps if s.slot == convo.active_step), None)
            if step is not None:
                ok, cleaned = step.validator(nlu.normalized or "")
                if not ok and not convo.slots.get(step.slot):
                    return BotReply(text=cleaned, intent=flow.intent, slots=dict(convo.slots))
                if ok:
                    convo.slots[step.slot] = cleaned

        # Find the next still-missing slot.
        nxt = flow.next_step(convo)
        if nxt is None:
            convo.active_step = None
            done_intent = flow.intent
            text_out = flow.on_complete(convo)
            # Flow finished: reset so the next message starts fresh.
            convo.active_intent = None
            convo.slots = {}
            return BotReply(text=text_out, intent=done_intent, slots={})

        convo.active_step = nxt.slot
        return BotReply(text=nxt.prompt, intent=flow.intent, slots=dict(convo.slots))

    def _finish(self, convo: Conversation, reply: BotReply) -> BotReply:
        self.memory.save(convo)
        return reply
