"""Tests for kerobot.

These encode real lessons from running a support bot in production:
slot-filling must skip already-known slots, loop detection must be fuzzy (not
exact), free-text steps must reject sentences, and escalation must fire on the
right signals. They run fully offline (MockProvider / keyword NLU).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kerobot import (
    EscalationService,
    Flow,
    FlowRegistry,
    FlowStep,
    Nlu,
    ReplyEngine,
)
from kerobot.escalation import EscalationConfig
from kerobot.flows import default_entity_validator, looks_like_sentence_not_entity
from kerobot.nlu import IntentSpec, normalize
from kerobot.types import Conversation


# --- fixtures ------------------------------------------------------------
def build_engine():
    intents = [
        IntentSpec("track_order", ["track", "where", "order"], ["order_id", "email"]),
        IntentSpec("refund", ["refund", "money back"], ["order_id"]),
        IntentSpec("hours", ["hours", "open"], []),
        IntentSpec("general", [], []),
    ]
    nlu = Nlu(intents)

    flows = FlowRegistry([
        Flow(
            intent="track_order",
            steps=[
                FlowStep("order_id", "What's your order number?"),
                FlowStep("email", "What's the email on the order?"),
            ],
            on_complete=lambda c: f"Order {c.slots['order_id']} is on its way.",
        ),
        Flow(
            intent="refund",
            steps=[FlowStep("order_id", "What's the order number?")],
            on_complete=lambda c: f"Refund started for {c.slots['order_id']}.",
        ),
    ])

    def answer(intent, convo, nlu):
        if intent == "hours":
            return "Mon-Fri 9-6."
        return "How can I help?"

    return ReplyEngine(nlu=nlu, flows=flows, answer_fn=answer)


# --- normalization -------------------------------------------------------
def test_normalize_squeezes_repeats_and_applies_typos():
    assert normalize("heeelllooo") == "heelloo"
    assert normalize("trak my oder", {"trak": "track", "oder": "order"}) == "track my order"


def test_normalize_preserves_repeated_digits():
    # Regression: repeated digits in ids/phone numbers must NOT be squeezed.
    assert normalize("track order 1111") == "track order 1111"
    assert normalize("ref 0000 9999") == "ref 0000 9999"


def test_switch_does_not_capture_trigger_message_as_slot():
    # Regression: switching flow mid-step must not treat the switching message
    # ("i want a refund") as the answer to the abandoned flow's pending slot.
    eng = build_engine()
    eng.handle("sw", "track my order")          # pending step: order_id
    reply = eng.handle("sw", "i want a refund")  # switch -> must ASK, not capture
    assert reply.intent == "refund"
    assert "order number" in reply.text.lower()
    assert "refund" not in reply.text.lower() or "?" in reply.text


# --- NLU keyword fallback ------------------------------------------------
def test_keyword_nlu_picks_intent():
    nlu = Nlu([IntentSpec("refund", ["refund"], []), IntentSpec("general", [], [])])
    res = nlu.analyze("I want a refund please")
    assert res.intent == "refund"
    assert res.confidence > 0.3


def test_nlu_extracts_email_and_order_slots():
    nlu = Nlu([IntentSpec("track_order", ["track"], ["order_id", "email"])])
    res = nlu.analyze("track order 12345 jane@example.com")
    assert res.slots.get("email") == "jane@example.com"
    assert res.slots.get("order_id") == "12345"


# --- flow slot-filling ---------------------------------------------------
def test_flow_asks_for_missing_slot():
    eng = build_engine()
    reply = eng.handle("c1", "I want to track my order")
    assert reply.intent == "track_order"
    assert "order number" in reply.text.lower()


def test_flow_skips_slots_provided_up_front():
    """User volunteers everything: the bot should not re-ask, it should finish."""
    eng = build_engine()
    reply = eng.handle("c2", "track order 9981 for sam@example.com")
    assert "9981" in reply.text
    assert "on its way" in reply.text.lower()


def test_flow_collects_slots_across_turns():
    eng = build_engine()
    eng.handle("c3", "where is my order")
    eng.handle("c3", "5567")
    reply = eng.handle("c3", "sam@example.com")
    assert "5567" in reply.text


# --- context switching ---------------------------------------------------
def test_context_switch_abandons_old_flow():
    eng = build_engine()
    eng.handle("c4", "track my order")           # enters track_order
    reply = eng.handle("c4", "actually I want a refund")  # confident switch
    assert reply.intent == "refund"


# --- sentence-as-entity guard -------------------------------------------
def test_sentence_not_entity_detection():
    assert looks_like_sentence_not_entity("what is my order status?")
    assert looks_like_sentence_not_entity("this is a very long thing with many words here")
    assert not looks_like_sentence_not_entity("ABC1234")


def test_validator_rejects_question():
    ok, _ = default_entity_validator("what's going on?")
    assert ok is False
    ok, cleaned = default_entity_validator("ABC1234")
    assert ok is True and cleaned == "ABC1234"


# --- escalation ----------------------------------------------------------
def test_escalation_on_explicit_human_request():
    eng = build_engine()
    reply = eng.handle("c5", "this bot is bad, get me a real agent")
    assert reply.escalated is True


def test_escalation_fuzzy_loop_detection():
    """Near-identical bot replies should trip the loop detector, not just exact."""
    svc = EscalationService(EscalationConfig(loop_similarity=0.8))
    convo = Conversation(id="x")
    convo.add("bot", "Please send your order id.")
    convo.add("user", "huh?")
    convo.add("bot", "Please send the order id.")  # near-identical, not exact
    decision = svc.evaluate(convo, "what?")
    assert decision.escalate is True
    assert decision.reason == "bot_loop_detected"


def test_no_escalation_on_normal_message():
    eng = build_engine()
    reply = eng.handle("c6", "what are your hours")
    assert reply.escalated is False
    assert "9-6" in reply.text
