#!/usr/bin/env python3
"""A runnable e-commerce support bot built with kerobot.

Run it with no setup (offline keyword mode):

    python examples/support_bot.py

Or with an LLM for natural-language understanding:

    export OPENAI_API_KEY=sk-...
    python examples/support_bot.py --llm

Type messages at the prompt; type 'quit' to exit. Try:
    "where is my order ABC1234"
    "track order 5567 for jane@example.com"     (fills both slots at once)
    "actually I want a refund"                   (context switch mid-flow)
    "this is useless, get me a human"            (escalation)
"""

from __future__ import annotations

import argparse
import os
import sys

# Allow running from the repo root without installing.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kerobot import (  # noqa: E402
    EscalationService,
    Flow,
    FlowRegistry,
    FlowStep,
    Nlu,
    ReplyEngine,
)
from kerobot.nlu import IntentSpec  # noqa: E402
from kerobot.providers import MockProvider, OpenAIProvider  # noqa: E402

# --- domain knowledge: intents the bot understands -----------------------
INTENTS = [
    IntentSpec(
        name="track_order",
        keywords=["track", "where", "order status", "my order", "shipment", "delivery"],
        slots=["order_id", "email"],
        description="Customer wants the status of an existing order.",
    ),
    IntentSpec(
        name="refund",
        keywords=["refund", "money back", "return", "cancel order", "reimburse"],
        slots=["order_id"],
        description="Customer wants a refund or to return an item.",
    ),
    IntentSpec(
        name="hours",
        keywords=["open", "hours", "closing", "when are you"],
        slots=[],
        description="Customer asks about opening hours.",
    ),
    IntentSpec(
        name="general",
        keywords=[],
        slots=[],
        description="Anything else / greetings.",
    ),
]


# --- flows: multi-step, slot-filling -------------------------------------
def _complete_track(convo) -> str:
    oid = convo.slots.get("order_id", "your order")
    return (
        f"Thanks! Order {oid} is out for delivery and should arrive within "
        f"2 business days. Anything else I can help with?"
    )


def _complete_refund(convo) -> str:
    oid = convo.slots.get("order_id", "your order")
    return (
        f"I've started a refund request for order {oid}. You'll get an email "
        f"confirmation shortly. Is there anything else?"
    )


FLOWS = FlowRegistry([
    Flow(
        intent="track_order",
        steps=[
            FlowStep("order_id", "Sure — what's your order number?"),
            FlowStep("email", "And the email on the order, please?"),
        ],
        on_complete=_complete_track,
    ),
    Flow(
        intent="refund",
        steps=[FlowStep("order_id", "I can help with that — what's the order number?")],
        on_complete=_complete_refund,
    ),
])


# --- answers: non-flow intents -------------------------------------------
def answer(intent, convo, nlu) -> str:
    if intent == "hours":
        return "We're open Monday–Friday, 9am to 6pm."
    if intent == "general":
        return "Hi! I can help track an order, start a refund, or share our hours. What do you need?"
    return ""


def build_engine(use_llm: bool) -> ReplyEngine:
    provider = OpenAIProvider() if use_llm else MockProvider()
    nlu = Nlu(
        INTENTS,
        provider=provider,
        typo_map={"oder": "order", "refnd": "refund", "trak": "track"},
    )
    return ReplyEngine(
        nlu=nlu,
        flows=FLOWS,
        answer_fn=answer,
        escalation=EscalationService(),
        handoff_message="I'll connect you with a human agent now — one moment.",
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="kerobot support bot demo")
    ap.add_argument("--llm", action="store_true", help="use OpenAI for NLU")
    args = ap.parse_args()

    engine = build_engine(args.llm)
    mode = "LLM" if args.llm else "offline keyword"
    print(f"kerobot demo ({mode} mode). Type 'quit' to exit.\n")

    convo_id = "demo-1"
    while True:
        try:
            text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if text.lower() in ("quit", "exit"):
            break
        if not text:
            continue
        reply = engine.handle(convo_id, text)
        tag = " [handoff]" if reply.escalated else ""
        print(f"bot> {reply.text}{tag}\n")


if __name__ == "__main__":
    main()
