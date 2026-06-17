# kerobot

[![Tests](https://img.shields.io/badge/tests-14%20passing-brightgreen)]() [![Python](https://img.shields.io/badge/python-3.9%2B-blue)]() [![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**AI livechat auto-reply framework — build customer-support assistants that actually close conversations instead of looping forever or trapping users in rigid menus.**

Most support bots fail the same ways: they don't understand informal, misspelled messages; they re-ask for things the customer already said; they can't tell when a user changes the subject; and they never know when to give up and fetch a human. `kerobot` is a small, dependency-free Python framework that handles all four, built from patterns that held up running a live support bot across thousands of real conversations.

It runs **offline out of the box** (deterministic keyword understanding, zero config, zero cost) and upgrades to **LLM-powered understanding** by setting one environment variable.

```
NLU (intent + slots) ─► flow routing ─► slot-filling ─► escalation ─► memory
```

---

## Why it exists

| Real-world failure | What kerobot does |
|---|---|
| "wher is my oder" — typo/slang not understood | Light normalization + typo map before classification |
| Bot asks for the order number the user already gave | Slot-filling **skips** any slot already known |
| User switches topic mid-flow, bot stays stuck | Context-switch detection abandons the stale flow |
| Bot repeats itself in a dead loop | **Fuzzy** loop detection (not exact match) triggers handoff |
| Free-text step accepts "what's going on?" as a name | Sentence-vs-entity guard rejects it and re-prompts |
| Conversation drags on, customer gets angry | Frustration + length signals escalate to a human |

These aren't hypothetical — each is a guard added after it broke in production.

---

## Quick start

No dependencies, no API key:

```bash
git clone https://github.com/YuuGR1337/kerobot
cd kerobot
python examples/support_bot.py
```

```
you> where is my order
bot> Sure — what's your order number?
you> 5567
bot> And the email on the order, please?
you> jane@example.com
bot> Thanks! Order 5567 is out for delivery and should arrive within 2 business days.

you> actually I want a refund
bot> I can help with that — what's the order number?
you> 8899
bot> I've started a refund request for order 8899.

you> this is useless, get me a human
bot> I'll connect you with a human agent now — one moment. [handoff]
```

Note the customer can provide everything at once and skip the questions entirely:

```
you> track order 9981 for sam@example.com
bot> Thanks! Order 9981 is out for delivery and should arrive within 2 business days.
```

### With an LLM (better understanding of messy input)

```bash
pip install openai
export OPENAI_API_KEY=sk-...
python examples/support_bot.py --llm
```

The LLM handles intent + slot extraction in a single cheap call; if it's
unavailable or unsure, the bot automatically falls back to keyword rules. **It
never goes dead.**

---

## Build your own bot

```python
from kerobot import Nlu, ReplyEngine, Flow, FlowRegistry, FlowStep
from kerobot.nlu import IntentSpec

intents = [
    IntentSpec("track_order", ["track", "where", "order"], slots=["order_id", "email"]),
    IntentSpec("hours", ["hours", "open"]),
    IntentSpec("general"),
]

flows = FlowRegistry([
    Flow(
        intent="track_order",
        steps=[
            FlowStep("order_id", "What's your order number?"),
            FlowStep("email", "What's the email on the order?"),
        ],
        on_complete=lambda c: f"Order {c.slots['order_id']} is on its way!",
    ),
])

def answer(intent, convo, nlu):
    if intent == "hours":
        return "We're open Mon–Fri, 9am–6pm."
    return "Hi! I can track an order or share our hours. What do you need?"

engine = ReplyEngine(nlu=Nlu(intents), flows=flows, answer_fn=answer)

reply = engine.handle("conversation-123", "where is my order")
print(reply.text)        # -> "What's your order number?"
print(reply.escalated)   # -> False
```

That's the whole API surface for a working bot: declare intents, declare flows, write an `answer` function for everything else.

---

## How it works

- **NLU** (`kerobot/nlu.py`) — normalizes the message, then classifies intent and extracts slots. LLM first, keyword rules as fallback and offline default.
- **Flows** (`kerobot/flows.py`) — ordered slot-filling steps that skip already-known slots and validate input.
- **Router** (`kerobot/router.py`) — continues the active flow or switches when the user confidently changes intent.
- **Escalation** (`kerobot/escalation.py`) — fuzzy loop detection, frustration and human-request signals, length ceiling.
- **Memory** (`kerobot/memory.py`) — per-conversation state with TTL; swap for Redis/DB by reimplementing three methods.
- **Engine** (`kerobot/engine.py`) — wires it all together for one turn.

Each piece is injected into the engine, so you can replace or test any of them in isolation.

---

## Testing

```bash
pip install pytest
pytest
```

The suite (runs fully offline) covers slot-skipping, cross-turn slot collection, context switching, the sentence-vs-entity guard, and fuzzy loop escalation.

---

## Roadmap

- Persistent memory backends (Redis, Postgres)
- Built-in webhook adapters (web widget, Telegram, WhatsApp)
- Multi-language normalization packs

## License

MIT — see [LICENSE](LICENSE).
