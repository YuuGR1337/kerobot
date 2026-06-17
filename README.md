# kerobot

[![Tests](https://img.shields.io/badge/tests-14%20passing-brightgreen)]() [![Python](https://img.shields.io/badge/python-3.9%2B-blue)]() [![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**Your support bot loops forever, re-asks for the order number the customer already gave, and never knows when to fetch a human — so people give up and your inbox fills with "is anyone there?"**

kerobot is an AI livechat auto-reply framework that fixes exactly those failures: it understands informal, misspelled messages; remembers what the customer already said and skips ahead; notices when they change the subject; and hands off to a human the moment it's stuck — instead of looping. A small, dependency-free Python framework built from patterns that held up across thousands of real support conversations.

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

## Use cases

kerobot is a good fit if you're trying to:

- build an **AI customer-support chatbot** that escalates to a human instead of looping
- add a **Python auto-reply bot** to a website live-chat widget, Telegram, or WhatsApp
- replace a rigid **menu/keyword FAQ bot** with one that understands natural language
- handle **order tracking, refunds, and support triage** automatically
- prototype a **conversational AI assistant** without paying for a SaaS chatbot platform
- run an **on-premise / self-hosted chatbot** with no vendor lock-in and an offline fallback

Think of it as a lightweight, self-hosted alternative to hosted chatbot builders — you own the code and the data.

---

## Roadmap

- Persistent memory backends (Redis, Postgres)
- Built-in webhook adapters (web widget, Telegram, WhatsApp)
- Multi-language normalization packs

## License

MIT — see [LICENSE](LICENSE).
