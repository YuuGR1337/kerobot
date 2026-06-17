"""
kerobot — a framework for AI auto-reply / customer-service assistants.

Turns repetitive support chat into a structured pipeline:
NLU (intent + slots) -> flow routing -> slot-filling -> escalation -> memory.

Designed to degrade gracefully: works with an LLM provider for natural-language
understanding, and falls back to deterministic keyword rules when no model is
configured, so it runs offline out of the box.
"""

from .types import Message, NluResult, BotReply, Conversation
from .nlu import Nlu
from .flows import Flow, FlowStep, FlowRegistry
from .router import FlowRouter
from .escalation import EscalationService
from .memory import ConversationMemory
from .engine import ReplyEngine
from .providers import LLMProvider, MockProvider, OpenAIProvider

__version__ = "0.1.0"

__all__ = [
    "Message",
    "NluResult",
    "BotReply",
    "Conversation",
    "Nlu",
    "Flow",
    "FlowStep",
    "FlowRegistry",
    "FlowRouter",
    "EscalationService",
    "ConversationMemory",
    "ReplyEngine",
    "LLMProvider",
    "MockProvider",
    "OpenAIProvider",
]
