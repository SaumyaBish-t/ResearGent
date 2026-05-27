"""
LLM provider abstraction.

Public surface — everything else in the codebase uses ONLY these:
    from src.llm import chat, embed, ModelTier
"""

from src.config import ModelTier
from src.llm.provider import chat, embed, get_client, list_status

__all__ = ["chat", "embed", "get_client", "list_status", "ModelTier"]
