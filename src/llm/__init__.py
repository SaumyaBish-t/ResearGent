"""
LLM provider abstraction.

Public surface:
    from src.llm import chat, embed, ModelTier
    from src.llm.observability import load_records, summarize
"""

from src.config import ModelTier
from src.llm.provider import chat, embed, get_client, list_status

__all__ = ["chat", "embed", "get_client", "list_status", "ModelTier"]
