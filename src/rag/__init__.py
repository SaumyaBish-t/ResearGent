"""RAG pipelines. Phase 1: naive (retrieve -> stuff -> generate)."""

from src.rag.naive import RAGAnswer, naive_rag

__all__ = ["RAGAnswer", "naive_rag"]
