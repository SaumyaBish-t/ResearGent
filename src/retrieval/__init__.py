"""Retrieval — Phase 1 ships naive dense-only retrieval. Phase 2 adds hybrid."""

from src.retrieval.naive import RetrievedChunk, naive_retrieve

__all__ = ["RetrievedChunk", "naive_retrieve"]
