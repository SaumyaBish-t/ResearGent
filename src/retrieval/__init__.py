"""Retrieval — naive (dense), BM25, hybrid (dense+BM25+RRF), and web (Tavily)."""

from src.retrieval.naive import RetrievedChunk, naive_retrieve
from src.retrieval.hybrid import HybridChunk, hybrid_retrieve
from src.retrieval.web import WebChunk, web_search

__all__ = [
    "RetrievedChunk",
    "naive_retrieve",
    "HybridChunk",
    "hybrid_retrieve",
    "WebChunk",
    "web_search",
]
