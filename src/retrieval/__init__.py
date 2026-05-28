"""Retrieval — local (dense + BM25), web (Tavily cascade), open-domain papers."""

from src.retrieval.naive import RetrievedChunk, naive_retrieve
from src.retrieval.hybrid import HybridChunk, hybrid_retrieve
from src.retrieval.web import WebChunk, web_search
from src.retrieval.papers import PaperChunk, discover_papers

__all__ = [
    "RetrievedChunk", "naive_retrieve",
    "HybridChunk", "hybrid_retrieve",
    "WebChunk", "web_search",
    "PaperChunk", "discover_papers",
]
