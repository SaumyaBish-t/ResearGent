"""
Web retrieval — Tavily-backed search for the Corrective RAG fallback path.

When the Critic flags local retrieval as low-confidence AND the rewrite
budget is exhausted, this fires. Tavily returns relevance-scored snippets
that we wrap in a `WebChunk` matching `HybridChunk`'s public shape so the
generator doesn't have to know where its evidence came from.

Why Tavily specifically
-----------------------
  - Designed for LLM agents — returns content snippets, not raw HTML
  - Free tier: 1000 searches/month, plenty for development
  - Sub-second latency
  - Returns a per-result `score` we can use for ranking
  - No JS rendering needed for most results (faster than Playwright)

Phase 5+ can add Playwright for deep-dive scraping when Tavily's snippets
aren't enough; the WebChunk interface stays the same.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from src.config import settings


@dataclass
class WebChunk:
    """Mirrors HybridChunk's public interface so the generator can mix them."""

    text: str
    url: str
    title: str
    score: float  # Tavily's relevance score, [0, 1]

    # ---- Public interface shared with HybridChunk ----
    @property
    def source_file(self) -> str:
        """For dedup keys + display. Use the URL as the 'file' identifier."""
        return self.url

    @property
    def page_number(self) -> int:
        """Web has no pages — sentinel value so existing code doesn't break."""
        return 0

    @property
    def chunk_index(self) -> int:
        """No chunking on web results — they ARE the chunk. -1 sentinel."""
        return -1

    @property
    def doc_title(self) -> str:
        return self.title

    @property
    def citation(self) -> str:
        return self.url

    @property
    def signal(self) -> str:
        return "web"


def web_search(query: str, *, max_results: int = 5) -> list[WebChunk]:
    """
    Run a Tavily search and return WebChunk records.

    Returns empty list if TAVILY_API_KEY is unset OR the API call fails —
    failures are non-fatal because this is a *fallback* path. The agent
    should still produce a graceful "I don't know" rather than crashing
    when the web is unreachable.
    """
    if not settings.tavily_api_key:
        return []

    # Lazy import: tavily-python pulls in requests + bs4; ~200ms on cold start.
    # Defer so non-web paths don't pay the cost.
    try:
        from tavily import TavilyClient
    except Exception:
        return []

    t0 = time.perf_counter()
    try:
        client = TavilyClient(api_key=settings.tavily_api_key)
        resp = client.search(
            query=query,
            max_results=max_results,
            search_depth="advanced",       # better snippet quality, +~1s latency
            include_answer=False,          # we synthesize ourselves
            include_raw_content=False,     # snippets are enough; raw blows up tokens
        )
    except Exception:
        return []

    out: list[WebChunk] = []
    for r in resp.get("results", [])[:max_results]:
        content = (r.get("content") or "").strip()
        if not content:
            continue
        out.append(
            WebChunk(
                text=content,
                url=r.get("url", ""),
                title=r.get("title", "") or "",
                score=float(r.get("score") or 0.0),
            )
        )
    # Discard timing locally; the agent node logs the duration into the trace.
    _ = time.perf_counter() - t0
    return out
