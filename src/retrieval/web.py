"""
Web retrieval — cascaded across multiple providers for resilience.

Why cascade
-----------
Tavily's free tier (1000/mo) sounds generous but a single user running 10
research queries/day can burn through it in a week. When that quota hits,
the agent should silently fall through to the next provider instead of
failing.

Cascade order (configurable via settings.web_search_cascade)
------------------------------------------------------------
  1. Tavily      — agent-tuned snippets, 1000/mo free, highest quality
  2. Serper.dev  — real Google SERP, 2500 free signup credits, $1/1k after
  3. DuckDuckGo  — no key, no quota, rate-limited (~1 req/sec). Guaranteed
                   working fallback so the agent NEVER hard-fails on web.

Each provider returns the same `WebChunk` shape so the rest of the codebase
(generator, citation builder) doesn't know or care which provider fired.

Observability
-------------
Each web-search attempt logs to `data/llm_calls.jsonl` (same file as LLM
calls) as op="web_search" with provider name, duration, results count,
ok flag, error. Lets you answer "how often did we fall through to DDG?"
from `researgent stats`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import httpx

from src.config import ModelTier, settings
from src.llm import observability as obs


@dataclass
class WebChunk:
    """Mirrors HybridChunk's public interface so the generator can mix them."""

    text: str
    url: str
    title: str
    score: float  # provider-reported relevance, normalized to [0, 1]
    provider: str = ""  # which web search provider served this chunk

    # ---- Public interface shared with HybridChunk ----
    @property
    def source_file(self) -> str:
        return self.url

    @property
    def page_number(self) -> int:
        return 0

    @property
    def chunk_index(self) -> int:
        return -1

    @property
    def doc_title(self) -> str:
        return self.title

    @property
    def citation(self) -> str:
        return self.url

    @property
    def signal(self) -> str:
        return f"web:{self.provider}" if self.provider else "web"


# ---------------------------------------------------------------------------
# Individual providers — each is a pure function (query, max_results) -> chunks
# Raises on error so the cascade orchestrator can decide what to do.
# ---------------------------------------------------------------------------


def _provider_tavily(query: str, max_results: int) -> list[WebChunk]:
    if not settings.tavily_api_key:
        raise RuntimeError("tavily_api_key not set")
    from tavily import TavilyClient  # lazy import

    client = TavilyClient(api_key=settings.tavily_api_key)
    resp = client.search(
        query=query,
        max_results=max_results,
        search_depth="advanced",
        include_answer=False,
        include_raw_content=False,
    )
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
                provider="tavily",
            )
        )
    return out


def _provider_serper(query: str, max_results: int) -> list[WebChunk]:
    """
    Serper.dev — raw Google SERP. Returns organic results with title + snippet.

    Score normalization: Serper doesn't expose a relevance score, so we
    synthesize one from rank position (1.0 for rank 1, decaying linearly).
    """
    if not settings.serper_api_key:
        raise RuntimeError("serper_api_key not set")

    r = httpx.post(
        "https://google.serper.dev/search",
        headers={
            "X-API-KEY": settings.serper_api_key,
            "Content-Type": "application/json",
        },
        json={"q": query, "num": max_results},
        timeout=15.0,
    )
    r.raise_for_status()
    data = r.json()
    organic = data.get("organic") or []

    out: list[WebChunk] = []
    n = max(len(organic), 1)
    for i, item in enumerate(organic[:max_results]):
        snippet = (item.get("snippet") or "").strip()
        if not snippet:
            continue
        # Rank 1 -> 1.0; last rank -> ~0.2. Bounded so all are clearly "web".
        rank_score = max(0.2, 1.0 - (i / n) * 0.8)
        out.append(
            WebChunk(
                text=snippet,
                url=item.get("link", ""),
                title=item.get("title", "") or "",
                score=rank_score,
                provider="serper",
            )
        )
    return out


def _provider_duckduckgo(query: str, max_results: int) -> list[WebChunk]:
    """
    DuckDuckGo via the `ddgs` library — HTML scrape, no API key needed.

    DDG doesn't return a relevance score, so we use rank-position scoring
    just like Serper. Rate limit is ~1 req/sec; the library handles backoff.
    """
    from ddgs import DDGS  # lazy import

    out: list[WebChunk] = []
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=max_results))

    n = max(len(results), 1)
    for i, r in enumerate(results[:max_results]):
        body = (r.get("body") or "").strip()
        if not body:
            continue
        rank_score = max(0.2, 1.0 - (i / n) * 0.8)
        out.append(
            WebChunk(
                text=body,
                url=r.get("href", "") or r.get("url", ""),
                title=r.get("title", "") or "",
                score=rank_score,
                provider="duckduckgo",
            )
        )
    return out


# ---------------------------------------------------------------------------
# Cascade orchestration
# ---------------------------------------------------------------------------


_PROVIDERS: dict[str, tuple[Callable[[str, int], list[WebChunk]], Callable[[], bool]]] = {
    "tavily":     (_provider_tavily,     lambda: bool(settings.tavily_api_key)),
    "serper":     (_provider_serper,     lambda: bool(settings.serper_api_key)),
    # DuckDuckGo is always considered "configured" — no key required.
    "duckduckgo": (_provider_duckduckgo, lambda: True),
}


def web_search(query: str, *, max_results: int = 5) -> list[WebChunk]:
    """
    Search the web with cascade fallback across configured providers.

    Walks `settings.web_search_cascade` in order. For each provider:
      - skip if not configured
      - call the provider, time it, log to observability
      - if it raises (quota, network, parse error) -> log + try next
      - if it returns empty -> log + try next
      - if it returns results -> return them

    Returns empty list ONLY if every provider failed or returned nothing —
    a graceful degradation that lets the agent's no_answer node take over.
    """
    cascade = settings.web_search_cascade or list(_PROVIDERS.keys())
    last_error: str | None = None

    for step, name in enumerate(cascade):
        entry = _PROVIDERS.get(name)
        if entry is None:
            continue
        fn, is_configured = entry
        if not is_configured():
            continue

        # Observability uses the EMBED tier label as a stand-in for "non-LLM
        # external call" since we don't have a dedicated WEB tier in the
        # enum. The provider name disambiguates in the stats view.
        try:
            with obs.track(
                op="web_search", tier=ModelTier.EMBED,
                provider=name, model="search", cascade_step=step,
            ) as ctx:
                t0 = time.perf_counter()
                results = fn(query, max_results)
                ctx["extra"] = {
                    "results": len(results),
                    "query_chars": len(query),
                    "wall_ms": int((time.perf_counter() - t0) * 1000),
                }
            if results:
                return results
            # Empty but no exception — try next.
            last_error = f"{name} returned 0 results"
        except Exception as e:
            last_error = f"{name}: {type(e).__name__}: {str(e)[:120]}"
            continue

    # Surface the last failure reason in a structured way the caller can log.
    # We don't raise — web_fallback should always be graceful. Returning
    # empty lets the agent's downstream nodes handle the no-evidence case.
    if last_error:
        _ = last_error  # available for future hook; intentionally not raised
    return []
