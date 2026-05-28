"""
Web fallback node — Tavily search when local retrieval has been exhausted.

Activates only after the rewrite budget is spent and the Critic still
reports low confidence. This is the "we can't answer from the corpus,
let's try the open web" branch of the Corrective RAG decision tree.

Per-sub-question search + merge
-------------------------------
We run one Tavily search per sub-question (same shape as the local
retriever). Results are slotted into the existing `chunks_by_subq` dict
under the original (NOT rewritten) sub-question key so the generator
sees a unified evidence structure.

Web results are MERGED with whatever weak local chunks already exist
rather than replacing them. The Critic may not have flagged the chunks
themselves as bad — sometimes the corpus just doesn't have the answer
in addition to having partial info.
"""

from __future__ import annotations

import time
from typing import Any

from src.agent.state import AgentState
from src.config import settings
from src.retrieval import web_search


def web_fallback(state: AgentState) -> dict[str, Any]:
    """
    Tavily search per sub-question; merge results into chunks_by_subq.

    If TAVILY_API_KEY is unset, this is a no-op — `web_search` returns
    [] and the generator runs with whatever local chunks survived. The
    user will see in the trace that web fallback was attempted but
    nothing came back.
    """
    sub_qs = list(state.get("sub_questions") or [state["question"]])
    chunks_by_subq = dict(state.get("chunks_by_subq") or {})
    timings: list[dict[str, Any]] = []
    total_added = 0

    for sq in sub_qs:
        t0 = time.perf_counter()
        # Prefer the rewritten query if we have one — the original sub-q is the
        # one that already failed local retrieval, and the rewrite has been
        # tuned for retrieval.
        rewritten = (state.get("rewritten_queries") or {}).get(sq)
        query_to_use = rewritten or sq

        web_hits = web_search(query_to_use, max_results=3)
        dur_ms = int((time.perf_counter() - t0) * 1000)

        if web_hits:
            existing = chunks_by_subq.get(sq) or []
            chunks_by_subq[sq] = list(existing) + list(web_hits)
            total_added += len(web_hits)

        timings.append(
            {
                "node": "web_fallback",
                "sub_q": sq[:80],
                "query_used": query_to_use[:80],
                "results": len(web_hits),
                "duration_ms": dur_ms,
            }
        )

    return {
        "chunks_by_subq": chunks_by_subq,
        "web_used": True,
        "trace": timings,
        # If still nothing AND no API key, surface the reason for the trace
        # without breaking the run — the generator's no_answer handles the
        # case of completely empty chunks gracefully.
        **({"error": "tavily_api_key_unset"} if not settings.tavily_api_key else {}),
    }
