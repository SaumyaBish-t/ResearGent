"""
Paper discovery node — searches arXiv + Semantic Scholar when the local
corpus + retries didn't surface enough evidence.

This sits BEFORE web_fallback in the CRAG cascade because:
  - Paper abstracts are denser + more authoritative than web snippets
  - arXiv/SS are free and fast (no quota concerns vs Tavily's 1000/mo)
  - Citation counts let us prefer well-known foundational work over
    obscure pre-prints

What it adds to state
---------------------
  - chunks_by_subq: PaperChunks merged into the original-question key
    (NOT distributed per sub-question — papers are a top-level evidence
    source, not a per-sub-q retrieval)
  - papers_discovered: list of {title, citation, score} for trace/display
  - papers_used: bool flag
"""

from __future__ import annotations

import re
import time
from typing import Any

from src.agent.state import AgentState
from src.config import ModelTier
from src.llm import chat
from src.retrieval import discover_papers


# Don't blow out the prompt with too many papers — abstract per paper is
# ~200 tokens, 5 papers = ~1000 tokens, comfortable budget.
DEFAULT_MAX_PAPERS = 5

# Tight FAST-tier prompt to convert a verbose user question into a keyword
# query that academic search engines actually like.
_QUERY_EXTRACT_SYSTEM = """Convert a user's research question into a short, \
keyword-rich academic search query suitable for arXiv and Semantic Scholar.

Rules:
  - 3-8 words maximum
  - Use TECHNICAL TERMS the literature would actually contain
  - Drop interrogative wrapping ("what are the", "how does", "explain")
  - Drop timeframe qualifiers ("latest", "2025", "recent") — academic
    search engines order by relevance not recency for short queries
  - Keep specific named entities, acronyms, methods

Output ONLY the search query, no quotes, no explanation."""


def _extract_search_query(question: str) -> str:
    """Use the FAST tier to shorten verbose questions for academic search."""
    # Heuristic short-circuit: if the question is already short and entity-rich,
    # skip the LLM call.
    word_count = len(re.findall(r"\w+", question))
    if word_count <= 6:
        return question

    try:
        cleaned = chat(
            messages=[
                {"role": "system", "content": _QUERY_EXTRACT_SYSTEM},
                {"role": "user", "content": question},
            ],
            tier=ModelTier.FAST,
            temperature=0.0,
            max_tokens=40,
        ).strip().strip('"').strip("'")
        # Drop any trailing newline / explanation the model may have added.
        cleaned = cleaned.split("\n")[0].strip()
        if cleaned and len(cleaned) < len(question):
            return cleaned
    except Exception:
        pass
    return question


def discover(state: AgentState) -> dict[str, Any]:
    """Search arXiv + Semantic Scholar and merge results into state."""
    question = state["question"]
    chunks_by_subq = dict(state.get("chunks_by_subq") or {})

    # Verbose questions retrieve poorly from arXiv/SS keyword search.
    # Distill to a short query first.
    search_query = _extract_search_query(question)

    t0 = time.perf_counter()
    papers = discover_papers(search_query, max_results=DEFAULT_MAX_PAPERS)
    dur_ms = int((time.perf_counter() - t0) * 1000)

    if not papers:
        return {
            "papers_used": True,
            "papers_discovered": [],
            "trace": [
                {
                    "node": "paper_discovery",
                    "duration_ms": dur_ms,
                    "search_query": search_query[:80],
                    "results": 0,
                    "note": "no papers found",
                }
            ],
        }

    # Attach discovered papers under the ORIGINAL question key so the
    # generator sees them grouped with whatever local chunks survived.
    # We use the original question (not a sub-q) because paper discovery
    # is top-level evidence — the abstracts are usually broad enough to
    # cover multiple sub-questions.
    existing = list(chunks_by_subq.get(question) or [])
    chunks_by_subq[question] = existing + list(papers)

    discovered_summary = [
        {
            "title": p.title[:120],
            "citation": p.citation,
            "year": p.year,
            "venue": p.venue,
            "source": p.source,
            "score": round(p.score, 3),
            "citations": p.citations,
        }
        for p in papers
    ]

    return {
        "chunks_by_subq": chunks_by_subq,
        "papers_used": True,
        "papers_discovered": discovered_summary,
        # Reset critic-related state so the rewriter budget doesn't
        # accidentally prevent re-grading the freshly-added evidence.
        "rewrite_attempts": 0,
        "trace": [
            {
                "node": "paper_discovery",
                "duration_ms": dur_ms,
                "search_query": search_query[:80],
                "results": len(papers),
                "providers": sorted({p.source for p in papers}),
                "top_score": round(max(p.score for p in papers), 3),
            }
        ],
    }
