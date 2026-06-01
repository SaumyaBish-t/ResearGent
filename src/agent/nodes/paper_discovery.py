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

from src.agent.artifacts import persist_mixed
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
    thread_id = state.get("run_id") or ""
    existing_refs = dict(state.get("chunk_refs_by_subq") or {})

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

    # Persist paper chunks as ephemeral artifacts and merge their refs
    # under the ORIGINAL question key. We use the original question (not
    # a sub-q) because paper discovery is top-level evidence — the
    # abstracts are usually broad enough to cover multiple sub-questions.
    new_refs = persist_mixed(thread_id, {question: list(papers)})
    merged_refs = dict(existing_refs)
    merged_refs[question] = list(existing_refs.get(question) or []) + list(
        new_refs.get(question) or []
    )

    # Dedupe the display summary by (arxiv_id, title) — after
    # `_expand_with_semantic_chunks` each PDF-enriched paper appears once
    # per kept slice (up to 5), which makes the discovery list look like
    # we found "5 copies of a Hanabi paper" when we really found one paper
    # sliced 5 ways. Keep the FIRST occurrence of each unique paper so the
    # highest-scoring slice's score is the one displayed.
    discovered_summary: list[dict[str, Any]] = []
    seen_papers: set[tuple[str, str]] = set()
    for p in papers:
        key = (
            (p.arxiv_id or "").strip().lower(),
            re.sub(r"\W+", " ", (p.title or "").lower()).strip(),
        )
        if key in seen_papers:
            continue
        seen_papers.add(key)
        discovered_summary.append(
            {
                "title": p.title[:120],
                "citation": p.citation,
                "year": p.year,
                "venue": p.venue,
                "source": p.source,
                "score": round(p.score, 3),
                "citations": p.citations,
            }
        )

    return {
        "chunk_refs_by_subq": merged_refs,
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
