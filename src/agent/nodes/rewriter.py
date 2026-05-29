"""
Rewriter node — rewrites sub-questions when the Critic flagged retrieval
as low/medium confidence and the retry budget hasn't been exhausted.

Why rewrite at all?
-------------------
Retrieval failures often come from query phrasing, not corpus gaps:
  - "What is X?"           -> matches definitions of X poorly because
                              the corpus says "X is a method for ..." not
                              "X means ..."
  - "How does Y handle Z"  -> if Y is uncommon term, retrieval may miss
                              the right section; "Y's approach to Z"
                              works better
  - Vague pronouns / context drift

A second-pass rewrite using the FAST tier (sub-second, cheap) often
recovers retrieval quality at a small cost. We bound retries strictly via
`settings.crag_max_rewrites` to avoid infinite loops on truly unanswerable
queries.

Only rewrites sub-questions that PRODUCED LOW RELEVANCE
-------------------------------------------------------
We don't rewrite sub-Qs that already got good chunks. The Critic dropped
"irrelevant" chunks; if a sub-Q still has chunks after that filter, it's
fine. Only sub-Qs with zero surviving chunks get rewritten. This keeps
the retry surface small and predictable.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

from src.agent.artifacts import persist_mixed
from src.agent.state import AgentState
from src.config import ModelTier
from src.llm import chat
from src.retrieval import hybrid_retrieve


_SYSTEM = """You rewrite poor retrieval queries to be more effective for hybrid \
search (dense embeddings + BM25 keyword match) over an academic corpus.

Given a question that produced poor retrieval, write a SINGLE rewritten query that:
  - Uses concrete domain terms the corpus would literally contain
  - Avoids vague pronouns ("it", "the model") — name things explicitly
  - Adds synonyms for low-frequency terms (e.g., "RAG" + "retrieval-augmented")
  - Stays a question or noun phrase, NOT a full sentence

Output ONLY a JSON object, no preamble:
{
  "rewritten": "your rewritten query",
  "reasoning": "one short sentence on why this should retrieve better"
}"""


_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except Exception:
        pass
    m = _JSON_OBJ_RE.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def _rewrite_one(sub_q: str) -> tuple[str, str, int]:
    """Return (rewritten_query, reasoning, ms). Falls back to original on parse fail."""
    t0 = time.perf_counter()
    raw = chat(
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": f"Original question: {sub_q}"},
        ],
        tier=ModelTier.FAST,
        temperature=0.2,  # tiny bit of variation so retries actually try something new
        max_tokens=200,
    )
    dur_ms = int((time.perf_counter() - t0) * 1000)
    parsed = _extract_json(raw) or {}
    rewritten = str(parsed.get("rewritten") or "").strip()
    reasoning = str(parsed.get("reasoning") or "").strip()
    if not rewritten or rewritten == sub_q:
        return sub_q, "no useful rewrite produced; keeping original", dur_ms
    return rewritten, reasoning, dur_ms


def rewrite_and_retry(state: AgentState) -> dict[str, Any]:
    """
    Rewrite sub-questions whose chunk list emptied out (or stayed weak),
    re-run hybrid retrieval, and accumulate.

    Bumps `rewrite_attempts` by 1 so the graph's edge predicate knows when
    to stop trying.
    """
    refs_by_subq = dict(state.get("chunk_refs_by_subq") or {})
    thread_id = state.get("run_id") or ""
    rewritten_map = dict(state.get("rewritten_queries") or {})
    timings: list[dict[str, Any]] = []

    # Choose targets: sub-questions with zero surviving refs after Critic.
    targets = [sq for sq, ch in refs_by_subq.items() if not ch]

    if not targets:
        # Critic verdict was medium not because of empty sub-Qs but because
        # of partial-relevance dilution. Rewrite ALL sub-Qs in this case.
        targets = list(refs_by_subq.keys()) or state.get("sub_questions") or [state["question"]]

    new_chunks_for_persist: dict[str, list] = {}
    new_attempt = int(state.get("rewrite_attempts") or 0) + 1

    for orig in targets:
        rewritten, reasoning, rw_ms = _rewrite_one(orig)
        rewritten_map[orig] = rewritten

        t0 = time.perf_counter()
        per_q_k = 4  # slightly larger than the planner's per-subq-k to give the rewrite room
        hits = hybrid_retrieve(rewritten, k=per_q_k)
        ret_ms = int((time.perf_counter() - t0) * 1000)

        # Capture in memory for the batch persist call below.
        new_chunks_for_persist[orig] = hits

        timings.append(
            {
                "node": "rewriter",
                "attempt": new_attempt,
                "orig_sub_q": orig[:80],
                "rewritten": rewritten[:80],
                "rewrite_ms": rw_ms,
                "retrieve_ms": ret_ms,
                "new_hits": len(hits),
                "reasoning": reasoning[:120],
            }
        )

    # Persist all fresh retrievals as refs (local hybrid hits become free
    # local refs via chroma_id). Then merge over the prior ref map.
    merged_refs = dict(refs_by_subq)
    if new_chunks_for_persist:
        new_refs = persist_mixed(thread_id, new_chunks_for_persist)
        merged_refs.update(new_refs)

    return {
        "chunk_refs_by_subq": merged_refs,
        "rewritten_queries": rewritten_map,
        "rewrite_attempts": new_attempt,
        "trace": timings,
    }
