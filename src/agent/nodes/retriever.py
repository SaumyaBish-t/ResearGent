"""
Retriever node — runs hybrid retrieval per sub-question.

Reuses Phase 2's hybrid_retrieve directly. Sequential per sub-q for now;
Phase 4 can fan out in parallel via LangGraph's Send pattern.

Phase 13 pointer pattern
------------------------
Input state carries `chunk_refs_by_subq: dict[sub_q, list[ChunkRef]]`.
We hydrate ONLY if we need the old chunks' metadata (for graph-expansion
exclusion); otherwise we skip hydration entirely and treat refs as
opaque pointers. Newly-retrieved chunks come back as HybridChunk with
populated `chroma_id` — converted to refs by `persist_mixed()` before
returning. Net result: state never carries chunk text.

Per-sub-Q `k` tuning
--------------------
For a 1-sub-q (simple) question, we use the user's full `k`. For an N-sub-q
plan, we shrink per-sub-q top-k so the prompt context stays bounded — but
we ensure at least 2 chunks per sub-q so the generator has some grounding.
The math: per_subq_k = max(2, ceil(k / N)). Total prompt context grows
slowly with N rather than linearly.
"""

from __future__ import annotations

import math
import time
from typing import Any

from src.agent.artifacts import hydrate, persist_mixed
from src.agent.state import AgentState
from src.config import settings
from src.retrieval import expand_via_wikilinks, hybrid_retrieve

# Total target chunks to pass to the generator. Tunable per call from the
# CLI; the node uses this when state doesn't carry an explicit k.
DEFAULT_TOTAL_K = 8


def retrieve(state: AgentState) -> dict[str, Any]:
    """
    Retrieve chunks for every sub-question — IDEMPOTENT.

    Skips sub-questions that already have refs in state. This matters most
    on reflection-loopbacks: when Phase 5's Reflector adds 2 new sub-Qs to
    an existing list of 4, we should only retrieve for the 2 new ones, not
    re-do the 4 already-answered ones.
    """
    sub_qs = state.get("sub_questions") or [state["question"]]
    total_k = int(state.get("k") or DEFAULT_TOTAL_K)  # type: ignore[arg-type]
    per_subq_k = max(2, math.ceil(total_k / max(1, len(sub_qs))))
    thread_id = state.get("run_id") or ""

    # Idempotency check is on REFS, not chunks — no hydration needed.
    existing_refs = dict(state.get("chunk_refs_by_subq") or {})
    to_retrieve = [sq for sq in sub_qs if not existing_refs.get(sq)]
    skipped = [sq for sq in sub_qs if existing_refs.get(sq)]

    # We hold chunks in memory only for the duration of this node call.
    # Pre-existing sub-Qs get hydrated so graph-expansion can read their
    # metadata (source_file, chunk_index, wikilinks). Fresh retrievals
    # come back as HybridChunk objects directly.
    chunks_by_subq: dict[str, list] = {}
    if existing_refs:
        chunks_by_subq.update(hydrate(existing_refs))

    timings: list[dict[str, Any]] = []
    if skipped:
        timings.append(
            {
                "node": "retriever",
                "skipped_idempotent": len(skipped),
                "to_retrieve": len(to_retrieve),
            }
        )

    # `doc_id_scope` lets callers restrict retrieval to a registry doc subset.
    doc_id_scope: list[str] | None = state.get("doc_id_scope")  # type: ignore[assignment]

    for sq in to_retrieve:
        t0 = time.perf_counter()
        hits = hybrid_retrieve(sq, k=per_subq_k, doc_ids=doc_id_scope)
        chunks_by_subq[sq] = hits
        timings.append(
            {
                "node": "retriever",
                "sub_q": sq[:80],
                "k": per_subq_k,
                "hits": len(hits),
                "duration_ms": int((time.perf_counter() - t0) * 1000),
            }
        )

    # ---- Phase 10: knowledge-graph expansion ----
    if settings.graph_expansion_enabled:
        all_seeds = []
        exclude: set[tuple[str, int]] = set()
        for sq_chunks in chunks_by_subq.values():
            for c in sq_chunks:
                all_seeds.append(c)
                exclude.add((
                    getattr(c, "source_file", "") or "",
                    getattr(c, "chunk_index", -1),
                ))

        any_have_links = any(
            getattr(c, "wikilinks", None) for c in all_seeds
        )
        if any_have_links:
            t0 = time.perf_counter()
            extras = expand_via_wikilinks(
                all_seeds,
                max_extra=settings.graph_expansion_max_extra_chunks,
                exclude_keys=exclude,
            )
            dur_ms = int((time.perf_counter() - t0) * 1000)
            timings.append(
                {
                    "node": "retriever",
                    "graph_expansion": True,
                    "seeds_with_links": sum(
                        1 for c in all_seeds if getattr(c, "wikilinks", None)
                    ),
                    "extras": len(extras),
                    "mutual": sum(1 for e in extras if e.is_mutual),
                    "duration_ms": dur_ms,
                }
            )
            if extras:
                top_key = state.get("question") or next(iter(chunks_by_subq.keys()), "")
                if top_key:
                    chunks_by_subq[top_key] = list(chunks_by_subq.get(top_key) or []) + extras

    # Convert in-memory chunks back to refs for the checkpoint. Local
    # (HybridChunk) chunks become free refs via chroma_id; web/paper/graph
    # chunks get one INSERT each in `agent_artifacts`.
    refs_out = persist_mixed(thread_id, chunks_by_subq)
    return {"chunk_refs_by_subq": refs_out, "trace": timings}


def has_any_chunks(state: AgentState) -> bool:
    """Edge predicate — true iff retrieval surfaced at least one ref anywhere."""
    by_q = state.get("chunk_refs_by_subq") or {}
    return any(len(v) > 0 for v in by_q.values())
