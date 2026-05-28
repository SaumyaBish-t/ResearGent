"""
Retriever node — runs hybrid retrieval per sub-question.

Reuses Phase 2's hybrid_retrieve directly. Sequential per sub-q for now;
Phase 4 can fan out in parallel via LangGraph's Send pattern.

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

from src.agent.state import AgentState
from src.config import settings
from src.retrieval import expand_via_wikilinks, hybrid_retrieve

# Total target chunks to pass to the generator. Tunable per call from the
# CLI; the node uses this when state doesn't carry an explicit k.
DEFAULT_TOTAL_K = 8


def retrieve(state: AgentState) -> dict[str, Any]:
    """
    Retrieve chunks for every sub-question — IDEMPOTENT.

    Skips sub-questions that already have chunks in state. This matters most
    on reflection-loopbacks: when Phase 5's Reflector adds 2 new sub-Qs to
    an existing list of 4, we should only retrieve for the 2 new ones, not
    re-do the 4 already-answered ones. Without this, each loop's retrieve
    + critic cost grows linearly with the cumulative sub-question count.

    The chunks for an existing sub-Q are deterministic (same hybrid_retrieve
    with same query = same chunks) so skipping is safe.
    """
    sub_qs = state.get("sub_questions") or [state["question"]]
    total_k = int(state.get("k") or DEFAULT_TOTAL_K)  # type: ignore[arg-type]
    per_subq_k = max(2, math.ceil(total_k / max(1, len(sub_qs))))

    # Identify which sub-Qs need fresh retrieval. An existing sub-Q is "done"
    # if it already has at least one chunk in chunks_by_subq.
    existing = dict(state.get("chunks_by_subq") or {})
    to_retrieve = [sq for sq in sub_qs if not existing.get(sq)]
    skipped = [sq for sq in sub_qs if existing.get(sq)]

    chunks_by_subq: dict[str, list] = dict(existing)
    timings: list[dict[str, Any]] = []

    if skipped:
        timings.append(
            {
                "node": "retriever",
                "skipped_idempotent": len(skipped),
                "to_retrieve": len(to_retrieve),
            }
        )

    for sq in to_retrieve:
        t0 = time.perf_counter()
        hits = hybrid_retrieve(sq, k=per_subq_k)
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
    # After direct retrieval, walk wikilink edges to surface structurally-
    # related chunks the embedder may have ranked low. Only fires when at
    # least one retrieved chunk carries wikilinks (i.e. came from a vault
    # note) — pure-PDF corpora skip this entirely with no extra cost.
    if settings.graph_expansion_enabled:
        # Collect seeds across all sub-questions + exclude-keys so expansion
        # doesn't re-surface chunks the retriever already has.
        all_seeds = []
        exclude: set[tuple[str, int]] = set()
        for sq_chunks in chunks_by_subq.values():
            for c in sq_chunks:
                all_seeds.append(c)
                exclude.add((c.source_file, c.chunk_index))

        # Skip the call entirely if no seed has any wikilinks — common short-
        # circuit that saves the store-scan cost on PDF-only retrievals.
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
            # Attach all extras under the ORIGINAL question key so the
            # generator sees them grouped as "related context" rather than
            # tied to a specific sub-question.
            if extras:
                top_key = state.get("question") or next(iter(chunks_by_subq.keys()), "")
                if top_key:
                    chunks_by_subq[top_key] = list(chunks_by_subq.get(top_key) or []) + extras

    return {"chunks_by_subq": chunks_by_subq, "trace": timings}


def has_any_chunks(state: AgentState) -> bool:
    """Edge predicate — true iff retrieval surfaced at least one chunk anywhere."""
    by_q = state.get("chunks_by_subq") or {}
    return any(len(v) > 0 for v in by_q.values())
