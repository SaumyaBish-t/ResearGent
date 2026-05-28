"""
Agent state — the typed object that flows through every node in the graph.

Why a TypedDict
---------------
LangGraph's StateGraph requires a TypedDict (or dataclass) so it knows how
to merge node return values into the running state. Each node returns a
dict containing ONLY the fields it wants to update; LangGraph merges that
into the existing state.

For list/dict fields we use Annotated[..., operator.add] / `dict | dict`
reducers so concurrent nodes (Phase 4+) can write without trampling each
other. For Phase 3 we don't strictly need them, but designing for fan-out
now means Phase 4's Critic and Web-Scraper can be added without touching
the schema.

Field reference (mental model)
------------------------------
    question              the original user question
    sub_questions         planner output; >=1 entry, may equal [question]
    is_complex            true if planner decomposed into multiple sub-Qs
    chunks_by_subq        {sub_q: [HybridChunk, ...]} from retriever
    draft_answer          generator's synthesized markdown answer
    citation_map          {S<n>: HybridChunk}; survives across answer & report
    error                 set by NoAnswer node when retrieval finds nothing
    run_id                stable id for this query, used for checkpoint lookup
    trace                 append-only log of node entries (for debugging)
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from src.retrieval import HybridChunk


def _merge_chunks_by_subq(
    a: dict[str, list[HybridChunk]],
    b: dict[str, list[HybridChunk]],
) -> dict[str, list[HybridChunk]]:
    """Reducer: per-sub-question chunk lists merge by union, not overwrite."""
    out = dict(a)
    for k, v in b.items():
        out[k] = v  # latest writer wins per sub-q (deterministic in sequential mode)
    return out


class AgentState(TypedDict, total=False):
    # ---- Inputs ----
    question: str
    run_id: str

    # ---- Planner outputs ----
    sub_questions: list[str]
    is_complex: bool
    planner_reasoning: str

    # ---- Retriever outputs ----
    chunks_by_subq: Annotated[dict[str, list[HybridChunk]], _merge_chunks_by_subq]

    # ---- Generator outputs ----
    draft_answer: str
    citation_map: dict[str, HybridChunk]

    # ---- Flow control / observability ----
    error: str | None
    trace: Annotated[list[dict[str, Any]], operator.add]
