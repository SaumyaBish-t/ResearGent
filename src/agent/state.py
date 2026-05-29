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

Lean-state contract (Phase 12)
------------------------------
Every field in this TypedDict gets serialized into the LangGraph
PostgresSaver checkpoint AT EVERY NODE BOUNDARY. With a 500 MB free-tier
budget, payload discipline matters more than any other optimisation.

Rules every node MUST follow:
  1. NEVER put raw HTML, full PDF text, or unbounded provider responses
     into state. Web/paper chunks are already capped (see web.py:_clip
     and the paper-discovery abstract-only contract); new sources must
     match that pattern.
  2. Lean metadata only: URLs, chunk_ids, page numbers, score, signal,
     short snippet. The full document text lives in ChromaDB; state
     carries the pointer.
  3. Don't accumulate. `chunks_by_subq` is the biggest line item — its
     reducer overwrites per sub-question rather than appending, which
     is what stops a reflection loop from doubling state every iteration.
  4. Anything debug-only (full provider responses, intermediate prompts)
     belongs in the JSONL observability log, NOT in state.

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

from src.retrieval import GraphChunk, HybridChunk, PaperChunk, WebChunk

# A "context chunk" can come from local hybrid retrieval, web fallback,
# academic paper discovery, OR knowledge-graph expansion. All four expose
# the same public interface (text, citation, signal, source_file, ...) so
# the generator and citation builder don't care about origin.
ContextChunk = HybridChunk | WebChunk | PaperChunk | GraphChunk


def _merge_chunks_by_subq(
    a: dict[str, list[ContextChunk]],
    b: dict[str, list[ContextChunk]],
) -> dict[str, list[ContextChunk]]:
    """Reducer: latest writer wins per sub-q (deterministic in sequential mode)."""
    out = dict(a)
    for k, v in b.items():
        out[k] = v
    return out


class AgentState(TypedDict, total=False):
    # ---- Inputs ----
    question: str
    run_id: str
    # Optional: restrict retrieval to a specific set of registry doc_ids
    # (UUID strings). When unset, the entire corpus is searched. Lets the
    # caller scope a query to "just my uploads" / "just my notes" / etc.
    # without code changes elsewhere.
    doc_id_scope: list[str]

    # ---- Planner outputs ----
    sub_questions: list[str]
    is_complex: bool
    planner_reasoning: str

    # ---- Retriever outputs ----
    chunks_by_subq: Annotated[dict[str, list[ContextChunk]], _merge_chunks_by_subq]

    # ---- Critic outputs (Phase 4) ----
    # Overall verdict on the current retrieval round.
    #   "high"   -> proceed to generator
    #   "medium" -> rewrite & retry if budget left, else proceed
    #   "low"    -> rewrite & retry if budget left, else web fallback
    confidence: str          # "high" | "medium" | "low"
    critic_reasoning: str    # one-line explanation for the trace

    # ---- Rewriter / loop control (Phase 4) ----
    rewrite_attempts: int    # bounded by settings.crag_max_rewrites
    rewritten_queries: dict[str, str]   # original sub_q -> rewritten sub_q

    # ---- Web fallback (Phase 4) ----
    web_used: bool

    # ---- Paper discovery (Phase 7) ----
    papers_used: bool
    # Lightweight summary of discovered papers for trace/display (not the
    # full PaperChunk objects — those live in chunks_by_subq).
    papers_discovered: list[dict]

    # ---- Self-reflection (Phase 5) ----
    # How many times the Reflector has triggered a loopback. Bounded by
    # settings.reflection_max_iterations to prevent infinite refinement loops.
    reflection_attempts: int
    # One-line gap descriptions surfaced by the latest Reflector pass — kept
    # for trace/display so users can see WHY a reflection loop triggered.
    reflection_gaps: list[str]
    # Follow-up sub-questions appended to sub_questions on the latest loop.
    # Surfaced separately so the AgentResult formatter can show what changed.
    reflection_follow_ups: list[str]

    # ---- Generator outputs ----
    draft_answer: str
    citation_map: dict[str, ContextChunk]

    # ---- Flow control / observability ----
    error: str | None
    trace: Annotated[list[dict[str, Any]], operator.add]
