"""
Agent state — the typed object that flows through every node in the graph.

Why a TypedDict
---------------
LangGraph's StateGraph requires a TypedDict (or dataclass) so it knows how
to merge node return values into the running state. Each node returns a
dict containing ONLY the fields it wants to update; LangGraph merges that
into the existing state.

Lean-state contract (Phase 12 + Phase 13)
-----------------------------------------
Every field in this TypedDict gets serialized into the LangGraph
PostgresSaver checkpoint AT EVERY NODE BOUNDARY. With a 500 MB free-tier
budget, payload discipline matters more than any other optimisation.

Phase 13 made this contract strict: **state holds REFERENCES, not chunk
text**. Where Phase 12 still carried full `HybridChunk` / `WebChunk` /
etc. objects through `chunks_by_subq` (worst case ~200KB/snapshot ×
~10 snapshots/run = 2MB/run), Phase 13 carries `ChunkRef` pointers
(~80 bytes each) and stores chunk text either in ChromaDB (for local
chunks) or in the `agent_artifacts` table (for ephemeral web/paper/
graph chunks). New per-snapshot cost: ~3KB. Per-run: ~30KB. The 500 MB
budget now buys ~15,000 runs instead of ~250.

Rules every node MUST follow:
  1. NEVER put chunk `text` into state. Pass refs; hydrate at node entry,
     persist refs at node exit. The helpers in `src.agent.artifacts`
     enforce this — see `hydrate(refs_by_subq)`, `persist_local()`,
     `persist_ephemeral()`.
  2. NEVER put raw HTML, full PDF text, or unbounded provider responses
     anywhere in the graph state. They're capped at the ChunkRef boundary.
  3. Don't accumulate. `chunk_refs_by_subq`'s reducer overwrites per
     sub-question rather than appending — stops reflection loops from
     doubling state every iteration.
  4. Anything debug-only (full provider responses, intermediate prompts)
     belongs in the JSONL observability log, NOT in state.

The only TEXT field that lives in state is `draft_answer` — bounded by
the generator's max_tokens (≈4-6KB), and the whole point of running the
agent in the first place.

Field reference (mental model)
------------------------------
    question              the original user question
    sub_questions         planner output; >=1 entry, may equal [question]
    is_complex            true if planner decomposed into multiple sub-Qs
    chunk_refs_by_subq    {sub_q: [ChunkRef-as-dict, ...]} — pointer-based
    draft_answer          generator's synthesized markdown answer
    citation_refs         {S<n>: ChunkRef-as-dict}; hydrated only at format-time
    error                 set by NoAnswer node when retrieval finds nothing
    run_id                stable id for this query, used for checkpoint lookup
    trace                 append-only log of node entries (for debugging)
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


def _merge_refs_by_subq(
    a: dict[str, list[Any]],
    b: dict[str, list[Any]],
) -> dict[str, list[Any]]:
    """
    Reducer for `chunk_refs_by_subq`.

    Latest writer wins per sub-q (deterministic in sequential mode).
    Crucial that this is OVERWRITE not APPEND — reflection loops re-run
    retrieval for the same sub-questions, and an append reducer would
    double the ref count every iteration.

    The list items are stored as plain dicts (`{"kind": ..., "id": ...}`)
    rather than ChunkRef instances because PostgresSaver's JSON serializer
    round-trips dataclasses as dicts anyway, and accepting dicts here means
    the type is robust against version skew.
    """
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

    # Phase 15: optional restriction to one or more registered domain ids
    # (e.g. ["agentic_ai", "time_series"]). When unset the agent searches
    # across every domain bucket. Set either by the CLI's `--domain` flag
    # (explicit user intent) or by the planner's keyword auto-router
    # (implicit, only when the query has strong domain signals).
    domain_scope: list[str]

    # ---- Planner outputs ----
    sub_questions: list[str]
    is_complex: bool
    planner_reasoning: str

    # ---- Retriever outputs (Phase 13 pointer form) ----
    # Each value is a list of ChunkRef-shaped dicts: {"kind": str, "id": str}.
    # Hydration to text happens inside each consuming node via
    # `src.agent.artifacts.hydrate()`.
    chunk_refs_by_subq: Annotated[dict[str, list[dict[str, str]]], _merge_refs_by_subq]

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
    # full PaperChunk objects — those live in `agent_artifacts` via refs).
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
    # Pointer form of citation map: {"S1": {"kind": ..., "id": ...}, ...}.
    # Hydrated for display in run.py / stream.py / vault_writer.py via
    # `src.agent.artifacts.hydrate_one()`.
    citation_refs: dict[str, dict[str, str]]

    # ---- Flow control / observability ----
    error: str | None
    trace: Annotated[list[dict[str, Any]], operator.add]
