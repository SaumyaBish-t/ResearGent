"""Agent state — the typed object that flows through every node in the graph.

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
    doc_id_scope: list[str]

    # Phase 15: optional restriction to one or more registered domain ids
    domain_scope: list[str]

    # ---- Planner outputs ----
    sub_questions: list[str]
    is_complex: bool
    planner_reasoning: str

    # ---- Retriever outputs (Phase 13 pointer form) ----
    chunk_refs_by_subq: Annotated[dict[str, list[dict[str, str]]], _merge_refs_by_subq]

    # ---- Critic outputs (Phase 4) ----
    confidence: str          # "high" | "medium" | "low"
    critic_score: float      # weighted score from the last Critic wave (0.0–1.0)
    paper_critic_score: float # same score, calculated from paper chunks only
    critic_reasoning: str    # one-line explanation for the trace

    # ---- Rewriter / loop control (Phase 4) ----
    rewrite_attempts: int    # bounded by settings.crag_max_rewrites
    rewritten_queries: dict[str, str]   # original sub_q -> rewritten sub_q

    # ---- Web fallback (Phase 4) ----
    web_used: bool

    # ---- Paper discovery (Phase 7) ----
    papers_used: bool
    papers_discovered: list[dict]

    # ---- Self-reflection (Phase 5) ----
    reflection_attempts: int
    reflection_gaps: list[str]
    reflection_follow_ups: list[str]

    # ---- Generator outputs ----
    draft_answer: str
    citation_refs: dict[str, dict[str, str]]

    # Claim verification verdicts: {S1: "supports"|"contradicts"|"unclear", ...}
    citation_verdicts: dict[str, str]

    # ---- Memory keeper output (Phase 18) ----
    memory_payload: dict | None

    # ---- Provenance Check (Phase 19) ----
    provenance_dropped_count: int   # retracted chunks hard-blocked this run
    provenance_flagged_count: int   # corrected/concern chunks passed through with a caveat

    # ---- Originality Check (Phase 20) ----
    originality_report: dict | None   # {score, flags: [...], sentences_checked}

    # ---- Results Ingestion (Phase 21) ----
    # User-supplied ground truth (their own experiment results). Copy-locked:
    # no node may rewrite, summarize, or paraphrase the values inside this
    # dict — only the surrounding prose is LLM-generated.
    user_results: dict | None   # {"summary": str, "metrics": [{"name","value","unit"}...],
                                 #  "dataset": str, "methodology_notes": str}

    # ---- Flow control / observability ----
    error: str | None
    trace: Annotated[list[dict[str, Any]], operator.add]
