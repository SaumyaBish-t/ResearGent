"""
Streaming wrapper around the agent graph.

Wraps `graph.stream(input, stream_mode="updates")` and translates each
LangGraph node-update into a clean StreamEvent the UI can render.

Stream event types
------------------
  run_started        — at the very beginning, includes question + run_id
  node_start         — when a node begins (currently inferred from updates)
  node_complete      — when a node returns its state update
  final              — terminal event with the materialized AgentResult dict
  error              — if the graph throws

We DON'T expose raw state on the wire because state contains chunk objects
that are heavy + serialization-fiddly. Instead we surface compact summaries
the UI actually needs (chunk counts, citation map, confidence, etc.).
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Iterator

from src.agent.graph import build_graph
from src.agent.state import AgentState


def _summarize_node_update(node_name: str, update: dict[str, Any]) -> dict[str, Any]:
    """
    Take the raw state delta a node returned and surface only UI-friendly fields.

    The full state can contain dozens of HybridChunk objects which are
    expensive to serialize and not useful in the streaming view (the UI
    shows them at the end). Per-node summary keeps SSE messages small.
    """
    summary: dict[str, Any] = {}

    if node_name == "planner":
        summary["is_complex"] = update.get("is_complex")
        summary["sub_questions"] = update.get("sub_questions") or []
        summary["reasoning"] = update.get("planner_reasoning") or ""

    elif node_name == "retriever":
        cbs = update.get("chunks_by_subq") or {}
        summary["total_chunks"] = sum(len(v) for v in cbs.values())
        summary["per_subq_counts"] = {k[:60]: len(v) for k, v in cbs.items()}

    elif node_name == "critic":
        summary["confidence"] = update.get("confidence")
        summary["reasoning"] = (update.get("critic_reasoning") or "")[:200]
        # Pull the latest critic trace entry for grade counts
        trace = update.get("trace") or []
        if trace:
            for entry in reversed(trace):
                if entry.get("node") == "critic" and "grades" in entry:
                    summary["grades"] = entry.get("grades")
                    summary["chunks_in"] = entry.get("chunks_in")
                    summary["chunks_kept"] = entry.get("chunks_kept")
                    break

    elif node_name == "rewriter":
        rewrites = update.get("rewritten_queries") or {}
        summary["rewrite_attempt"] = update.get("rewrite_attempts")
        summary["rewritten_count"] = len(rewrites)

    elif node_name == "web_fallback":
        cbs = update.get("chunks_by_subq") or {}
        web_count = 0
        providers: set[str] = set()
        for chunks in cbs.values():
            for c in chunks:
                if getattr(c, "provider", "") or (getattr(c, "url", "") and not getattr(c, "chunk_index", 0) >= 0):
                    web_count += 1
                    p = getattr(c, "provider", "")
                    if p:
                        providers.add(p)
        summary["web_chunks_added"] = web_count
        summary["providers_used"] = sorted(providers)

    elif node_name == "generator":
        summary["answer_chars"] = len(update.get("draft_answer") or "")
        summary["n_sources"] = len(update.get("citation_map") or {})

    elif node_name == "reflector":
        summary["attempt"] = update.get("reflection_attempts")
        summary["gaps_found"] = bool(update.get("reflection_follow_ups"))
        summary["follow_ups"] = list(update.get("reflection_follow_ups") or [])
        summary["gaps"] = list(update.get("reflection_gaps") or [])

    elif node_name == "no_answer":
        summary["reason"] = update.get("error") or "no_chunks_retrieved"

    return summary


def stream_agent(
    question: str,
    *,
    k: int = 8,
    run_id: str | None = None,
) -> Iterator[dict[str, Any]]:
    """
    Run the agent and yield one event dict per node transition.

    No checkpointer here — streaming runs are ephemeral by default. The
    blocking `run_agent` path uses the checkpointer for replay/audit.
    """
    rid = run_id or uuid.uuid4().hex[:12]
    graph = build_graph(use_checkpointer=False)

    initial: AgentState = {
        "question": question,
        "run_id": rid,
        "k": k,  # type: ignore[typeddict-unknown-key]
    }

    yield {
        "type": "run_started",
        "run_id": rid,
        "question": question,
        "ts": time.time(),
    }

    final_state: AgentState = {}
    try:
        # stream_mode="updates" yields {node_name: state_delta} dicts per step.
        for chunk in graph.stream(initial, stream_mode="updates"):
            for node_name, update in chunk.items():
                # Accumulate into our local view of final state for the
                # terminal event.
                if isinstance(update, dict):
                    final_state.update(update)  # type: ignore[arg-type]
                yield {
                    "type": "node_complete",
                    "node": node_name,
                    "summary": _summarize_node_update(node_name, update or {}),
                    "ts": time.time(),
                }
    except Exception as e:
        yield {
            "type": "error",
            "error": f"{type(e).__name__}: {e}",
            "ts": time.time(),
        }
        return

    # Final terminal event with the fully materialized answer.
    citation_map = final_state.get("citation_map") or {}
    sources_payload = []
    for tag, c in sorted(citation_map.items(), key=lambda kv: int(kv[0][1:])):
        rrf = getattr(c, "rrf_score", None)
        score = getattr(c, "score", None)
        sources_payload.append(
            {
                "tag": tag,
                "citation": c.citation,
                "signal": c.signal,
                "rrf_score": rrf,
                "score": score,
                "preview": c.text[:300],
            }
        )

    yield {
        "type": "final",
        "run_id": rid,
        "answer": final_state.get("draft_answer") or "",
        "sources": sources_payload,
        "sub_questions": final_state.get("sub_questions") or [question],
        "is_complex": bool(final_state.get("is_complex")),
        "confidence": final_state.get("confidence") or "",
        "rewrite_attempts": int(final_state.get("rewrite_attempts") or 0),
        "web_used": bool(final_state.get("web_used")),
        "reflection_attempts": int(final_state.get("reflection_attempts") or 0),
        "error": final_state.get("error"),
        "ts": time.time(),
    }
