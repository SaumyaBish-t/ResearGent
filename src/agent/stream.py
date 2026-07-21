"""Streaming wrapper around the agent graph.

Wraps `graph.stream(input, stream_mode="updates")` and translates each
LangGraph node-update into a clean StreamEvent the UI can render.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Iterator

from src.agent.graph import build_graph
from src.agent.state import AgentState


def _summarize_node_update(node_name: str, update: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}

    if node_name == "planner":
        summary["is_complex"] = update.get("is_complex")
        summary["sub_questions"] = update.get("sub_questions") or []
        summary["reasoning"] = update.get("planner_reasoning") or ""

    elif node_name == "retriever":
        cbs = update.get("chunk_refs_by_subq") or {}
        summary["total_chunks"] = sum(len(v) for v in cbs.values())
        summary["per_subq_counts"] = {k[:60]: len(v) for k, v in cbs.items()}
        graph_count = sum(
            1
            for refs in cbs.values()
            for r in (refs or [])
            if (isinstance(r, dict) and r.get("kind") == "graph")
        )
        if graph_count:
            summary["graph_expanded"] = graph_count

    elif node_name == "critic":
        summary["confidence"] = update.get("confidence")
        summary["score"] = update.get("critic_score")
        summary["reasoning"] = (update.get("critic_reasoning") or "")[:200]
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
        cbs = update.get("chunk_refs_by_subq") or {}
        web_count = sum(
            1
            for refs in cbs.values()
            for r in (refs or [])
            if (isinstance(r, dict) and r.get("kind") == "web")
        )
        summary["web_chunks_added"] = web_count
        trace = update.get("trace") or []
        providers: set[str] = set()
        for e in trace:
            if e.get("node") == "web_fallback":
                providers.update(e.get("providers") or [])
        summary["providers_used"] = sorted(providers)

    elif node_name == "paper_discovery":
        papers = update.get("papers_discovered") or []
        cbs = update.get("chunk_refs_by_subq") or {}
        paper_count = sum(
            1
            for refs in cbs.values()
            for r in (refs or [])
            if (isinstance(r, dict) and r.get("kind") == "paper")
        )
        summary["papers_found"] = len(papers)
        summary["paper_chunks_added"] = paper_count

    elif node_name == "verifier":
        verdicts = update.get("citation_verdicts") or {}
        summary["claims_checked"] = len(verdicts)
        supports = sum(1 for v in verdicts.values() if v == "supports")
        contradicts = sum(1 for v in verdicts.values() if v == "contradicts")
        summary["supports"] = supports
        summary["contradicts"] = contradicts

    elif node_name == "memory_keeper":
        payload = update.get("memory_payload") or {}
        summary["topics_extracted"] = len(payload.get("topics") or [])
        summary["entities_extracted"] = len(payload.get("entities") or [])
        summary["gaps_extracted"] = len(payload.get("gaps") or [])

    elif node_name == "generator":
        summary["answer_chars"] = len(update.get("draft_answer") or "")
        summary["n_sources"] = len(update.get("citation_refs") or {})

    elif node_name == "reflector":
        summary["attempt"] = update.get("reflection_attempts")
        summary["gaps_found"] = bool(update.get("reflection_follow_ups"))
        summary["follow_ups"] = list(update.get("reflection_follow_ups") or [])
        summary["gaps"] = list(update.get("reflection_gaps") or [])

    elif node_name == "no_answer":
        summary["reason"] = update.get("error") or "no_chunks_retrieved"

    elif node_name == "llm_reasoning":
        summary["method"] = "prior-only"

    return summary


def stream_agent(
    question: str,
    *,
    k: int = 8,
    run_id: str | None = None,
    domain_scope: list[str] | None = None,
) -> Iterator[dict[str, Any]]:
    rid = run_id or uuid.uuid4().hex[:12]
    graph = build_graph(use_checkpointer=False)

    initial: AgentState = {
        "question": question,
        "run_id": rid,
        "k": k,  # type: ignore[typeddict-unknown-key]
    }
    if domain_scope:
        initial["domain_scope"] = domain_scope

    yield {
        "type": "run_started",
        "run_id": rid,
        "question": question,
        "ts": time.time(),
    }

    final_state: AgentState = {}
    try:
        for chunk in graph.stream(initial, stream_mode="updates"):
            for node_name, update in chunk.items():
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

    # Memory extraction payload (if any)
    memory_payload = final_state.get("memory_payload")
    if memory_payload:
        yield {
            "type": "memory_extracted",
            "payload": memory_payload,
            "ts": time.time(),
        }

    # Final terminal event with fully materialized answer.
    from src.agent.artifacts import hydrate_one

    citation_refs = final_state.get("citation_refs") or {}
    ordered_tags = sorted(citation_refs.keys(), key=lambda t: int(t[1:])) if citation_refs else []
    hydrated = hydrate_one([citation_refs[t] for t in ordered_tags]) if ordered_tags else []
    citation_map = dict(zip(ordered_tags, hydrated))
    citation_verdicts = final_state.get("citation_verdicts") or {}
    sources_payload = []
    for tag, c in citation_map.items():
        entry = {
            "tag": tag,
            "citation": c.citation,
            "signal": c.signal,
            "rrf_score": None,
            "score": None,
            "preview": c.text[:300],
            "doc_title": c.doc_title,
            "url": c.url or "",
            "source_file": c.source_file,
        }
        if c.authors:
            entry["authors"] = c.authors
        if c.year is not None:
            entry["year"] = c.year
        if c.venue:
            entry["venue"] = c.venue
        if c.arxiv_id:
            entry["arxiv_id"] = c.arxiv_id
        v = citation_verdicts.get(str(tag))
        if v:
            entry["verdict"] = v
        sources_payload.append(entry)

    yield {
        "type": "final",
        "run_id": rid,
        "answer": final_state.get("draft_answer") or "",
        "sources": sources_payload,
        "sub_questions": final_state.get("sub_questions") or [question],
        "is_complex": bool(final_state.get("is_complex")),
        "confidence": final_state.get("confidence") or "",
        "score": float(final_state.get("critic_score") or 0.0),
        "rewrite_attempts": int(final_state.get("rewrite_attempts") or 0),
        "web_used": bool(final_state.get("web_used")),
        "papers_used": bool(final_state.get("papers_used")),
        "reflection_attempts": int(final_state.get("reflection_attempts") or 0),
        "error": final_state.get("error"),
        "ts": time.time(),
    }

    # ---- Auto-save ----
    from src.agent.save import auto_save_run, should_auto_save
    from src.config import settings as _settings

    saved_path = auto_save_run(
        question=question,
        answer=final_state.get("draft_answer") or "",
        sources=citation_map,
        sub_questions=final_state.get("sub_questions") or [question],
        is_complex=bool(final_state.get("is_complex")),
        confidence=final_state.get("confidence") or "",
        rewrite_attempts=int(final_state.get("rewrite_attempts") or 0),
        web_used=bool(final_state.get("web_used")),
        papers_used=bool(final_state.get("papers_used")),
        reflection_attempts=int(final_state.get("reflection_attempts") or 0),
        run_id=rid,
        error=final_state.get("error"),
        score=float(final_state.get("critic_score") or 0.0),
        domain_scope=list(final_state.get("domain_scope") or []) or None,
    )
    if saved_path is not None:
        yield {
            "type": "saved",
            "path": str(saved_path),
            "ts": time.time(),
        }
    else:
        if not _settings.auto_save_to_notes:
            reason = "auto_save_disabled"
        elif final_state.get("error") == "no_sources_used_llm_priors":
            reason = "llm_priors_no_sources"
        elif not (
            (final_state.get("domain_scope") or [])
            and len(final_state.get("domain_scope") or []) == 1
        ):
            n = len(final_state.get("domain_scope") or [])
            reason = f"no_single_domain_{n}_resolved"
        elif not should_auto_save(
            confidence=final_state.get("confidence") or "",
            error=final_state.get("error"),
            score=float(final_state.get("critic_score") or 0.0),
        ):
            reason = (
                f"confidence_{final_state.get('confidence') or 'unknown'}_"
                f"score_{final_state.get('critic_score') or 0.0:.3f}_"
                f"below_{_settings.auto_save_min_confidence}/"
                f"{_settings.auto_save_min_score}"
            )
        else:
            reason = "no_notes_folder_configured"
        yield {
            "type": "save_skipped",
            "reason": reason,
            "ts": time.time(),
        }