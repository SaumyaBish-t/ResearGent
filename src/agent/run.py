"""
Public entry point for running the agent graph.

Usage:
    from src.agent import run_agent
    result = run_agent("Compare CRAG and Self-RAG", k=8)
    print(result.answer)
    for tag, c in result.sources.items():
        print(tag, c.citation)
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from src.agent.graph import build_graph
from src.agent.artifacts import HydratedChunk
from src.agent.state import AgentState


@dataclass
class AgentResult:
    """Materialized agent run, ready for display."""

    question: str
    answer: str
    sources: dict[str, HydratedChunk]
    sub_questions: list[str]
    is_complex: bool
    planner_reasoning: str
    trace: list[dict[str, Any]]
    run_id: str
    # Phase 4 additions
    confidence: str = ""
    critic_score: float = 0.0   # latest weighted critic score (0.0–1.0)
    rewrite_attempts: int = 0
    web_used: bool = False
    rewritten_queries: dict[str, str] | None = None
    # Phase 15: domain scope used for retrieval (and now for save routing)
    domain_scope: list[str] | None = None
    # Phase 5 additions
    reflection_attempts: int = 0
    reflection_gaps: list[str] | None = None
    reflection_follow_ups: list[str] | None = None
    # Phase 7 additions — open-domain
    papers_used: bool = False
    papers_discovered: list[dict] | None = None
    error: str | None = None

    def formatted(self) -> str:
        """Markdown-friendly render with sources footer + CRAG metadata + trace."""
        lines = []
        if self.is_complex:
            lines.append(f"_Decomposed into {len(self.sub_questions)} sub-questions:_")
            for i, sq in enumerate(self.sub_questions, start=1):
                lines.append(f"  {i}. {sq}")
            if self.planner_reasoning:
                lines.append(f"  ({self.planner_reasoning})")
            lines.append("")

        # CRAG + Discovery + Reflection status line — only when noteworthy
        crag_flags = []
        if self.confidence:
            crag_flags.append(f"conf={self.confidence}")
        if self.rewrite_attempts:
            crag_flags.append(f"rewrites={self.rewrite_attempts}")
        if self.papers_used:
            n = len(self.papers_discovered or [])
            crag_flags.append(f"papers={n}")
        if self.web_used:
            crag_flags.append("web=YES")
        if self.reflection_attempts:
            crag_flags.append(f"reflections={self.reflection_attempts}")
        if self.error == "no_sources_used_llm_priors":
            crag_flags.append("LLM_PRIORS_ONLY")
        if crag_flags:
            lines.append(f"_CRAG: {'  '.join(crag_flags)}_")
            lines.append("")

        # Show discovered papers when discovery fired
        if self.papers_discovered:
            lines.append("_Paper discovery (arXiv + Semantic Scholar):_")
            for p in self.papers_discovered:
                year = f" ({p.get('year')})" if p.get("year") else ""
                cit = p.get("citation") or ""
                title = (p.get("title") or "")[:90]
                src = p.get("source") or ""
                score = p.get("score") or 0
                cites = p.get("citations")
                cit_str = f", {cites} citations" if cites else ""
                lines.append(f"  + {title}{year}  [{cit}] (score={score:.2f}, {src}{cit_str})")
            lines.append("")

        # Show reflection diagnostics when the Reflector actually triggered a loop
        if self.reflection_follow_ups:
            lines.append("_Reflector found gaps and added follow-up sub-questions:_")
            for fu in self.reflection_follow_ups:
                lines.append(f"  + {fu}")
            if self.reflection_gaps:
                for g in self.reflection_gaps:
                    lines.append(f"    (gap: {g})")
            lines.append("")

        lines.append(self.answer)

        if self.sources:
            lines.append("")
            lines.append("Sources:")
            for tag, c in sorted(self.sources.items(), key=lambda kv: int(kv[0][1:])):
                # Origin format varies: HybridChunk has rrf_score, WebChunk has none
                rrf = getattr(c, "rrf_score", None)
                origin = f"signal={c.signal}"
                if rrf is not None:
                    origin += f", rrf={rrf:.4f}"
                else:
                    score = getattr(c, "score", None)
                    if score is not None:
                        origin += f", web_score={score:.2f}"
                lines.append(f"  [{tag}] {c.citation}  ({origin})")

        if self.trace:
            lines.append("")
            lines.append("Trace:")
            for ev in self.trace:
                node = ev.get("node", "?")
                rest = ", ".join(f"{k}={v}" for k, v in ev.items() if k != "node")
                lines.append(f"  {node}: {rest}")

        return "\n".join(lines)


def run_agent(
    question: str,
    *,
    k: int = 8,
    run_id: str | None = None,
    use_checkpointer: bool = True,
    domain_scope: list[str] | None = None,
) -> AgentResult:
    """
    Execute the agent graph end-to-end.

    `run_id` doubles as the checkpoint thread_id — pass the same id again
    to resume from the last checkpoint (useful for crash recovery / replay).

    `domain_scope` (Phase 15) hard-restricts retrieval to one or more
    registered domain ids (`agentic_ai`, `quant_finance`, `time_series`).
    When None, the planner's keyword auto-router decides; explicit values
    take precedence and skip the inference step.
    """
    graph = build_graph(use_checkpointer=use_checkpointer)
    rid = run_id or uuid.uuid4().hex[:12]

    initial: AgentState = {
        "question": question,
        "run_id": rid,
        # k is consumed by the retriever node; not a state field, passed in via initial.
        "k": k,  # type: ignore[typeddict-unknown-key]
    }
    if domain_scope:
        initial["domain_scope"] = list(domain_scope)

    # LangGraph requires a thread_id when using a checkpointer, so the run is
    # addressable for replay. Without a checkpointer this is ignored.
    config = {"configurable": {"thread_id": rid}} if use_checkpointer else None

    final: AgentState = graph.invoke(initial, config=config)  # type: ignore[arg-type]

    # Hydrate citation refs once for the result struct. Downstream consumers
    # (note auto-saver, CLI formatter) want chunks with text/citation, not
    # pointers — the boundary between lean state and human-facing output.
    from src.agent.artifacts import hydrate_one
    citation_refs = final.get("citation_refs") or {}
    ordered_tags = sorted(citation_refs.keys(), key=lambda t: int(t[1:])) if citation_refs else []
    hydrated_chunks = hydrate_one([citation_refs[t] for t in ordered_tags]) if ordered_tags else []
    sources = dict(zip(ordered_tags, hydrated_chunks))

    return AgentResult(
        question=question,
        answer=final.get("draft_answer", "") or "",
        sources=sources,
        sub_questions=final.get("sub_questions") or [question],
        is_complex=bool(final.get("is_complex")),
        planner_reasoning=final.get("planner_reasoning") or "",
        trace=final.get("trace") or [],
        run_id=rid,
        confidence=str(final.get("confidence") or ""),
        critic_score=float(final.get("critic_score") or 0.0),
        rewrite_attempts=int(final.get("rewrite_attempts") or 0),
        domain_scope=list(final.get("domain_scope") or []) or None,
        web_used=bool(final.get("web_used")),
        rewritten_queries=final.get("rewritten_queries") or {},
        reflection_attempts=int(final.get("reflection_attempts") or 0),
        reflection_gaps=final.get("reflection_gaps") or [],
        reflection_follow_ups=final.get("reflection_follow_ups") or [],
        papers_used=bool(final.get("papers_used")),
        papers_discovered=final.get("papers_discovered") or [],
        error=final.get("error"),
    )
