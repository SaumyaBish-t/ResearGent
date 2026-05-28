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
from src.agent.state import AgentState, ContextChunk


@dataclass
class AgentResult:
    """Materialized agent run, ready for display."""

    question: str
    answer: str
    sources: dict[str, ContextChunk]
    sub_questions: list[str]
    is_complex: bool
    planner_reasoning: str
    trace: list[dict[str, Any]]
    run_id: str
    # Phase 4 additions
    confidence: str = ""
    rewrite_attempts: int = 0
    web_used: bool = False
    rewritten_queries: dict[str, str] | None = None
    # Phase 5 additions
    reflection_attempts: int = 0
    reflection_gaps: list[str] | None = None
    reflection_follow_ups: list[str] | None = None
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

        # CRAG + Reflection status line — only when there's something noteworthy
        crag_flags = []
        if self.confidence:
            crag_flags.append(f"conf={self.confidence}")
        if self.rewrite_attempts:
            crag_flags.append(f"rewrites={self.rewrite_attempts}")
        if self.web_used:
            crag_flags.append("web_fallback=YES")
        if self.reflection_attempts:
            crag_flags.append(f"reflections={self.reflection_attempts}")
        if crag_flags:
            lines.append(f"_CRAG: {'  '.join(crag_flags)}_")
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
) -> AgentResult:
    """
    Execute the agent graph end-to-end.

    `run_id` doubles as the checkpoint thread_id — pass the same id again
    to resume from the last checkpoint (useful for crash recovery / replay).
    """
    graph = build_graph(use_checkpointer=use_checkpointer)
    rid = run_id or uuid.uuid4().hex[:12]

    initial: AgentState = {
        "question": question,
        "run_id": rid,
        # k is consumed by the retriever node; not a state field, passed in via initial.
        "k": k,  # type: ignore[typeddict-unknown-key]
    }

    # LangGraph requires a thread_id when using a checkpointer, so the run is
    # addressable for replay. Without a checkpointer this is ignored.
    config = {"configurable": {"thread_id": rid}} if use_checkpointer else None

    final: AgentState = graph.invoke(initial, config=config)  # type: ignore[arg-type]

    return AgentResult(
        question=question,
        answer=final.get("draft_answer", "") or "",
        sources=final.get("citation_map") or {},
        sub_questions=final.get("sub_questions") or [question],
        is_complex=bool(final.get("is_complex")),
        planner_reasoning=final.get("planner_reasoning") or "",
        trace=final.get("trace") or [],
        run_id=rid,
        confidence=str(final.get("confidence") or ""),
        rewrite_attempts=int(final.get("rewrite_attempts") or 0),
        web_used=bool(final.get("web_used")),
        rewritten_queries=final.get("rewritten_queries") or {},
        reflection_attempts=int(final.get("reflection_attempts") or 0),
        reflection_gaps=final.get("reflection_gaps") or [],
        reflection_follow_ups=final.get("reflection_follow_ups") or [],
        error=final.get("error"),
    )
