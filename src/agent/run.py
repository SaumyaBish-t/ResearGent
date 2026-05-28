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
from src.agent.state import AgentState
from src.retrieval import HybridChunk


@dataclass
class AgentResult:
    """Materialized agent run, ready for display."""

    question: str
    answer: str
    sources: dict[str, HybridChunk]
    sub_questions: list[str]
    is_complex: bool
    planner_reasoning: str
    trace: list[dict[str, Any]]
    run_id: str
    error: str | None = None

    def formatted(self) -> str:
        """Markdown-friendly render with sources footer + per-node trace summary."""
        lines = []
        if self.is_complex:
            lines.append(f"_Decomposed into {len(self.sub_questions)} sub-questions:_")
            for i, sq in enumerate(self.sub_questions, start=1):
                lines.append(f"  {i}. {sq}")
            if self.planner_reasoning:
                lines.append(f"  ({self.planner_reasoning})")
            lines.append("")
        lines.append(self.answer)

        if self.sources:
            lines.append("")
            lines.append("Sources:")
            for tag, c in sorted(self.sources.items(), key=lambda kv: int(kv[0][1:])):
                origin = f"signal={c.signal}, rrf={c.rrf_score:.4f}"
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
        error=final.get("error"),
    )
