r"""Compile the agent graph.

Phase 5 topology (Plan + CRAG + Self-Reflection — the full agentic loop):

    START -> planner -> retriever -> critic
                          ▲             │
                          │     ┌───────┼─────────┐
                          │ high conf  med/low &  med/low &
                          │            retries OK retries done
                          │     │         │           │
                          │     │         ▼           ▼
                          │     │     rewriter   web_fallback
                          │     │         │           │
                          │     │         └─► critic  │
                          │     │           (loop ≤N) │
                          │     │                     ▼
                          │     │                 generator
                          │     │                     │
                          │     └────────► generator ◄┘
                          │                     │
                          │                     ▼
                          │                  verifier
                          │                     │
                          │                     ▼
                          │                 reflector
                          │                     │
                          │             ┌───────┼────────┐
                          │         gaps found            accept
                          │         budget left            OR
                          │             │             budget exhausted
                          └─────────────┘                  │
                          (loop with follow-up             ▼
                           sub-questions, ≤N iters)   memory_keeper
                                                          │
                                                          ▼
                                                        END

Failure paths (kept from earlier phases):
  - retriever returns empty -> fallback cascade
  - web_fallback returns empty -> no_answer -> END

Phase 17: Verifier node between generator and reflector.
Phase 18: Memory keeper after reflector accepts.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from src.agent.nodes import (
    critic,
    generator,
    llm_reasoning,
    memory_keeper,
    paper_discovery,
    planner,
    reflector,
    retriever,
    rewriter,
    verifier,
    web_fallback,
)
from src.agent.state import AgentState
from src.config import settings


def _route_after_retriever(state: AgentState) -> str:
    if retriever.has_any_chunks(state):
        return "critic"

    if settings.paper_discovery_enabled and not state.get("papers_used"):
        return "paper_discovery"
    if settings.tavily_api_key and not state.get("web_used"):
        return "web_fallback"
    if settings.llm_reasoning_fallback_enabled:
        return "llm_reasoning"
    return "no_answer"


def _route_after_critic(state: AgentState) -> str:
    conf = state.get("confidence") or "low"
    attempts = int(state.get("rewrite_attempts") or 0)
    max_rewrites = settings.crag_max_rewrites

    if conf == "high":
        return "generator"

    if attempts < max_rewrites:
        return "rewriter"

    papers_tried = bool(state.get("papers_used"))
    if not papers_tried and settings.paper_discovery_enabled:
        return "paper_discovery"

    web_already_tried = bool(state.get("web_used"))
    have_web_key = bool(settings.tavily_api_key)
    if not web_already_tried and have_web_key:
        return "web_fallback"

    chunks_by_subq = state.get("chunk_refs_by_subq") or {}
    if any(chunks_by_subq.values()):
        return "generator"

    return "web_fallback"


def _route_after_papers(state: AgentState) -> str:
    web_already_tried = bool(state.get("web_used"))
    have_web_key = bool(settings.tavily_api_key)
    if not web_already_tried and have_web_key:
        return "web_fallback"
    return "critic"


def _route_after_web(state: AgentState) -> str:
    chunks_by_subq = state.get("chunk_refs_by_subq") or {}
    if any(chunks_by_subq.values()):
        return "critic"
    if settings.llm_reasoning_fallback_enabled:
        return "llm_reasoning"
    return "no_answer"


def _route_after_reflector(state: AgentState) -> str:
    follow_ups = state.get("reflection_follow_ups") or []
    attempts = int(state.get("reflection_attempts") or 0)
    existing_subq_count = len(state.get("sub_questions") or [])

    if not follow_ups or attempts >= settings.reflection_max_iterations:
        return "end"

    if existing_subq_count > settings.reflection_max_subq_total:
        return "end"

    return "retriever"


def build_graph(use_checkpointer: bool = True):
    """Construct + compile the agent graph with optional checkpointer."""
    g = StateGraph(AgentState)

    # Phase 3 nodes
    g.add_node("planner", planner.plan)
    g.add_node("retriever", retriever.retrieve)
    g.add_node("generator", generator.generate)
    g.add_node("no_answer", generator.no_answer)
    # Phase 4 nodes
    g.add_node("critic", critic.critique)
    g.add_node("rewriter", rewriter.rewrite_and_retry)
    g.add_node("web_fallback", web_fallback.web_fallback)
    # Phase 5 node
    g.add_node("reflector", reflector.reflect)
    # Phase 7 nodes — open-domain
    g.add_node("paper_discovery", paper_discovery.discover)
    g.add_node("llm_reasoning", llm_reasoning.reason)
    # Phase 17: claim verification
    g.add_node("verifier", verifier.verify)
    # Phase 18: memory extraction
    g.add_node("memory_keeper", memory_keeper.extract_memory)

    g.add_edge(START, "planner")
    g.add_edge("planner", "retriever")
    g.add_conditional_edges(
        "retriever",
        _route_after_retriever,
        {
            "critic": "critic",
            "paper_discovery": "paper_discovery",
            "web_fallback": "web_fallback",
            "llm_reasoning": "llm_reasoning",
            "no_answer": "no_answer",
        },
    )
    g.add_conditional_edges(
        "critic",
        _route_after_critic,
        {
            "generator": "generator",
            "rewriter": "rewriter",
            "paper_discovery": "paper_discovery",
            "web_fallback": "web_fallback",
        },
    )
    g.add_edge("rewriter", "critic")
    g.add_conditional_edges(
        "paper_discovery",
        _route_after_papers,
        {"critic": "critic", "web_fallback": "web_fallback"},
    )
    g.add_conditional_edges(
        "web_fallback",
        _route_after_web,
        {
            "critic": "critic",
            "no_answer": "no_answer",
            "llm_reasoning": "llm_reasoning",
        },
    )
    g.add_edge("llm_reasoning", END)
    # Phase 17: generator → verifier → reflector
    g.add_edge("generator", "verifier")
    g.add_edge("verifier", "reflector")
    # Phase 18: reflector → memory_keeper → END
    g.add_conditional_edges(
        "reflector",
        _route_after_reflector,
        {"retriever": "retriever", "end": "memory_keeper"},
    )
    g.add_edge("memory_keeper", END)
    g.add_edge("no_answer", END)

    if use_checkpointer:
        if settings.resolve_database_url():
            from src.db import get_checkpointer

            return g.compile(checkpointer=get_checkpointer())
        from langgraph.checkpoint.memory import MemorySaver

        return g.compile(checkpointer=MemorySaver())

    return g.compile()