"""
Compile the agent graph.

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
                          │                 reflector
                          │                     │
                          │             ┌───────┼────────┐
                          │         gaps found            accept
                          │         budget left            OR
                          │             │             budget exhausted
                          └─────────────┘                  │
                          (loop with follow-up             ▼
                           sub-questions, ≤N iters)       END

Failure paths (kept from earlier phases):
  - retriever returns empty -> no_answer -> END
  - web_fallback returns empty -> no_answer -> END
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from langgraph.graph import END, START, StateGraph

from src.agent.nodes import (
    critic,
    generator,
    planner,
    reflector,
    retriever,
    rewriter,
    web_fallback,
)
from src.agent.state import AgentState
from src.config import settings

CHECKPOINT_DIR = Path("data") / "agent_state"
CHECKPOINT_PATH = CHECKPOINT_DIR / "checkpoints.sqlite"


def _route_after_retriever(state: AgentState) -> str:
    """Branch right after the very first retrieve — empty = early no_answer."""
    return "critic" if retriever.has_any_chunks(state) else "no_answer"


def _route_after_critic(state: AgentState) -> str:
    """
    The CRAG decision point.

    high           -> generator (corpus is sufficient, ship it)
    medium + retry -> rewriter  (we can do better, try once more)
    low + retry    -> rewriter  (definitely missing; rewrite + try again)
    medium/low + no retries left:
      - web not used AND tavily key set -> web_fallback (corpus failed us)
      - web already used / no key       -> generator (best effort with surviving chunks)
                                           or no_answer (if nothing survived)

    Why prefer web over best-effort generator
    -----------------------------------------
    Earlier policy went to generator whenever ANY chunks survived. But "medium"
    after 2 rewrites means the Critic explicitly said "these chunks don't
    really answer the question." Producing an answer from them either:
      a) hallucinates a connection that isn't there, or
      b) says "I don't know" with a confident citation footer
    Both are worse than trying the web first. The web result might still be
    bad, but at least we tried the right tool.
    """
    conf = state.get("confidence") or "low"
    attempts = int(state.get("rewrite_attempts") or 0)
    max_rewrites = settings.crag_max_rewrites

    if conf == "high":
        return "generator"

    if attempts < max_rewrites:
        return "rewriter"

    # Budget exhausted with non-high confidence.
    web_already_tried = bool(state.get("web_used"))
    have_web_key = bool(settings.tavily_api_key)

    if not web_already_tried and have_web_key:
        return "web_fallback"

    # Best effort: surviving chunks or graceful no_answer.
    chunks_by_subq = state.get("chunks_by_subq") or {}
    if any(chunks_by_subq.values()):
        return "generator"
    return "web_fallback"  # last desperate try (returns empty if no key, then no_answer)


def _route_after_web(state: AgentState) -> str:
    """After web fallback: if we now have ANY chunks, generate; else no_answer."""
    chunks_by_subq = state.get("chunks_by_subq") or {}
    return "generator" if any(chunks_by_subq.values()) else "no_answer"


def _route_after_reflector(state: AgentState) -> str:
    """
    Phase 5 decision point — accept the draft or loop back for more retrieval.

    Loop back IFF ALL of:
      - Reflector flagged gaps AND produced actionable follow-up questions
      - We're strictly under the audit budget (< not <=, so max N means
        exactly N audit calls)
      - Total sub-questions wouldn't blow past the hard ceiling — runaway
        decomposition is worse than an imperfect answer

    Otherwise END the run.
    """
    follow_ups = state.get("reflection_follow_ups") or []
    attempts = int(state.get("reflection_attempts") or 0)
    existing_subq_count = len(state.get("sub_questions") or [])

    # Strict < — max=2 means exactly 2 reflector audits, 1 loopback.
    if not follow_ups or attempts >= settings.reflection_max_iterations:
        return "end"

    # Hard cap on sub-question explosion. Without this, each loop adds N
    # follow-ups and the retriever + critic costs grow super-linearly.
    if existing_subq_count > settings.reflection_max_subq_total:
        return "end"

    return "retriever"


def build_graph(use_checkpointer: bool = True):
    """Construct + compile the agent graph with optional SQLite checkpointer."""
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

    g.add_edge(START, "planner")
    g.add_edge("planner", "retriever")
    g.add_conditional_edges(
        "retriever",
        _route_after_retriever,
        {"critic": "critic", "no_answer": "no_answer"},
    )
    g.add_conditional_edges(
        "critic",
        _route_after_critic,
        {
            "generator": "generator",
            "rewriter": "rewriter",
            "web_fallback": "web_fallback",
        },
    )
    # Rewriter loops back to critic so the new chunks get graded.
    g.add_edge("rewriter", "critic")
    g.add_conditional_edges(
        "web_fallback",
        _route_after_web,
        {"generator": "generator", "no_answer": "no_answer"},
    )
    # Phase 5: generator -> reflector -> (retriever loop OR END)
    g.add_edge("generator", "reflector")
    g.add_conditional_edges(
        "reflector",
        _route_after_reflector,
        {"retriever": "retriever", "end": END},
    )
    g.add_edge("no_answer", END)

    if use_checkpointer:
        from langgraph.checkpoint.sqlite import SqliteSaver

        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(CHECKPOINT_PATH), check_same_thread=False)
        checkpointer = SqliteSaver(conn)
        return g.compile(checkpointer=checkpointer)

    return g.compile()
