"""
Compile the agent graph.

Phase 4 topology (Corrective RAG):

    START -> planner -> retriever -> critic
                                       │
                              ┌────────┼─────────┐
                          high conf  med/low &  med/low &
                                    retries OK   retries done
                              │          │             │
                              │          ▼             ▼
                              │      rewriter     web_fallback
                              │          │             │
                              │          └──► critic   │
                              │            (loop ≤N)   │
                              │                        ▼
                              │                    generator
                              │                        │
                              └─────► generator <──────┘
                                          │
                                          ▼
                                         END

  No-chunks path (kept from Phase 3):
    if retriever (or web_fallback) leaves chunks_by_subq totally empty,
    we still route to `no_answer` for a graceful "I don't know".

Splicing in Phase 5's Reflector
-------------------------------
Reflector will sit between generator and END. It reads the draft,
identifies gaps, and either accepts (-> END) or loops back to retriever
with a new sub-question. The edges from generator change from
`generator -> END` to `generator -> reflector` with one new conditional
edge — no other node touches.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from langgraph.graph import END, START, StateGraph

from src.agent.nodes import critic, generator, planner, retriever, rewriter, web_fallback
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
      - some chunks survived -> generator (best effort with what we have)
      - no chunks survived   -> web_fallback (last resort)
    """
    conf = state.get("confidence") or "low"
    attempts = int(state.get("rewrite_attempts") or 0)
    max_rewrites = settings.crag_max_rewrites

    if conf == "high":
        return "generator"

    if attempts < max_rewrites:
        return "rewriter"

    # Budget exhausted.
    chunks_by_subq = state.get("chunks_by_subq") or {}
    if any(chunks_by_subq.values()):
        return "generator"  # best effort
    return "web_fallback"


def _route_after_web(state: AgentState) -> str:
    """After web fallback: if we now have ANY chunks, generate; else no_answer."""
    chunks_by_subq = state.get("chunks_by_subq") or {}
    return "generator" if any(chunks_by_subq.values()) else "no_answer"


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
    g.add_edge("generator", END)
    g.add_edge("no_answer", END)

    if use_checkpointer:
        from langgraph.checkpoint.sqlite import SqliteSaver

        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(CHECKPOINT_PATH), check_same_thread=False)
        checkpointer = SqliteSaver(conn)
        return g.compile(checkpointer=checkpointer)

    return g.compile()
