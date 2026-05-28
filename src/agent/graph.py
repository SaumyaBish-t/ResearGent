"""
Compile the agent graph.

Phase 3 topology:

    START -> planner -> retriever
                            │
              ┌─────────────┴──────────────┐
        no chunks                       chunks ok
              │                            │
              ▼                            ▼
         no_answer                     generator
              │                            │
              └─────────────┬──────────────┘
                            ▼
                           END

Designed so Phase 4's additions (Critic, Web-Scraper, Reflector) splice in
between existing nodes without restructuring edges:

  Phase 4 will become:
    retriever -> critic
                    │
        ┌───────────┼────────────┐
   chunks ok    low conf      no chunks
        │           │             │
        ▼           ▼             ▼
    generator   web_scraper   no_answer
                    │
                    └──> retriever (loop with rewritten query, bounded)

Checkpointer
------------
SqliteSaver persists EVERY node transition to data/agent_state/checkpoints.sqlite.
That gives us:
  - Replay any past run by run_id
  - Time-travel debugging (jump to any node state)
  - Crash recovery: if a node throws, the next run from same thread_id
    resumes from the last completed checkpoint
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph

from src.agent.nodes import generator, planner, retriever
from src.agent.state import AgentState

CHECKPOINT_DIR = Path("data") / "agent_state"
CHECKPOINT_PATH = CHECKPOINT_DIR / "checkpoints.sqlite"


def _route_after_retriever(state: AgentState) -> str:
    """Conditional edge — retrieval branch."""
    return "generate" if retriever.has_any_chunks(state) else "no_answer"


def build_graph(use_checkpointer: bool = True):
    """
    Construct the StateGraph and (optionally) attach the SQLite checkpointer.

    `use_checkpointer=False` is useful in tests / one-shot scripts where
    we don't want SQLite file I/O.
    """
    g = StateGraph(AgentState)

    g.add_node("planner", planner.plan)
    g.add_node("retriever", retriever.retrieve)
    g.add_node("generator", generator.generate)
    g.add_node("no_answer", generator.no_answer)

    g.add_edge(START, "planner")
    g.add_edge("planner", "retriever")
    g.add_conditional_edges(
        "retriever",
        _route_after_retriever,
        {"generate": "generator", "no_answer": "no_answer"},
    )
    g.add_edge("generator", END)
    g.add_edge("no_answer", END)

    if use_checkpointer:
        # Lazy import — sqlite_saver module touches the DB on import in some
        # versions; we want the import cost only when actually used.
        from langgraph.checkpoint.sqlite import SqliteSaver
        import sqlite3

        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False because we may use the same connection across
        # threads in async/CLI contexts. Safe — LangGraph serializes writes.
        conn = sqlite3.connect(str(CHECKPOINT_PATH), check_same_thread=False)
        checkpointer = SqliteSaver(conn)
        return g.compile(checkpointer=checkpointer)

    return g.compile()
