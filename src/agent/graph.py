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
    novelty_check,
    originality_check,
    paper_discovery,
    planner,
    provenance_check,
    reflector,
    retriever,
    rewriter,
    review_gate,
    verifier,
    web_fallback,
)
from src.agent.state import AgentState
from src.config import settings

PAPER_CONFIDENCE_THRESHOLD = 0.70

# Cached result of the connectivity probe below. `None` = not probed yet.
_PG_CHECKPOINTER_OK: bool | None = None


def _postgres_checkpointer_available() -> bool:
    """
    True when the shared pool can ACTUALLY hand out a connection.

    `settings.resolve_database_url()` only says a URL is configured — it
    says nothing about reachability. Building a `PostgresSaver` over an
    unreachable database succeeds (the pool connects lazily) and then the
    run dies MID-STREAM on the first checkpoint write. Probing once and
    caching the answer lets us fall back to an in-process checkpointer up
    front instead of failing halfway through a run.
    """
    global _PG_CHECKPOINTER_OK
    if _PG_CHECKPOINTER_OK is not None:
        return _PG_CHECKPOINTER_OK
    if not settings.resolve_database_url():
        _PG_CHECKPOINTER_OK = False
        return False
    try:
        from src.db import get_pool

        with get_pool().connection(timeout=3):
            _PG_CHECKPOINTER_OK = True
    except Exception:
        _PG_CHECKPOINTER_OK = False
    return _PG_CHECKPOINTER_OK


def _web_search_available() -> bool:
    configured = {
        "tavily": bool(settings.tavily_api_key),
        "serper": bool(settings.serper_api_key),
        "duckduckgo": True,
    }
    cascade = settings.web_search_cascade or list(configured)
    return any(configured.get(provider, False) for provider in cascade)


def _route_after_retriever(state: AgentState) -> str:
    if retriever.has_any_chunks(state):
        return "critic"

    if settings.paper_discovery_enabled and not state.get("papers_used"):
        return "paper_discovery"
    if _web_search_available() and not state.get("web_used"):
        return "web_fallback"
    if settings.llm_reasoning_fallback_enabled:
        return "llm_reasoning"
    return "no_answer"


def _route_after_planner(state: AgentState) -> str:
    """Search academic papers before local retrieval and web fallback."""
    if settings.paper_discovery_enabled and not state.get("papers_used"):
        return "paper_discovery"
    return "retriever"


def _route_after_critic(state: AgentState) -> str:
    conf = state.get("confidence") or "low"
    attempts = int(state.get("rewrite_attempts") or 0)
    max_rewrites = settings.crag_max_rewrites
    paper_score = float(state.get("paper_critic_score") or 0.0)
    papers_tried = bool(state.get("papers_used"))
    web_already_tried = bool(state.get("web_used"))
    chunks_by_subq = state.get("chunk_refs_by_subq") or {}
    has_chunks = any(chunks_by_subq.values())

    if papers_tried and not web_already_tried:
        if paper_score < PAPER_CONFIDENCE_THRESHOLD and _web_search_available():
            return "web_fallback"
        if paper_score >= PAPER_CONFIDENCE_THRESHOLD:
            return "generator"

    if web_already_tried:
        if has_chunks:
            return "generator"
        if settings.llm_reasoning_fallback_enabled:
            return "llm_reasoning"
        return "no_answer"

    if conf == "high":
        return "generator"

    if attempts < max_rewrites:
        return "rewriter"

    if not papers_tried and settings.paper_discovery_enabled:
        return "paper_discovery"

    have_web_key = _web_search_available()
    if not web_already_tried and have_web_key:
        return "web_fallback"

    if has_chunks:
        return "generator"

    if settings.llm_reasoning_fallback_enabled:
        return "llm_reasoning"
    return "no_answer"


def _route_after_web(state: AgentState) -> str:
    chunks_by_subq = state.get("chunk_refs_by_subq") or {}
    if any(chunks_by_subq.values()):
        return "provenance_check"
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
    # Phase 19: provenance check
    g.add_node("provenance_check", provenance_check.provenance_check)
    # Phase 20: originality check
    g.add_node("originality_check", originality_check.originality_check)
    # Phase 24: novelty check + human review gate
    g.add_node("novelty_check", novelty_check.novelty_check)
    g.add_node("human_review_gate", review_gate.human_review_gate)

    g.add_edge(START, "planner")
    g.add_conditional_edges(
        "planner",
        _route_after_planner,
        {"paper_discovery": "paper_discovery", "retriever": "retriever"},
    )
    g.add_conditional_edges(
        "retriever",
        _route_after_retriever,
        {
            "critic": "provenance_check",   # CHANGED — was "critic"
            "paper_discovery": "paper_discovery",
            "web_fallback": "web_fallback",
            "llm_reasoning": "llm_reasoning",
            "no_answer": "no_answer",
        },
    )
    # Phase 24: provenance -> novelty_check -> human_review_gate -> critic
    #
    # The Critic's text says it only sees sources that passed Provenance
    # AND Novelty. That is enforced structurally: the Novelty Check
    # filters `chunk_refs_by_subq` in place, and the Critic reads that
    # same field — so whatever the Novelty Check drops never reaches it.
    #
    # The Human Review Gate sits between the Novelty Check and the Critic
    # as a side branch: it interrupts the graph when the novelty
    # similarity is high, and on resume the graph continues to the Critic.
    g.add_edge("provenance_check", "novelty_check")
    g.add_edge("novelty_check", "human_review_gate")
    g.add_edge("human_review_gate", "critic")
    g.add_conditional_edges(
        "critic",
        _route_after_critic,
        {
            "generator": "generator",
            "rewriter": "rewriter",
            "paper_discovery": "paper_discovery",
            "web_fallback": "web_fallback",
            "llm_reasoning": "llm_reasoning",
            "no_answer": "no_answer",
        },
    )
    g.add_edge("rewriter", "critic")
    g.add_edge("paper_discovery", "retriever")
    g.add_conditional_edges(
        "web_fallback",
        _route_after_web,
        {
            "provenance_check": "provenance_check",
            "no_answer": "no_answer",
            "llm_reasoning": "llm_reasoning",
        },
    )
    g.add_edge("llm_reasoning", END)
    # Phase 17/20/24: generator -> verifier -> reflector -> originality_check
    #
    # The Originality Check sits BETWEEN Answer Generation and the Vault.
    # It used to run before the Reflector, which meant it could validate a
    # draft that was not yet final. The Reflector does not rewrite the
    # draft (only the Generator does), so by the time the Reflector
    # returns, `draft_answer` is final — and the originality report
    # describes exactly the text that will be written to the vault.
    g.add_edge("generator", "verifier")
    g.add_edge("verifier", "reflector")
    # Phase 18: reflector -> originality_check -> memory_keeper -> END
    #
    # The Originality Check now sits on the reflector's accept path, so the
    # report describes exactly the final text that reaches the vault.
    g.add_conditional_edges(
        "reflector",
        _route_after_reflector,
        {
            "retriever": "retriever",
            "end": "originality_check",
        },
    )
    g.add_edge("originality_check", "memory_keeper")
    g.add_edge("memory_keeper", END)
    g.add_edge("no_answer", END)

    if use_checkpointer:
        from langgraph.checkpoint.memory import MemorySaver

        # Prefer the durable Postgres checkpointer, but only when the
        # database is actually reachable. Otherwise fall back to an
        # IN-PROCESS checkpointer: interrupt/resume still works within one
        # server process, and local runs without a database keep working.
        if _postgres_checkpointer_available():
            from src.db import get_checkpointer

            return g.compile(checkpointer=get_checkpointer())

        return g.compile(checkpointer=MemorySaver())

    return g.compile()
