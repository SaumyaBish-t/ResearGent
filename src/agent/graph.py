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

from langgraph.graph import END, START, StateGraph

from src.agent.nodes import (
    critic,
    generator,
    llm_reasoning,
    paper_discovery,
    planner,
    reflector,
    retriever,
    rewriter,
    web_fallback,
)
from src.agent.state import AgentState
from src.config import settings

def _route_after_retriever(state: AgentState) -> str:
    """
    Branch right after the very first retrieve.

    Phase 15 fix: empty retrieval is NOT a terminal state.

    The old behaviour ("hits == 0 -> no_answer") orphaned the entire
    Critic-driven fallback ladder (rewriter / paper_discovery /
    web_fallback / llm_reasoning), which is exactly the cascade that
    exists to RESCUE empty retrieval. The visible symptom was a
    domain-scoped query returning 0 chunks and the agent giving up
    immediately, even though the persona contract pins Stage-2 live
    Semantic Scholar specifically for this case.

    New ladder when retrieval came back empty:
      1. paper_discovery — Stage-2 live arXiv + S2 (if enabled).
      2. web_fallback    — Tavily/Serper/DDG cascade (if a key is set).
      3. llm_reasoning   — last-resort priors (if enabled).
      4. no_answer       — graceful "I don't know" with no sources.

    When retrieval found ANYTHING, we go straight to Critic — unchanged.
    """
    if retriever.has_any_chunks(state):
        return "critic"

    # Empty retrieval → escalate. We don't bother calling the Critic on
    # zero chunks (it has nothing to grade) and skip straight to the
    # fallback most appropriate for the persona's Stage 1/Stage 2 model.
    if settings.paper_discovery_enabled and not state.get("papers_used"):
        return "paper_discovery"
    if settings.tavily_api_key and not state.get("web_used"):
        return "web_fallback"
    if settings.llm_reasoning_fallback_enabled:
        return "llm_reasoning"
    return "no_answer"


def _route_after_critic(state: AgentState) -> str:
    """
    The CRAG decision point (Phase 7 extends Phase 4's logic).

    high                  -> generator  (corpus is sufficient, ship it)
    medium + budget left  -> rewriter   (we can do better, try once more)
    low + budget left     -> rewriter   (definitely missing; rewrite + retry)
    medium/low + exhausted:
      - papers not yet tried + enabled -> paper_discovery  (academic literature)
      - web not yet tried + key set    -> web_fallback     (recent advances)
      - both failed but chunks survive -> generator        (best effort)
      - everything empty               -> web_fallback     (last desperate try)

    Priority order for fallbacks (NEW in Phase 7):
      paper_discovery -> web_fallback -> generator -> llm_reasoning

    Why papers BEFORE web
    ---------------------
    For technical/research questions, peer-reviewed (or pre-print) abstracts
    are denser + more authoritative than web snippets. arXiv + Semantic
    Scholar are free with no quota concerns vs Tavily's 1000/mo budget.
    """
    conf = state.get("confidence") or "low"
    attempts = int(state.get("rewrite_attempts") or 0)
    max_rewrites = settings.crag_max_rewrites

    if conf == "high":
        return "generator"

    if attempts < max_rewrites:
        return "rewriter"

    # Budget exhausted with non-high confidence — escalate through fallbacks.

    # 1. Try academic paper discovery first (free, dense, authoritative).
    papers_tried = bool(state.get("papers_used"))
    if not papers_tried and settings.paper_discovery_enabled:
        return "paper_discovery"

    # 2. Then web fallback for recent / general content.
    web_already_tried = bool(state.get("web_used"))
    have_web_key = bool(settings.tavily_api_key)
    if not web_already_tried and have_web_key:
        return "web_fallback"

    # 3. Best effort: surviving chunks or graceful next step.
    chunks_by_subq = state.get("chunk_refs_by_subq") or {}
    if any(chunks_by_subq.values()):
        return "generator"

    # 4. Last-desperate web try. When web_already_tried, this is a one-pass
    # waste (web_fallback will return empty and fall through to no_answer /
    # llm_reasoning) — accepted because it's bounded and keeps the routing
    # logic simple. Total max overhead: one extra web_fallback call.
    return "web_fallback"


def _route_after_papers(state: AgentState) -> str:
    """
    After paper_discovery completes.

    Policy: when a web-search key is configured AND web hasn't run yet,
    ALWAYS fold web results into the same chunk pool before grading. The
    Generator then synthesises a curated answer citing BOTH paper abstracts
    and recent web snippets — paper-only pools were over-confidently graded
    HIGH on tangentially-related papers, and the Critic's relaxed external
    threshold sometimes shipped refusal-style answers as "high" verdicts.

    Falls back to direct critic when web is unavailable / already tried.
    """
    web_already_tried = bool(state.get("web_used"))
    have_web_key = bool(settings.tavily_api_key)
    if not web_already_tried and have_web_key:
        return "web_fallback"
    return "critic"


def _route_after_web(state: AgentState) -> str:
    """
    After web fallback. Three outcomes:
      - chunks present                 -> critic (RE-GRADE including web chunks)
      - empty + llm_reasoning enabled  -> llm_reasoning (LAST-RESORT priors)
      - empty + llm_reasoning disabled -> no_answer (graceful "I don't know")

    Why re-grade through critic (and not straight to generator)
    -----------------------------------------------------------
    Without this loopback, the final confidence shown to the user — and
    used by auto-save — reflected only the OLD pre-web chunks. Web chunks
    are typically cleaner and more on-topic than the local retrieval that
    failed; re-grading produces an accurate confidence score that reflects
    the evidence the generator will ACTUALLY see.

    Loop-termination guarantee
    --------------------------
    The critic's routing already knows `web_used=True`, so it won't loop
    back to web_fallback again. Rewriter is also guarded by the rewrite
    budget which is exhausted by the time we get here. So the only
    onward edges from this second critic pass are:
      - high           -> generator  (the happy path; auto-save fires)
      - medium/low     -> generator (best-effort with current chunks)
    Bounded.
    """
    chunks_by_subq = state.get("chunk_refs_by_subq") or {}
    if any(chunks_by_subq.values()):
        return "critic"
    if settings.llm_reasoning_fallback_enabled:
        return "llm_reasoning"
    return "no_answer"


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
    # Phase 7 nodes — open-domain
    g.add_node("paper_discovery", paper_discovery.discover)
    g.add_node("llm_reasoning", llm_reasoning.reason)

    g.add_edge(START, "planner")
    g.add_edge("planner", "retriever")
    g.add_conditional_edges(
        "retriever",
        _route_after_retriever,
        {
            "critic": "critic",
            # Phase 15 fix: empty retrieval escalates through the same
            # fallback ladder the Critic would have triggered, instead of
            # terminating at no_answer.
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
    # Rewriter loops back to critic so the new chunks get graded.
    g.add_edge("rewriter", "critic")
    # Paper discovery: when a web key is configured, ALWAYS chain into
    # web_fallback first so the Critic sees a merged paper+web pool. This
    # produces a curated answer that cites both sources and prevents the
    # earlier failure mode of confidently shipping a refusal from a
    # tangentially-related paper set.
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
    # LLM reasoning is a terminal node — no further nodes can salvage
    # an answer that wasn't grounded in any retrieval. Goes straight to END.
    g.add_edge("llm_reasoning", END)
    # Phase 5: generator -> reflector -> (retriever loop OR END)
    g.add_edge("generator", "reflector")
    g.add_conditional_edges(
        "reflector",
        _route_after_reflector,
        {"retriever": "retriever", "end": END},
    )
    g.add_edge("no_answer", END)

    if use_checkpointer:
        # Checkpointer selection — Postgres when configured, else in-memory.
        # Sqlite is gone: a managed PG free tier survives restarts (the
        # whole point of Phase 12), and shipping two saver code paths just
        # rots. Local dev without DATABASE_URL silently uses MemorySaver
        # so `researgent research ...` still works for one-off runs.
        if settings.resolve_database_url():
            from src.db import get_checkpointer

            return g.compile(checkpointer=get_checkpointer())

        from langgraph.checkpoint.memory import MemorySaver

        return g.compile(checkpointer=MemorySaver())

    return g.compile()
