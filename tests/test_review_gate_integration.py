"""
Integration test: the Human Review Gate really PAUSES and really RESUMES.

The unit tests in `test_review_gate.py` stub `interrupt()`, which proves the
node's logic but NOT that LangGraph actually suspends the graph and picks
up the human's decision on resume.

This test wires the REAL node into a REAL LangGraph graph with a REAL
checkpointer and asserts the full cycle:

    1. Run the graph -> it stops at the gate, with the payload visible.
    2. Resume with approval -> the graph continues and reaches the end.
    3. Resume with rejection -> the graph stops with the rejection error.

Run:
    PYTHONPATH= .venv/Scripts/python.exe -m pytest tests/test_review_gate_integration.py -v
"""

import sys

sys.path.insert(0, ".")

from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from src.agent.nodes.review_gate import human_review_gate
from src.agent.state import AgentState


def _critic(state: AgentState) -> dict[str, Any]:
    """
    A stand-in for the real Critic that records that it ran.

    It writes `confidence` — a REAL `AgentState` field. A made-up field
    like `reached_critic` would be silently DROPPED by LangGraph, because
    the state schema only carries declared keys, and the test would then
    fail for a reason that has nothing to do with the gate.
    """
    return {"confidence": "high"}


def _build_graph():
    """
    Build a graph over the REAL `AgentState`.

    Using the real state schema matters: `trace` is declared with an
    `operator.add` reducer, and a test-local `trace: list` would declare a
    different channel type for the same key, which LangGraph rejects.
    """
    g = StateGraph(AgentState)
    g.add_node("human_review_gate", human_review_gate)
    g.add_node("critic", _critic)
    g.add_edge(START, "human_review_gate")
    g.add_edge("human_review_gate", "critic")
    g.add_edge("critic", END)
    return g.compile(checkpointer=MemorySaver())


HIGH_SIM = {
    "novelty_review_request": {
        "similarity": 0.78,
        "verdict": "not_novel",
        "rationale": "The result reproduces prior work.",
        "papers": [{"title": "A Paper", "citation": "A Paper p.1"}],
    },
    "novelty_report": {"verdict": "not_novel", "overall_similarity": 0.78},
}

LOW_SIM = {
    "novelty_review_request": None,
    "novelty_report": {"verdict": "novel", "overall_similarity": 0.12},
}


def test_graph_pauses_at_the_gate_and_exposes_the_payload():
    graph = _build_graph()
    cfg = {"configurable": {"thread_id": "t-pause"}}

    chunks = list(graph.stream(HIGH_SIM, cfg, stream_mode="updates"))

    # The run stopped AT the gate — the critic never ran.
    assert any("__interrupt__" in c for c in chunks)

    snap = graph.get_state(cfg)
    assert snap.next == ("human_review_gate",)
    assert snap.interrupts
    payload = snap.interrupts[0].value
    assert payload["similarity"] == 0.78
    assert payload["verdict"] == "not_novel"
    assert payload["papers"]
    # The critic has NOT run yet.
    assert snap.values.get("confidence") is None


def test_graph_resumes_after_approval_and_reaches_the_critic():
    graph = _build_graph()
    cfg = {"configurable": {"thread_id": "t-approve"}}

    list(graph.stream(HIGH_SIM, cfg, stream_mode="updates"))

    # The human confirms.
    list(graph.stream(Command(resume={"approved": True}), cfg, stream_mode="updates"))

    final = graph.get_state(cfg)
    assert final.values.get("confidence") == "high"
    assert final.values.get("novelty_review_approved") is True
    assert final.values.get("novelty_review_decision") == "approved"
    assert not final.values.get("error")
    assert final.next == ()


def test_graph_resumes_after_rejection_and_stops():
    graph = _build_graph()
    cfg = {"configurable": {"thread_id": "t-reject"}}

    list(graph.stream(HIGH_SIM, cfg, stream_mode="updates"))
    list(graph.stream(Command(resume={"approved": False}), cfg, stream_mode="updates"))

    final = graph.get_state(cfg)
    assert final.values.get("novelty_review_approved") is False
    assert final.values.get("novelty_review_decision") == "rejected"
    assert final.values.get("error") == "novelty_review_rejected"


def test_graph_does_not_pause_below_the_threshold():
    """Below the threshold the graph runs straight through."""
    graph = _build_graph()
    cfg = {"configurable": {"thread_id": "t-low"}}

    chunks = list(graph.stream(LOW_SIM, cfg, stream_mode="updates"))

    assert not any("__interrupt__" in c for c in chunks)
    final = graph.get_state(cfg)
    assert final.values.get("confidence") == "high"
    assert final.next == ()
