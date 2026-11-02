"""
Tests for the Human Review Gate node (Phase 24).

The gate is a side branch under Novelty Check:

    Novelty Check ──▶ [Human Review Gate] ──▶ Critic
          │                     │
          └── similarity > 50% ─┘ (pause; human confirms; resume)

Implementation note
-------------------
The pause uses LangGraph's `interrupt()`, which raises a `GraphInterrupt`
when the node runs. On the FIRST pass the node interrupts BEFORE writing
its update, so the pause payload must be carried IN the interrupt value
itself (not in a state field). On RESUME the node re-runs from the top,
`interrupt()` returns the human's decision, and the node then writes the
approval into state.

These tests exercise both halves of that cycle.

Run:
    PYTHONPATH= .venv/Scripts/python.exe -m pytest tests/test_review_gate.py -v
"""

import sys

sys.path.insert(0, ".")

import pytest

from src.agent.nodes import review_gate as node


# ---------------------------------------------------------------------------
# The interrupt payload — what the human is shown.
# ---------------------------------------------------------------------------


def test_gate_interrupts_with_a_review_request(monkeypatch):
    """Above the threshold, the gate must pause and describe why.

    The pause is a `GraphInterrupt` — it propagates out of the node so
    LangGraph can suspend the graph. The payload the human sees travels
    INSIDE the interrupt value, because the node's state update is never
    written on the first pass.
    """
    monkeypatch.setattr(node.settings, "novelty_review_timeout_seconds", 0)

    calls: list[dict] = []

    def fake_interrupt(payload):
        calls.append(payload)
        raise node.GraphInterrupt(payload)

    monkeypatch.setattr(node, "interrupt", fake_interrupt)

    with pytest.raises(node.GraphInterrupt):
        node.human_review_gate({
            "novelty_review_request": {
                "similarity": 0.78,
                "verdict": "not_novel",
                "rationale": "The result reproduces prior work.",
                "papers": [{"title": "A Paper", "citation": "A Paper p.1"}],
            },
            "novelty_report": {"verdict": "not_novel", "overall_similarity": 0.78},
        })

    # The interrupt carries the payload the UI renders.
    assert len(calls) == 1
    payload = calls[0]
    assert payload["similarity"] == 0.78
    assert payload["verdict"] == "not_novel"
    assert payload["papers"]


def test_gate_does_not_interrupt_below_the_threshold(monkeypatch):
    """Below the threshold the gate is a no-op — no pause."""
    monkeypatch.setattr(node.settings, "novelty_review_timeout_seconds", 0)

    def boom(_payload):
        raise AssertionError("interrupt() must not be called below threshold")

    monkeypatch.setattr(node, "interrupt", boom)

    out = node.human_review_gate({
        "novelty_review_request": None,
        "novelty_report": {"verdict": "novel", "overall_similarity": 0.12},
    })

    assert out["novelty_review_approved"] is True
    assert out["novelty_review_decision"] == "auto_approved"


def test_gate_does_not_interrupt_without_a_report(monkeypatch):
    def boom(_payload):
        raise AssertionError("interrupt() must not be called with no report")

    monkeypatch.setattr(node, "interrupt", boom)

    out = node.human_review_gate({
        "novelty_review_request": None,
        "novelty_report": None,
    })

    assert out["novelty_review_approved"] is True


# ---------------------------------------------------------------------------
# The resume path — the human confirmed.
# ---------------------------------------------------------------------------


def test_gate_approves_when_the_human_confirms(monkeypatch):
    monkeypatch.setattr(node.settings, "novelty_review_timeout_seconds", 0)

    monkeypatch.setattr(node, "interrupt", lambda _payload: {"approved": True})

    out = node.human_review_gate({
        "novelty_review_request": {"similarity": 0.78, "verdict": "not_novel"},
        "novelty_report": {"verdict": "not_novel", "overall_similarity": 0.78},
    })

    assert out["novelty_review_approved"] is True
    assert out["novelty_review_decision"] == "approved"


def test_gate_rejects_when_the_human_declines(monkeypatch):
    monkeypatch.setattr(node.settings, "novelty_review_timeout_seconds", 0)
    monkeypatch.setattr(node, "interrupt", lambda _payload: {"approved": False})

    out = node.human_review_gate({
        "novelty_review_request": {"similarity": 0.78, "verdict": "not_novel"},
        "novelty_report": {"verdict": "not_novel", "overall_similarity": 0.78},
    })

    assert out["novelty_review_approved"] is False
    assert out["novelty_review_decision"] == "rejected"


def test_gate_accepts_a_bare_boolean_decision(monkeypatch):
    """The resume payload may be just `true`/`false`."""
    monkeypatch.setattr(node.settings, "novelty_review_timeout_seconds", 0)
    monkeypatch.setattr(node, "interrupt", lambda _payload: True)

    out = node.human_review_gate({
        "novelty_review_request": {"similarity": 0.78},
        "novelty_report": {"verdict": "not_novel"},
    })

    assert out["novelty_review_approved"] is True


def test_gate_accepts_a_string_decision(monkeypatch):
    monkeypatch.setattr(node.settings, "novelty_review_timeout_seconds", 0)
    monkeypatch.setattr(node, "interrupt", lambda _payload: "approve")

    out = node.human_review_gate({
        "novelty_review_request": {"similarity": 0.78},
        "novelty_report": {"verdict": "not_novel"},
    })

    assert out["novelty_review_approved"] is True


# ---------------------------------------------------------------------------
# Rejection is a real outcome — the run must not silently continue.
# ---------------------------------------------------------------------------


def test_rejection_sets_an_error(monkeypatch):
    monkeypatch.setattr(node.settings, "novelty_review_timeout_seconds", 0)
    monkeypatch.setattr(node, "interrupt", lambda _payload: {"approved": False})

    out = node.human_review_gate({
        "novelty_review_request": {"similarity": 0.78},
        "novelty_report": {"verdict": "not_novel"},
    })

    assert out["error"] == "novelty_review_rejected"


def test_approval_does_not_set_an_error(monkeypatch):
    monkeypatch.setattr(node.settings, "novelty_review_timeout_seconds", 0)
    monkeypatch.setattr(node, "interrupt", lambda _payload: {"approved": True})

    out = node.human_review_gate({
        "novelty_review_request": {"similarity": 0.78},
        "novelty_report": {"verdict": "not_novel"},
    })

    assert out.get("error") is None


# ---------------------------------------------------------------------------
# Trace.
# ---------------------------------------------------------------------------


def test_gate_emits_a_trace(monkeypatch):
    monkeypatch.setattr(node.settings, "novelty_review_timeout_seconds", 0)
    monkeypatch.setattr(node, "interrupt", lambda _payload: {"approved": True})

    out = node.human_review_gate({
        "novelty_review_request": {"similarity": 0.78},
        "novelty_report": {"verdict": "not_novel"},
    })

    assert out["trace"]
    assert out["trace"][-1]["node"] == "human_review_gate"
    assert out["trace"][-1]["decision"] == "approved"
