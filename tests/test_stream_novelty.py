"""
Tests for the Phase 24 SSE / stream plumbing.

The Novelty Check and the Human Review Gate must be visible to the
frontend as they happen:

  * `novelty_check`  — the verdict, the similarity, the matched metrics.
  * `human_review_gate` — the pause request, so the UI can show the
    "are you sure?" prompt.

These tests pin the event summaries and the run-result fields.

Run:
    PYTHONPATH= .venv/Scripts/python.exe -m pytest tests/test_stream_novelty.py -v
"""

import sys

sys.path.insert(0, ".")

from src.agent.stream import _summarize_node_update


NOVELTY_UPDATE = {
    "novelty_report": {
        "verdict": "improvement",
        "is_novel": True,
        "overall_similarity": 0.62,
        "approach_similarity": 0.58,
        "result_similarity": 0.71,
        "requires_review": True,
        "rationale": "Improves on every comparable metric.",
        "compared_papers": 4,
        "matched_metrics": [
            {
                "name": "BLEU-4",
                "value": 38.1,
                "paper_name": "BLEU-4",
                "paper_value": 34.8,
                "direction": "better",
                "delta": 3.3,
            }
        ],
    },
    "novelty_dropped_refs": {"q": [{"kind": "paper", "id": "dup"}]},
    "novelty_review_request": {"similarity": 0.62, "verdict": "improvement"},
    "trace": [
        {
            "node": "novelty_check",
            "verdict": "improvement",
            "compared_papers": 4,
            "dropped_sources": 1,
            "overall_similarity": 0.62,
        }
    ],
}


# ---------------------------------------------------------------------------
# The novelty_check event summary.
# ---------------------------------------------------------------------------


def test_novelty_check_summary_exposes_the_verdict():
    s = _summarize_node_update("novelty_check", NOVELTY_UPDATE)
    assert s["verdict"] == "improvement"


def test_novelty_check_summary_exposes_the_similarity():
    s = _summarize_node_update("novelty_check", NOVELTY_UPDATE)
    assert s["overall_similarity"] == 0.62


def test_novelty_check_summary_exposes_the_matched_metrics():
    s = _summarize_node_update("novelty_check", NOVELTY_UPDATE)
    assert s["matched_metrics"]
    assert s["matched_metrics"][0]["name"] == "BLEU-4"


def test_novelty_check_summary_exposes_the_dropped_count():
    s = _summarize_node_update("novelty_check", NOVELTY_UPDATE)
    assert s["dropped_sources"] == 1


def test_novelty_check_summary_exposes_the_review_flag():
    s = _summarize_node_update("novelty_check", NOVELTY_UPDATE)
    assert s["requires_review"] is True


def test_novelty_check_summary_handles_a_passthrough():
    """A run with no user results must not crash the summariser."""
    s = _summarize_node_update("novelty_check", {"novelty_report": None})
    assert s.get("verdict") is None


# ---------------------------------------------------------------------------
# The human_review_gate event summary.
# ---------------------------------------------------------------------------


def test_review_gate_summary_exposes_the_decision():
    s = _summarize_node_update(
        "human_review_gate",
        {
            "novelty_review_approved": True,
            "novelty_review_decision": "approved",
            "trace": [
                {
                    "node": "human_review_gate",
                    "decision": "approved",
                    "similarity": 0.62,
                    "threshold": 0.5,
                    "verdict": "improvement",
                    "n_papers": 2,
                }
            ],
        },
    )
    assert s["decision"] == "approved"


def test_review_gate_summary_exposes_the_similarity():
    s = _summarize_node_update(
        "human_review_gate",
        {
            "novelty_review_decision": "approved",
            "trace": [
                {
                    "node": "human_review_gate",
                    "decision": "approved",
                    "similarity": 0.62,
                    "threshold": 0.5,
                }
            ],
        },
    )
    assert s["similarity"] == 0.62


def test_review_gate_summary_handles_an_auto_approval():
    s = _summarize_node_update(
        "human_review_gate",
        {
            "novelty_review_decision": "auto_approved",
            "trace": [
                {
                    "node": "human_review_gate",
                    "decision": "auto_approved",
                    "reason": "below_threshold",
                }
            ],
        },
    )
    assert s["decision"] == "auto_approved"


def test_review_gate_summary_handles_a_rejection():
    s = _summarize_node_update(
        "human_review_gate",
        {
            "novelty_review_approved": False,
            "novelty_review_decision": "rejected",
            "error": "novelty_review_rejected",
            "trace": [
                {"node": "human_review_gate", "decision": "rejected"}
            ],
        },
    )
    assert s["decision"] == "rejected"


# ---------------------------------------------------------------------------
# The run result carries the novelty fields.
# ---------------------------------------------------------------------------


def test_run_result_type_has_novelty_fields():
    """The run result must expose the novelty verdict to the API layer."""
    from src.agent.run import AgentResult

    fields = AgentResult.__dataclass_fields__
    assert "novelty_report" in fields
    assert "novelty_review_approved" in fields
    assert "novelty_review_decision" in fields
