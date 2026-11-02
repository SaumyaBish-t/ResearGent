"""
Human Review Gate node — Phase 24.

A side branch under Novelty Check. When the novelty similarity is high,
the pipeline PAUSES and a human decides whether the run may continue.

    Novelty Check ──┬──▶ Critic
                    │
                    └──▶ Human Review Gate ──▶ (resume) Critic
                              │
                              └── similarity > threshold → PAUSE

How the pause works
-------------------
LangGraph's `interrupt()` raises `GraphInterrupt` the first time the node
runs, which suspends the graph and surfaces the payload to the caller.
When the caller resumes with `Command(resume=<decision>)`, the node
RE-RUNS from the top and `interrupt()` RETURNS that decision instead of
raising.

Two consequences drive this implementation:

  * The payload must travel INSIDE the interrupt value. A state field
    written before the interrupt would be discarded, because the node's
    update never lands on the first pass.

  * The node must be idempotent across the two passes — everything it
    needs must come from state or from the interrupt value, never from
    something computed on the first pass.

The resume decision is accepted in several shapes so the API can send
whatever is natural: a bool, a dict, or a string.
"""

from __future__ import annotations

from typing import Any

from langgraph.errors import GraphInterrupt
from langgraph.types import interrupt

from src.agent.state import AgentState
from src.config import settings


# The node's name in the graph — also the key the frontend waits on.
NODE_NAME = "human_review_gate"


def _parse_decision(raw: Any) -> bool:
    """
    Interpret the human's decision.

    Accepts:
      * `True` / `False`
      * `{"approved": True}` / `{"approved": False}`
      * `"approve"` / `"approved"` / `"yes"` / `"confirm"`
      * `"reject"` / `"rejected"` / `"no"` / `"deny"`

    Anything unrecognised is treated as a REJECTION — when in doubt the
    run stops, because proceeding without a clear confirmation is exactly
    the failure this gate exists to prevent.
    """
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, dict):
        for key in ("approved", "approve", "confirm", "accept"):
            if key in raw:
                return bool(raw[key])
        decision = raw.get("decision")
        if decision is not None:
            return _parse_decision(decision)
        return False
    if isinstance(raw, str):
        s = raw.strip().lower()
        if s in ("approve", "approved", "yes", "y", "confirm", "confirmed", "ok"):
            return True
        if s in ("reject", "rejected", "no", "n", "deny", "denied", "cancel"):
            return False
    return False


def _build_payload(state: AgentState) -> dict[str, Any]:
    """The review request the human sees."""
    request = dict(state.get("novelty_review_request") or {})
    report = state.get("novelty_report") or {}

    payload: dict[str, Any] = {
        "node": NODE_NAME,
        "similarity": request.get(
            "similarity", report.get("overall_similarity", 0.0)
        ),
        "verdict": request.get("verdict", report.get("verdict", "unknown")),
        "rationale": request.get("rationale", report.get("rationale", "")),
        "threshold": request.get("threshold", settings.novelty_review_threshold),
        "papers": request.get("papers", []),
        "matched_metrics": report.get("matched_metrics", []),
        "approach_similarity": report.get("approach_similarity", 0.0),
        "result_similarity": report.get("result_similarity", 0.0),
        "question": "Does this comparison look right? Approve to continue.",
    }
    return payload


def human_review_gate(state: AgentState) -> dict[str, Any]:
    """
    Pause the run for human confirmation when novelty similarity is high.

    Returns a state update recording the decision. When the human rejects
    the comparison, the run is stopped with `error="novelty_review_rejected"`
    rather than continuing with a claim the user did not endorse.
    """
    request = state.get("novelty_review_request")
    report = state.get("novelty_report")

    # ---- Nothing to review ------------------------------------------------
    if not request and not report:
        return {
            "novelty_review_approved": True,
            "novelty_review_decision": "auto_approved",
            "trace": [
                {
                    "node": NODE_NAME,
                    "decision": "auto_approved",
                    "reason": "no_novelty_report",
                }
            ],
        }

    similarity = float(
        request.get("similarity", report.get("overall_similarity", 0.0))
        if request
        else report.get("overall_similarity", 0.0)
    )
    threshold = float(settings.novelty_review_threshold)

    # ---- Below the threshold: flow straight through ------------------------
    if similarity < threshold:
        return {
            "novelty_review_approved": True,
            "novelty_review_decision": "auto_approved",
            "trace": [
                {
                    "node": NODE_NAME,
                    "decision": "auto_approved",
                    "similarity": round(similarity, 4),
                    "threshold": threshold,
                    "reason": "below_threshold",
                }
            ],
        }

    # ---- Above the threshold: pause ----------------------------------------
    # First pass: `interrupt()` raises GraphInterrupt and the payload is
    # surfaced to the caller. Resume pass: it returns the human's decision.
    payload = _build_payload(state)
    decision = interrupt(payload)

    approved = _parse_decision(decision)
    decision_label = "approved" if approved else "rejected"

    update: dict[str, Any] = {
        "novelty_review_approved": approved,
        "novelty_review_decision": decision_label,
        "trace": [
            {
                "node": NODE_NAME,
                "decision": decision_label,
                "similarity": round(similarity, 4),
                "threshold": threshold,
                "verdict": payload.get("verdict"),
                "n_papers": len(payload.get("papers") or []),
            }
        ],
    }

    if not approved:
        # The human declined. Stop the run — do NOT generate an answer that
        # presents a contested comparison as settled.
        update["error"] = "novelty_review_rejected"

    return update


__all__ = ["NODE_NAME", "GraphInterrupt", "human_review_gate"]
