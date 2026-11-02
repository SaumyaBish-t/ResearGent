"""
Tests for the Phase 24 API endpoints.

Three endpoints:

  * `POST /api/results`              — ingest the user's OWN results.
                                       Returns the normalised, validated
                                       payload plus the derived queries.
  * `GET  /api/research-with-results` — SSE research run that carries the
                                       user's results into the pipeline.
  * `POST /api/novelty/review`       — resume a run paused at the Human
                                       Review Gate.

These are contract tests — they check the routes exist, validate their
input, and shape their output. They do NOT run a full agent (that needs
network + DB).

Run:
    PYTHONPATH= .venv/Scripts/python.exe -m pytest tests/test_api_novelty.py -v
"""

import sys

sys.path.insert(0, ".")

from fastapi.testclient import TestClient

from src.api.app import create_app


VALID_RESULTS = {
    "summary": "Sparse attention with learned pruning masks.",
    "approach": "Sparse attention with learned pruning masks.",
    "dataset": "WMT14",
    "metrics": [{"name": "BLEU-4", "value": 38.1}],
}


def _client() -> TestClient:
    app = create_app()
    # The auth dependency is overridden so the contract is tested without
    # a live session; auth itself has its own test suite.
    from src.auth.users import User

    app.dependency_overrides[
        __import__("src.auth.deps", fromlist=["current_user"]).current_user
    ] = lambda: User(
        id="11111111-1111-1111-1111-111111111111",
        email="t@example.test",
        name="Test",
        picture=None,
        is_admin=False,
    )
    return TestClient(app)


# ---------------------------------------------------------------------------
# The routes exist.
# ---------------------------------------------------------------------------


def test_ingest_results_route_exists():
    routes = {r.path for r in _client().app.routes}
    assert "/api/results" in routes


def test_research_with_results_route_exists():
    routes = {r.path for r in _client().app.routes}
    assert "/api/research-with-results" in routes


def test_review_resume_route_exists():
    routes = {r.path for r in _client().app.routes}
    assert "/api/novelty/review" in routes


# ---------------------------------------------------------------------------
# Ingest validates and normalises.
# ---------------------------------------------------------------------------


def test_ingest_accepts_valid_results():
    r = _client().post("/api/results", json=VALID_RESULTS)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True
    assert body["results"]["metrics"][0]["name"] == "BLEU-4"


def test_ingest_returns_the_derived_search_queries():
    r = _client().post("/api/results", json=VALID_RESULTS)
    body = r.json()
    assert body["results"]["search_queries"]
    assert any(
        "sparse attention" in q.lower() for q in body["results"]["search_queries"]
    )


def test_ingest_returns_the_ingestion_id():
    r = _client().post("/api/results", json=VALID_RESULTS)
    assert r.json()["results"]["ingestion_id"]


def test_ingest_keeps_the_value_verbatim():
    """The user's number is ground truth — it must not be rounded."""
    payload = {**VALID_RESULTS, "metrics": [{"name": "BLEU-4", "value": "38.10"}]}
    r = _client().post("/api/results", json=payload)
    assert r.json()["results"]["metrics"][0]["value"] == "38.10"


def test_ingest_rejects_an_empty_payload():
    r = _client().post("/api/results", json={})
    assert r.status_code == 422


def test_ingest_rejects_results_without_metrics():
    r = _client().post("/api/results", json={"summary": "no numbers"})
    assert r.status_code == 422


def test_ingest_rejects_a_non_numeric_metric():
    payload = {**VALID_RESULTS, "metrics": [{"name": "BLEU", "value": "great"}]}
    r = _client().post("/api/results", json=payload)
    assert r.status_code == 422


def test_ingest_rejects_duplicate_metrics():
    payload = {
        **VALID_RESULTS,
        "metrics": [
            {"name": "BLEU-4", "value": 38.1},
            {"name": "bleu_4", "value": 40.0},
        ],
    }
    r = _client().post("/api/results", json=payload)
    assert r.status_code == 422


def test_ingest_error_message_is_actionable():
    r = _client().post("/api/results", json={"summary": "s", "metrics": []})
    assert r.status_code == 422
    body = r.json()
    assert body["detail"]


# ---------------------------------------------------------------------------
# The review-resume endpoint validates its decision.
# ---------------------------------------------------------------------------


def test_review_requires_a_thread_id():
    r = _client().post("/api/novelty/review", json={"approved": True})
    assert r.status_code == 422


def test_review_rejects_an_unknown_thread():
    r = _client().post(
        "/api/novelty/review",
        json={"thread_id": "no-such-thread", "approved": True},
    )
    assert r.status_code in (404, 400)


def test_review_accepts_a_boolean_decision():
    """A paused run resumes with a plain approve/reject."""
    r = _client().post(
        "/api/novelty/review",
        json={"thread_id": "no-such-thread", "approved": False},
    )
    # The thread is unknown here — the point is the DECISION SHAPE is
    # accepted (it fails on the thread, not on the payload).
    assert r.status_code != 422
