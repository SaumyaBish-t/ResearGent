"""
Tests for the Novelty Check node (Phase 24).

The node's job:

  1. Read the user's OWN results (ground truth) from state.
  2. Compare their APPROACH and their NUMBERS against the retrieved papers.
  3. Emit a `novelty_report`.
  4. Emit the refs that PASSED, so the Critic downstream only sees
     sources that cleared both Provenance and Novelty.

These tests use FAKE chunks and a stubbed embedder so they run without
network access or a database.

Run:
    PYTHONPATH= .venv/Scripts/python.exe -m pytest tests/test_novelty_node.py -v
"""

import sys

sys.path.insert(0, ".")

from src.agent.nodes import novelty_check as node
from src.agent.artifacts import HydratedChunk


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


def _paper_chunk(text: str, *, title: str = "Paper") -> HydratedChunk:
    return HydratedChunk(
        text=text,
        citation=f"{title} p.1",
        doc_title=title,
        signal="paper:semantic_scholar",
        arxiv_id="2401.00001",
    )


def _stub_embed(monkeypatch, *, same: bool = False):
    """
    Replace the node's embedder with a deterministic stub.

    `same=True` makes every embedding identical (maximal similarity);
    `same=False` makes them orthogonal (minimal similarity).
    """
    calls: list[list[str]] = []

    def fake_embed(texts):
        calls.append(list(texts))
        out = []
        for i, _ in enumerate(texts):
            # First text is the user's, rest are papers.
            out.append([1.0, 0.0] if (same or i == 0) else [0.0, 1.0])
        return out

    monkeypatch.setattr(node, "embed", fake_embed)
    return calls


USER_RESULTS = {
    "summary": "Sparse attention with learned pruning masks.",
    "approach": "Sparse attention with learned pruning masks.",
    "dataset": "WMT14",
    "metrics": [
        {"name": "BLEU-4", "value": 38.1, "numeric_value": 38.1},
    ],
}


# ---------------------------------------------------------------------------
# No user results -> the node is a no-op passthrough.
# ---------------------------------------------------------------------------


def test_no_user_results_is_a_passthrough(monkeypatch):
    monkeypatch.setattr(node.settings, "novelty_check_enabled", True)
    refs = {"q": [{"kind": "paper", "id": "a1"}]}

    out = node.novelty_check({"chunk_refs_by_subq": refs})

    assert out["chunk_refs_by_subq"] == refs
    assert out.get("novelty_report") is None


def test_disabled_novelty_check_is_a_passthrough(monkeypatch):
    monkeypatch.setattr(node.settings, "novelty_check_enabled", False)
    refs = {"q": [{"kind": "paper", "id": "a1"}]}

    out = node.novelty_check({
        "chunk_refs_by_subq": refs,
        "user_results": USER_RESULTS,
    })

    assert out["chunk_refs_by_subq"] == refs
    assert out.get("novelty_report") is None


# ---------------------------------------------------------------------------
# The report.
# ---------------------------------------------------------------------------


def test_report_is_produced_when_results_and_papers_exist(monkeypatch):
    monkeypatch.setattr(node.settings, "novelty_check_enabled", True)
    _stub_embed(monkeypatch, same=True)
    monkeypatch.setattr(node, "hydrate", lambda _refs: {
        "q": [_paper_chunk("Our method reaches 34.8 BLEU-4 on WMT14.")]
    })

    out = node.novelty_check({
        "chunk_refs_by_subq": {"q": [{"kind": "paper", "id": "a1"}]},
        "user_results": USER_RESULTS,
    })

    report = out["novelty_report"]
    assert report is not None
    assert report["verdict"] in ("novel", "improvement", "not_novel", "inconclusive")
    assert 0.0 <= report["overall_similarity"] <= 1.0
    assert report["compared_papers"] == 1


def test_report_records_the_user_metric_comparison(monkeypatch):
    """The user's BLEU-4 38.1 vs the paper's 34.8 must be recorded."""
    monkeypatch.setattr(node.settings, "novelty_check_enabled", True)
    _stub_embed(monkeypatch, same=True)
    monkeypatch.setattr(node, "hydrate", lambda _refs: {
        "q": [_paper_chunk("Our method reaches 34.8 BLEU-4 on WMT14.")]
    })

    out = node.novelty_check({
        "chunk_refs_by_subq": {"q": [{"kind": "paper", "id": "a1"}]},
        "user_results": USER_RESULTS,
    })

    matched = out["novelty_report"]["matched_metrics"]
    assert len(matched) == 1
    assert matched[0]["direction"] == "better"


def test_trace_is_emitted(monkeypatch):
    monkeypatch.setattr(node.settings, "novelty_check_enabled", True)
    _stub_embed(monkeypatch, same=True)
    monkeypatch.setattr(node, "hydrate", lambda _refs: {
        "q": [_paper_chunk("Reaches 34.8 BLEU-4.")]
    })

    out = node.novelty_check({
        "chunk_refs_by_subq": {"q": [{"kind": "paper", "id": "a1"}]},
        "user_results": USER_RESULTS,
    })

    trace = out["trace"]
    assert trace
    assert trace[-1]["node"] == "novelty_check"
    assert "verdict" in trace[-1]


# ---------------------------------------------------------------------------
# The novelty screen — which sources survive to the Critic.
# ---------------------------------------------------------------------------


def test_refs_are_filtered_by_novelty(monkeypatch):
    """A paper whose approach matches closely is a novelty duplicate and
    must NOT reach the Critic."""
    monkeypatch.setattr(node.settings, "novelty_check_enabled", True)
    monkeypatch.setattr(node.settings, "novelty_approach_threshold", 0.50)
    _stub_embed(monkeypatch, same=True)
    monkeypatch.setattr(node, "hydrate", lambda _refs: {
        "q": [
            _paper_chunk("Sparse attention with learned pruning masks reaches 34.8 BLEU-4."),
            _paper_chunk("A survey of graph neural networks for molecules."),
        ]
    })

    out = node.novelty_check({
        "chunk_refs_by_subq": {
            "q": [
                {"kind": "paper", "id": "duplicate"},
                {"kind": "paper", "id": "distinct"},
            ]
        },
        "user_results": USER_RESULTS,
    })

    surviving = [r["id"] for r in out["chunk_refs_by_subq"]["q"]]
    assert "duplicate" not in surviving
    assert out["novelty_dropped_refs"]["q"]
    assert any(r["id"] == "duplicate" for r in out["novelty_dropped_refs"]["q"])


def test_all_refs_survive_when_novelty_is_disabled(monkeypatch):
    monkeypatch.setattr(node.settings, "novelty_check_enabled", False)
    refs = {"q": [{"kind": "paper", "id": "a1"}]}
    out = node.novelty_check({
        "chunk_refs_by_subq": refs,
        "user_results": USER_RESULTS,
    })
    assert out["chunk_refs_by_subq"] == refs


# ---------------------------------------------------------------------------
# The review gate trigger.
# ---------------------------------------------------------------------------


def test_requires_review_is_set_when_similarity_is_high(monkeypatch):
    monkeypatch.setattr(node.settings, "novelty_check_enabled", True)
    monkeypatch.setattr(node.settings, "novelty_review_threshold", 0.50)
    _stub_embed(monkeypatch, same=True)
    monkeypatch.setattr(node, "hydrate", lambda _refs: {
        "q": [_paper_chunk("Sparse attention with learned pruning masks reaches 34.8 BLEU-4.")]
    })

    out = node.novelty_check({
        "chunk_refs_by_subq": {"q": [{"kind": "paper", "id": "a1"}]},
        "user_results": USER_RESULTS,
    })

    assert out["novelty_report"]["requires_review"] is True
    assert out["novelty_review_request"] is not None
    assert out["novelty_review_request"]["similarity"] > 0


def test_no_review_request_when_similarity_is_low(monkeypatch):
    monkeypatch.setattr(node.settings, "novelty_check_enabled", True)
    monkeypatch.setattr(node.settings, "novelty_review_threshold", 0.50)
    _stub_embed(monkeypatch, same=False)
    monkeypatch.setattr(node, "hydrate", lambda _refs: {
        "q": [_paper_chunk("A survey of graph neural networks for molecules.")]
    })

    out = node.novelty_check({
        "chunk_refs_by_subq": {"q": [{"kind": "paper", "id": "a1"}]},
        "user_results": USER_RESULTS,
    })

    assert out["novelty_report"]["requires_review"] is False
    assert out.get("novelty_review_request") is None


# ---------------------------------------------------------------------------
# Robustness.
# ---------------------------------------------------------------------------


def test_empty_chunk_set_still_produces_a_report(monkeypatch):
    monkeypatch.setattr(node.settings, "novelty_check_enabled", True)
    monkeypatch.setattr(node, "hydrate", lambda _refs: {})

    out = node.novelty_check({
        "chunk_refs_by_subq": {"q": []},
        "user_results": USER_RESULTS,
    })

    assert out["novelty_report"] is not None
    assert out["novelty_report"]["compared_papers"] == 0


def test_embed_failure_does_not_crash_the_run(monkeypatch):
    """A dead embedder must degrade gracefully, not kill the pipeline."""
    monkeypatch.setattr(node.settings, "novelty_check_enabled", True)

    def boom(_texts):
        raise RuntimeError("embedder down")

    monkeypatch.setattr(node, "embed", boom)
    monkeypatch.setattr(node, "hydrate", lambda _refs: {
        "q": [_paper_chunk("Reaches 34.8 BLEU-4.")]
    })

    out = node.novelty_check({
        "chunk_refs_by_subq": {"q": [{"kind": "paper", "id": "a1"}]},
        "user_results": USER_RESULTS,
    })

    assert out["chunk_refs_by_subq"]["q"]
    assert out["novelty_report"] is not None


# ---------------------------------------------------------------------------
# The matched-metric SHAPE contract (regression)
# ---------------------------------------------------------------------------


def test_matched_metrics_use_the_flat_shape_the_frontend_reads(monkeypatch):
    """
    The backend must emit ONE canonical shape for a matched metric.

    This was a real integration bug: `novelty_report.matched_metrics` used
    a NESTED form (`user_metric` / `paper_metric` sub-dicts) while
    `novelty_review_request` used a FLAT one. The frontend reads the flat
    keys, so the novelty card rendered `undefined` for every value.

    Pin the flat keys the UI actually reads.
    """
    monkeypatch.setattr(node.settings, "novelty_check_enabled", True)
    _stub_embed(monkeypatch, same=True)
    monkeypatch.setattr(node, "hydrate", lambda _refs: {
        "q": [_paper_chunk("Our method reaches 34.8 BLEU-4 on WMT14.")]
    })

    out = node.novelty_check({
        "chunk_refs_by_subq": {"q": [{"kind": "paper", "id": "a1"}]},
        "user_results": USER_RESULTS,
    })

    report_metrics = out["novelty_report"]["matched_metrics"]
    assert report_metrics, "no matched metrics in the report"
    row = report_metrics[0]
    for key in ("name", "value", "paper_name", "paper_value", "direction"):
        assert key in row, f"report matched_metric is missing the flat key {key!r}"

    # The per-paper rows in the review request must use the SAME shape.
    request = out["novelty_review_request"]
    if request and request.get("papers"):
        paper_rows = request["papers"][0].get("metrics") or []
        if paper_rows:
            assert set(paper_rows[0].keys()) == set(row.keys()), (
                "review-request metric rows drifted from the report shape"
            )


def test_summary_matched_metrics_are_also_flat(monkeypatch):
    """The SSE summary reads the report, so it inherits the flat shape."""
    from src.agent.stream import _summarize_node_update

    monkeypatch.setattr(node.settings, "novelty_check_enabled", True)
    _stub_embed(monkeypatch, same=True)
    monkeypatch.setattr(node, "hydrate", lambda _refs: {
        "q": [_paper_chunk("Our method reaches 34.8 BLEU-4 on WMT14.")]
    })

    out = node.novelty_check({
        "chunk_refs_by_subq": {"q": [{"kind": "paper", "id": "a1"}]},
        "user_results": USER_RESULTS,
    })
    s = _summarize_node_update("novelty_check", out)
    assert s["matched_metrics"], "summary dropped the matched metrics"
    assert "value" in s["matched_metrics"][0]
    assert "paper_value" in s["matched_metrics"][0]
