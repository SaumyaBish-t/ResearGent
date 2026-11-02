"""
Tests for the Results Ingestion store + Novelty Check node contract (Phase 24).

These define the contract for the two new pieces:

  * `src.novelty.ingest` — the Results Ingestion store. The user's OWN
    results are GROUND TRUTH: they are stored, validated, and frozen. No
    node may rewrite, round, or paraphrase a value inside them.

  * `src.agent.nodes.novelty_check` — the Novelty Check node. It scores
    the user's approach + results against the retrieved papers and emits
    a report plus the refs that SURVIVED the novelty screen.

Run:
    PYTHONPATH= .venv/Scripts/python.exe -m pytest tests/test_novelty_ingest.py -v
"""

import sys

sys.path.insert(0, ".")

from src.novelty.ingest import (
    IngestionError,
    build_search_queries,
    normalize_user_results,
    validate_user_results,
)


# ---------------------------------------------------------------------------
# Validation — the gate on what may enter the pipeline as ground truth.
# ---------------------------------------------------------------------------


def test_validate_accepts_wellformed_results():
    results = {
        "summary": "Our new pruning heuristic on the retrieval corpus.",
        "metrics": [
            {"name": "BLEU-4", "value": 38.1},
            {"name": "Accuracy", "value": 0.94, "unit": "%"},
        ],
        "dataset": "WMT14 en-de",
        "methodology_notes": "Contrastive fine-tuning with hard negatives.",
    }
    out = validate_user_results(results)
    assert out["metrics"][0]["name"] == "BLEU-4"
    assert float(out["metrics"][0]["value"]) == 38.1


def test_validate_rejects_empty_payload():
    try:
        validate_user_results({})
    except IngestionError as e:
        assert "summary" in str(e).lower() or "metrics" in str(e).lower()
    else:
        raise AssertionError("expected IngestionError for empty payload")


def test_validate_rejects_results_without_metrics():
    try:
        validate_user_results({"summary": "no numbers here", "metrics": []})
    except IngestionError:
        pass
    else:
        raise AssertionError("expected IngestionError for empty metric list")


def test_validate_rejects_metric_without_name():
    try:
        validate_user_results({"summary": "s", "metrics": [{"value": 1.0}]})
    except IngestionError as e:
        assert "name" in str(e).lower()
    else:
        raise AssertionError("expected IngestionError for unnamed metric")


def test_validate_rejects_metric_with_non_numeric_value():
    try:
        validate_user_results({"summary": "s", "metrics": [{"name": "BLEU", "value": "great"}]})
    except IngestionError as e:
        assert "numeric" in str(e).lower()
    else:
        raise AssertionError("expected IngestionError for non-numeric value")


def test_validate_rejects_duplicate_metric_names():
    """Two 'BLEU-4' entries with different numbers are ambiguous."""
    try:
        validate_user_results({
            "summary": "s",
            "metrics": [
                {"name": "BLEU-4", "value": 38.1},
                {"name": "bleu_4", "value": 40.0},
            ],
        })
    except IngestionError as e:
        assert "duplicate" in str(e).lower()
    else:
        raise AssertionError("expected IngestionError for duplicate metric")


# ---------------------------------------------------------------------------
# Normalisation — the stored form is canonical and copy-locked.
# ---------------------------------------------------------------------------


def test_normalize_keeps_metric_values_verbatim():
    """The VALUE is ground truth — normalisation must not round it."""
    out = normalize_user_results({
        "summary": "s",
        "metrics": [{"name": "BLEU-4", "value": "38.10"}],
    })
    # The raw string is preserved exactly; only a parsed float is added.
    assert out["metrics"][0]["value"] == "38.10"
    assert float(out["metrics"][0]["numeric_value"]) == 38.1


def test_normalize_adds_missing_approach_from_methodology():
    out = normalize_user_results({
        "summary": "s",
        "metrics": [{"name": "BLEU", "value": 1.0}],
        "methodology_notes": "We use a diffusion model over the molecular graph.",
    })
    assert out["approach"]


def test_normalize_adds_missing_summary_from_approach():
    out = normalize_user_results({
        "approach": "A diffusion model over the molecular graph.",
        "metrics": [{"name": "BLEU", "value": 1.0}],
    })
    assert out["summary"]


def test_normalize_does_not_overwrite_an_explicit_approach():
    out = normalize_user_results({
        "summary": "s",
        "approach": "my explicit approach",
        "metrics": [{"name": "BLEU", "value": 1.0}],
    })
    assert out["approach"] == "my explicit approach"


def test_normalize_attaches_an_ingestion_id():
    out = normalize_user_results({"summary": "s", "metrics": [{"name": "B", "value": 1}]})
    assert out["ingestion_id"]
    assert out["ingested_at"]


# ---------------------------------------------------------------------------
# Search-query derivation — R feeds the Retriever, and only the Retriever.
# ---------------------------------------------------------------------------


def test_build_search_queries_uses_the_approach():
    queries = build_search_queries({
        "summary": "Pruning heuristics for retrieval",
        "approach": "sparse attention with learned pruning masks",
        "metrics": [{"name": "BLEU-4", "value": 38.1}],
        "dataset": "WMT14",
    })
    assert any("sparse attention" in q.lower() for q in queries)


def test_build_search_queries_includes_the_dataset():
    queries = build_search_queries({
        "summary": "s",
        "approach": "sparse attention",
        "metrics": [{"name": "BLEU-4", "value": 38.1}],
        "dataset": "WMT14 en-de",
    })
    assert any("wmt14" in q.lower() for q in queries)


def test_build_search_queries_includes_metric_names():
    queries = build_search_queries({
        "summary": "s",
        "approach": "sparse attention",
        "metrics": [{"name": "BLEU-4", "value": 38.1}],
    })
    assert any("bleu" in q.lower() for q in queries)


def test_build_search_queries_are_deduplicated():
    queries = build_search_queries({
        "summary": "same",
        "approach": "same approach",
        "metrics": [],
    })
    assert len(queries) == len(set(queries))


def test_build_search_queries_for_empty_results_is_empty():
    assert build_search_queries({}) == []
