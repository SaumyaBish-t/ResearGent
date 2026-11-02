"""
Tests for extracting reported metrics from paper text (Phase 24).

The Novelty Check compares the user's numbers against the numbers the
retrieved papers REPORT. To do that we must pull structured metrics out
of free-form PDF text:

    "Our method reaches 34.8 BLEU-4 on the WMT14 test set."
        -> {"name": "BLEU-4", "value": 34.8}

    "Table 2: Accuracy 94.2%, F1 0.912"
        -> [{"name": "Accuracy", "value": 94.2},
            {"name": "F1", "value": 0.912}]

These are the RED-phase tests for the extractor.

Run:
    PYTHONPATH= .venv/Scripts/python.exe -m pytest tests/test_novelty_extract.py -v
"""

import sys

sys.path.insert(0, ".")

from src.novelty.extraction import (
    extract_metrics,
    extract_metrics_from_text,
    extract_named_value,
    known_metric_names,
)


# ---------------------------------------------------------------------------
# Named-value extraction — "METRIC = value" / "METRIC: value".
# ---------------------------------------------------------------------------

def test_extract_named_value_with_equals():
    m = extract_named_value("We obtain BLEU-4 = 34.8 on the test set.")
    assert m is not None
    assert m["name"] == "BLEU-4"
    assert float(m["value"]) == 34.8


def test_extract_named_value_with_colon():
    m = extract_named_value("Results — Accuracy: 94.2%")
    assert m is not None
    assert m["name"] == "Accuracy"
    assert float(m["value"]) == 94.2


def test_extract_named_value_returns_none_without_name():
    assert extract_named_value("We obtain 34.8 on the test set.") is None


def test_extract_named_value_returns_none_without_number():
    assert extract_named_value("We obtain BLEU-4 = N/A") is None


# ---------------------------------------------------------------------------
# Metric extraction from a sentence — "value METRIC" (the paper style).
# ---------------------------------------------------------------------------

def test_extract_metrics_value_before_metric():
    out = extract_metrics_from_text("The model achieves 34.8 BLEU-4 overall.")
    names = [m["name"] for m in out]
    assert "BLEU-4" in names
    hit = next(m for m in out if m["name"] == "BLEU-4")
    assert float(hit["value"]) == 34.8


def test_extract_metrics_finds_multiple_metrics_in_a_line():
    out = extract_metrics_from_text("Accuracy 94.2% and F1 0.912 on the test set.")
    by_name = {m["name"]: float(m["value"]) for m in out}
    assert "Accuracy" in by_name
    assert "F1" in by_name


def test_extract_metrics_returns_empty_when_nothing_present():
    assert extract_metrics_from_text("We train a transformer on Wikipedia.") == []


def test_extract_metrics_ignores_years_as_metric_values():
    """'In 2024 we published' must not be read as a metric."""
    out = extract_metrics_from_text("In 2024 we published a new benchmark.")
    assert out == []


def test_extract_metrics_does_not_extract_bare_unknown_numbers():
    """A number with no metric name nearby is not a metric."""
    out = extract_metrics_from_text("We trained for 100 epochs on 8 GPUs.")
    assert out == []


# ---------------------------------------------------------------------------
# Whole-document extraction.
# ---------------------------------------------------------------------------

def test_extract_metrics_dedupes_by_name_keeping_first():
    text = (
        "Our method reaches 34.8 BLEU-4.\n"
        "Ablations report 33.1 BLEU-4.\n"
    )
    out = extract_metrics(text)
    bleu = [m for m in out if m["name"] == "BLEU-4"]
    assert len(bleu) == 1
    assert float(bleu[0]["value"]) == 34.8


def test_extract_metrics_caps_per_metric_count():
    """A long paper mentioning BLEU 50 times must not flood the report."""
    text = "\n".join(f"step {i}: 30.{i % 9} BLEU-4" for i in range(60))
    out = extract_metrics(text)
    bleu = [m for m in out if m["name"] == "BLEU-4"]
    assert len(bleu) <= 3


def test_extract_metrics_records_source_line():
    text = "The baseline reaches 12.1 BLEU-4.\n"
    out = extract_metrics(text)
    assert out
    assert "source" in out[0]


# ---------------------------------------------------------------------------
# Known metric vocabulary.
# ---------------------------------------------------------------------------

def test_known_metric_names_includes_common_benchmarks():
    names = {n.lower() for n in known_metric_names()}
    assert "bleu" in names or "bleu-4" in names
    assert "rouge" in names or "rouge-l" in names
    assert "accuracy" in names
    assert "f1" in names
    assert "perplexity" in names


def test_known_metric_names_are_normalised():
    """Every advertised name must survive normalisation unchanged-ish."""
    for n in known_metric_names():
        assert n.strip() != ""
