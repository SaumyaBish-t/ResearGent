"""
Results Ingestion store — Phase 24.

The user's OWN experimental results are GROUND TRUTH. This module is the
only place they enter the pipeline, and it enforces that contract:

  * validated  — a metric must have a name and a numeric value; duplicate
                 metric names are rejected as ambiguous.
  * frozen     — values are stored VERBATIM. Normalisation adds a parsed
                 `numeric_value` alongside the raw value but never rounds,
                 re-formats, or rewrites what the user reported.
  * queryable  — the ingested result carries the search queries the
                 Retriever should use to find the related prior work.

The stored dict is what the `user_results` state field holds. Nodes may
READ it; no node may WRITE to it.

Run:
    PYTHONPATH= .venv/Scripts/python.exe -m pytest tests/test_novelty_ingest.py -v
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from src.novelty.scoring import normalize_metric_name, parse_numeric


class IngestionError(ValueError):
    """Raised when a user-supplied result payload is not usable."""


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_user_results(results: Any) -> dict[str, Any]:
    """
    Validate a raw user-results payload and return it unchanged.

    Raises `IngestionError` with an actionable message when the payload
    cannot be used. The returned dict is the ORIGINAL object — validation
    never mutates the caller's data.
    """
    if not isinstance(results, dict):
        raise IngestionError(
            "Results must be an object with 'summary' and 'metrics' keys."
        )

    summary = (results.get("summary") or results.get("approach") or "").strip()
    if not summary:
        raise IngestionError(
            "Results need a 'summary' (or 'approach') describing what you did."
        )

    metrics = results.get("metrics")
    if not isinstance(metrics, list) or not metrics:
        raise IngestionError(
            "Results need a non-empty 'metrics' list, e.g. "
            '[{"name": "BLEU-4", "value": 38.1}].'
        )

    seen: dict[str, str] = {}
    for i, m in enumerate(metrics):
        if not isinstance(m, dict):
            raise IngestionError(f"Metric #{i + 1} must be an object.")
        name = (m.get("name") or "").strip()
        if not name:
            raise IngestionError(f"Metric #{i + 1} is missing a 'name'.")
        value = m.get("value")
        if parse_numeric(value) is None:
            raise IngestionError(
                f"Metric '{name}' must have a numeric value (got {value!r})."
            )
        key = normalize_metric_name(name)
        if key in seen:
            raise IngestionError(
                f"Duplicate metric '{name}' (same as '{seen[key]}'). "
                "Report each metric once so comparisons are unambiguous."
            )
        seen[key] = name

    return results


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


def normalize_user_results(results: dict[str, Any]) -> dict[str, Any]:
    """
    Produce the canonical stored form of a user-results payload.

    The stored form adds, without ever altering the user's own numbers:

      * `ingestion_id` — a stable id for this ingestion.
      * `ingested_at`  — when it was ingested.
      * `approach`     — the described method (from `approach`, else
                         derived from `methodology_notes` / `summary`).
      * `numeric_value` on each metric — the PARSED number, kept alongside
                         the original `value` so the original stays intact.

    The `value` field is copied VERBATIM. A user who wrote "38.10" keeps
    "38.10"; the pipeline compares against 38.1 but reports 38.10.
    """
    validated = validate_user_results(results)

    out: dict[str, Any] = dict(validated)

    # ---- Identity ----
    out["ingestion_id"] = validated.get("ingestion_id") or uuid.uuid4().hex[:12]
    out["ingested_at"] = validated.get("ingested_at") or datetime.now(
        timezone.utc
    ).isoformat(timespec="seconds")

    # ---- Approach (the method, for the novelty comparison) ----
    approach = (validated.get("approach") or "").strip()
    if not approach:
        approach = (validated.get("methodology_notes") or "").strip()
    if not approach:
        approach = (validated.get("summary") or "").strip()
    out["approach"] = approach

    # ---- Summary (fall back to the approach) ----
    summary = (validated.get("summary") or "").strip()
    if not summary:
        summary = approach
    out["summary"] = summary

    # ---- Metrics: keep `value` verbatim, add `numeric_value` ----
    metrics: list[dict[str, Any]] = []
    for m in validated.get("metrics") or []:
        entry = dict(m)
        entry["name"] = (m.get("name") or "").strip()
        entry["value"] = m.get("value")          # VERBATIM
        entry["numeric_value"] = parse_numeric(m.get("value"))
        metrics.append(entry)
    out["metrics"] = metrics

    # ---- Search queries for the Retriever ----
    out["search_queries"] = build_search_queries(out)

    return out


# ---------------------------------------------------------------------------
# Search-query derivation — R feeds the Retriever, and only the Retriever.
# ---------------------------------------------------------------------------


def build_search_queries(results: dict[str, Any]) -> list[str]:
    """
    Derive the retrieval queries used to find the related prior work.

    Built from the user's OWN description — the approach, the dataset, and
    the metric names — so the Retriever looks for papers doing the same
    THING, which is what the Novelty Check must compare against.

    Queries are deduplicated and capped; no LLM is consulted (the user's
    words are the ground truth, and a paraphrase would blur what was
    actually claimed).
    """
    if not results:
        return []

    queries: list[str] = []

    def _add(q: str) -> None:
        q = (q or "").strip()
        if q and q.lower() not in {x.lower() for x in queries}:
            queries.append(q)

    approach = (results.get("approach") or "").strip()
    summary = (results.get("summary") or "").strip()
    dataset = (results.get("dataset") or "").strip()
    methodology = (results.get("methodology_notes") or "").strip()

    if approach:
        _add(approach)
    if methodology and methodology.lower() != approach.lower():
        _add(methodology)
    if summary and summary.lower() != approach.lower():
        _add(summary)

    # Dataset-qualified queries — "sparse attention WMT14" finds papers
    # that did the SAME method on the SAME benchmark, which is the right
    # comparison for a novelty claim.
    if dataset and approach:
        _add(f"{approach} {dataset}")

    # Metric-qualified queries — a paper that reports the same metric is
    # the only one whose number we can meaningfully compare against.
    metric_names = [
        (m.get("name") or "").strip()
        for m in (results.get("metrics") or [])
        if (m.get("name") or "").strip()
    ]
    if metric_names:
        _add(f"{approach} {metric_names[0]}".strip())
        if dataset:
            _add(f"{metric_names[0]} {dataset}")

    return queries[:5]
