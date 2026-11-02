"""Novelty Check — scoring primitives for user-supplied results (Phase 24)."""

from src.novelty.scoring import (
    APPROACH_CONTENT_OVERLAP,
    APPROACH_SIMILARITY_THRESHOLD,
    METRIC_EQUAL_RELATIVE_TOLERANCE,
    REVIEW_SIMILARITY_THRESHOLD,
    MatchDirection,
    MetricMatch,
    NoveltyDecision,
    NoveltyVerdict,
    compare_metric,
    content_overlap,
    cosine,
    decide_verdict,
    match_metrics,
    normalize_metric_name,
    parse_numeric,
    score_approach_similarity,
)

__all__ = [
    "APPROACH_CONTENT_OVERLAP",
    "APPROACH_SIMILARITY_THRESHOLD",
    "METRIC_EQUAL_RELATIVE_TOLERANCE",
    "REVIEW_SIMILARITY_THRESHOLD",
    "MatchDirection",
    "MetricMatch",
    "NoveltyDecision",
    "NoveltyVerdict",
    "compare_metric",
    "content_overlap",
    "cosine",
    "decide_verdict",
    "match_metrics",
    "normalize_metric_name",
    "parse_numeric",
    "score_approach_similarity",
]
