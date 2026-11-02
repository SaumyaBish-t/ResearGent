"""
Tests for the Novelty Check scoring engine (Phase 24).

The Novelty Check answers two questions about a user's own experimental
results, compared against the retrieved papers:

  1. APPROACH similarity — does the user's approach match what a paper did?
  2. RESULT similarity  — does the user's number beat/match/lose to the
                          paper's reported number on the same metric?

And it must reach a VERDICT: is the result novel, an improvement, or
already-known (not novel)?

These tests are the RED phase of TDD — they define the contract before the
implementation exists.

Run:
    PYTHONPATH= .venv/Scripts/python.exe -m pytest tests/test_novelty_scoring.py -v
"""

import sys

sys.path.insert(0, ".")

from src.novelty.scoring import (
    MatchDirection,
    MetricMatch,
    NoveltyVerdict,
    compare_metric,
    decide_verdict,
    normalize_metric_name,
    parse_numeric,
    score_approach_similarity,
)


# ---------------------------------------------------------------------------
# Metric-name normalisation — the basis for matching user metrics to paper
# metrics. "BLEU-4", "bleu 4", "BLEU_4" and "BLEU score (4-gram)" are the
# same metric; "BLEU" alone is not (different tokenisation).
# ---------------------------------------------------------------------------

def test_normalize_metric_name_strips_punctuation_and_case():
    assert normalize_metric_name("BLEU-4") == "bleu 4"
    assert normalize_metric_name("bleu_4") == "bleu 4"
    assert normalize_metric_name("  BLEU 4  ") == "bleu 4"


def test_normalize_metric_name_drops_generic_suffixes():
    """'F1 score' and 'F1' are the same metric."""
    assert normalize_metric_name("F1 score") == "f1"
    assert normalize_metric_name("Accuracy Score") == "accuracy"


def test_normalize_metric_name_keeps_distinct_variants_distinct():
    """BLEU-4 and BLEU are DIFFERENT metrics — must not collapse."""
    assert normalize_metric_name("BLEU-4") != normalize_metric_name("BLEU")


def test_normalize_metric_name_expands_synonyms():
    """'exact match' and 'EM' refer to the same metric family."""
    assert normalize_metric_name("exact match") == normalize_metric_name("EM")


# ---------------------------------------------------------------------------
# Numeric parsing — user results arrive as strings ("94.2", "0.912").
# ---------------------------------------------------------------------------

def test_parse_numeric_reads_plain_float():
    assert parse_numeric("94.2") == 94.2
    assert parse_numeric(94.2) == 94.2


def test_parse_numeric_strips_percent_sign():
    assert parse_numeric("94.2%") == 94.2


def test_parse_numeric_reads_paper_style_bracket():
    """Papers write BLEU as 'BLEU = 34.8' or 'BLEU: 34.8'."""
    assert parse_numeric("34.8") == 34.8


def test_parse_numeric_returns_none_for_non_numeric():
    assert parse_numeric("not a number") is None
    assert parse_numeric("") is None
    assert parse_numeric(None) is None


def test_parse_numeric_handles_scientific_and_negative():
    assert parse_numeric("-2.5") == -2.5
    assert parse_numeric("1e-3") == 0.001


# ---------------------------------------------------------------------------
# Metric matching — comparing a user metric against a paper metric.
# ---------------------------------------------------------------------------

def test_compare_metric_identical_metric_and_value_is_not_better():
    m = compare_metric(
        user_metric={"name": "BLEU-4", "value": 34.8},
        paper_metric={"name": "BLEU-4", "value": 34.8},
    )
    assert m.matched is True
    assert m.direction == MatchDirection.EQUAL
    assert m.delta == 0.0


def test_compare_metric_user_better_is_improvement():
    m = compare_metric(
        user_metric={"name": "BLEU-4", "value": 38.1},
        paper_metric={"name": "BLEU-4", "value": 34.8},
    )
    assert m.matched is True
    assert m.direction == MatchDirection.BETTER
    assert m.delta > 0


def test_compare_metric_user_worse_is_worse():
    m = compare_metric(
        user_metric={"name": "BLEU-4", "value": 31.0},
        paper_metric={"name": "BLEU-4", "value": 34.8},
    )
    assert m.matched is True
    assert m.direction == MatchDirection.WORSE
    assert m.delta < 0


def test_compare_metric_different_metric_is_unmatched():
    m = compare_metric(
        user_metric={"name": "BLEU-4", "value": 38.1},
        paper_metric={"name": "ROUGE-L", "value": 41.0},
    )
    assert m.matched is False
    assert m.direction == MatchDirection.UNKNOWN


def test_compare_metric_ignores_metric_without_numeric_value():
    m = compare_metric(
        user_metric={"name": "BLEU-4", "value": "n/a"},
        paper_metric={"name": "BLEU-4", "value": 34.8},
    )
    assert m.matched is False


def test_compare_metric_supports_lower_is_better_direction():
    """Latency/perplexity/loss: LOWER is better."""
    m = compare_metric(
        user_metric={"name": "perplexity", "value": 10.0},
        paper_metric={"name": "perplexity", "value": 12.5},
        higher_is_better=False,
    )
    assert m.direction == MatchDirection.BETTER
    assert m.delta < 0  # raw delta stays signed by the raw difference


# ---------------------------------------------------------------------------
# Approach similarity — does the user's method match the paper's method?
# ---------------------------------------------------------------------------

def test_score_approach_similarity_identical_approach_is_high():
    s = score_approach_similarity(
        user_approach="We fine-tune a transformer encoder with contrastive learning.",
        paper_approach="We fine-tune a transformer encoder with contrastive learning.",
    )
    assert s >= 0.95


def test_score_approach_similarity_unrelated_approach_is_low():
    s = score_approach_similarity(
        user_approach="We use a graph neural network over a molecular graph.",
        paper_approach="We evaluate hardware prefetchers on a CPU trace.",
    )
    assert s < 0.5


def test_score_approach_similarity_is_symmetric():
    a = "We propose a diffusion model for protein folding."
    b = "Protein folding with diffusion models."
    assert abs(score_approach_similarity(a, b) - score_approach_similarity(b, a)) < 1e-6


def test_score_approach_similarity_empty_input_is_zero():
    assert score_approach_similarity("", "some approach") == 0.0
    assert score_approach_similarity("some approach", "") == 0.0


# ---------------------------------------------------------------------------
# Verdict — combining approach + result similarity into a decision.
# ---------------------------------------------------------------------------

def test_decide_verdict_novel_when_low_similarity_and_better():
    v = decide_verdict(
        approach_similarity=0.20,
        result_matches=[
            MetricMatch(
                user_metric={"name": "BLEU-4", "value": 38.1},
                paper_metric={"name": "BLEU-4", "value": 34.8},
                matched=True,
                direction=MatchDirection.BETTER,
                delta=3.3,
            )
        ],
    )
    assert v.verdict == NoveltyVerdict.NOVEL
    assert v.is_novel is True


def test_decide_verdict_improvement_when_approach_matches_but_result_better():
    v = decide_verdict(
        approach_similarity=0.82,
        result_matches=[
            MetricMatch(
                user_metric={"name": "BLEU-4", "value": 38.1},
                paper_metric={"name": "BLEU-4", "value": 34.8},
                matched=True,
                direction=MatchDirection.BETTER,
                delta=3.3,
            )
        ],
    )
    assert v.verdict == NoveltyVerdict.IMPROVEMENT
    assert v.is_novel is True


def test_decide_verdict_not_novel_when_approach_matches_and_result_not_better():
    v = decide_verdict(
        approach_similarity=0.82,
        result_matches=[
            MetricMatch(
                user_metric={"name": "BLEU-4", "value": 34.8},
                paper_metric={"name": "BLEU-4", "value": 34.8},
                matched=True,
                direction=MatchDirection.EQUAL,
                delta=0.0,
            )
        ],
    )
    assert v.verdict == NoveltyVerdict.NOT_NOVEL
    assert v.is_novel is False


def test_decide_verdict_not_novel_when_result_is_worse():
    v = decide_verdict(
        approach_similarity=0.80,
        result_matches=[
            MetricMatch(
                user_metric={"name": "BLEU-4", "value": 30.0},
                paper_metric={"name": "BLEU-4", "value": 34.8},
                matched=True,
                direction=MatchDirection.WORSE,
                delta=-4.8,
            )
        ],
    )
    assert v.verdict == NoveltyVerdict.NOT_NOVEL
    assert v.is_novel is False


def test_decide_verdict_inconclusive_when_no_metric_matched():
    """No comparable metric at all — we cannot claim novelty OR duplication."""
    v = decide_verdict(approach_similarity=0.90, result_matches=[])
    assert v.verdict == NoveltyVerdict.INCONCLUSIVE
    assert v.is_novel is False


def test_decide_verdict_novel_when_approach_distinct_and_nothing_comparable():
    """Distinct approach + no comparable metric in the corpus = nothing found.

    This is the 'we looked and found no prior work doing this' case — it is
    genuinely novel, unlike the ambiguous case above where the approach
    matched but the numbers could not be compared.
    """
    v = decide_verdict(approach_similarity=0.12, result_matches=[])
    assert v.verdict == NoveltyVerdict.NOVEL
    assert v.is_novel is True


def test_decide_verdict_result_similarity_is_high_for_equal_numbers():
    """Equal numbers => the result reproduces prior work => high similarity.

    Note the field being asserted: `result_similarity` is the numeric
    reproduction score (1.0 here), while `overall_similarity` BLENDS that
    with the approach score. A distinct approach (0.30) with identical
    numbers gives overall=0.65, which is still above the 0.50 review
    threshold — so the Human Review Gate fires, which is exactly right for
    "different method, same numbers".
    """
    v = decide_verdict(
        approach_similarity=0.30,
        result_matches=[
            MetricMatch(
                user_metric={"name": "BLEU-4", "value": 34.8},
                paper_metric={"name": "BLEU-4", "value": 34.8},
                matched=True,
                direction=MatchDirection.EQUAL,
                delta=0.0,
            )
        ],
    )
    assert v.result_similarity >= 0.9
    # Different method + identical numbers must still reach the review gate.
    assert v.overall_similarity > 0.50
    assert v.requires_review is True


def test_decide_verdict_result_similarity_lower_for_distant_numbers():
    """A large improvement means the number is NOT a reproduction."""
    v = decide_verdict(
        approach_similarity=0.30,
        result_matches=[
            MetricMatch(
                user_metric={"name": "BLEU-4", "value": 60.0},
                paper_metric={"name": "BLEU-4", "value": 34.8},
                matched=True,
                direction=MatchDirection.BETTER,
                delta=25.2,
            )
        ],
    )
    assert v.result_similarity < 0.4
    assert v.overall_similarity < 0.6


def test_decide_verdict_high_similarity_triggers_review():
    """Above the review threshold the verdict must ask for human review."""
    v = decide_verdict(
        approach_similarity=0.78,
        result_matches=[
            MetricMatch(
                user_metric={"name": "BLEU-4", "value": 34.8},
                paper_metric={"name": "BLEU-4", "value": 34.8},
                matched=True,
                direction=MatchDirection.EQUAL,
                delta=0.0,
            )
        ],
    )
    assert v.requires_review is True
