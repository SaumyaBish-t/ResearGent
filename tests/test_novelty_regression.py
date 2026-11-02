"""
Regression tests for the Novelty Check (Phase 24) against labelled cases.

These enforce the same standard the originality check was held to: the
verdict must be validated against hand-annotated cases with MEASURED
precision/recall, not an LLM's confidence score.

Run:
    PYTHONPATH= .venv/Scripts/python.exe -m pytest tests/test_novelty_regression.py -v
"""

import sys

sys.path.insert(0, ".")

from src.eval.integrity import EmbedCache
from src.novelty.scoring import (
    APPROACH_SIMILARITY_THRESHOLD,
    REVIEW_SIMILARITY_THRESHOLD,
    MatchDirection,
    decide_verdict,
    match_metrics,
    score_approach_similarity,
)
from tests.novelty_cases import CASES, NoveltyCase


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Shared embedding cache so the suite embeds each case's text ONCE.
_CACHE = EmbedCache()


def _approach_similarity(case: NoveltyCase) -> float:
    """
    Approach similarity for a case, using the PRODUCTION signal.

    The production node scores the approach with embedding cosine AND
    content-word overlap. This suite uses the same pair of signals via
    the real EMBED tier — the same convention the originality regression
    suite follows (see `tests/test_originality_regression.py`).
    """
    user_vec, paper_vec = _CACHE.embed([case.user_approach, case.paper_approach])
    return score_approach_similarity(
        case.user_approach,
        case.paper_approach,
        vectors=(user_vec, paper_vec),
    )


def _decision(case: NoveltyCase):
    matches = match_metrics(
        user_metrics=case.user_metrics,
        paper_metrics=case.paper_metrics,
        higher_is_better=not case.lower_is_better,
    )
    return decide_verdict(
        approach_similarity=_approach_similarity(case),
        result_matches=matches,
    )


# ---------------------------------------------------------------------------
# Threshold contract — the shipped defaults are named constants.
# ---------------------------------------------------------------------------


def test_thresholds_are_named_constants():
    assert APPROACH_SIMILARITY_THRESHOLD == 0.50
    assert REVIEW_SIMILARITY_THRESHOLD == 0.50


# ---------------------------------------------------------------------------
# Headline: measured precision / recall over the labelled set.
# ---------------------------------------------------------------------------


def test_headline_precision_recall():
    """
    "Is it novel?" as a binary question over the labelled set.

    POSITIVE = the verdict says the result is novel (novel or improvement).
    NEGATIVE = the verdict says not novel or inconclusive.

    The threshold is chosen to maximise F1 subject to a specificity floor
    (no false accusation of non-novelty), because wrongly calling a
    genuine reproduction "novel" is the costly error here.
    """
    tp = fp = tn = fn = 0
    for case in CASES:
        if case.ambiguous:
            continue
        decision = _decision(case)
        predicted_novel = decision.is_novel
        actual_novel = case.expected_verdict in ("novel", "improvement")
        if predicted_novel and actual_novel:
            tp += 1
        elif predicted_novel and not actual_novel:
            fp += 1
        elif not predicted_novel and actual_novel:
            fn += 1
        else:
            tn += 1

    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0

    print(
        f"\n  novelty labelled: TP={tp} FP={fp} TN={tn} FN={fn} "
        f"prec={prec:.3f} rec={rec:.3f} f1={f1:.3f} spec={spec:.3f}"
    )

    assert prec >= 0.85, f"precision {prec:.3f} below floor"
    assert rec >= 0.85, f"recall {rec:.3f} below floor"
    assert f1 >= 0.85, f"F1 {f1:.3f} below floor"
    assert spec >= 0.85, f"specificity {spec:.3f} below floor"


# ---------------------------------------------------------------------------
# Per-label behaviour
# ---------------------------------------------------------------------------


def test_every_reproduction_case_is_not_novel():
    """A result that reproduces prior work must never be called novel."""
    for case in CASES:
        if case.ambiguous or case.label != "reproduction":
            continue
        d = _decision(case)
        assert not d.is_novel, f"{case.case_id}: reproduction wrongly called novel"
        assert d.verdict.value == "not_novel", case.case_id


def test_every_improvement_case_is_novel():
    """Beating prior work on the same metric IS a novel contribution."""
    for case in CASES:
        if case.ambiguous or case.label != "improvement":
            continue
        d = _decision(case)
        assert d.is_novel, f"{case.case_id}: improvement wrongly rejected"
        assert d.verdict.value == "improvement", (
            f"{case.case_id}: got {d.verdict.value}, expected improvement"
        )


def test_every_regression_case_is_not_novel():
    """Losing to prior work on the same metric is not a novel result."""
    for case in CASES:
        if case.ambiguous or case.label != "regression":
            continue
        d = _decision(case)
        assert not d.is_novel, f"{case.case_id}: regression wrongly called novel"


def test_every_distinct_case_is_novel():
    """No comparable prior metric => nothing to duplicate => novel."""
    for case in CASES:
        if case.ambiguous or case.label != "distinct":
            continue
        d = _decision(case)
        assert d.is_novel, f"{case.case_id}: distinct case wrongly rejected"
        assert d.verdict.value == "novel", (
            f"{case.case_id}: got {d.verdict.value}, expected novel"
        )


# ---------------------------------------------------------------------------
# Review-gate behaviour
# ---------------------------------------------------------------------------


def test_reproduction_cases_trigger_review():
    """An exact reproduction MUST reach the Human Review Gate."""
    for case in CASES:
        if case.ambiguous or case.label != "reproduction":
            continue
        d = _decision(case)
        assert d.requires_review, (
            f"{case.case_id}: reproduction did not trigger the review gate"
        )


def test_same_approach_improvement_triggers_review():
    """
    Same method, better number -> STILL pauses for human confirmation.

    This follows the spec literally: "If similarity is above 50%, the
    pipeline pauses." An improvement that reuses a published method's
    approach carries high similarity, and a human genuinely should check
    it — "same method, better number" might be a real improvement, or it
    might be a mis-specified baseline or a tuned-on-test evaluation. The
    gate is where that judgement belongs; the pipeline must not decide it.
    """
    for case in CASES:
        if case.ambiguous or case.label != "improvement":
            continue
        if _approach_similarity(case) < 0.80:
            continue  # a genuinely different method is a different question
        d = _decision(case)
        assert d.requires_review, (
            f"{case.case_id}: same-approach improvement did not reach review"
        )


def test_distinct_novel_results_do_not_trigger_review():
    """
    A result with a distinct approach and no comparable metric flows
    straight through — the Human Review Gate must not stall genuinely
    novel work.
    """
    for case in CASES:
        if case.ambiguous or case.label != "distinct":
            continue
        d = _decision(case)
        assert not d.requires_review, (
            f"{case.case_id}: distinct result wrongly paused for review"
        )


def test_review_gate_threshold_is_respected():
    """requires_review must agree with the configured threshold."""
    for case in CASES:
        if case.ambiguous:
            continue
        d = _decision(case)
        expected = d.overall_similarity >= REVIEW_SIMILARITY_THRESHOLD
        assert d.requires_review is expected, case.case_id


# ---------------------------------------------------------------------------
# Guard rails
# ---------------------------------------------------------------------------


def test_distinct_metrics_never_match():
    """'Win Rate' must not match 'EM' — different metric families."""
    for case in CASES:
        if case.ambiguous or case.label != "distinct":
            continue
        matches = match_metrics(
            user_metrics=case.user_metrics,
            paper_metrics=case.paper_metrics,
            higher_is_better=not case.lower_is_better,
        )
        assert not any(m.matched for m in matches), (
            f"{case.case_id}: unrelated metrics were matched"
        )


def test_lower_is_better_metrics_are_directed_correctly():
    """FID 12.4 beats FID 18.9 — lower is better."""
    for case in CASES:
        if case.ambiguous or case.label != "improvement":
            continue
        if not case.lower_is_better:
            continue
        matches = match_metrics(
            user_metrics=case.user_metrics,
            paper_metrics=case.paper_metrics,
            higher_is_better=False,
        )
        matched = [m for m in matches if m.matched]
        assert matched, case.case_id
        assert matched[0].direction == MatchDirection.BETTER, case.case_id


def test_ambiguous_cases_are_excluded_from_headline_metrics():
    """The labelled set must actually mark its ambiguous cases."""
    assert any(c.ambiguous for c in CASES)
    assert any(not c.ambiguous for c in CASES)
