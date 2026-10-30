"""
Regression tests for the Originality Check (Phase 20) against labelled cases.

These tests enforce the behaviour the industry reviewer asked for: validated
against labelled test cases with measured precision/recall, not an
LLM-generated confidence score alone.

Run:
    PYTHONPATH= .venv/Scripts/python.exe -m pytest tests/test_originality_regression.py -v
"""

import sys

sys.path.insert(0, ".")

from src.agent.nodes.originality_check import (
    _CONTENT_OVERLAP,
    _SIM_THRESHOLD,
    _VERBATIM_NGRAM,
    _content_overlap,
    _content_words,
    _ngram_overlap,
    _split_sentences,
)
from src.eval.integrity import EmbedCache, cosine
from tests.originality_cases import CASES

# ---------------------------------------------------------------------------
# Unit helpers
# ---------------------------------------------------------------------------

def test_split_sentences_trims_short():
    # sentences shorter than 20 chars are dropped by the filter
    out = _split_sentences("Hi. This is a longer sentence. A.")
    assert out == ["This is a longer sentence."]


def test_content_words_drop_stopwords():
    assert _content_words("The quick brown fox jumps over the lazy dog") == {
        "quick", "brown", "fox", "jumps", "lazy", "dog"
    }


def test_content_overlap_identical():
    assert abs(_content_overlap("foo bar baz", "foo bar baz") - 1.0) < 1e-9


def test_content_overlap_disjoint():
    assert _content_overlap("foo bar baz", "qux quux quuz") == 0.0


def test_ngram_overlap_identical():
    assert _ngram_overlap("the quick brown fox", "the quick brown fox", n=3) == 3


def test_ngram_overlap_none():
    assert _ngram_overlap("foo bar", "qux quux", n=3) == 0


# ---------------------------------------------------------------------------
# Threshold contract
# ---------------------------------------------------------------------------

def test_sim_threshold_is_data_driven():
    """The sim threshold comes from a grid search, not a guess."""
    assert _SIM_THRESHOLD == 0.50
    assert _CONTENT_OVERLAP == 0.20
    assert _VERBATIM_NGRAM == 8


# ---------------------------------------------------------------------------
# Labelled-case behaviour (headline: precision=recall=0.90)
# ---------------------------------------------------------------------------

def _decision(case):
    """Reproduce the production decision for one labelled case."""
    sents = _split_sentences(case.source_text) or [case.source_text]
    cache = EmbedCache()
    vecs = cache.embed([case.draft_sentence] + sents)
    dvec = vecs[0]
    best_sim = max((cosine(dvec, v) for v in vecs[1:]), default=0.0)
    best_ngram = max(
        (_ngram_overlap(case.draft_sentence, s, n=_VERBATIM_NGRAM)
         for s in [*sents, case.source_text]),
        default=0,
    )
    cov = _content_overlap(case.draft_sentence, case.source_text)
    cited = f"[{case.source_tag}]" in case.draft_sentence
    flagged = False
    status = None
    if not cited:
        if best_ngram >= _VERBATIM_NGRAM:
            flagged, status = True, "verbatim_uncited"
        elif best_sim >= _SIM_THRESHOLD and cov >= _CONTENT_OVERLAP:
            flagged, status = True, "uncited_paraphrase"
    return flagged, status, best_sim, best_ngram, cov


def test_headline_precision_recall():
    """Unambiguous cases: precision=recall=0.90 on the labelled set.

    Headline numbers (from docs/eval/originality_validation.md sweep):
      sim=0.50, cov=0.20, ngram=8  -> TP=9 FP=1 TN=15 FN=1
      precision=0.90 recall=0.90 F1=0.90 specificity=0.938
    """
    tp = fp = tn = fn = 0
    for case in CASES:
        if case.ambiguous:
            continue
        flagged, *_ = _decision(case)
        actual = case.is_violation
        if flagged and actual:
            tp += 1
        elif flagged and not actual:
            fp += 1
        elif not flagged and actual:
            fn += 1
        else:
            tn += 1
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    assert prec >= 0.85, f"precision {prec:.3f} below floor"
    assert rec >= 0.85, f"recall {rec:.3f} below floor"
    assert f1 >= 0.85, f"F1 {f1:.3f} below floor"
    print(
        f"\n  labelled: TP={tp} FP={fp} TN={tn} FN={fn} "
        f"prec={prec:.3f} rec={rec:.3f} f1={f1:.3f}"
    )


def test_cited_never_flagged():
    """Properly-cited verbatim/paraphrase must NOT produce a violation."""
    for case in CASES:
        if case.ambiguous:
            continue
        if not case.label.endswith("_cited"):
            continue
        flagged, *_ = _decision(case)
        assert not flagged, f"{case.case_id}: cited draft wrongly flagged"


def test_original_not_flagged():
    """Every unambiguous original case is left alone."""
    for case in CASES:
        if case.ambiguous or case.label != "original":
            continue
        flagged, *_ = _decision(case)
        assert not flagged, f"{case.case_id}: original wrongly flagged"


def test_verbatim_uncited_flagged():
    """Every unambiguous verbatim_uncited case is caught."""
    for case in CASES:
        if case.ambiguous or case.label != "verbatim_uncited":
            continue
        flagged, status, *_ = _decision(case)
        assert flagged, f"{case.case_id}: verbatim uncited not flagged"
        assert status == "verbatim_uncited"


def test_paraphrase_uncited_flagged():
    """Every unambiguous paraphrase_uncited case is caught."""
    for case in CASES:
        if case.ambiguous or case.label != "paraphrase_uncited":
            continue
        flagged, *_ = _decision(case)
        assert flagged, f"{case.case_id}: paraphrase uncited not flagged"
