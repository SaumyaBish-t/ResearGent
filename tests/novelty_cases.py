"""
Labelled cases for the Novelty Check (Phase 24).

Each case pairs a USER result (their own experiment) with a PAPER's
reported result on the same metric, and states what the Novelty Check
SHOULD conclude. The cases are hand-annotated so the thresholds can be
justified by measured precision/recall rather than an LLM's opinion.

Labels
------
  "reproduction"   — the user's number matches the paper's. Not novel.
  "improvement"    — the user's number beats the paper's on the same
                     metric. A legitimate contribution.
  "regression"     — the user's number loses to the paper's. Not novel.
  "distinct"       — no comparable metric exists in the paper (the paper
                     measured something else entirely). Nothing to
                     duplicate.
  "ambiguous"      — genuinely unclear; excluded from headline metrics.

The `approach` fields carry the paper's described method and the user's
described method, so the approach-similarity half of the decision is
exercised too.

Run:
    PYTHONPATH= .venv/Scripts/python.exe -m pytest tests/test_novelty_regression.py -v
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class NoveltyCase:
    """One labelled user-result vs paper-result comparison."""

    case_id: str
    label: str                    # see Labels above
    ambiguous: bool = False
    # The user's own result.
    user_approach: str = ""
    user_metrics: list[dict] = field(default_factory=list)
    # The paper's reported result.
    paper_approach: str = ""
    paper_metrics: list[dict] = field(default_factory=list)
    # Metrics where a LOWER number is the better result.
    lower_is_better: bool = False
    # The expected verdict.
    expected_verdict: str = ""
    # The expected review-gate behaviour.
    expect_review: bool | None = None


CASES: list[NoveltyCase] = [
    # -----------------------------------------------------------------------
    # REPRODUCTION — the user's number is the paper's number.
    # -----------------------------------------------------------------------
    NoveltyCase(
        case_id="rep_01",
        label="reproduction",
        user_approach=(
            "We fine-tune a BERT encoder with contrastive learning for "
            "sentence retrieval and evaluate on the STS-B test set."
        ),
        user_metrics=[{"name": "BLEU-4", "value": 34.8}],
        paper_approach=(
            "We fine-tune a BERT encoder with contrastive learning for "
            "sentence retrieval and evaluate on the STS-B test set."
        ),
        paper_metrics=[{"name": "BLEU-4", "value": 34.8}],
        expected_verdict="not_novel",
        expect_review=True,
    ),
    NoveltyCase(
        case_id="rep_02",
        label="reproduction",
        user_approach=(
            "A graph convolutional network classifies molecules by their "
            "substructure graph, trained with a cross-entropy loss."
        ),
        user_metrics=[{"name": "Accuracy", "value": 0.912}],
        paper_approach=(
            "We train a graph convolutional network on molecular graphs "
            "using cross-entropy loss and report classification accuracy."
        ),
        paper_metrics=[{"name": "Accuracy", "value": 0.913}],
        expected_verdict="not_novel",
        expect_review=True,
    ),
    NoveltyCase(
        case_id="rep_03",
        label="reproduction",
        user_approach=(
            "A recurrent encoder-decoder summarises documents, evaluated "
            "with ROUGE-L against reference summaries."
        ),
        user_metrics=[{"name": "ROUGE-L", "value": 41.2}],
        paper_approach=(
            "We evaluate a recurrent encoder-decoder summarisation model "
            "using ROUGE-L scores on the benchmark."
        ),
        paper_metrics=[{"name": "ROUGE-L", "value": 41.2}],
        expected_verdict="not_novel",
        expect_review=True,
    ),

    # -----------------------------------------------------------------------
    # IMPROVEMENT — same metric, strictly better number.
    # -----------------------------------------------------------------------
    NoveltyCase(
        case_id="imp_01",
        label="improvement",
        user_approach=(
            "We fine-tune a BERT encoder with contrastive learning for "
            "sentence retrieval and evaluate on the STS-B test set."
        ),
        user_metrics=[{"name": "BLEU-4", "value": 38.1}],
        paper_approach=(
            "We fine-tune a BERT encoder with contrastive learning for "
            "sentence retrieval and evaluate on the STS-B test set."
        ),
        paper_metrics=[{"name": "BLEU-4", "value": 34.8}],
        expected_verdict="improvement",
        expect_review=False,
    ),
    NoveltyCase(
        case_id="imp_02",
        label="improvement",
        user_approach=(
            "A diffusion model generates 3D molecular conformers, scored "
            "with FID against reference structures."
        ),
        user_metrics=[{"name": "FID", "value": 12.4}],
        paper_approach=(
            "We generate 3D molecular conformers with a diffusion model and "
            "report the FID of the generated structures."
        ),
        paper_metrics=[{"name": "FID", "value": 18.9}],
        lower_is_better=True,
        expected_verdict="improvement",
        expect_review=False,
    ),
    NoveltyCase(
        case_id="imp_03",
        label="improvement",
        user_approach=(
            "A transformer encoder is pre-trained with a masked language "
            "modelling objective, then fine-tuned for classification."
        ),
        user_metrics=[{"name": "Accuracy", "value": 94.2}],
        paper_approach=(
            "We pre-train a transformer encoder with masked language "
            "modelling and fine-tune it for downstream classification."
        ),
        paper_metrics=[{"name": "Accuracy", "value": 91.6}],
        expected_verdict="improvement",
        expect_review=False,
    ),

    # -----------------------------------------------------------------------
    # REGRESSION — the user's number loses to the paper's.
    # -----------------------------------------------------------------------
    NoveltyCase(
        case_id="reg_01",
        label="regression",
        user_approach=(
            "A graph convolutional network classifies molecules by their "
            "substructure graph, trained with a cross-entropy loss."
        ),
        user_metrics=[{"name": "Accuracy", "value": 0.874}],
        paper_approach=(
            "We train a graph convolutional network on molecular graphs "
            "using cross-entropy loss and report classification accuracy."
        ),
        paper_metrics=[{"name": "Accuracy", "value": 0.913}],
        expected_verdict="not_novel",
        expect_review=False,
    ),
    NoveltyCase(
        case_id="reg_02",
        label="regression",
        user_approach=(
            "We evaluate a recurrent encoder-decoder summarisation model "
            "using ROUGE-L scores on the benchmark."
        ),
        user_metrics=[{"name": "ROUGE-L", "value": 38.0}],
        paper_approach=(
            "A recurrent encoder-decoder summarises documents, evaluated "
            "with ROUGE-L against reference summaries."
        ),
        paper_metrics=[{"name": "ROUGE-L", "value": 41.2}],
        expected_verdict="not_novel",
        expect_review=False,
    ),

    # -----------------------------------------------------------------------
    # DISTINCT — the paper measured something else entirely, so there is
    # nothing to duplicate. The user's approach may or may not resemble it.
    # -----------------------------------------------------------------------
    NoveltyCase(
        case_id="dis_01",
        label="distinct",
        user_approach=(
            "We tune a preference model with direct preference optimisation "
            "on human feedback and measure win rate against a baseline."
        ),
        user_metrics=[{"name": "Win Rate", "value": 62.5}],
        paper_approach=(
            "We scale a retrieval-augmented generation pipeline and report "
            "its exact-match score on a QA benchmark."
        ),
        paper_metrics=[{"name": "EM", "value": 41.8}],
        expected_verdict="novel",
        expect_review=False,
    ),
    NoveltyCase(
        case_id="dis_02",
        label="distinct",
        user_approach=(
            "A recurrent encoder-decoder summarises documents, evaluated "
            "with ROUGE-L against reference summaries."
        ),
        user_metrics=[{"name": "ROUGE-L", "value": 44.0}],
        paper_approach=(
            "We pre-train a transformer encoder with masked language "
            "modelling and fine-tune it for downstream classification."
        ),
        paper_metrics=[{"name": "Accuracy", "value": 91.6}],
        expected_verdict="novel",
        expect_review=False,
    ),

    # -----------------------------------------------------------------------
    # AMBIGUOUS — excluded from headline metrics, kept as a sanity case.
    # -----------------------------------------------------------------------
    NoveltyCase(
        case_id="amb_01",
        label="ambiguous",
        ambiguous=True,
        user_approach=(
            "We propose a neural architecture for sequence labelling and "
            "evaluate it on a shared benchmark."
        ),
        user_metrics=[{"name": "F1", "value": 0.88}],
        paper_approach=(
            "We also propose a neural architecture for sequence labelling "
            "and evaluate it on a shared benchmark."
        ),
        paper_metrics=[{"name": "F1", "value": 0.88}],
        expected_verdict="not_novel",
    ),
]
