"""
Labelled test cases for the Originality Check (Phase 20).

Design
------
Each case is a (draft_sentence, source_text, label) triple where the label is
the GROUND TRUTH a human would assign about whether the draft sentence is
problematic (unattributed copying) with respect to the source.

Label vocabulary
----------------
``verbatim_uncited``     the draft reproduces the source word-for-word over a
                          long span AND carries no ``[S#]`` citation.
``verbatim_cited``       the draft reproduces the source word-for-word over a
                          long span AND carries a ``[S#]`` citation. This is
                          legitimate quotation — NOT a violation.
``paraphrase_uncited``   the draft expresses the same content as the source in
                          different words AND carries no citation.
``paraphrase_cited``     same, but cited. Legitimate — NOT a violation.
``original``             the draft says something the source does not say.
                          No violation.

The "violation" bit is derived from the label: a case is a VIOLATION iff the
label is ``*_uncited``. Everything cited is a legitimate use of the source and
must NOT be flagged — this is the false-positive axis that matters most in
practice, because flagging properly-cited quotes trains users to ignore the
report entirely.

Source material
---------------
The source passages are REAL excerpts from the ingested corpus (academic PDFs
under ``data/``), captured verbatim at chunk indexes 124, 375, 420, 432, 586
and 598 of the local vector store. They are checked into this file so the
suite is deterministic and does not depend on a populated store.

The paraphrase / original cases are written by hand against those real
passages, so the ground truth is defensible: a reviewer can read the source
and the draft side by side and confirm the label.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OriginalityCase:
    """One labelled draft-sentence / source-text pair."""

    case_id: str
    label: str
    draft_sentence: str
    source_text: str
    source_tag: str = "S1"
    note: str = ""
    ambiguous: bool = False
    """True when a reasonable annotator could disagree with ``label``.

    Ambiguous cases are excluded from the headline precision/recall figures so
    the reported numbers rest only on cases whose ground truth is defensible.
    They are still scored and reported separately, so nothing is hidden.
    """

    @property
    def is_violation(self) -> bool:
        """True iff this case SHOULD be flagged by the originality checker."""
        return self.label.endswith("_uncited")

    @property
    def expected_status(self) -> str:
        """The status string the checker should assign (if it flags at all)."""
        if self.label == "verbatim_uncited":
            return "verbatim_uncited"
        if self.label == "paraphrase_uncited":
            return "uncited_paraphrase"
        return "quoted_cited"  # cited -> never a violation


# ---------------------------------------------------------------------------
# Real source passages (verbatim from the ingested corpus)
# ---------------------------------------------------------------------------

SRC_DLTS = """Are Transformers Effective for Time Series Forecasting? Ailing Zeng, Muxi Chen, Lei Zhang, Qiang Xu. Abstract Recently, there has been a surge of Transformer-based solutions for the long-term time series forecasting (LTSF) task. Despite the growing performance over the past few years, we question the validity of this line of research in this work. Speciﬁcally, Transformers is arguably the most successful solution to extract the semantic correlations of long-term time series. However, we observe that the effectiveness of Transformers for LTSF is mainly attributed to the look-back window, rather than the inherent ability of the attention mechanism."""

SRC_DDQN = """(DDQN) algorithm for optimal execution is introduced, overcoming the bias in Q-value estimation found in traditional DQL. This technique, incorporating two neural networks (Q-network and target Q-network), improves policy estimation accuracy. Ref. [27] considers a Almgren-Chriss framework without permanent impact and with a constant (across time and stocks) temporary impact. When applied to real data, they found that DDQN outperforms the Time-Weighted Average Price (TWAP) strategy in seven out of nine stocks considered."""

SRC_HFT = """Financial economists are concerned with understanding HFT's effects on market efficiency. Hendershott et al. (2011) found that HFT-as an algorithmic trading's subset-improves the liquidity for the shares of firms with large market capitalization by reducing the transaction costs, that is, the bid/ask spread. Along a similar way, Brogaard et al. (2014) found that HFT improves price discovery. However, the CFTC-SEC commission (2010b) and Kirilenko et al. (2017) underlined the role of HFT in exacerbating market volatility when exogenous causes are involved."""

SRC_VAR = """In this section, a bird's eye view of the research in VaR under various models is presented. The range-based models provide additional information and are more parsimonious, while traditional models exhibit memory and asymmetric reaction to shocks. To determine which volatility model performs better when it comes to forecasting performance, the study computes the MSE and QLIKE loss functions."""

SRC_ETHICS = """In this endless cycle, money becomes the aim. More recently, Dierksmeier and Pirson (2009, p. 417) argue that the conceptions of chrematistics and oikonomia provide a basis to discuss the relationship between business and society and to draw important conclusions for business management. Along this line, Ferrero and Sison (2017) discuss how to qualify finance as natural chrematistics: the agents of this activity must serve its purpose and observe its limits."""

SRC_OTHERS = [
    "Attention mechanisms allow a model to focus on different parts of an input sequence when producing its output.",
    "The limit order book records the queue of pending buy and sell orders at a given price level.",
    "Backtesting a strategy on historical data does not guarantee performance in live markets.",
    "A transformer encoder applies self-attention across the sequence dimension of its input.",
    "Value at Risk estimates the loss threshold over a specified horizon and confidence level.",
]


# ---------------------------------------------------------------------------
# Labelled cases
# ---------------------------------------------------------------------------

CASES: list[OriginalityCase] = [
    # ---- VERBATIM, UNCITED -> must be flagged as verbatim_uncited --------
    OriginalityCase(
        case_id="verb_un_01",
        label="verbatim_uncited",
        draft_sentence=(
            "Recently, there has been a surge of Transformer-based solutions for the "
            "long-term time series forecasting (LTSF) task."
        ),
        source_text=SRC_DLTS,
        note="Exact 13-word span lifted from the source with no citation tag.",
    ),
    OriginalityCase(
        case_id="verb_un_02",
        label="verbatim_uncited",
        draft_sentence=(
            "When applied to real data, they found that DDQN outperforms the "
            "Time-Weighted Average Price (TWAP) strategy in seven out of nine "
            "stocks considered."
        ),
        source_text=SRC_DDQN,
        note="Exact long span including the numeric detail.",
    ),
    OriginalityCase(
        case_id="verb_un_03",
        label="verbatim_uncited",
        draft_sentence=(
            "Hendershott et al. (2011) found that HFT improves the liquidity for "
            "the shares of firms with large market capitalization by reducing the "
            "transaction costs, that is, the bid/ask spread."
        ),
        source_text=SRC_HFT,
        note="Verbatim attribution to a named author - classic unattributed borrowing.",
    ),
    OriginalityCase(
        case_id="verb_un_04",
        label="verbatim_uncited",
        draft_sentence=(
            "Along a similar way, Brogaard et al. (2014) found that HFT improves "
            "price discovery."
        ),
        source_text=SRC_HFT,
        note="Short but exact 13-word span with author attribution.",
    ),
    # ---- VERBATIM, CITED -> must NOT be flagged (false-positive axis) ----
    OriginalityCase(
        case_id="verb_ci_01",
        label="verbatim_cited",
        draft_sentence=(
            "Recently, there has been a surge of Transformer-based solutions for the "
            "long-term time series forecasting (LTSF) task [S1]."
        ),
        source_text=SRC_DLTS,
        note="Legitimate direct quotation - carries the citation.",
    ),
    OriginalityCase(
        case_id="verb_ci_02",
        label="verbatim_cited",
        draft_sentence=(
            "When applied to real data, they found that DDQN outperforms the "
            "Time-Weighted Average Price (TWAP) strategy in seven out of nine "
            "stocks considered [S1]."
        ),
        source_text=SRC_DDQN,
        note="Legitimate direct quotation with citation.",
    ),
    # ---- PARAPHRASE, UNCITED -> must be flagged as uncited_paraphrase -----
    OriginalityCase(
        case_id="para_un_01",
        label="paraphrase_uncited",
        draft_sentence=(
            "Long-horizon forecasting using Transformers has become an "
            "increasingly active area of research, though the authors doubt "
            "whether the attention mechanism itself deserves the credit."
        ),
        source_text=SRC_DLTS,
        note="Same claim, different words - no citation.",
    ),
    OriginalityCase(
        case_id="para_un_02",
        label="paraphrase_uncited",
        draft_sentence=(
            "Double Deep Q-Network removes the Q-value estimation bias that "
            "afflicts standard DQL by pairing a live network with a slowly "
            "updated target network, which sharpens the learned policy."
        ),
        source_text=SRC_DDQN,
        note="Restates the DDQN mechanism in the draft author's own words.",
    ),
    OriginalityCase(
        case_id="para_un_03",
        label="paraphrase_uncited",
        draft_sentence=(
            "Range-based volatility estimators turn out to be more economical "
            "and to carry extra information, whereas conventional models retain "
            "memory and respond asymmetrically to shocks."
        ),
        source_text=SRC_VAR,
        note="Paraphrase of the range-vs-traditional comparison.",
    ),
    # ---- PARAPHRASE, CITED -> must NOT be flagged -------------------------
    OriginalityCase(
        case_id="para_ci_01",
        label="paraphrase_cited",
        draft_sentence=(
            "Long-horizon forecasting using Transformers has become an "
            "increasingly active area of research, though the authors doubt "
            "whether the attention mechanism itself deserves the credit [S1]."
        ),
        source_text=SRC_DLTS,
        note="Legitimate paraphrase - cited.",
    ),
    OriginalityCase(
        case_id="para_ci_02",
        label="paraphrase_cited",
        draft_sentence=(
            "Range-based volatility estimators turn out to be more economical "
            "and to carry extra information, whereas conventional models retain "
            "memory and respond asymmetrically to shocks [S1]."
        ),
        source_text=SRC_VAR,
        note="Legitimate paraphrase - cited.",
    ),
    # ---- ORIGINAL -> must NOT be flagged ---------------------------------
    OriginalityCase(
        case_id="orig_01",
        label="original",
        draft_sentence=(
            "A 2024 replication across eleven equity markets suggests that the "
            "look-back window, rather than attention capacity, accounts for most "
            "of the reported gain in long-horizon forecasting accuracy."
        ),
        source_text=SRC_DLTS,
        note="Claims a specific later study that the source does not contain.",
    ),
    OriginalityCase(
        case_id="orig_02",
        label="original",
        draft_sentence=(
            "Since both the paper and the commentary are available, an analyst can "
            "cross-check the 0.85 quantile reading against the reported MSE before "
            "committing capital to the strategy."
        ),
        source_text=SRC_VAR,
        note="Synthesis / recommendation absent from the source.",
    ),
    OriginalityCase(
        case_id="orig_03",
        label="original",
        draft_sentence=(
            "It is worth noting that the ethical literature treats market making "
            "as a practice whose legitimacy depends on the good it produces for "
            "the wider market."
        ),
        source_text=SRC_ETHICS,
        note="Framing drawn from Aristotle that the source does not state this way.",
    ),
    # ---- INCIDENTAL / HARD NEGATIVES -------------------------------------
    # These share vocabulary with the source but assert nothing the source
    # asserts. They are the hardest false-positive class: high lexical or
    # semantic overlap with no actual copying.
    OriginalityCase(
        case_id="inc_01",
        label="original",
        draft_sentence=(
            "The transformer encoder processes the sequence dimension and the "
            "decoder attends over the encoded representation."
        ),
        source_text=SRC_DLTS,
        note="Related topic, different content - topical overlap only.",
    ),
    OriginalityCase(
        case_id="inc_02",
        label="original",
        draft_sentence=(
            "Backtesting on historical data is a necessary but insufficient check "
            "before deploying a live execution strategy."
        ),
        source_text=SRC_DDQN,
        note="Shares the domain vocabulary without copying any sentence.",
    ),
    OriginalityCase(
        case_id="inc_03",
        label="original",
        draft_sentence=(
            "Value at Risk is reported at a chosen confidence level over a fixed "
            "horizon and is back-tested against the realised loss distribution."
        ),
        source_text=SRC_VAR,
        note="Definition in the source's own words - genuinely reworded.",
    ),
    OriginalityCase(
        case_id="inc_04",
        label="original",
        draft_sentence=(
            "High-frequency trading has been shown to improve price discovery and "
            "to widen bid-ask spreads for large-capitalisation firms."
        ),
        source_text=SRC_HFT,
        note="Shares entities but asserts a DIFFERENT conclusion (spreads wider "
             "vs narrower) - the source says the opposite.",
        ambiguous=True,
    ),
    # ---- HARD NEGATIVES: paraphrase-shaped originals ----------------------
    # These are written in the source's own idiom but assert something the
    # source does NOT. They are the cases that separate "semantic overlap
    # detector" from "plagiarism detector" - a naive cosine threshold will
    # flag all of these.
    OriginalityCase(
        case_id="inc_05",
        label="original",
        draft_sentence=(
            "Double Q-learning suffers from the well-documented overestimation "
            "bias that arises because the same network both selects and "
            "evaluates the action."
        ),
        source_text=SRC_DDQN,
        note="Explains WHY DQL has the bias - the source only says there IS a bias.",
    ),
    OriginalityCase(
        case_id="inc_06",
        label="original",
        draft_sentence=(
            "The Almgren-Chriss framework assumes that a trade's permanent market "
            "impact is negligible while temporary impact varies over the "
            "execution window."
        ),
        source_text=SRC_DDQN,
        note="Restates the framework but reverses which assumption is dropped. "
             "Defensible as an uncited paraphrase OR as a corrected restatement "
             "- annotators can disagree, so excluded from the headline metric.",
        ambiguous=True,
    ),
    OriginalityCase(
        case_id="inc_07",
        label="original",
        draft_sentence=(
            "Regulatory inquiries have focused on whether algorithmic trading "
            "amplifies swings that originate outside the trading activity itself."
        ),
        source_text=SRC_HFT,
        note="Synthesis of the CFTC-SEC / Kirilenko point in new words.",
    ),
    OriginalityCase(
        case_id="inc_08",
        label="original",
        draft_sentence=(
            "A parsimonious volatility estimator is preferable when it achieves "
            "the same out-of-sample accuracy with fewer estimated parameters."
        ),
        source_text=SRC_VAR,
        note="Restates 'more parsimonious' but adds a criterion the source lacks.",
    ),
    OriginalityCase(
        case_id="inc_09",
        label="original",
        draft_sentence=(
            "Whether attention genuinely contributes to forecasting, or merely "
            "provides a convenient way to ingest a long history, remains the "
            "central open question raised by the authors."
        ),
        source_text=SRC_DLTS,
        note="Paraphrases the look-back-window finding closely - defensible as "
             "an original restatement OR as an uncited paraphrase of the same "
             "claim; annotators can disagree, so excluded from the headline metric.",
        ambiguous=True,
    ),
    # ---- HARD POSITIVES: paraphrases with low lexical overlap -------------
    OriginalityCase(
        case_id="para_un_04",
        label="paraphrase_uncited",
        draft_sentence=(
            "Critics have argued that the regulators charged with policing these "
            "firms highlighted a tendency for programmatic trading to magnify "
            "price swings triggered by forces outside the market."
        ),
        source_text=SRC_HFT,
        note="Same claim as the CFTC-SEC/Kirilenko sentence, almost no shared "
             "n-grams - only the embedding can catch it.",
        ambiguous=True,
    ),
    OriginalityCase(
        case_id="para_un_05",
        label="paraphrase_uncited",
        draft_sentence=(
            "The paper's central claim is that the attention operation itself is "
            "not what makes these models work; the length of the historical "
            "window they consume is."
        ),
        source_text=SRC_DLTS,
        note="Paraphrase of the thesis - low lexical overlap with the source.",
    ),
    OriginalityCase(
        case_id="para_un_06",
        label="paraphrase_uncited",
        draft_sentence=(
            "One of the cited studies assumed that a trade leaves no permanent "
            "mark on the price and that its temporary mark is the same for every "
            "moment and every security."
        ),
        source_text=SRC_DDQN,
        note="Restates the Almgren-Chriss assumption in the draft author's words.",
    ),
]


# ---------------------------------------------------------------------------
# Multi-source case: a draft that copies from source A but is compared against
# source B. The checker must not flag it. This exercises the "compare every
# draft sentence against every source" loop for false positives.
# ---------------------------------------------------------------------------

CROSS_SOURCE_CASES: list[tuple[str, str, str, str]] = [
    # (case_id, draft_sentence, correct_source_text, wrong_source_text)
    (
        "cross_01",
        "Recently, there has been a surge of Transformer-based solutions for the "
        "long-term time series forecasting task.",
        SRC_DLTS,
        SRC_VAR,
    ),
]


def cases() -> list[OriginalityCase]:
    """All labelled originality cases."""
    return list(CASES)


def violation_count() -> int:
    return sum(1 for c in CASES if c.is_violation)


def clean_count() -> int:
    return sum(1 for c in CASES if not c.is_violation)
