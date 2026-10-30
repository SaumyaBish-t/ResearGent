"""
Labelled test cases for the Verifier node's claim-grounding task.

The Verifier grades each ``[S#]``-tagged sentence in the draft against the
text of the source it cites, returning one of:

``supports``     the source contains the specific fact the sentence asserts.
``contradicts``  the source says something incompatible with the assertion.
``unclear``      the source is on-topic but neither confirms nor denies.

Label vocabulary for this suite
--------------------------------
``supports``    the claim must be graded supports. A grade of ``unclear``
                here is a FALSE NEGATIVE - the source does contain the fact.
``contradicts`` the claim must be graded contradicts. Grading it ``unclear``
                is a FALSE NEGATIVE: the source explicitly disagrees, and
                "unclear" hides a detected fabrication.
``unclear``     the claim must be graded unclear. Grading it ``supports``
                is a FALSE POSITIVE - the source does NOT contain the fact,
                and calling it supported launders an unsupported claim.

These are the labels a careful human annotator would assign by reading the
source and the claim side by side. They are deliberately split into the three
verdict classes so that a detector which simply always says "supports" scores
badly - a trivial detector must not pass this suite.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GroundingCase:
    """One labelled claim / source-text pair."""

    case_id: str
    expected_verdict: str  # supports | contradicts | unclear
    claim: str
    source_text: str
    tag: str = "S1"
    note: str = ""


# ---------------------------------------------------------------------------
# Real source passages (verbatim from the ingested corpus)
# ---------------------------------------------------------------------------

SRC_DLTS = """Are Transformers Effective for Time Series Forecasting? Ailing Zeng, Muxi Chen, Lei Zhang, Qiang Xu. Abstract Recently, there has been a surge of Transformer-based solutions for the long-term time series forecasting (LTSF) task. Despite the growing performance over the past few years, we question the validity of this line of research in this work. Speciﬁcally, Transformers is arguably the most successful solution to extract the semantic correlations of long-term time series. However, we observe that the effectiveness of Transformers for LTSF is mainly attributed to the look-back window, rather than the inherent ability of the attention mechanism."""

SRC_DDQN = """(DDQN) algorithm for optimal execution is introduced, overcoming the bias in Q-value estimation found in traditional DQL. This technique, incorporating two neural networks (Q-network and target Q-network), improves policy estimation accuracy. Ref. [27] considers a Almgren-Chriss framework without permanent impact and with a constant (across time and stocks) temporary impact. When applied to real data, they found that DDQN outperforms the Time-Weighted Average Price (TWAP) strategy in seven out of nine stocks considered."""

SRC_HFT = """Financial economists are concerned with understanding HFT's effects on market efficiency. Hendershott et al. (2011) found that HFT-as an algorithmic trading's subset-improves the liquidity for the shares of firms with large market capitalization by reducing the transaction costs, that is, the bid/ask spread. Along a similar way, Brogaard et al. (2014) found that HFT improves price discovery. However, the CFTC-SEC commission (2010b) and Kirilenko et al. (2017) underlined the role of HFT in exacerbating market volatility when exogenous causes are involved."""

SRC_VAR = """In this section, a bird's eye view of the research in VaR under various models is presented. The range-based models provide additional information and are more parsimonious, while traditional models exhibit memory and asymmetric reaction to shocks. To determine which volatility model performs better when it comes to forecasting performance, the study computes the MSE and QLIKE loss functions."""

SRC_ETHICS = """In this endless cycle, money becomes the aim. More recently, Dierksmeier and Pirson (2009, p. 417) argue that the conceptions of chrematistics and oikonomia provide a basis to discuss the relationship between business and society and to draw important conclusions for business management. Along this line, Ferrero and Sison (2017) discuss how to qualify finance as natural chrematistics: the agents of this activity must serve its purpose and observe its limits."""


GROUNDING_CASES: list[GroundingCase] = [
    # ---- SUPPORTS: the source contains the fact --------------------------
    GroundingCase(
        case_id="sup_01",
        expected_verdict="supports",
        claim=(
            "The effectiveness of Transformers for long-term time series "
            "forecasting is mainly attributed to the look-back window rather than "
            "to the attention mechanism itself."
        ),
        source_text=SRC_DLTS,
        note="Direct restatement of the paper's central finding.",
    ),
    GroundingCase(
        case_id="sup_02",
        expected_verdict="supports",
        claim=(
            "DDQN uses two neural networks, a Q-network and a target Q-network."
        ),
        source_text=SRC_DDQN,
        note="Explicitly enumerated in the source.",
    ),
    GroundingCase(
        case_id="sup_03",
        expected_verdict="supports",
        claim=(
            "DDQN outperformed the TWAP strategy in seven of the nine stocks "
            "considered."
        ),
        source_text=SRC_DDQN,
        note="Exact numeric claim present in the source.",
    ),
    GroundingCase(
        case_id="sup_04",
        expected_verdict="supports",
        claim=(
            "HFT improves price discovery."
        ),
        source_text=SRC_HFT,
        note="Attributed to Brogaard et al. (2014) in the source.",
    ),
    GroundingCase(
        case_id="sup_05",
        expected_verdict="supports",
        claim=(
            "HFT reduces transaction costs, specifically the bid-ask spread, for "
            "shares of firms with large market capitalization."
        ),
        source_text=SRC_HFT,
        note="Directly supported.",
    ),
    GroundingCase(
        case_id="sup_06",
        expected_verdict="supports",
        claim=(
            "The study compares volatility models using the MSE and QLIKE loss "
            "functions."
        ),
        source_text=SRC_VAR,
        note="Explicitly stated.",
    ),
    GroundingCase(
        case_id="sup_07",
        expected_verdict="supports",
        claim=(
            "Dierksmeier and Pirson (2009) argue that the conceptions of "
            "chrematistics and oikonomia provide a basis for discussing the "
            "relationship between business and society."
        ),
        source_text=SRC_ETHICS,
        note="Explicitly attributed with the correct year.",
    ),
    # ---- CONTRADICTS: the source disagrees -------------------------------
    GroundingCase(
        case_id="con_01",
        expected_verdict="contradicts",
        claim=(
            "The authors conclude that the attention mechanism, rather than the "
            "look-back window, is what makes Transformers effective for "
            "long-term forecasting."
        ),
        source_text=SRC_DLTS,
        note="The source says the OPPOSITE - the look-back window is the cause.",
    ),
    GroundingCase(
        case_id="con_02",
        expected_verdict="contradicts",
        claim=(
            "DDQN outperformed TWAP in all nine stocks considered."
        ),
        source_text=SRC_DDQN,
        note="Source says seven out of nine - a different number.",
    ),
    GroundingCase(
        case_id="con_03",
        expected_verdict="contradicts",
        claim=(
            "HFT widens the bid-ask spread for shares of firms with large market "
            "capitalization."
        ),
        source_text=SRC_HFT,
        note="Source says HFT REDUCES the spread - inverted conclusion.",
    ),
    GroundingCase(
        case_id="con_04",
        expected_verdict="contradicts",
        claim=(
            "Traditional volatility models provide more information and are more "
            "parsimonious than range-based models."
        ),
        source_text=SRC_VAR,
        note="Source attributes those properties to the RANGE-based models.",
    ),
    GroundingCase(
        case_id="con_05",
        expected_verdict="contradicts",
        claim=(
            "The paper finds that HFT reduces market volatility."
        ),
        source_text=SRC_HFT,
        note="Source says HFT EXACERBATES volatility when exogenous causes are involved.",
    ),
    # ---- UNCLEAR: on-topic but the source does not settle it --------------
    GroundingCase(
        case_id="unc_01",
        expected_verdict="unclear",
        claim=(
            "The paper's authors affiliated with the Chinese University of Hong "
            "Kong released the code on GitHub under an MIT license."
        ),
        source_text=SRC_DLTS,
        note="The affiliation is in the source; the licensing claim is not.",
    ),
    GroundingCase(
        case_id="unc_02",
        expected_verdict="unclear",
        claim=(
            "The DDQN experiments were run on the FI-2010 benchmark dataset."
        ),
        source_text=SRC_DDQN,
        note="The dataset is never named in this excerpt.",
    ),
    GroundingCase(
        case_id="unc_03",
        expected_verdict="unclear",
        claim=(
            "The authors recommend a 95 percent confidence level for the VaR "
            "back-testing procedure."
        ),
        source_text=SRC_VAR,
        note="No confidence level is stated in the excerpt.",
    ),
    GroundingCase(
        case_id="unc_04",
        expected_verdict="unclear",
        claim=(
            "The market makers analysed in the study are all headquartered in "
            "Europe."
        ),
        source_text=SRC_HFT,
        note="The excerpt does not cover the headquarters of the firms.",
    ),
    GroundingCase(
        case_id="unc_05",
        expected_verdict="unclear",
        claim=(
            "The paper reports that the model was trained for 200 epochs on a "
            "single NVIDIA A100 GPU."
        ),
        source_text=SRC_DDQN,
        note="Training details are absent from the excerpt.",
    ),
]


def cases() -> list[GroundingCase]:
    return list(GROUNDING_CASES)


def expected_counts() -> dict[str, int]:
    out = {"supports": 0, "contradicts": 0, "unclear": 0}
    for c in GROUNDING_CASES:
        out[c.expected_verdict] += 1
    return out
