"""
Novelty scoring engine — Phase 24.

Pure, side-effect-free scoring primitives for the Novelty Check node. Kept
separate from the LangGraph node so the maths can be unit-tested and
threshold-swept without touching the graph, the DB, or the network.

What it answers
---------------
Given a user's OWN experimental results (the "ground truth" from Results
Ingestion) and the papers the Retriever surfaced, two questions:

  1. APPROACH SIMILARITY — does the user's described approach match the
     approach a paper used? (embedding cosine + content-word overlap)

  2. RESULT SIMILARITY  — on a metric that BOTH report, is the user's
     number better, equal, or worse than the paper's?

These combine into a verdict:

  NOVEL        — approach is distinct AND (no comparable metric found OR
                 the user beats every comparable metric).
  IMPROVEMENT  — approach matches a prior paper, but the user's numbers
                 beat it. This is a legitimate, publishable contribution.
  NOT_NOVEL    — approach matches AND the user's numbers reproduce or lose
                 to the prior work. The result is already known.
  INCONCLUSIVE — the approach matches but no metric could be compared, so
                 neither claim can be honestly made.

Design constraints
------------------
* NO LLM is consulted for the numbers. The user's metrics and the paper's
  metrics are compared deterministically. An LLM may EXPLAIN the verdict,
  but it never decides whether 38.1 beats 34.8.
* Metric matching is by NORMALISED NAME, never by fuzzy string distance, so
  "BLEU-4" never matches "ROUGE-L" by accident.
* Every threshold lives here as a named module constant so a sweep can
  grid-search them and the shipped defaults can be justified by data.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Sequence


# ---------------------------------------------------------------------------
# Thresholds — the knobs a sweep can grid-search.
# ---------------------------------------------------------------------------

# Approach-similarity cosine at/above which the user's method is considered
# the SAME method as a prior paper's.
APPROACH_SIMILARITY_THRESHOLD = 0.50

# Result-similarity at/above which the Human Review Gate pauses the run.
# Per the spec: "If similarity is above 50%, the pipeline pauses".
REVIEW_SIMILARITY_THRESHOLD = 0.50

# Content-word overlap required alongside the cosine before the approach is
# called a match. Guards against embedding noise (cosine alone is generous
# for short texts).
APPROACH_CONTENT_OVERLAP = 0.20

# Relative difference in a metric that still counts as "the same number".
# 2% relative tolerance — papers round to 1-2 decimal places, so 34.8 vs
# 35.0 is the same reported result, not a meaningful difference.
METRIC_EQUAL_RELATIVE_TOLERANCE = 0.02

# A metric is only comparable when both sides carry a numeric value.
_MIN_ABS_DELTA_FOR_DIRECTION = 1e-9


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class MatchDirection(str, Enum):
    """How the user's metric compares to the paper's, on a shared metric."""

    BETTER = "better"
    EQUAL = "equal"
    WORSE = "worse"
    UNKNOWN = "unknown"


class NoveltyVerdict(str, Enum):
    """The overall outcome of the Novelty Check."""

    NOVEL = "novel"
    IMPROVEMENT = "improvement"
    NOT_NOVEL = "not_novel"
    INCONCLUSIVE = "inconclusive"


# ---------------------------------------------------------------------------
# Metric-name normalisation
# ---------------------------------------------------------------------------

# Generic words that carry no metric identity — dropped so "F1 score" and
# "F1" normalise to the same token.
_GENERIC_METRIC_WORDS = {
    "score", "scores", "metric", "metrics", "result", "results", "value",
    "values", "rate", "ratio", "acc", "test", "eval", "evaluation",
}

# Metric families with common spellings. Normalising the family name keeps
# "exact match" and "EM" comparable while still keeping the variant intact.
_METRIC_SYNONYMS: dict[str, str] = {
    "exact match": "em",
    "exact-match": "em",
    "exactmatch": "em",
    "f1": "f1",
    "f1 score": "f1",
    "f1-score": "f1",
    "rouge": "rouge",
    "bleu": "bleu",
    "accuracy": "accuracy",
    "acc": "accuracy",
    "perplexity": "perplexity",
    "ppl": "perplexity",
}


def normalize_metric_name(name: str) -> str:
    """
    Canonicalise a metric name so two spellings of the same metric compare.

    Rules:
      * lower-case, strip surrounding whitespace
      * punctuation (`-`, `_`, `.`, `/`) becomes a space
      * generic trailing words ("score", "metric") are dropped
      * known synonyms collapse onto a family token ("EM", "exact match")
      * the VARIANT survives ("bleu 4" != "bleu")

    >>> normalize_metric_name("BLEU-4")
    'bleu 4'
    >>> normalize_metric_name("F1 score")
    'f1'
    >>> normalize_metric_name("exact match")
    'em'
    """
    if not name:
        return ""
    text = str(name).strip().lower()
    # Punctuation -> space.
    text = re.sub(r"[^\w\s]+", " ", text, flags=re.UNICODE)
    text = re.sub(r"[\s_]+", " ", text).strip()

    # Synonym fold BEFORE generic-word stripping so "exact match" -> "em".
    if text in _METRIC_SYNONYMS:
        return _METRIC_SYNONYMS[text]

    words = [w for w in text.split() if w]
    # Drop generic words anywhere in the name.
    words = [w for w in words if w not in _GENERIC_METRIC_WORDS]
    if not words:
        words = [w for w in text.split() if w]
    if not words:
        return ""

    folded = " ".join(words)
    # A single remaining word may itself be a synonym family ("accuracy").
    if len(words) == 1 and words[0] in _METRIC_SYNONYMS:
        return _METRIC_SYNONYMS[words[0]]
    return folded


# ---------------------------------------------------------------------------
# Numeric parsing
# ---------------------------------------------------------------------------

_NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def parse_numeric(value: Any) -> float | None:
    """
    Extract a float from a user-supplied or paper-reported metric value.

    Accepts numbers and strings like "94.2", "94.2%", "BLEU = 34.8",
    "1e-3", "-2.5". Returns None when there is no number at all.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    m = _NUM_RE.search(text)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Metric matching
# ---------------------------------------------------------------------------


@dataclass
class MetricMatch:
    """The comparison of one user metric against one paper metric."""

    user_metric: dict[str, Any]
    paper_metric: dict[str, Any]
    matched: bool
    direction: MatchDirection
    delta: float
    # Relative difference, used to decide EQUAL vs a real difference.
    relative_delta: float = 0.0
    paper_label: str = ""

    def __post_init__(self) -> None:
        """
        Derive `relative_delta` when the caller did not supply it.

        Without this, a hand-constructed `MetricMatch(delta=25.2,
        relative_delta=0.0)` would report itself as an IDENTICAL result,
        because the relative delta is what the similarity maths reads. The
        dataclass must never be able to disagree with itself.
        """
        if self.relative_delta == 0.0 and self.delta:
            u_val = parse_numeric(self.user_metric.get("value"))
            p_val = parse_numeric(self.paper_metric.get("value"))
            if u_val is not None and p_val is not None:
                scale = max(abs(p_val), abs(u_val), _MIN_ABS_DELTA_FOR_DIRECTION)
                self.relative_delta = abs(self.delta) / scale

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_metric": self.user_metric,
            "paper_metric": self.paper_metric,
            "matched": self.matched,
            "direction": self.direction.value,
            "delta": round(self.delta, 6),
            "relative_delta": round(self.relative_delta, 6),
            "paper_label": self.paper_label,
        }


def _metrics_match_by_name(user_name: str, paper_name: str) -> bool:
    """True when two metric names denote the SAME metric (variant included)."""
    a = normalize_metric_name(user_name)
    b = normalize_metric_name(paper_name)
    if not a or not b:
        return False
    if a == b:
        return True
    # A metric with a variant ("bleu 4") only matches the same variant.
    # A bare family ("bleu") matches the family only — never "bleu 4".
    return False


def compare_metric(
    *,
    user_metric: dict[str, Any],
    paper_metric: dict[str, Any],
    higher_is_better: bool = True,
) -> MetricMatch:
    """
    Compare one user metric against one paper metric.

    `higher_is_better=False` is for latency / perplexity / loss / error,
    where a SMALLER number is the better result.
    """
    user_name = str(user_metric.get("name", ""))
    paper_name = str(paper_metric.get("name", ""))
    paper_label = str(paper_metric.get("label", "") or paper_name)

    if not _metrics_match_by_name(user_name, paper_name):
        return MetricMatch(
            user_metric=user_metric,
            paper_metric=paper_metric,
            matched=False,
            direction=MatchDirection.UNKNOWN,
            delta=0.0,
            paper_label=paper_label,
        )

    u_val = parse_numeric(user_metric.get("value"))
    p_val = parse_numeric(paper_metric.get("value"))
    if u_val is None or p_val is None:
        return MetricMatch(
            user_metric=user_metric,
            paper_metric=paper_metric,
            matched=False,
            direction=MatchDirection.UNKNOWN,
            delta=0.0,
            paper_label=paper_label,
        )

    delta = u_val - p_val
    scale = max(abs(p_val), abs(u_val), _MIN_ABS_DELTA_FOR_DIRECTION)
    relative = abs(delta) / scale

    if relative <= METRIC_EQUAL_RELATIVE_TOLERANCE:
        direction = MatchDirection.EQUAL
    else:
        better_when_higher = higher_is_better
        is_better = (delta > 0) if better_when_higher else (delta < 0)
        direction = MatchDirection.BETTER if is_better else MatchDirection.WORSE

    return MetricMatch(
        user_metric=user_metric,
        paper_metric=paper_metric,
        matched=True,
        direction=direction,
        delta=delta,
        relative_delta=relative,
        paper_label=paper_label,
    )


def match_metrics(
    *,
    user_metrics: Sequence[dict[str, Any]],
    paper_metrics: Sequence[dict[str, Any]],
    higher_is_better: bool = True,
) -> list[MetricMatch]:
    """
    Match every user metric against every paper metric.

    A user metric can match at most ONE paper metric per paper — the first
    one with the same normalised name. Unmatched user metrics are returned
    as unmatched entries so the caller can report "no comparable metric".
    """
    matches: list[MetricMatch] = []
    used_paper_indices: set[int] = set()

    for u_metric in user_metrics or []:
        found = False
        for idx, p_metric in enumerate(paper_metrics or []):
            if idx in used_paper_indices:
                continue
            m = compare_metric(
                user_metric=u_metric,
                paper_metric=p_metric,
                higher_is_better=higher_is_better,
            )
            if m.matched:
                matches.append(m)
                used_paper_indices.add(idx)
                found = True
                break
        if not found:
            matches.append(
                MetricMatch(
                    user_metric=u_metric,
                    paper_metric={},
                    matched=False,
                    direction=MatchDirection.UNKNOWN,
                    delta=0.0,
                )
            )
    return matches


# ---------------------------------------------------------------------------
# Approach similarity
# ---------------------------------------------------------------------------

_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "of", "to", "in", "on", "at",
    "for", "with", "from", "by", "as", "is", "are", "was", "were", "be",
    "we", "our", "us", "they", "their", "it", "its", "this", "that", "these",
    "those", "then", "than", "so", "such", "can", "could", "may", "might",
    "must", "shall", "should", "will", "would", "have", "has", "had", "do",
    "does", "did", "also", "more", "most", "other", "into", "over", "under",
    "between", "within", "without", "across", "per", "via", "using", "used",
    "use", "both", "each", "any", "all", "some", "which", "who", "when",
    "where", "why", "how", "what", "whose", "new", "novel", "show", "shows",
    "shown", "find", "finds", "found", "propose", "proposed", "presents",
    "present", "paper", "work", "result", "results", "evaluate", "evaluated",
}


def _content_words(text: str) -> set[str]:
    """Stopword-filtered content words for overlap scoring."""
    return {
        w
        for w in re.findall(r"[a-z][a-z0-9\-]*", (text or "").lower())
        if w not in _STOPWORDS and len(w) > 2
    }


def content_overlap(a: str, b: str) -> float:
    """Jaccard overlap of stopword-filtered content words (0.0 - 1.0)."""
    aw, bw = _content_words(a), _content_words(b)
    if not aw or not bw:
        return 0.0
    return len(aw & bw) / len(aw | bw)


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity of two vectors (0.0 when either is empty)."""
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def score_approach_similarity(
    user_approach: str,
    paper_approach: str,
    *,
    vectors: tuple[Sequence[float], Sequence[float]] | None = None,
) -> float:
    """
    Score how similar two described approaches are, in [0.0, 1.0].

    Combines two signals:
      * embedding cosine (catches paraphrase) — supplied via `vectors` to
        keep this function pure and testable; when absent, the content
        overlap alone is used.
      * content-word Jaccard (catches shared technique vocabulary)

    The final score is the MEAN of the two available signals, so a pair
    that shares vocabulary but embeds oddly still scores meaningfully.
    """
    if not (user_approach or "").strip() or not (paper_approach or "").strip():
        return 0.0

    overlap = content_overlap(user_approach, paper_approach)

    if vectors is not None:
        va, vb = vectors
        sim = cosine(va, vb)
        # A high cosine with no shared content words is embedding noise;
        # damp it rather than trusting it outright.
        if overlap < APPROACH_CONTENT_OVERLAP and sim > APPROACH_SIMILARITY_THRESHOLD:
            sim *= 0.5
        return max(0.0, min(1.0, (sim + overlap) / 2.0))

    return max(0.0, min(1.0, overlap))


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------


@dataclass
class NoveltyDecision:
    """The Novelty Check's decision, ready for the report."""

    verdict: NoveltyVerdict
    is_novel: bool
    overall_similarity: float
    approach_similarity: float
    result_similarity: float
    requires_review: bool
    matched_metrics: list[MetricMatch] = field(default_factory=list)
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "is_novel": self.is_novel,
            "overall_similarity": round(self.overall_similarity, 4),
            "approach_similarity": round(self.approach_similarity, 4),
            "result_similarity": round(self.result_similarity, 4),
            "requires_review": self.requires_review,
            "matched_metrics": [m.to_dict() for m in self.matched_metrics],
            "rationale": self.rationale,
        }


def _result_similarity(matches: Iterable[MetricMatch]) -> float:
    """
    How similar are the RESULTS to prior work?

    1.0  — every comparable metric reproduces the prior number exactly.
    0.0  — every comparable metric is far from the prior number.

    Computed per-metric as `1 - min(1, relative_delta / 0.25)` so a 25%
    difference scores 0 and an identical number scores 1, then averaged
    over the matched metrics. Unmatched metrics are ignored entirely.
    """
    matched = [m for m in matches if m.matched]
    if not matched:
        return 0.0
    scores = []
    for m in matched:
        # A result that BEATS the prior work is the opposite of a
        # reproduction, so it should score LOW similarity.
        scores.append(max(0.0, 1.0 - min(1.0, m.relative_delta / 0.25)))
    return sum(scores) / len(scores)


def decide_verdict(
    *,
    approach_similarity: float,
    result_matches: Sequence[MetricMatch],
    approach_threshold: float = APPROACH_SIMILARITY_THRESHOLD,
    review_threshold: float = REVIEW_SIMILARITY_THRESHOLD,
) -> NoveltyDecision:
    """
    Combine approach + result similarity into the Novelty Check verdict.

    Decision table
    -------------
    approach < threshold, no comparable metric      -> NOVEL
    approach < threshold, user beats the metric      -> NOVEL
    approach < threshold, user ties the metric        -> NOT_NOVEL
    approach < threshold, user loses the metric       -> NOT_NOVEL
    approach >= threshold, no comparable metric       -> INCONCLUSIVE
    approach >= threshold, user beats the metric      -> IMPROVEMENT
    approach >= threshold, user ties the metric        -> NOT_NOVEL
    approach >= threshold, user loses the metric       -> NOT_NOVEL

    `requires_review` is True whenever the OVERALL similarity reaches the
    review threshold — that is the signal the Human Review Gate uses to
    pause the pipeline.
    """
    matched = [m for m in result_matches if m.matched]
    result_sim = _result_similarity(result_matches)

    # Overall similarity blends how close the METHOD is and how close the
    # NUMBERS are. An identical method with identical numbers is a full
    # reproduction (1.0); a distinct method with different numbers is not
    # (low).
    overall = 0.5 * approach_similarity + 0.5 * result_sim

    approach_matches = approach_similarity >= approach_threshold

    if not matched:
        if approach_matches:
            verdict = NoveltyVerdict.INCONCLUSIVE
            rationale = (
                "Approach resembles prior work, but no directly comparable "
                "metric was found — novelty cannot be established."
            )
        else:
            verdict = NoveltyVerdict.NOVEL
            rationale = (
                "No prior work found with a comparable approach, and no "
                "comparable metric was reported."
            )
    else:
        better = [m for m in matched if m.direction == MatchDirection.BETTER]
        equal_or_worse = [
            m for m in matched
            if m.direction in (MatchDirection.EQUAL, MatchDirection.WORSE)
        ]
        if equal_or_worse:
            verdict = NoveltyVerdict.NOT_NOVEL
            rationale = (
                f"{len(equal_or_worse)} of {len(matched)} comparable metric(s) "
                "reproduce or fall below the prior reported result."
            )
        elif better:
            if approach_matches:
                verdict = NoveltyVerdict.IMPROVEMENT
                rationale = (
                    "Approach matches prior work, but the result improves on "
                    "every comparable metric — a genuine improvement."
                )
            else:
                verdict = NoveltyVerdict.NOVEL
                rationale = (
                    "Approach is distinct from prior work and the result "
                    "improves on every comparable metric."
                )
        else:
            verdict = NoveltyVerdict.INCONCLUSIVE
            rationale = "Comparable metrics found, but no direction could be established."

    is_novel = verdict in (NoveltyVerdict.NOVEL, NoveltyVerdict.IMPROVEMENT)
    requires_review = overall >= review_threshold

    return NoveltyDecision(
        verdict=verdict,
        is_novel=is_novel,
        overall_similarity=overall,
        approach_similarity=approach_similarity,
        result_similarity=result_sim,
        requires_review=requires_review,
        matched_metrics=matched,
        rationale=rationale,
    )
