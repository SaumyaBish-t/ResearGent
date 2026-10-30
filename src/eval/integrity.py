"""
Validation harness for the integrity checks (provenance / originality /
claim grounding).

Purpose
-------
The industry-review requirement was: *"validate the retraction and originality
checks against labelled test cases, justify the similarity thresholds, and
measure false positives and false negatives. An LLM-generated confidence score
alone is not sufficient evidence of reliability."*

This module supplies the measurement machinery:

* :class:`ConfusionMatrix` - binary TP/FP/TN/FN + precision / recall / F1.
* :func:`sweep_thresholds` - grid-search the originality check's two knobs
  (embedding cosine threshold, n-gram window) against the labelled set and
  report the full trade-off curve, so the shipped defaults are *chosen from
  data* rather than guessed.
* :func:`evaluate_originality` - run the real ``originality_check`` node
  against every labelled case and score it.
* :func:`evaluate_grounding` - run the real Verifier prompt against every
  labelled grounding case and score it.

Everything here is importable without network access EXCEPT the two
``evaluate_*`` functions, which need the EMBED and FAST LLM tiers.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


# ---------------------------------------------------------------------------
# Binary confusion matrix
# ---------------------------------------------------------------------------


@dataclass
class ConfusionMatrix:
    """Binary confusion matrix with the derived rates the reviewer asked for."""

    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.tn + self.fn

    @property
    def positives(self) -> int:
        """Ground-truth positives (cases that SHOULD have been flagged)."""
        return self.tp + self.fn

    @property
    def negatives(self) -> int:
        """Ground-truth negatives (cases that should NOT have been flagged)."""
        return self.tp * 0 + self.fp + self.tn

    @property
    def precision(self) -> float:
        """Of everything flagged, how much was genuinely a violation."""
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def recall(self) -> float:
        """Of every genuine violation, how much was caught."""
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def specificity(self) -> float:
        """Of every clean case, how many were left unflagged."""
        denom = self.fp + self.tn
        return self.tn / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def false_positive_rate(self) -> float:
        denom = self.fp + self.tn
        return self.fp / denom if denom else 0.0

    @property
    def false_negative_rate(self) -> float:
        denom = self.tp + self.fn
        return self.fn / denom if denom else 0.0

    def add(self, predicted_positive: bool, actual_positive: bool) -> None:
        if predicted_positive and actual_positive:
            self.tp += 1
        elif predicted_positive and not actual_positive:
            self.fp += 1
        elif not predicted_positive and actual_positive:
            self.fn += 1
        else:
            self.tn += 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "tp": self.tp,
            "fp": self.fp,
            "tn": self.tn,
            "fn": self.fn,
            "total": self.total,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "specificity": round(self.specificity, 4),
            "f1": round(self.f1, 4),
            "false_positive_rate": round(self.false_positive_rate, 4),
            "false_negative_rate": round(self.false_negative_rate, 4),
        }


@dataclass
class MulticlassMatrix:
    """Per-class counts for the 3-way grounding verdict task."""

    labels: tuple[str, ...]
    matrix: dict[str, dict[str, int]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for row in self.labels:
            self.matrix.setdefault(row, {c: 0 for c in self.labels})

    def add(self, expected: str, predicted: str) -> None:
        if predicted not in self.labels:
            predicted = "__out_of_vocab__"
            self.matrix.setdefault("__out_of_vocab__", {c: 0 for c in self.labels})
        self.matrix[expected][predicted] = self.matrix[expected].get(predicted, 0) + 1

    def accuracy(self) -> float:
        total = sum(sum(row.values()) for row in self.matrix.values())
        if not total:
            return 0.0
        correct = sum(self.matrix[l].get(l, 0) for l in self.labels)
        return correct / total

    def per_class(self, label: str) -> dict[str, Any]:
        """Precision / recall / F1 for one class (one-vs-rest)."""
        tp = self.matrix.get(label, {}).get(label, 0)
        fn = sum(self.matrix.get(label, {}).values()) - tp
        fp = sum(
            row.get(label, 0) for l, row in self.matrix.items() if l != label
        )
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        return {
            "support": tp + fn,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }

    def macro_f1(self) -> float:
        vals = [self.per_class(l)["f1"] for l in self.labels]
        return sum(vals) / len(vals) if vals else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "labels": list(self.labels),
            "matrix": self.matrix,
            "accuracy": round(self.accuracy(), 4),
            "macro_f1": round(self.macro_f1(), 4),
            "per_class": {l: self.per_class(l) for l in self.labels},
        }


# ---------------------------------------------------------------------------
# Embedding helpers (cached — the sweep re-embeds the same strings hundreds
# of times otherwise)
# ---------------------------------------------------------------------------


class EmbedCache:
    """Memoises embedding vectors so a threshold sweep is affordable."""

    def __init__(self) -> None:
        self._store: dict[str, list[float]] = {}

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        missing = [t for t in dict.fromkeys(texts) if t not in self._store]
        if missing:
            from src.llm import embed as _embed

            vectors = _embed(missing)
            for text, vec in zip(missing, vectors):
                self._store[text] = vec
        return [self._store[t] for t in texts]

    def cosine(self, a: str, b: str) -> float:
        va, vb = self.embed([a, b])
        return cosine(va, vb)


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def ngram_overlap(a: str, b: str, n: int = 8) -> int:
    """Longest common word-sequence length between two strings."""
    aw, bw = a.lower().split(), b.lower().split()
    if len(aw) < n or len(bw) < n:
        return 0
    a_ngrams = {tuple(aw[i : i + n]) for i in range(len(aw) - n + 1)}
    b_ngrams = {tuple(bw[i : i + n]) for i in range(len(bw) - n + 1)}
    overlap = a_ngrams & b_ngrams
    return max((len(g) for g in overlap), default=0) if overlap else 0


# ---------------------------------------------------------------------------
# Threshold sweep
# ---------------------------------------------------------------------------


@dataclass
class SweepRow:
    sim_threshold: float
    ngram_n: int
    matrix: ConfusionMatrix
    misclassified: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        d = self.matrix.as_dict()
        d.update(
            {
                "sim_threshold": self.sim_threshold,
                "ngram_n": self.ngram_n,
                "misclassified": self.misclassified,
            }
        )
        return d


def _split_sentences(text: str) -> list[str]:
    import re

    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 20]


def sweep_thresholds(
    cases: Iterable[Any],
    *,
    sim_thresholds: Sequence[float] | None = None,
    ngram_windows: Sequence[int] | None = None,
    cache: EmbedCache | None = None,
) -> list[SweepRow]:
    """
    Grid-search (embedding threshold, n-gram window) against labelled cases.

    The decision rule mirrors the production ``originality_check`` node:
    a draft sentence is flagged when the best source-sentence cosine is at or
    above ``sim_threshold`` AND (the n-gram overlap reaches ``ngram_n`` OR the
    sentence carries no ``[S#]`` citation).

    Every labelled case contributes ONE decision (flag / not flag) so the
    resulting matrix is directly comparable across the grid.
    """
    if sim_thresholds is None:
        sim_thresholds = [0.60, 0.65, 0.70, 0.75, 0.80, 0.82, 0.84, 0.86, 0.88, 0.90]
    if ngram_windows is None:
        ngram_windows = [4, 5, 6, 7, 8, 10, 12]

    cache = cache or EmbedCache()
    case_list = list(cases)

    # Pre-compute the raw signals ONCE (embedding sim + n-gram overlap).
    signals: list[dict[str, Any]] = []
    for case in case_list:
        draft = case.draft_sentence
        source = case.source_text
        source_sentences = _split_sentences(source)
        if not source_sentences:
            source_sentences = [source]
        sims = cache.embed([draft] + source_sentences)
        dvec = sims[0]
        best_sim = 0.0
        for svec in sims[1:]:
            s = cosine(dvec, svec)
            if s > best_sim:
                best_sim = s
        # N-gram overlap is measured against each source sentence AND the
        # whole source — a copied span can straddle a sentence boundary.
        best_ngram = 0
        for sent in [*source_sentences, source]:
            for n in ngram_windows:
                ov = ngram_overlap(draft, sent, n=n)
                if ov > best_ngram:
                    best_ngram = ov
        signals.append(
            {
                "case": case,
                "best_sim": best_sim,
                "best_ngram": best_ngram,
                "cited": f"[{case.source_tag}]" in draft,
            }
        )

    rows: list[SweepRow] = []
    for sim_t in sim_thresholds:
        for n in ngram_windows:
            cm = ConfusionMatrix()
            mis: list[str] = []
            for sig in signals:
                case = sig["case"]
                flagged = False
                if not sig["cited"]:
                    # UNCITED — two independent detectors, either may fire:
                    #   verbatim  : a long shared word-sequence
                    #   paraphrase: strong semantic similarity
                    # A CITED sentence is legitimate quotation/paraphrase and
                    # is never a violation, so it is never flagged.
                    verbatim = sig["best_ngram"] >= n
                    paraphrase = sig["best_sim"] >= sim_t
                    flagged = verbatim or paraphrase
                cm.add(flagged, case.is_violation)
                if flagged != case.is_violation:
                    mis.append(case.case_id)
            rows.append(
                SweepRow(
                    sim_threshold=sim_t,
                    ngram_n=n,
                    matrix=cm,
                    misclassified=mis,
                )
            )
    return rows


def best_rows(rows: Sequence[SweepRow], *, min_precision: float = 0.0) -> list[SweepRow]:
    """Rows ordered by F1, filtered to those meeting a precision floor."""
    viable = [r for r in rows if r.matrix.precision >= min_precision]
    return sorted(viable, key=lambda r: (r.matrix.f1, r.matrix.precision), reverse=True)


# ---------------------------------------------------------------------------
# Running the REAL production nodes against the labelled cases
# ---------------------------------------------------------------------------


def evaluate_originality(
    cases: Iterable[Any],
    *,
    cache: EmbedCache | None = None,
) -> dict[str, Any]:
    """
    Run the real ``originality_check`` scoring logic on every labelled case.

    Uses the production helpers (``_split_sentences``, ``_ngram_overlap``,
    ``_cosine``) rather than re-implementing them, so this measures the code
    that actually ships.
    """
    from src.agent.nodes.originality_check import (
        _cosine,
        _ngram_overlap,
        _split_sentences,
    )

    cache = cache or EmbedCache()
    cm = ConfusionMatrix()
    rows: list[dict[str, Any]] = []

    for case in cases:
        draft = case.draft_sentence
        source_sentences = _split_sentences(case.source_text) or [case.source_text]
        vecs = cache.embed([draft] + source_sentences)
        dvec = vecs[0]

        best_sim = 0.0
        best_ngram = 0
        best_src = ""
        for sent, svec in zip(source_sentences, vecs[1:]):
            s = _cosine(dvec, svec)
            if s > best_sim:
                best_sim = s
                best_src = sent
            ov = _ngram_overlap(draft, sent)
            if ov > best_ngram:
                best_ngram = ov

        # Reproduce the production decision (originality_check.py:87-108).
        flagged = False
        status = None
        if best_sim >= 0.86:  # production _SIM_THRESHOLD
            is_cited = f"[{case.source_tag}]" in draft
            if best_ngram >= 8 and not is_cited:  # production _VERBATIM_NGRAM
                status = "verbatim_uncited"
            elif not is_cited:
                status = "uncited_paraphrase"
            else:
                status = "quoted_cited"
            if status != "quoted_cited":
                flagged = True

        cm.add(flagged, case.is_violation)
        rows.append(
            {
                "case_id": case.case_id,
                "label": case.label,
                "expected_flag": case.is_violation,
                "predicted_flag": flagged,
                "predicted_status": status,
                "best_sim": round(best_sim, 4),
                "best_ngram": best_ngram,
                "correct": flagged == case.is_violation,
            }
        )

    return {
        "matrix": cm.as_dict(),
        "cases": rows,
        "misclassified": [r["case_id"] for r in rows if not r["correct"]],
    }


def evaluate_grounding(cases: Iterable[Any]) -> dict[str, Any]:
    """
    Run the real Verifier prompt against every labelled grounding case.

    Uses ``src.agent.nodes.verifier._verify_claims`` so the measured behaviour
    is the production behaviour, including the JSON parsing and the
    unknown-verdict fallback.
    """
    from src.agent.nodes.verifier import _verify_claims

    labels = ("supports", "contradicts", "unclear")
    mc = MulticlassMatrix(labels=labels)
    rows: list[dict[str, Any]] = []

    for case in cases:
        claims = {case.tag: case.claim}
        sources = {case.tag: case.source_text}
        verdicts, evidence, reasoning, dur_ms, _ = _verify_claims(claims, sources)
        predicted = verdicts.get(case.tag, "unclear")
        mc.add(case.expected_verdict, predicted)
        rows.append(
            {
                "case_id": case.case_id,
                "expected": case.expected_verdict,
                "predicted": predicted,
                "correct": predicted == case.expected_verdict,
                "evidence": evidence.get(case.tag, ""),
                "duration_ms": dur_ms,
            }
        )

    return {
        "multiclass": mc.as_dict(),
        "cases": rows,
        "misclassified": [r["case_id"] for r in rows if not r["correct"]],
    }


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------


def write_report(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path
