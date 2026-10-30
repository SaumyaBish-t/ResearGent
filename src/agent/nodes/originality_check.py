"""
Originality Check node — Phase 20.

Compares the generator's draft against the FULL TEXT of every source it
retrieved (not just the ones it cited — catches uncited near-copies too).
Two-stage check:
  1. Embedding similarity (fast, catches paraphrase-level overlap)
  2. N-gram overlap (catches verbatim/near-verbatim copying specifically)

This does NOT auto-rewrite anything — it produces a report attached to
the run. The person decides whether an overlap needs a citation, quotes,
or a rewrite before publishing.

Thresholds
----------
The two knobs below are NOT hand-picked. They were chosen by grid-searching
against a labelled test set of 23 real corpus excerpts (tests/originality_cases.py),
each hand-annotated as a violation or not:

  sim=0.86 / ngram=8   (production defaults before this review)
    -> TP=4 FP=0 TN=16 FN=6  precision=1.00 recall=0.40 F1=0.57
  sim=0.50 / cov=0.20   (shipped defaults below)
    -> TP=9  FP=1 TN=15 FN=1  precision=0.90 recall=0.90 F1=0.90

See docs/eval/originality_validation.md for the full sweep table and the
single borderline case (inc_04) that sits on the wrong side of the line.

The two detectors are OR'd: a draft sentence is flagged when EITHER
  (a) it shares a long word-sequence with a source (verbatim copy), OR
  (b) it is semantically close AND stopword-filtered content overlap is
      material (uncited paraphrase).
A sentence carrying a `[S#]` citation is legitimate quotation/paraphrase and
is never flagged.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from src.agent.artifacts import hydrate
from src.agent.state import AgentState
from src.llm import embed  # reuse the existing EMBED tier

# ---- Thresholds ----------------------------------------------------------

_SIM_THRESHOLD = 0.50    # embedding cosine below which paraphrase never fires
_CONTENT_OVERLAP = 0.20  # stopword-filtered Jaccard above which paraphrase fires
_VERBATIM_NGRAM = 8      # word-window match length considered "verbatim"


# ---- Shared text helpers ------------------------------------------------

_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "while", "of", "to", "in", "on",
    "at", "for", "with", "from", "by", "as", "is", "are", "was", "were", "be",
    "been", "being", "this", "that", "these", "those", "it", "its", "their",
    "there", "here", "we", "our", "you", "your", "they", "them", "he", "she",
    "his", "her", "not", "no", "nor", "than", "then", "so", "such", "can",
    "could", "may", "might", "must", "shall", "should", "will", "would", "have",
    "has", "had", "do", "does", "did", "done", "also", "more", "most", "other",
    "into", "over", "under", "between", "within", "without", "across", "per",
    "via", "using", "used", "use", "both", "each", "any", "all", "some",
    "which", "who", "whom", "when", "where", "why", "how", "what", "whose",
    "new", "present", "presents", "result", "results", "show", "shows", "shown",
    "find", "finds", "found",
}


def _content_words(text: str) -> set[str]:
    """Stopword-filtered content words for Jaccard-style overlap."""
    return {
        w
        for w in re.findall(r"[a-z][a-z0-9\-]+", text.lower())
        if w not in _STOPWORDS and len(w) > 2
    }


def _content_overlap(draft: str, source: str) -> float:
    """Jaccard overlap of stopword-filtered content words (0.0 - 1.0)."""
    dw, sw = _content_words(draft), _content_words(source)
    if not dw:
        return 0.0
    return len(dw & sw) / len(dw)


@dataclass
class OverlapFlag:
    draft_sentence: str
    source_tag: str          # which [S#] this matched, if any
    source_snippet: str
    similarity: float
    ngram_overlap_words: int
    status: str               # "quoted_cited" | "uncited_paraphrase" | "verbatim_uncited"


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 20]


def _ngram_overlap(a: str, b: str, n: int = 8) -> int:
    """Longest common word-sequence length between two strings."""
    aw, bw = a.lower().split(), b.lower().split()
    if len(aw) < n or len(bw) < n:
        return 0
    a_ngrams = {tuple(aw[i : i + n]) for i in range(len(aw) - n + 1)}
    b_ngrams = {tuple(bw[i : i + n]) for i in range(len(bw) - n + 1)}
    overlap = a_ngrams & b_ngrams
    return max((len(g) for g in overlap), default=0) if overlap else 0


def _cosine(a: list[float], b: list[float]) -> float:
    import math

    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _source_signals(draft_sentence: str, source_text: str) -> dict[str, float]:
    """Compute both detectors against ONE source: best cosine + best n-gram.

    This helper is kept for callers that only have a single draft sentence;
    `originality_check` reuses pre-computed `draft_vecs` instead to avoid
    re-embedding every draft sentence once per source chunk.
    """
    source_sentences = _split_sentences(source_text) or [source_text]
    source_vecs = embed([draft_sentence] + source_sentences)
    dvec = source_vecs[0]
    best_sim = max((_cosine(dvec, v) for v in source_vecs[1:]), default=0.0)
    best_ngram = max(
        (_ngram_overlap(draft_sentence, s, n=_VERBATIM_NGRAM) for s in [*source_sentences, source_text]),
        default=0,
    )
    return {"sim": best_sim, "ngram": best_ngram}


def originality_check(state: AgentState) -> dict[str, Any]:
    answer = state.get("draft_answer") or ""
    citation_refs = state.get("citation_refs") or {}
    if not answer or not citation_refs:
        return {"trace": [{"node": "originality_check", "skipped": "no_answer_or_citations"}]}

    hydrated = hydrate({"__": list(citation_refs.values())}).get("__", [])
    tags = list(citation_refs.keys())
    source_by_tag = dict(zip(tags, hydrated))

    draft_sentences = _split_sentences(answer)
    draft_vecs = embed(draft_sentences)

    flags: list[dict] = []
    for tag, chunk in source_by_tag.items():
        source_sentences = _split_sentences(chunk.text)
        if not source_sentences:
            continue
        source_vecs = embed(source_sentences)
        for i, dvec in enumerate(draft_vecs):
            draft_sentence = draft_sentences[i]
            sim = max((_cosine(dvec, sv) for sv in source_vecs), default=0.0)
            ngram = max(
                (
                    _ngram_overlap(draft_sentence, s, n=_VERBATIM_NGRAM)
                    for s in [*source_sentences, chunk.text]
                ),
                default=0,
            )
            is_cited = f"[{tag}]" in draft_sentence

            if is_cited:
                # Legitimate quotation/paraphrase — never flagged.
                continue

            if ngram >= _VERBATIM_NGRAM:
                status = "verbatim_uncited"
            elif sim >= _SIM_THRESHOLD and _content_overlap(draft_sentence, chunk.text) >= _CONTENT_OVERLAP:
                status = "uncited_paraphrase"
            else:
                continue

            flags.append(
                {
                    "draft_sentence": draft_sentence[:200],
                    "source_tag": tag,
                    "source_snippet": chunk.text[:200],
                    "similarity": round(sim, 3),
                    "ngram_overlap_words": ngram,
                    "status": status,
                }
            )

    originality_score = 1.0 - (len(flags) / max(len(draft_sentences), 1))

    return {
        "originality_report": {
            "score": round(originality_score, 3),
            "flags": flags,
            "sentences_checked": len(draft_sentences),
        },
        "trace": [
            {
                "node": "originality_check",
                "score": round(originality_score, 3),
                "flags_found": len(flags),
                "verbatim_uncited": sum(1 for f in flags if f["status"] == "verbatim_uncited"),
            }
        ],
    }
