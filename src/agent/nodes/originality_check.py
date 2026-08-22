"""
Originality Check node — Phase 20.

Compares the generator's draft against the FULL TEXT of every source it
retrieved (not just the ones it cited — catches uncited near-copies too).
Two-stage check:
  1. Embedding similarity (fast, catches paraphrase-level overlap)
  2. N-gram overlap on any chunk pair above the similarity threshold
     (catches verbatim/near-verbatim copying specifically)

This does NOT auto-rewrite anything — it produces a report attached to
the run. The person decides whether an overlap needs a citation, quotes,
or a rewrite before publishing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.agent.artifacts import HydratedChunk, hydrate
from src.agent.state import AgentState
from src.llm import embed  # reuse the existing EMBED tier


@dataclass
class OverlapFlag:
    draft_sentence: str
    source_tag: str          # which [S#] this matched, if any
    source_snippet: str
    similarity: float
    ngram_overlap_words: int
    status: str               # "quoted_cited" | "uncited_paraphrase" | "verbatim_uncited"


def _split_sentences(text: str) -> list[str]:
    import re
    # Reuse whatever sentence splitter the repo already has (check
    # src/agent/nodes/verifier.py's `_extract_tagged_claims` — it does
    # ad-hoc sentence-boundary detection with the same separator list;
    # factor that logic into a shared `src/agent/text_utils.py` helper
    # used by BOTH verifier.py and this module, rather than duplicating it).
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip()) > 20]


def _ngram_overlap(a: str, b: str, n: int = 8) -> int:
    """Longest common word-sequence length between two strings, capped check."""
    aw, bw = a.lower().split(), b.lower().split()
    a_ngrams = {tuple(aw[i:i+n]) for i in range(len(aw) - n + 1)}
    b_ngrams = {tuple(bw[i:i+n]) for i in range(len(bw) - n + 1)}
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


_SIM_THRESHOLD = 0.86   # embedding cosine similarity to flag a candidate pair
_VERBATIM_NGRAM = 8     # word-window match length considered "verbatim"


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
            for j, svec in enumerate(source_vecs):
                sim = _cosine(dvec, svec)
                if sim < _SIM_THRESHOLD:
                    continue
                ngram = _ngram_overlap(draft_sentences[i], source_sentences[j], n=_VERBATIM_NGRAM)
                is_cited = f"[{tag}]" in draft_sentences[i]
                if ngram >= _VERBATIM_NGRAM and not is_cited:
                    status = "verbatim_uncited"
                elif not is_cited:
                    status = "uncited_paraphrase"
                else:
                    status = "quoted_cited"
                if status != "quoted_cited":
                    flags.append({
                        "draft_sentence": draft_sentences[i][:200],
                        "source_tag": tag,
                        "source_snippet": source_sentences[j][:200],
                        "similarity": round(sim, 3),
                        "ngram_overlap_words": ngram,
                        "status": status,
                    })

    originality_score = 1.0 - (len(flags) / max(len(draft_sentences), 1))

    return {
        "originality_report": {
            "score": round(originality_score, 3),
            "flags": flags,
            "sentences_checked": len(draft_sentences),
        },
        "trace": [{
            "node": "originality_check",
            "score": round(originality_score, 3),
            "flags_found": len(flags),
            "verbatim_uncited": sum(1 for f in flags if f["status"] == "verbatim_uncited"),
        }],
    }