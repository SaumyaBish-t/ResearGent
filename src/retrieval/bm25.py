"""
BM25 lexical index — the "sparse" half of hybrid retrieval.

Why BM25 specifically
---------------------
BM25 (Okapi BM25) is the workhorse lexical scoring function used by every
serious search engine for the last 30 years. It scores a document against a
query by:
  - rewarding query-term frequency in the document (TF)
  - penalizing terms that appear in many documents (IDF)
  - normalizing for document length (so long docs don't dominate)

It catches things dense embeddings miss:
  - Exact-string identifiers ("CVE-2024-3094", "ASC 606", "RAG-Sequence")
  - Acronyms and product names
  - Rare jargon and code snippets
  - Numbers and dates
  - Misspellings that an embedder smooths over silently

Persistence
-----------
We persist BOTH:
  - the `BM25Okapi` object itself (which holds the IDF + doc-length stats)
  - the parallel arrays of chunk ids + metadata + raw text

both as one pickle file per (provider, embed-model) — same naming key as the
Chroma collection. Re-ingesting the corpus replaces these files atomically.

Why pickle + not a DB? BM25 is fundamentally an in-memory structure. The
underlying scoring requires the whole corpus tokens in RAM. SQLite or
similar would just add latency for no benefit at this scale.
"""

from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from pathlib import Path

from rank_bm25 import BM25Okapi

from src.store import DB_PATH, collection_name_for_current_embedder

# Where pickled BM25 indexes live. Sibling directory to the Chroma DB.
BM25_DIR = DB_PATH.parent / "bm25_idx"

# Small English stopword list — kept minimal because BM25's IDF already
# down-weights common words. Removing too many stopwords HURTS retrieval
# on short queries ("what is the X").
_STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "and", "or", "but",
    "is", "are", "was", "were", "be", "been", "being", "this", "that", "these",
    "those", "it", "its", "as", "by", "with", "from",
}

_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-]+")


def tokenize(text: str) -> list[str]:
    """
    Lowercase + alphanumeric tokenization with hyphen/underscore preserved.

    Hyphen preservation matters for academic + technical text — "FlashAttention-2"
    should stay one token, not become {"flashattention", "2"}. Same for
    "ASC-606", "RAG-Sequence", "self-RAG".
    """
    return [t.lower() for t in _TOKEN_RE.findall(text) if t.lower() not in _STOPWORDS]


# ---------------------------------------------------------------------------
# On-disk format
# ---------------------------------------------------------------------------


@dataclass
class _BM25Payload:
    """What we pickle to disk. Keep flat — no nested objects."""

    bm25: BM25Okapi
    ids: list[str]           # chunk ids, parallel to bm25 internal docs
    texts: list[str]         # original chunk text, for return
    metadatas: list[dict]    # per-chunk metadata (source_file, page, etc.)


def _index_path() -> Path:
    BM25_DIR.mkdir(parents=True, exist_ok=True)
    return BM25_DIR / f"{collection_name_for_current_embedder()}.pkl"


# ---------------------------------------------------------------------------
# Build / load
# ---------------------------------------------------------------------------


def build_index(ids: list[str], texts: list[str], metadatas: list[dict]) -> None:
    """
    Rebuild the BM25 index from scratch and persist it.

    Called after a full corpus ingest. (For per-doc incremental updates we'd
    need a different data structure — BM25's IDF depends on the full corpus,
    so adding one doc cheaply isn't possible.)
    """
    if not ids:
        # Empty index — write a sentinel so callers know we tried.
        _index_path().write_bytes(pickle.dumps(None))
        return

    tokenized = [tokenize(t) for t in texts]
    bm25 = BM25Okapi(tokenized)
    payload = _BM25Payload(bm25=bm25, ids=ids, texts=texts, metadatas=metadatas)
    _index_path().write_bytes(pickle.dumps(payload))


_cache: _BM25Payload | None | object = object()  # sentinel for "not yet loaded"


def _load() -> _BM25Payload | None:
    """Lazy-load + memoize the index for this process."""
    global _cache
    if _cache is not None and not isinstance(_cache, object) or _cache is None:
        # Already loaded (either payload or explicit None)
        if not isinstance(_cache, object):
            return _cache
    p = _index_path()
    if not p.exists():
        _cache = None
        return None
    try:
        _cache = pickle.loads(p.read_bytes())
    except Exception:
        _cache = None
    return _cache  # type: ignore[return-value]


def invalidate_cache() -> None:
    """Drop the in-memory cache so the next call reloads from disk."""
    global _cache
    _cache = object()


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------


@dataclass
class BM25Hit:
    chunk_id: str
    text: str
    metadata: dict
    score: float  # raw BM25 score — NOT bounded [0,1]; only relative order is meaningful


def search(query: str, k: int = 10) -> list[BM25Hit]:
    """
    Return top-k BM25 hits for `query`. Empty list if no index built yet.

    Returns ALL chunks ranked, then slices to k — `rank_bm25` doesn't expose
    a top-k API directly, and the corpus is small enough that this is fine.
    """
    payload = _load()
    if payload is None:
        return []

    q_tokens = tokenize(query)
    if not q_tokens:
        return []

    scores = payload.bm25.get_scores(q_tokens)
    # argsort descending — top-k by score
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [
        BM25Hit(
            chunk_id=payload.ids[i],
            text=payload.texts[i],
            metadata=payload.metadatas[i],
            score=float(scores[i]),
        )
        for i in ranked
        if scores[i] > 0  # drop zero-score hits — they have no query overlap
    ]


def index_stats() -> dict:
    """Diagnostic — what's in the persisted index?"""
    p = _index_path()
    if not p.exists():
        return {"exists": False, "path": str(p)}
    payload = _load()
    if payload is None:
        return {"exists": True, "path": str(p), "chunks": 0}
    return {
        "exists": True,
        "path": str(p),
        "chunks": len(payload.ids),
        "size_kb": int(p.stat().st_size / 1024),
    }
