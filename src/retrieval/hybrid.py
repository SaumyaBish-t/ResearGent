"""
Hybrid retrieval: dense + BM25 fused with Reciprocal Rank Fusion.

Why RRF and not weighted score fusion?
--------------------------------------
The naive approach to combining two retrievers is `final_score = α * dense + β * bm25`.
This fails because dense scores (cosine similarity) live in [0, 1] but BM25
scores are unbounded and depend on corpus statistics. Normalizing them is
brittle — any normalization scheme breaks on different corpora.

RRF sidesteps this entirely by using ONLY ranks, not scores:

    score(d) = Σ over rankings:  1 / (k + rank_i(d))

where k is a constant (60 in the original paper, also the default in
Elasticsearch, OpenSearch, Vespa). A document ranked #1 in dense and #2 in
BM25 gets `1/61 + 1/62 = 0.0325`. A document ranked #1 in only dense and
unranked in BM25 gets `1/61 = 0.0164`.

This means a document that consistently ranks well across BOTH retrievers
beats one that ranks great in only one. That's exactly the behavior we want
when fusing semantic + lexical signals.

References:
    Cormack et al., "Reciprocal Rank Fusion outperforms Condorcet and
    individual Rank Learning Methods", SIGIR 2009.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from src.retrieval import bm25 as bm25_idx
from src.retrieval.naive import RetrievedChunk, naive_retrieve

# RRF damping constant. 60 = standard from the original 2009 paper.
RRF_K = 60


@dataclass
class HybridChunk:
    """Returned chunk with provenance — knows where each ranker placed it."""

    text: str
    source_file: str
    page_number: int
    chunk_index: int
    rrf_score: float
    dense_rank: int | None       # 1-based; None if not in dense top-k
    bm25_rank: int | None        # 1-based; None if not in BM25 top-k
    dense_score: float | None    # cosine similarity if ranked
    bm25_score: float | None     # raw BM25 score if ranked
    doc_title: str = ""
    # Knowledge-graph fields — only populated for vault chunks. Used by
    # graph-expansion retrieval (Phase 10) to walk wikilink edges and
    # surface structurally-related context the embedder might have ranked low.
    wikilinks: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    # Chroma id — propagated from the dense-side result. Used by the
    # agent's pointer-based state so a HybridChunk can ride through a
    # checkpoint as an 80-byte ChunkRef instead of its full text.
    chroma_id: str = ""

    @property
    def citation(self) -> str:
        return f"{self.source_file} p.{self.page_number}"

    @property
    def signal(self) -> str:
        """Short label showing which retriever(s) surfaced this chunk."""
        if self.dense_rank and self.bm25_rank:
            return "BOTH"
        if self.dense_rank:
            return "dense"
        return "bm25"


def hybrid_retrieve(
    query: str,
    *,
    k: int = 5,
    pool_size: int | None = None,
    doc_ids: list[str] | None = None,
) -> list[HybridChunk]:
    """
    Run dense + BM25 in parallel-ish, fuse with RRF, return top-k.

    `pool_size` controls how many candidates each retriever returns BEFORE
    fusion. A larger pool gives RRF more documents to consider — useful when
    one retriever surfaces a great candidate that the other ranks low. Default
    `pool_size = 2*k` strikes a good balance for academic corpora.
    """
    pool = pool_size or max(20, k * 4)

    # ---- Retrieve from both sides ----
    # `doc_ids` is honoured by the dense side via Chroma metadata filtering.
    # BM25 doesn't speak that filter today, so we filter its hits post-hoc.
    # Same end result; small efficiency penalty (BM25 still scans the full
    # index). Worth tightening only if doc-scoped queries become hot.
    dense_hits = naive_retrieve(query, k=pool, doc_ids=doc_ids)
    bm25_hits = bm25_idx.search(query, k=pool)
    if doc_ids:
        allowed = set(doc_ids)
        bm25_hits = [h for h in bm25_hits if h.metadata.get("doc_id") in allowed]

    # ---- Index by chunk id (= "<doc_id>:<chunk_index>") for join ----
    # The naive retriever returns RetrievedChunk without an explicit id, so we
    # reconstruct it the same way the ingest pipeline does: doc_id is the
    # prefix on the Chroma id — we don't have it directly here, so we fall
    # back to (source_file, chunk_index) which is also unique.
    def _key_dense(c: RetrievedChunk) -> str:
        return f"{c.source_file}::{c.chunk_index}"

    def _key_bm25(h) -> str:
        m = h.metadata
        return f"{m.get('source_file','?')}::{m.get('chunk_index', -1)}"

    dense_ranks: dict[str, tuple[int, RetrievedChunk]] = {
        _key_dense(c): (rank, c) for rank, c in enumerate(dense_hits, start=1)
    }
    bm25_ranks: dict[str, tuple[int, "bm25_idx.BM25Hit"]] = {
        _key_bm25(h): (rank, h) for rank, h in enumerate(bm25_hits, start=1)
    }

    # ---- RRF accumulate ----
    scores: dict[str, float] = defaultdict(float)
    for key, (rank, _) in dense_ranks.items():
        scores[key] += 1.0 / (RRF_K + rank)
    for key, (rank, _) in bm25_ranks.items():
        scores[key] += 1.0 / (RRF_K + rank)

    # ---- Materialize top-k as HybridChunk records ----
    ranked_keys = sorted(scores.keys(), key=lambda k_: scores[k_], reverse=True)[:k]

    out: list[HybridChunk] = []
    for key in ranked_keys:
        d = dense_ranks.get(key)
        b = bm25_ranks.get(key)

        # Pull canonical fields from whichever side has the chunk.
        # Wikilinks + tags come from whichever path knows them: dense already
        # extracted them (see naive.py), bm25 has them in raw metadata.
        wikilinks: list[str] = []
        tags: list[str] = []
        chroma_id = ""
        if d is not None:
            _, dc = d
            text = dc.text
            source_file = dc.source_file
            page_number = dc.page_number
            chunk_index = dc.chunk_index
            doc_title = dc.doc_title
            wikilinks = list(dc.wikilinks)
            tags = list(dc.tags)
            chroma_id = dc.chroma_id
        else:
            assert b is not None
            _, bh = b
            text = bh.text
            m = bh.metadata
            source_file = m.get("source_file", "?")
            page_number = int(m.get("page_number", 0))
            chunk_index = int(m.get("chunk_index", 0))
            doc_title = m.get("doc_title", "") or ""
            wl_raw = m.get("wikilinks", "") or ""
            tg_raw = m.get("tags", "") or ""
            wikilinks = [w.strip() for w in wl_raw.split(",") if w.strip()]
            tags = [t.strip() for t in tg_raw.split(",") if t.strip()]
            # BM25 stores the Chroma id as `chunk_id` (set at ingest time).
            chroma_id = getattr(bh, "chunk_id", "") or ""

        out.append(
            HybridChunk(
                text=text,
                source_file=source_file,
                page_number=page_number,
                chunk_index=chunk_index,
                rrf_score=scores[key],
                dense_rank=d[0] if d else None,
                bm25_rank=b[0] if b else None,
                dense_score=d[1].score if d else None,
                bm25_score=b[1].score if b else None,
                doc_title=doc_title,
                wikilinks=wikilinks,
                tags=tags,
                chroma_id=chroma_id,
            )
        )
    return out
