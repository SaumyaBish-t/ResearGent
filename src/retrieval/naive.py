"""
Naive (dense-only) retrieval.

This is the *baseline*. In Phase 2 we'll add BM25 + RRF and benchmark against
this. Keeping it simple here is intentional — when hybrid retrieval wins on
your eval set, you'll see exactly how much it wins by.

NVIDIA NIM embeddings expect `input_type="query"` at query time (vs `"passage"`
at ingest time). The `embed()` helper currently defaults to "passage" — for
queries we pass `extra_body` ourselves via the lower-level client. This is a
small NVIDIA-specific wart; other providers ignore the field.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.config import ModelTier, settings
from src.llm import get_client
from src.store import get_or_create_papers_collection


@dataclass
class RetrievedChunk:
    text: str
    source_file: str
    page_number: int
    chunk_index: int
    score: float  # cosine similarity in [0, 1] (1 = identical)
    doc_title: str = ""
    # Knowledge-graph metadata (populated only when the chunk came from a
    # vault note that had wikilinks / tags ingested). Empty for PDF chunks.
    wikilinks: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    # Chroma id of the underlying chunk (`<content_hash>:<chunk_index>`).
    # Populated by `naive_retrieve` from the query result so pointer-based
    # state (see src/agent/artifacts.py) can rehydrate this chunk from
    # Chroma without ever putting `text` into the LangGraph checkpoint.
    chroma_id: str = ""

    @property
    def citation(self) -> str:
        return f"{self.source_file} p.{self.page_number}"


def _embed_query(query: str) -> list[float]:
    """Embed a single query with provider-appropriate input_type hint."""
    provider = settings.resolve_provider(ModelTier.EMBED)
    client = get_client(provider)
    model = (
        settings.nvidia_model_embed
        if provider == "nvidia"
        else settings.ollama_model_embed
    )

    extra_body = {"input_type": "query", "truncate": "END"} if provider == "nvidia" else None
    resp = client.embeddings.create(model=model, input=[query], extra_body=extra_body)
    return resp.data[0].embedding


def naive_retrieve(
    query: str,
    *,
    k: int = 5,
    doc_ids: list[str] | None = None,
    domains: list[str] | None = None,
) -> list[RetrievedChunk]:
    """
    Top-k dense retrieval from the currently active papers collection.

    `doc_ids` scopes the search to a specific set of registry doc_ids —
    e.g. "search only the user's uploaded PDFs", "search only notes
    tagged #research". The doc_ids come from the Postgres registry
    (see `src.registry.get_doc_ids_for_filter`). When None (the default),
    the entire corpus is searched.

    `domains` (Phase 15) scopes by the `domain` metadata stamped at ingest
    — e.g. ["quant_finance"] or ["agentic_ai", "time_series"]. ANDed with
    doc_ids when both are supplied (Chroma's `$and` operator).
    """
    col = get_or_create_papers_collection()
    if col.count() == 0:
        return []

    qvec = _embed_query(query)

    # Compose the Chroma `where` clause. Chroma's filter DSL requires an
    # explicit `$and` when combining multiple top-level keys, so we keep
    # the no-filter and single-filter cases on cheap fast paths.
    clauses: list[dict] = []
    if doc_ids:
        clauses.append({"doc_id": {"$in": doc_ids}})
    if domains:
        clauses.append({"domain": {"$in": domains}})

    if not clauses:
        where = None
    elif len(clauses) == 1:
        where = clauses[0]
    else:
        where = {"$and": clauses}

    res = col.query(
        query_embeddings=[qvec],
        n_results=k,
        include=["documents", "metadatas", "distances"],
        where=where,
    )

    ids = res["ids"][0] if res.get("ids") else []
    docs = res["documents"][0] if res.get("documents") else []
    metas = res["metadatas"][0] if res.get("metadatas") else []
    dists = res["distances"][0] if res.get("distances") else []

    out: list[RetrievedChunk] = []
    for chroma_id, doc, meta, dist in zip(ids, docs, metas, dists):
        # Chroma returns COSINE DISTANCE (1 - cos_sim) when space is "cosine".
        score = max(0.0, 1.0 - float(dist))
        # Vault chunks store wikilinks + tags as comma-joined strings — split
        # back into lists so downstream code (graph expansion, citation
        # rendering) sees structured data.
        wl_raw = meta.get("wikilinks", "") or ""
        tg_raw = meta.get("tags", "") or ""
        wikilinks = [w.strip() for w in wl_raw.split(",") if w.strip()]
        tags = [t.strip() for t in tg_raw.split(",") if t.strip()]
        out.append(
            RetrievedChunk(
                text=doc,
                source_file=meta.get("source_file", "?"),
                page_number=int(meta.get("page_number", 0)),
                chunk_index=int(meta.get("chunk_index", 0)),
                doc_title=meta.get("doc_title", "") or "",
                score=score,
                wikilinks=wikilinks,
                tags=tags,
                chroma_id=chroma_id,
            )
        )
    return out
