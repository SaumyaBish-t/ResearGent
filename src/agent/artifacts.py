"""
Pointer-based state management for chunk payloads.

Why this exists
---------------
Pre-refactor, `AgentState.chunks_by_subq` held the FULL text of every
retrieved chunk. LangGraph writes a checkpoint at every node boundary,
so a 5-sub-question × 10-chunk query carrying ~4KB per chunk consumed
~200KB per snapshot × ~10 snapshots per run ≈ 2 MB per run. A 500 MB
Postgres free tier hit its ceiling after a few hundred runs.

Fix: state holds only `ChunkRef` pointers. Nodes hydrate refs into
`HydratedChunk` instances at the start of their run, do their work in
memory, and persist refs (not chunks) back to state. Per-checkpoint
size drops from ~200KB to ~3KB — a ~70x improvement.

Storage strategy by chunk origin
--------------------------------
* `kind="local"`  — chunk already lives in ChromaDB (was ingested as
  part of the corpus). Ref is the Chroma id (`<content_hash>:<idx>`).
  Hydration is a free Chroma `get(ids=...)` lookup. No Postgres write.

* `kind="web" | "paper" | "graph"` — chunk is ephemeral, doesn't live
  in any persistent store. We write it to `agent_artifacts` as JSONB,
  ref is the row's UUID. Hydration is one indexed PG lookup.

Both kinds prune cleanly on the same TTL as checkpoints — the `db prune`
command deletes by `thread_id = ANY(stale_threads)`.

Why a unified HydratedChunk type
--------------------------------
Pre-refactor, downstream nodes consumed
`Union[HybridChunk, WebChunk, PaperChunk, GraphChunk]` and did property
access that "happened to work" because all four exposed `.text`,
`.citation`, `.doc_title`. The rank/signal fields on HybridChunk
(`rrf_score`, `dense_rank`, `bm25_rank`) were only ever read in CLI
bench output, never inside the agent. Collapsing into one type at the
hydration boundary makes the agent code uniformly typed and avoids
preserving fields nobody downstream uses.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from sqlalchemy import Column, DateTime, Index, String, func, select
from sqlalchemy.dialects.postgresql import JSONB, UUID

from src.config import settings
from src.registry import Base, session_scope


# ---------------------------------------------------------------------------
# Pointer + view types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChunkRef:
    """Tiny pointer carried in LangGraph state. ~70 bytes serialized."""

    kind: str    # "local" | "web" | "paper" | "graph"
    id: str      # Chroma chunk_id for "local", artifact UUID otherwise

    def to_dict(self) -> dict[str, str]:
        return {"kind": self.kind, "id": self.id}

    @classmethod
    def from_dict(cls, d: dict[str, str]) -> "ChunkRef":
        return cls(kind=d["kind"], id=d["id"])


@dataclass
class HydratedChunk:
    """
    What nodes see after `hydrate()`. The union of fields any downstream
    node actually USES — see module docstring for why we drop rank fields.
    """

    text: str
    citation: str
    doc_title: str = ""
    source_file: str = ""
    page_number: int = 0
    chunk_index: int = -1
    signal: str = ""              # "local" | "web:tavily" | "paper:arxiv" | ...
    wikilinks: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    url: str = ""                 # set for web/paper, empty for local


# ---------------------------------------------------------------------------
# Postgres table for non-Chroma chunks (web / paper / graph)
# ---------------------------------------------------------------------------


class AgentArtifact(Base):
    """One row per ephemeral chunk (web search result, paper abstract, ...)."""

    __tablename__ = "agent_artifacts"

    artifact_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    # Mirrors LangGraph's checkpoint thread_id so the TTL pruner can drop
    # artifacts in lockstep with checkpoints. Indexed for the prune scan.
    thread_id = Column(String(128), nullable=False, index=True)
    kind = Column(String(16), nullable=False)     # "web" | "paper" | "graph"
    payload = Column(JSONB, nullable=False)       # serialized HydratedChunk
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        index=True,
    )

    # Composite index — the prune query is "DELETE WHERE thread_id IN (...)";
    # this lets it skip the heap entirely.
    __table_args__ = (
        Index("agent_artifacts_thread_created_idx", "thread_id", "created_at"),
    )


# ---------------------------------------------------------------------------
# Local-chunk hydration (Chroma)
# ---------------------------------------------------------------------------
# Pulled out so the dependency on `src.store` stays lazy — keeps the
# import graph cheap when only the table model is wanted (e.g. db init).


def _hydrate_local(ids: list[str]) -> dict[str, HydratedChunk]:
    """Batch-fetch local chunks from Chroma. Missing ids are silently skipped."""
    if not ids:
        return {}
    from src.store import get_or_create_papers_collection

    col = get_or_create_papers_collection()
    res = col.get_by_ids(ids)
    full_ids = res.get("ids") or []
    docs = res.get("documents") or []
    metas = res.get("metadatas") or []

    out: dict[str, HydratedChunk] = {}
    for cid, doc, meta in zip(full_ids, docs, metas):
        wl_raw = meta.get("wikilinks", "") or ""
        tg_raw = meta.get("tags", "") or ""
        out[cid] = HydratedChunk(
            text=doc,
            citation=f"{meta.get('source_file','?')} p.{meta.get('page_number',0)}",
            doc_title=meta.get("doc_title", "") or "",
            source_file=meta.get("source_file", "?"),
            page_number=int(meta.get("page_number", 0)),
            chunk_index=int(meta.get("chunk_index", 0)),
            signal="local",
            wikilinks=[w.strip() for w in wl_raw.split(",") if w.strip()],
            tags=[t.strip() for t in tg_raw.split(",") if t.strip()],
        )
    return out


# ---------------------------------------------------------------------------
# Hydrate + persist — what the nodes call
# ---------------------------------------------------------------------------


def hydrate(refs_by_subq: dict[str, list[Any]]) -> dict[str, list[HydratedChunk]]:
    """
    Turn `{sub_q: [ChunkRef|dict, ...]}` into `{sub_q: [HydratedChunk, ...]}`.

    Accepts dict-form refs (LangGraph round-trips dataclasses as dicts when
    serializing through Postgres) as well as the dataclass form.

    Strategy:
      1. Collect ids by kind across all sub-questions
      2. Batch-fetch local chunks from Chroma (one call)
      3. Batch-fetch ephemeral chunks from Postgres (one query)
      4. Re-assemble preserving the original per-sub-question ordering

    Two queries total per node call, regardless of how many chunks.
    """
    if not refs_by_subq:
        return {}

    flat: list[ChunkRef] = []
    for refs in refs_by_subq.values():
        for r in refs or []:
            if isinstance(r, dict):
                r = ChunkRef.from_dict(r)
            flat.append(r)

    local_ids = [r.id for r in flat if r.kind == "local"]
    ephemeral_ids = [r.id for r in flat if r.kind != "local"]

    local_map = _hydrate_local(local_ids) if local_ids else {}
    ephemeral_map: dict[str, HydratedChunk] = {}
    if ephemeral_ids and settings.resolve_database_url():
        with session_scope() as s:
            ids_uuid = [uuid.UUID(i) for i in ephemeral_ids]
            stmt = select(AgentArtifact).where(AgentArtifact.artifact_id.in_(ids_uuid))
            for row in s.execute(stmt).scalars():
                ephemeral_map[str(row.artifact_id)] = HydratedChunk(**row.payload)

    out: dict[str, list[HydratedChunk]] = {}
    for sq, refs in refs_by_subq.items():
        bucket: list[HydratedChunk] = []
        for r in refs or []:
            if isinstance(r, dict):
                r = ChunkRef.from_dict(r)
            if r.kind == "local":
                hc = local_map.get(r.id)
            else:
                hc = ephemeral_map.get(r.id)
            if hc is not None:
                bucket.append(hc)
        out[sq] = bucket
    return out


def hydrate_one(refs: Iterable[Any]) -> list[HydratedChunk]:
    """Convenience: hydrate a flat list of refs (e.g. citation_refs)."""
    bucket = list(refs or [])
    return hydrate({"_": bucket}).get("_", [])


# ---------------------------------------------------------------------------
# Persist (dehydrate)
# ---------------------------------------------------------------------------


def persist_local(chroma_ids: list[str]) -> list[ChunkRef]:
    """Build refs for chunks that already live in Chroma. Pure transform."""
    return [ChunkRef(kind="local", id=cid) for cid in chroma_ids]


def _classify_chunk(c: Any) -> str:
    """Identify a chunk's origin without taking an import dependency on it."""
    name = type(c).__name__
    if name == "HybridChunk":
        return "local"
    if name == "WebChunk":
        return "web"
    if name == "PaperChunk":
        return "paper"
    if name == "GraphChunk":
        return "graph"
    if name == "HydratedChunk":
        # Allow already-hydrated input (e.g. critic dropping irrelevants).
        # The hydrated chunk's `signal` tells us where it came from.
        sig = (c.signal or "").split(":", 1)[0]
        return sig if sig in {"local", "web", "paper", "graph"} else "local"
    raise TypeError(f"Don't know how to persist chunk of type {name}")


def _to_hydrated(c: Any) -> HydratedChunk:
    """Flatten any chunk type into the storage view."""
    name = type(c).__name__
    if name == "HydratedChunk":
        return c
    if name == "HybridChunk":
        return HydratedChunk(
            text=c.text, citation=c.citation, doc_title=c.doc_title,
            source_file=c.source_file, page_number=c.page_number,
            chunk_index=c.chunk_index, signal="local",
            wikilinks=list(c.wikilinks), tags=list(c.tags),
        )
    if name == "WebChunk":
        return HydratedChunk(
            text=c.text, citation=c.citation, doc_title=c.title,
            source_file=c.url, signal=c.signal, url=c.url,
        )
    if name == "PaperChunk":
        return HydratedChunk(
            text=c.text, citation=c.citation, doc_title=c.doc_title,
            source_file=c.url or c.title, signal=c.signal,
            url=c.url or c.pdf_url,
        )
    if name == "GraphChunk":
        return HydratedChunk(
            text=c.text, citation=c.citation, doc_title=c.doc_title,
            source_file=c.source_file, page_number=c.page_number,
            chunk_index=c.chunk_index, signal="graph",
            wikilinks=list(c.wikilinks), tags=list(c.tags),
        )
    raise TypeError(f"Don't know how to flatten chunk of type {name}")


def persist_mixed(
    thread_id: str,
    chunks_by_subq: dict[str, list[Any]],
) -> dict[str, list[dict[str, str]]]:
    """
    Dehydrate a `{sub_q: [chunks...]}` dict for storage in agent state.

    Local chunks (anything carrying a `chroma_id`) become free refs.
    Everything else gets written to `agent_artifacts` in one batched
    transaction so a multi-sub-Q persist is exactly one PG round trip.

    Returns the same shape with ChunkRef-as-dict in place of chunks —
    that dict form is what LangGraph's serializer round-trips cleanly.
    """
    if not chunks_by_subq:
        return {}

    # Pass 1: route each chunk + collect ephemeral payloads.
    # `placements` records (sub_q, position_in_bucket) so we can stitch
    # newly-minted ephemeral ids back into the right slot post-insert.
    refs_out: dict[str, list[dict[str, str] | None]] = {
        sq: [None] * len(chunks or []) for sq, chunks in chunks_by_subq.items()
    }
    ephemeral_jobs: list[tuple[str, int, str, HydratedChunk]] = []

    for sq, chunks in chunks_by_subq.items():
        for i, c in enumerate(chunks or []):
            kind = _classify_chunk(c)
            if kind == "local":
                cid = getattr(c, "chroma_id", "") or ""
                if not cid:
                    # Defensive: HybridChunk without chroma_id (shouldn't
                    # happen post-Phase-13) — fall back to ephemeral so
                    # the chunk doesn't silently disappear.
                    ephemeral_jobs.append((sq, i, "local", _to_hydrated(c)))
                else:
                    refs_out[sq][i] = ChunkRef(kind="local", id=cid).to_dict()
            else:
                ephemeral_jobs.append((sq, i, kind, _to_hydrated(c)))

    # Pass 2: batch-insert ephemeral chunks if any. One transaction.
    if ephemeral_jobs:
        if not settings.resolve_database_url():
            raise RuntimeError(
                "Pointer-based state requires Postgres for ephemeral chunks. "
                "Set DATABASE_URL or disable web/paper/graph fallbacks."
            )
        with session_scope() as s:
            for sq, idx, kind, hc in ephemeral_jobs:
                row = AgentArtifact(
                    thread_id=thread_id,
                    kind=kind,
                    payload=asdict(hc),
                )
                s.add(row)
                s.flush()
                refs_out[sq][idx] = ChunkRef(kind=kind, id=str(row.artifact_id)).to_dict()

    # `None` slots can only remain if a chunk failed classification AND
    # ephemeral persistence both — by contract that doesn't happen. Filter
    # defensively so consumers see a clean list.
    return {sq: [r for r in bucket if r is not None] for sq, bucket in refs_out.items()}


def persist_ephemeral(
    *,
    thread_id: str,
    kind: str,
    chunks: list[HydratedChunk],
) -> list[ChunkRef]:
    """
    Write web/paper/graph chunks to `agent_artifacts`, return refs.

    Idempotency: we always insert fresh rows. Same chunk seen twice
    becomes two artifacts — costly, but agent nodes don't re-persist
    the same chunk twice in practice (web_fallback runs once, paper
    discovery runs once). If this ever loops, add a (thread_id,
    content_hash) UNIQUE constraint.
    """
    if not chunks:
        return []
    if not settings.resolve_database_url():
        # No PG configured — fall back to in-memory refs. This makes
        # MemorySaver runs still work; the chunks live in the state
        # blob like before. Acceptable for local dev.
        raise RuntimeError(
            "persist_ephemeral requires Postgres; called without DATABASE_URL"
        )

    refs: list[ChunkRef] = []
    with session_scope() as s:
        for c in chunks:
            payload = asdict(c)
            row = AgentArtifact(
                thread_id=thread_id,
                kind=kind,
                payload=payload,
            )
            s.add(row)
            s.flush()
            refs.append(ChunkRef(kind=kind, id=str(row.artifact_id)))
    return refs


# ---------------------------------------------------------------------------
# TTL pruner hook
# ---------------------------------------------------------------------------


def prune_artifacts_for_threads(stale_threads: list[str]) -> int:
    """Delete artifacts for the given thread_ids. Returns row count."""
    if not stale_threads:
        return 0
    from src.db import connection

    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            "DELETE FROM agent_artifacts WHERE thread_id = ANY(%s)",
            (stale_threads,),
        )
        return cur.rowcount or 0
