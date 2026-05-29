"""
documents_registry — the authoritative catalog of every ingested artifact.

What goes in
------------
Every PDF and every markdown note that survives ingestion gets one row here.
The row is the source of truth for:

  * which doc_ids exist (and therefore which ChromaDB metadata filters are
    valid — see the Retriever node)
  * where the raw bytes live (local path today, S3 URL when we move there)
  * the cheap stuff you want to surface in a UI without parsing PDFs again
    (filename, byte size, title, ingest timestamp)

What doesn't
------------
NO full text. NO chunk contents. NO embeddings. Those live in ChromaDB.
This table stays under a megabyte even with thousands of documents — it
holds pointers, not payloads.

Why SQLAlchemy here but raw psycopg in db.py / main.py
------------------------------------------------------
SQLAlchemy buys us declarative schema migrations (`Base.metadata.create_all`),
typed model objects for the few places we want them (the ingest pipeline),
and easy testability. For the LangGraph checkpoint pruner — pure DDL/DML
against the saver's own tables — raw psycopg is leaner. Mixing the two
intentionally: model layer for OUR tables, raw layer for borrowed tables.
"""

from __future__ import annotations

import datetime as _dt
import uuid
from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    String,
    Text,
    create_engine,
    select,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from src.config import settings


class Base(DeclarativeBase):
    """SQLAlchemy declarative base. One Base per logical schema family."""


class DocumentRecord(Base):
    """One row per ingested document (PDF or .md note)."""

    __tablename__ = "documents_registry"

    # UUID rather than the ingest-time hash so we never collide if two
    # different files happen to share a hash prefix. The hash is preserved
    # separately as `content_hash` for dedup detection.
    doc_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Stable hash of the file bytes — used to detect "you re-ingested the
    # same PDF" and skip work without scanning ChromaDB metadata.
    content_hash = Column(String(64), nullable=False, index=True, unique=True)

    # Display fields. `filename` is the basename, `title` is the parsed
    # title (PDF metadata or first H1 in a note); either can be None.
    filename = Column(String(512), nullable=False)
    title = Column(String(1024), nullable=True)

    # "pdf" | "note" | future formats. Makes Retriever's doc_id filters
    # type-aware (e.g. "search only my notes for this query").
    source_type = Column(String(32), nullable=False, default="pdf")

    # Raw bytes location. Local disk path today (`data/storage/<doc_id>.pdf`),
    # ready to swap for an S3 URL or signed-URL fetcher later — the column
    # name + nullable contract intentionally accommodates either.
    file_storage_url = Column(Text, nullable=False)

    # Byte size of the original file. Surfaces "how much disk is this
    # corpus taking" without listdir() loops.
    file_size = Column(BigInteger, nullable=False, default=0)

    # Number of chunks the ingester produced + persisted to ChromaDB.
    # Cheap sanity check: registry count of chunks * ~3KB per chunk ≈
    # corpus weight in vector store.
    chunk_count = Column(BigInteger, nullable=False, default=0)

    uploaded_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: _dt.datetime.now(_dt.timezone.utc),
    )

    # Free-form bag for source-type-specific metadata that doesn't deserve
    # its own column: arxiv_id, doi, vault sub-path, frontmatter, etc.
    extra = Column(JSONB, nullable=False, default=dict)


# ---------------------------------------------------------------------------
# Engine + session — share the pool indirectly via the libpq URL.
# ---------------------------------------------------------------------------
# We deliberately use a SEPARATE SQLAlchemy engine here rather than wrapping
# the psycopg_pool from src.db. Reasons:
#   * SQLAlchemy's own pool gives us clean Session lifecycle + automatic
#     reconnect; mixing pools is asking for connection-leak bugs.
#   * Two small pools (8 + 5 conns) is well under any free-tier limit.
#   * If pool pressure ever becomes a real concern, swap in
#     `psycopg.AsyncConnection.connect()` and wire SQLAlchemy's
#     `creator=` argument to share the db.py pool. Today we don't need it.

_engine = None
_SessionMaker: sessionmaker | None = None


def _get_engine():
    global _engine, _SessionMaker
    if _engine is None:
        url = settings.resolve_database_url()
        if not url:
            raise RuntimeError(
                "documents_registry needs Postgres. Set DATABASE_URL in .env."
            )
        # SQLAlchemy wants `postgresql+psycopg://` to pick psycopg v3.
        if url.startswith("postgresql://"):
            url = "postgresql+psycopg://" + url[len("postgresql://") :]
        _engine = create_engine(
            url,
            pool_size=5,
            max_overflow=2,
            pool_pre_ping=True,   # auto-recycle dead conns (free-tier idle kill)
            future=True,
        )
        _SessionMaker = sessionmaker(bind=_engine, expire_on_commit=False, future=True)
    return _engine


@contextmanager
def session_scope() -> Iterator[Session]:
    """`with session_scope() as s:` — commits on clean exit, rolls back on error."""
    _get_engine()  # ensures _SessionMaker is set
    assert _SessionMaker is not None
    s = _SessionMaker()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()


def init_schema() -> None:
    """Idempotently create the documents_registry table. Called by `db init`."""
    eng = _get_engine()
    Base.metadata.create_all(eng)


# ---------------------------------------------------------------------------
# CRUD helpers — the ingest pipeline + retriever talk to the registry
# through these, never raw SQL.
# ---------------------------------------------------------------------------


def upsert_document(
    *,
    content_hash: str,
    filename: str,
    title: str | None,
    source_type: str,
    file_storage_url: str,
    file_size: int,
    chunk_count: int,
    extra: dict | None = None,
) -> uuid.UUID:
    """
    Insert or update a registry row keyed by `content_hash`.

    Returns the row's doc_id (UUID) — the ingest pipeline then stamps this
    onto every Chroma metadata record so the Retriever can filter by it.
    """
    with session_scope() as s:
        existing = s.execute(
            select(DocumentRecord).where(DocumentRecord.content_hash == content_hash)
        ).scalar_one_or_none()

        if existing is not None:
            # Re-ingest of the same content: refresh the mutable fields, keep
            # the original doc_id so existing references stay valid.
            existing.filename = filename
            existing.title = title
            existing.file_storage_url = file_storage_url
            existing.file_size = file_size
            existing.chunk_count = chunk_count
            if extra is not None:
                existing.extra = extra
            return existing.doc_id

        row = DocumentRecord(
            content_hash=content_hash,
            filename=filename,
            title=title,
            source_type=source_type,
            file_storage_url=file_storage_url,
            file_size=file_size,
            chunk_count=chunk_count,
            extra=extra or {},
        )
        s.add(row)
        s.flush()  # populate row.doc_id before commit
        return row.doc_id


def list_documents(
    *,
    source_type: str | None = None,
    limit: int = 1000,
) -> list[DocumentRecord]:
    """Read the registry — for UI / debug. Detached objects, safe outside the session."""
    with session_scope() as s:
        stmt = select(DocumentRecord)
        if source_type:
            stmt = stmt.where(DocumentRecord.source_type == source_type)
        stmt = stmt.order_by(DocumentRecord.uploaded_at.desc()).limit(limit)
        return list(s.execute(stmt).scalars().all())


def get_doc_ids_for_filter(
    *,
    source_type: str | None = None,
    filenames: list[str] | None = None,
) -> list[str]:
    """
    Look up doc_ids matching a filter expression.

    The Retriever calls this when a query is scoped — e.g. "search only my
    notes" or "search only these uploads" — then passes the result as a
    ChromaDB metadata filter (`where={"doc_id": {"$in": [...]}}`).

    Returned as STRINGS because that's what we stamp into Chroma metadata
    (Chroma's filter syntax is string-typed).
    """
    with session_scope() as s:
        stmt = select(DocumentRecord.doc_id)
        if source_type:
            stmt = stmt.where(DocumentRecord.source_type == source_type)
        if filenames:
            stmt = stmt.where(DocumentRecord.filename.in_(filenames))
        return [str(r) for r in s.execute(stmt).scalars().all()]
