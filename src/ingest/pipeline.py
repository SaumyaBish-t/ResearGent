"""
End-to-end ingestion: directory of PDFs -> chunks -> embeddings -> Chroma.

Idempotency
-----------
We compute a stable `doc_id` from file contents. Before inserting a document
we delete any existing chunks for that doc_id, so re-ingesting the same PDF
*replaces* its chunks rather than duplicating them. If the file changes, the
hash changes, the doc_id changes, and you'll see it as a new document.

Batching
--------
We send embeddings in batches of `EMBED_BATCH` chunks. Most providers cap
the per-request batch size (NVIDIA at 96 inputs/req at the time of writing);
keeping it small also bounds memory and lets us surface progress smoothly.
"""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from src.config import settings
from src.ingest.chunker import Chunk, chunk_pages
from src.ingest.obsidian import VaultChunk, VaultNote, chunk_note, iter_vault_notes, parse_note
from src.ingest.pdf import Document, parse_pdf
from src.llm import ModelTier, embed
from src.retrieval import bm25 as bm25_idx
from src.store import get_or_create_papers_collection

# Where raw bytes live after ingestion. Today: local disk under data/storage/.
# Tomorrow: swap for an S3 URL by changing `_persist_raw()` only.
RAW_STORAGE_DIR = Path("data") / "storage"


def _persist_raw(src: Path, *, content_hash: str, ext: str) -> str:
    """
    Copy `src` into the canonical raw-storage location and return the URL.

    Naming by content_hash means re-ingesting the same file is a no-op on
    disk. Returning a `file://` URL keeps the column type future-proof —
    a later S3 migration just changes this function's return value.
    """
    RAW_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    dest = RAW_STORAGE_DIR / f"{content_hash}{ext}"
    if not dest.exists():
        # Copy not move — the ingest source dir (data/papers/) should
        # remain browseable; the storage dir is implementation detail.
        dest.write_bytes(src.read_bytes())
    return dest.resolve().as_uri()

console = Console()

# Batch size for embedding requests. Kept SMALL by default because NVIDIA NIM's
# nv-embed-v1 free endpoint caps around ~8K tokens per request — a batch of 32
# chunks at ~500 tokens each is ~16K tokens and silently stalls / times out.
# 8 chunks * ~500 tokens = ~4K tokens, well within every provider's limit.
# Override via INGEST_EMBED_BATCH env if you've measured your endpoint can handle more.
import os
EMBED_BATCH = int(os.environ.get("INGEST_EMBED_BATCH", "8"))


def _augment_text_with_entities(text: str, entities: list[str]) -> str:
    """
    Append a single-line entity manifest onto the chunk text BEFORE it is
    embedded and indexed.

    Why append instead of using a separate field?
    --------------------------------------------
    We get "GraphRAG-style" lexical and dense recall on the extracted
    entities for free: the same text the embedder sees is the text BM25
    tokenizes. A query mentioning "Reciprocal Rank Fusion" now lights up
    every chunk where GLiNER caught that phrase — even if the chunk's
    prose refers to it only as "the fusion step."

    Kept as a one-liner so it does not shift the chunk's dense vector
    meaningfully when entities are absent (empty list → no append).
    """
    if not entities:
        return text
    return f"{text}\n\n[Extracted Entities: {', '.join(entities)}]"


def _batched(items: list, n: int) -> Iterable[list]:
    for i in range(0, len(items), n):
        yield items[i : i + n]


def _delete_existing_doc(content_hash: str) -> int:
    """
    Remove any chunks already stored for this content hash. Returns count removed.

    Keyed on `content_hash` (not `doc_id`) so re-ingesting an identical file
    correctly replaces the prior chunks even though the doc_id (a registry
    UUID) is newly minted on each ingest call. Falls back to the legacy
    `doc_id == hash` lookup so corpora ingested before Phase 12 still clean
    up properly.
    """
    col = get_or_create_papers_collection()
    removed_ids: list[str] = []
    for filter_field in ("content_hash", "doc_id"):
        try:
            existing = col.get(where={filter_field: content_hash}, include=[])
        except Exception:
            continue
        removed_ids.extend(existing.get("ids") or [])
    if removed_ids:
        col.delete(ids=list(set(removed_ids)))
    return len(set(removed_ids))


def _embed_and_store(
    doc: Document,
    chunks: list[Chunk],
    *,
    registry_doc_id: str,
    domain: str | None = None,
    verbose: bool = True,
) -> int:
    """
    Embed chunks in batches, insert into Chroma. Returns count inserted.

    Prints per-batch latency so a stalled embedder is immediately visible
    rather than appearing as a generic hang.
    """
    if not chunks:
        return 0

    col = get_or_create_papers_collection()
    inserted = 0
    total_batches = (len(chunks) + EMBED_BATCH - 1) // EMBED_BATCH
    total_start = time.perf_counter()

    for batch_i, batch in enumerate(_batched(chunks, EMBED_BATCH), start=1):
        # Phase 14: embed the entity-augmented text, not the raw chunk.
        # `texts` is what hits both the dense embedder AND Chroma's stored
        # document field — so BM25 (which is rebuilt from Chroma docs) will
        # also tokenize the entity manifest.
        texts = [_augment_text_with_entities(c.text, c.entities) for c in batch]
        batch_tokens = sum(c.token_count for c in batch)

        t0 = time.perf_counter()
        try:
            vectors = embed(texts, tier=ModelTier.EMBED)
        except Exception as e:
            if verbose:
                console.print(
                    f"  [red]embed FAIL[/red] batch {batch_i}/{total_batches}  "
                    f"({len(batch)} chunks, ~{batch_tokens} tok): {type(e).__name__}: {e}"
                )
            raise
        dur_ms = int((time.perf_counter() - t0) * 1000)

        # Chroma id prefix uses the content hash so the same chunk has a
        # stable id across re-ingests (lets `_delete_existing_doc` find it).
        # `doc_id` in metadata is the REGISTRY UUID — what the Retriever
        # filters on. `content_hash` is the dedup key.
        ids = [f"{doc.doc_id}:{c.chunk_index}" for c in batch]
        metadatas = [
            {
                "doc_id": registry_doc_id,
                "content_hash": doc.doc_id,
                "source_file": c.source_file,
                "page_number": c.page_number,
                "chunk_index": c.chunk_index,
                "token_count": c.token_count,
                "doc_title": doc.title or "",
                # Phase 14: Chroma metadata is scalar-only (no list[str]), so
                # we comma-join — matches the existing convention for `tags`
                # and `wikilinks` on vault chunks. Retrieval-side filtering
                # can still match with a `$contains` predicate.
                "entities": ", ".join(c.entities),
                # Phase 15: domain bucket — "" means "uncategorised" so the
                # field is always present and the retriever can build a
                # uniform `$in` filter without special-casing absent keys.
                "domain": domain or "",
            }
            for c in batch
        ]
        col.add(ids=ids, embeddings=vectors, documents=texts, metadatas=metadatas)
        inserted += len(batch)

        if verbose:
            tps = int(batch_tokens / max(dur_ms, 1) * 1000)  # tokens / second
            console.print(
                f"  embed batch [cyan]{batch_i}/{total_batches}[/cyan]  "
                f"{len(batch)} chunks, ~{batch_tokens} tok  "
                f"[green]{dur_ms} ms[/green]  ({tps} tok/s)"
            )

    if verbose:
        total_s = time.perf_counter() - total_start
        console.print(f"  [bold]embedded {inserted} chunks in {total_s:.1f}s[/bold]")
    return inserted


def _rebuild_bm25_from_chroma() -> int:
    """
    Rebuild the BM25 index from the current Chroma collection contents.

    BM25's IDF depends on the FULL corpus, so per-doc incremental updates are
    cheap with embeddings (just insert a vector) but expensive for BM25 (must
    recompute IDF across all docs). We accept the cost — for academic corpora
    of a few hundred PDFs this completes in milliseconds.
    """
    col = get_or_create_papers_collection()
    n = col.count()
    if n == 0:
        bm25_idx.build_index([], [], [])
        bm25_idx.invalidate_cache()
        return 0

    # Pull all chunks. Chroma .get() with no `where` returns everything.
    # We do NOT need vectors here — only ids/texts/metadatas — so include=[]
    # is overridden to the minimum we actually need.
    res = col.get(include=["documents", "metadatas"])
    ids = res.get("ids") or []
    docs = res.get("documents") or []
    metas = res.get("metadatas") or []
    bm25_idx.build_index(ids=ids, texts=docs, metadatas=metas)
    bm25_idx.invalidate_cache()
    return len(ids)


def _infer_domain_from_path(path: Path) -> str | None:
    """
    Pull the domain id off a path that lives under `data/papers/<domain>/...`.

    We do this passively so the user can drop a PDF into a domain folder
    and `researgent ingest` Just Works without an explicit --domain flag.
    Returns None when the path isn't inside a registered domain subdir —
    the caller then falls back to whatever it was passed explicitly.
    """
    from src.domains import DOMAINS  # local import — avoids cycles + cold start cost

    try:
        rel = path.resolve().relative_to(PAPERS_ROOT.resolve())
    except ValueError:
        # Path isn't under data/papers/ at all (e.g. an ad-hoc PDF on the desktop).
        return None
    parts = rel.parts
    if not parts:
        return None
    candidate = parts[0]
    return candidate if candidate in DOMAINS else None


# `PAPERS_ROOT` is owned by `src.domains` (the single source of truth for
# corpus topology). Imported here lazily where used; we deliberately do not
# re-export it from this module to avoid two definitions drifting apart.
from src.domains import PAPERS_ROOT  # noqa: E402  (intentional bottom-up import)


def ingest_file(
    path: Path,
    *,
    domain: str | None = None,
    verbose: bool = True,
    rebuild_bm25: bool = True,
) -> dict:
    """
    Ingest a single PDF. Returns a result summary dict.

    `rebuild_bm25=False` skips the BM25 rebuild — useful when ingesting many
    PDFs in a row (the directory ingester rebuilds once at the end).

    `domain` (Phase 15): an optional registered domain id. When omitted, we
    auto-detect from the file's parent directory (data/papers/<domain>/...).
    The domain is stamped into every chunk's Chroma metadata AND into the
    registry row's `extra` JSONB so retrieval can later filter by it.
    """
    if verbose:
        console.print(f"[cyan]parsing[/cyan] {path.name}")
    doc = parse_pdf(path)

    # Domain resolution: explicit > inferred-from-path > None (uncategorised).
    # We don't ERROR on unknown domain strings here because the CLI already
    # validates them; ingest_file is also called directly from S2 seed code
    # which always passes a verified id.
    effective_domain = domain or _infer_domain_from_path(path)

    if verbose:
        dom_disp = effective_domain or "(none)"
        console.print(
            f"  pages: {doc.total_pages}  |  content_hash: {doc.doc_id}  |  domain: {dom_disp}"
        )

    chunks = chunk_pages(doc.pages)
    removed = _delete_existing_doc(doc.doc_id)

    # Persist raw bytes + register in PG BEFORE embedding so a mid-embed
    # crash still leaves a discoverable registry row pointing at the file.
    # Registration is gated on Postgres being configured — local-dev users
    # without DATABASE_URL fall back to using the content_hash as doc_id
    # so retrieval still works (just without registry-based filtering).
    file_size = path.stat().st_size
    registry_doc_id: str
    storage_url: str | None = None
    if settings.resolve_database_url():
        from src.registry import upsert_document

        storage_url = _persist_raw(path, content_hash=doc.doc_id, ext=".pdf")
        # Domain lives in `extra` rather than as a dedicated column so we
        # avoid a schema migration; the `extra` JSONB was already designed
        # as the bag for source-type-specific metadata. Queryable via
        # `extra->>'domain' = 'agentic_ai'` from SQL when needed.
        registry_uuid = upsert_document(
            content_hash=doc.doc_id,
            filename=path.name,
            title=doc.title,
            source_type="pdf",
            file_storage_url=storage_url,
            file_size=file_size,
            chunk_count=len(chunks),
            extra={"domain": effective_domain} if effective_domain else None,
        )
        registry_doc_id = str(registry_uuid)
    else:
        # No Postgres → keep Phase-11 behaviour. The content_hash doubles
        # as doc_id; nothing in the retrieval path requires UUID format.
        registry_doc_id = doc.doc_id

    if verbose:
        console.print(
            f"  chunks: {len(chunks)} (replaced {removed} existing)  "
            f"doc_id: {registry_doc_id}"
        )
        console.print("  [cyan]embedding[/cyan]...")

    inserted = _embed_and_store(
        doc, chunks, registry_doc_id=registry_doc_id, domain=effective_domain
    )

    bm25_count = 0
    if rebuild_bm25:
        if verbose:
            console.print("  [cyan]building BM25 index[/cyan]...")
        bm25_count = _rebuild_bm25_from_chroma()

    return {
        "source_file": doc.source_file,
        "doc_id": registry_doc_id,
        "content_hash": doc.doc_id,
        "title": doc.title,
        "domain": effective_domain,
        "pages": doc.total_pages,
        "chunks_inserted": inserted,
        "chunks_replaced": removed,
        "bm25_chunks": bm25_count,
        "storage_url": storage_url,
        "file_size": file_size,
    }


def ingest_directory(dir_path: Path, *, domain: str | None = None) -> list[dict]:
    """
    Ingest every PDF in a directory (non-recursive). BM25 rebuilds once at end.

    Per-batch progress is ALWAYS shown — even when processing many files —
    because silent embedding on a CPU-based embedder can take minutes per PDF
    and the user needs to see it's not hung.
    """
    pdfs = sorted(p for p in dir_path.glob("*.pdf"))
    if not pdfs:
        console.print(f"[yellow]No PDFs found in {dir_path}[/yellow]")
        return []

    # Pre-warm the embedder so users don't pay cold-load latency mid-progress.
    # Surfaces failures BEFORE we start parsing PDFs — much faster feedback.
    console.print("[cyan]pre-warming embedder...[/cyan]")
    try:
        t0 = time.perf_counter()
        embed(["warmup"], tier=ModelTier.EMBED)
        console.print(f"  [green]warm[/green]  ({int((time.perf_counter() - t0) * 1000)} ms)")
    except Exception as e:
        console.print(f"  [red]warmup FAIL[/red]: {type(e).__name__}: {e}")
        console.print("[red]Embedder is broken. Run `researgent doctor` to diagnose.[/red]")
        return [{"error": "embedder warmup failed"}]

    console.print(f"\n[bold]processing {len(pdfs)} PDFs[/bold]\n")
    results: list[dict] = []
    overall_start = time.perf_counter()

    for i, pdf in enumerate(pdfs, start=1):
        console.print(f"[bold blue]({i}/{len(pdfs)})[/bold blue] {pdf.name}")
        try:
            # verbose=True so users see live per-batch progress on every PDF.
            # BM25 rebuilds once at the end (it scans ALL chunks anyway).
            # `domain` here propagates the directory-level override; if None,
            # ingest_file's per-path auto-detection still runs.
            r = ingest_file(pdf, domain=domain, verbose=True, rebuild_bm25=False)
            results.append(r)
            console.print(
                f"  [green]done[/green]  {r['pages']} pages, {r['chunks_inserted']} chunks\n"
            )
        except Exception as e:
            console.print(f"  [red]FAIL[/red] {type(e).__name__}: {e}\n")
            results.append({"source_file": pdf.name, "error": str(e)})

    total_s = time.perf_counter() - overall_start
    console.print(f"[bold]all PDFs processed in {total_s:.1f}s[/bold]")
    console.print("\n[cyan]rebuilding BM25 index[/cyan]...")
    bm25_n = _rebuild_bm25_from_chroma()
    console.print(f"  [green]BM25[/green] indexed {bm25_n} chunks")
    return results


def ingest_all_domains(
    *,
    domain_ids: list[str] | None = None,
) -> dict[str, list[dict]]:
    """
    Phase 15 orchestrator: walk every registered domain subdir under
    `data/papers/<domain>/` and ingest each as a tagged batch.

    Why a single entry-point
    ------------------------
    Three benefits over `for d in domains: ingest_directory(...)`:
      1. ONE embedder warm-up cost amortised across all three domains
         (warm-up alone is ~1-2s on cold MiniLM / GLiNER caches; doing it
         per-domain triples that for no recall gain).
      2. ONE BM25 rebuild at the very end. BM25's IDF is corpus-wide, so
         rebuilding three times in a row is wasted work — only the last
         result matters.
      3. The returned dict is keyed by domain so the CLI can render a clean
         per-corpus summary table without re-iterating the filesystem.

    `domain_ids` lets callers run a subset (e.g. just `["agentic_ai"]`).
    Defaults to every registered domain.
    """
    from src.domains import DOMAINS, all_domain_ids

    target = domain_ids or all_domain_ids()
    out: dict[str, list[dict]] = {}

    # Single shared warm-up. We bypass ingest_directory's own warm-up because
    # we want to warm exactly once and surface failures BEFORE filesystem
    # walking — same reasoning as Phase 1's directory ingester.
    console.print("[cyan]pre-warming embedder...[/cyan]")
    try:
        t0 = time.perf_counter()
        embed(["warmup"], tier=ModelTier.EMBED)
        console.print(f"  [green]warm[/green]  ({int((time.perf_counter() - t0) * 1000)} ms)")
    except Exception as e:
        console.print(f"  [red]warmup FAIL[/red]: {type(e).__name__}: {e}")
        return {d: [{"error": "embedder warmup failed"}] for d in target}

    for dom_id in target:
        if dom_id not in DOMAINS:
            console.print(f"[red]skip unknown domain[/red] {dom_id}")
            out[dom_id] = [{"error": f"unknown domain {dom_id}"}]
            continue
        dom = DOMAINS[dom_id]
        dom.ingest_dir.mkdir(parents=True, exist_ok=True)
        pdfs = sorted(dom.ingest_dir.glob("*.pdf"))

        console.print(
            f"\n[bold]── {dom.label} ──[/bold]  "
            f"[dim]{dom.ingest_dir}[/dim]  ({len(pdfs)} PDFs)"
        )
        if not pdfs:
            out[dom_id] = []
            continue

        results: list[dict] = []
        for i, pdf in enumerate(pdfs, start=1):
            console.print(f"[bold blue]({i}/{len(pdfs)})[/bold blue] {pdf.name}")
            try:
                # rebuild_bm25=False — we do ONE rebuild at the bottom across
                # the entire corpus (BM25 IDF is corpus-wide).
                r = ingest_file(pdf, domain=dom_id, verbose=True, rebuild_bm25=False)
                results.append(r)
                console.print(
                    f"  [green]done[/green]  {r['pages']} pages, {r['chunks_inserted']} chunks\n"
                )
            except Exception as e:
                console.print(f"  [red]FAIL[/red] {type(e).__name__}: {e}\n")
                results.append({"source_file": pdf.name, "error": str(e)})
        out[dom_id] = results

    console.print("\n[cyan]rebuilding BM25 index[/cyan]...")
    bm25_n = _rebuild_bm25_from_chroma()
    console.print(f"  [green]BM25[/green] indexed {bm25_n} chunks across all domains")
    return out


# ---------------------------------------------------------------------------
# Obsidian vault ingestion (Phase 8)
# ---------------------------------------------------------------------------


def _embed_and_store_vault_chunks(
    note: VaultNote,
    chunks: list[VaultChunk],
    *,
    registry_doc_id: str,
    verbose: bool = True,
) -> int:
    """Embed vault chunks and insert into the same papers collection."""
    if not chunks:
        return 0

    col = get_or_create_papers_collection()
    inserted = 0
    total_batches = (len(chunks) + EMBED_BATCH - 1) // EMBED_BATCH
    total_start = time.perf_counter()

    for batch_i, batch in enumerate(_batched(chunks, EMBED_BATCH), start=1):
        # Phase 14: same entity-augmentation as the PDF path.
        texts = [_augment_text_with_entities(c.text, c.entities) for c in batch]
        batch_tokens = sum(c.token_count for c in batch)

        t0 = time.perf_counter()
        try:
            vectors = embed(texts, tier=ModelTier.EMBED)
        except Exception as e:
            if verbose:
                console.print(
                    f"  [red]embed FAIL[/red] batch {batch_i}/{total_batches}: {type(e).__name__}: {e}"
                )
            raise
        dur_ms = int((time.perf_counter() - t0) * 1000)

        ids = [f"{note.doc_id}:{c.chunk_index}" for c in batch]
        metadatas = [
            {
                "doc_id": registry_doc_id,
                "content_hash": note.doc_id,
                "source_file": c.source_file,
                "page_number": c.page_number,       # 0 for vault notes
                "chunk_index": c.chunk_index,
                "token_count": c.token_count,
                "doc_title": c.note_title,
                # Vault-specific — comma-joined for the simple metadata layer
                "source_type": "vault",
                "heading_path": c.heading_path,
                "tags": ",".join(c.tags),
                "wikilinks": ",".join(c.wikilinks),
                # Phase 14: GLiNER entities, comma-joined to fit Chroma's
                # scalar-only metadata. Same convention as `tags`/`wikilinks`.
                "entities": ", ".join(c.entities),
            }
            for c in batch
        ]
        col.add(ids=ids, embeddings=vectors, documents=texts, metadatas=metadatas)
        inserted += len(batch)

        if verbose:
            tps = int(batch_tokens / max(dur_ms, 1) * 1000)
            console.print(
                f"  embed batch [cyan]{batch_i}/{total_batches}[/cyan]  "
                f"{len(batch)} chunks, ~{batch_tokens} tok  "
                f"[green]{dur_ms} ms[/green]  ({tps} tok/s)"
            )

    if verbose:
        total_s = time.perf_counter() - total_start
        console.print(f"  [bold]embedded {inserted} chunks in {total_s:.1f}s[/bold]")
    return inserted


def ingest_vault(
    vault_path: Path,
    *,
    rebuild_bm25: bool = True,
    verbose: bool = True,
) -> list[dict]:
    """
    Walk an Obsidian vault and ingest every .md file.

    Re-running is idempotent: each note's chunks are keyed by `doc_id` (a
    content hash). Modifying a note changes its hash → its chunks get
    replaced cleanly on the next ingest. Unchanged notes are re-embedded
    (we don't skip; embedder is cheap on local Ollama) but the result
    is identical to what was already stored.
    """
    vault_path = vault_path.resolve()
    if not vault_path.exists() or not vault_path.is_dir():
        console.print(f"[red]Vault path not found: {vault_path}[/red]")
        return []

    notes = iter_vault_notes(vault_path)
    if not notes:
        console.print(f"[yellow]No .md notes found in {vault_path}[/yellow]")
        return []

    # Pre-warm — same reasoning as the PDF path.
    console.print("[cyan]pre-warming embedder...[/cyan]")
    try:
        t0 = time.perf_counter()
        embed(["warmup"], tier=ModelTier.EMBED)
        console.print(f"  [green]warm[/green]  ({int((time.perf_counter() - t0) * 1000)} ms)")
    except Exception as e:
        console.print(f"  [red]warmup FAIL[/red]: {type(e).__name__}: {e}")
        console.print("[red]Embedder is broken. Run `researgent doctor` to diagnose.[/red]")
        return [{"error": "embedder warmup failed"}]

    console.print(f"\n[bold]processing {len(notes)} notes from {vault_path}[/bold]\n")

    results: list[dict] = []
    overall_start = time.perf_counter()

    for i, note_path in enumerate(notes, start=1):
        rel = note_path.relative_to(vault_path)
        console.print(f"[bold blue]({i}/{len(notes)})[/bold blue] {rel}")
        try:
            note = parse_note(note_path, vault_path)
            chunks = chunk_note(note)
            removed = _delete_existing_doc(note.doc_id)

            # Register the note's bytes the same way as PDFs. The "raw"
            # representation of a markdown note is just the file itself.
            file_size = note_path.stat().st_size
            if settings.resolve_database_url():
                from src.registry import upsert_document

                storage_url = _persist_raw(
                    note_path, content_hash=note.doc_id, ext=".md"
                )
                registry_uuid = upsert_document(
                    content_hash=note.doc_id,
                    filename=str(rel),
                    title=note.title,
                    source_type="note",
                    file_storage_url=storage_url,
                    file_size=file_size,
                    chunk_count=len(chunks),
                    extra={"tags": note.tags, "wikilinks": note.wikilinks},
                )
                registry_doc_id = str(registry_uuid)
            else:
                registry_doc_id = note.doc_id

            if verbose:
                console.print(
                    f"  chunks: {len(chunks)}  tags: {len(note.tags)}  "
                    f"links: {len(note.wikilinks)}  (replaced {removed})  "
                    f"doc_id: {registry_doc_id}"
                )
            inserted = _embed_and_store_vault_chunks(
                note, chunks, registry_doc_id=registry_doc_id, verbose=verbose,
            )
            results.append({
                "source_file": str(rel),
                "doc_id": registry_doc_id,
                "content_hash": note.doc_id,
                "title": note.title,
                "tags": note.tags,
                "wikilinks": note.wikilinks,
                "chunks_inserted": inserted,
                "chunks_replaced": removed,
            })
            console.print(f"  [green]done[/green]  {note.title!r}: {inserted} chunks\n")
        except Exception as e:
            console.print(f"  [red]FAIL[/red] {type(e).__name__}: {e}\n")
            results.append({"source_file": str(rel), "error": str(e)})

    total_s = time.perf_counter() - overall_start
    console.print(f"[bold]vault ingest complete in {total_s:.1f}s[/bold]")

    if rebuild_bm25:
        console.print("\n[cyan]rebuilding BM25 index[/cyan]...")
        bm25_n = _rebuild_bm25_from_chroma()
        console.print(f"  [green]BM25[/green] indexed {bm25_n} chunks")
    return results
