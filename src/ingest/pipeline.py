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

from src.ingest.chunker import Chunk, chunk_pages
from src.ingest.pdf import Document, parse_pdf
from src.llm import ModelTier, embed
from src.retrieval import bm25 as bm25_idx
from src.store import get_or_create_papers_collection

console = Console()

# Batch size for embedding requests. Kept SMALL by default because NVIDIA NIM's
# nv-embed-v1 free endpoint caps around ~8K tokens per request — a batch of 32
# chunks at ~500 tokens each is ~16K tokens and silently stalls / times out.
# 8 chunks * ~500 tokens = ~4K tokens, well within every provider's limit.
# Override via INGEST_EMBED_BATCH env if you've measured your endpoint can handle more.
import os
EMBED_BATCH = int(os.environ.get("INGEST_EMBED_BATCH", "8"))


def _batched(items: list, n: int) -> Iterable[list]:
    for i in range(0, len(items), n):
        yield items[i : i + n]


def _delete_existing_doc(doc_id: str) -> int:
    """Remove any chunks already stored under this doc_id. Returns count removed."""
    col = get_or_create_papers_collection()
    try:
        existing = col.get(where={"doc_id": doc_id}, include=[])
    except Exception:
        return 0
    ids = existing.get("ids") or []
    if ids:
        col.delete(ids=ids)
    return len(ids)


def _embed_and_store(doc: Document, chunks: list[Chunk], *, verbose: bool = True) -> int:
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
        texts = [c.text for c in batch]
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

        ids = [f"{doc.doc_id}:{c.chunk_index}" for c in batch]
        metadatas = [
            {
                "doc_id": doc.doc_id,
                "source_file": c.source_file,
                "page_number": c.page_number,
                "chunk_index": c.chunk_index,
                "token_count": c.token_count,
                "doc_title": doc.title or "",
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


def ingest_file(path: Path, *, verbose: bool = True, rebuild_bm25: bool = True) -> dict:
    """
    Ingest a single PDF. Returns a result summary dict.

    `rebuild_bm25=False` skips the BM25 rebuild — useful when ingesting many
    PDFs in a row (the directory ingester rebuilds once at the end).
    """
    if verbose:
        console.print(f"[cyan]parsing[/cyan] {path.name}")
    doc = parse_pdf(path)

    if verbose:
        console.print(f"  pages: {doc.total_pages}  |  doc_id: {doc.doc_id}")

    chunks = chunk_pages(doc.pages)
    removed = _delete_existing_doc(doc.doc_id)

    if verbose:
        console.print(f"  chunks: {len(chunks)} (replaced {removed} existing)")
        console.print("  [cyan]embedding[/cyan]...")

    inserted = _embed_and_store(doc, chunks)

    bm25_count = 0
    if rebuild_bm25:
        if verbose:
            console.print("  [cyan]building BM25 index[/cyan]...")
        bm25_count = _rebuild_bm25_from_chroma()

    return {
        "source_file": doc.source_file,
        "doc_id": doc.doc_id,
        "title": doc.title,
        "pages": doc.total_pages,
        "chunks_inserted": inserted,
        "chunks_replaced": removed,
        "bm25_chunks": bm25_count,
    }


def ingest_directory(dir_path: Path) -> list[dict]:
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
            r = ingest_file(pdf, verbose=True, rebuild_bm25=False)
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
