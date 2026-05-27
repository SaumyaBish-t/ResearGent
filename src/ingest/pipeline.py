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
from src.store import get_or_create_papers_collection

console = Console()
EMBED_BATCH = 32


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


def _embed_and_store(doc: Document, chunks: list[Chunk]) -> int:
    """Embed chunks in batches, insert into Chroma. Returns count inserted."""
    if not chunks:
        return 0

    col = get_or_create_papers_collection()
    inserted = 0

    for batch in _batched(chunks, EMBED_BATCH):
        texts = [c.text for c in batch]
        vectors = embed(texts, tier=ModelTier.EMBED)

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

    return inserted


def ingest_file(path: Path, *, verbose: bool = True) -> dict:
    """Ingest a single PDF. Returns a result summary dict."""
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
    return {
        "source_file": doc.source_file,
        "doc_id": doc.doc_id,
        "title": doc.title,
        "pages": doc.total_pages,
        "chunks_inserted": inserted,
        "chunks_replaced": removed,
    }


def ingest_directory(dir_path: Path) -> list[dict]:
    """Ingest every PDF in a directory (non-recursive)."""
    pdfs = sorted(p for p in dir_path.glob("*.pdf"))
    if not pdfs:
        console.print(f"[yellow]No PDFs found in {dir_path}[/yellow]")
        return []

    results: list[dict] = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as prog:
        task = prog.add_task("ingesting PDFs", total=len(pdfs))
        for pdf in pdfs:
            try:
                r = ingest_file(pdf, verbose=False)
                results.append(r)
                prog.console.print(
                    f"  [green]ok[/green] {pdf.name}  "
                    f"({r['pages']} pages, {r['chunks_inserted']} chunks)"
                )
            except Exception as e:
                prog.console.print(f"  [red]fail[/red] {pdf.name}: {type(e).__name__}: {e}")
                results.append({"source_file": pdf.name, "error": str(e)})
            prog.advance(task)
    return results
