"""
PDF parsing — produces a list of Page records with text + page number.

We use PyMuPDF (imported as `fitz`) because it:
  - Handles multi-column academic layouts much better than pypdf
  - Preserves reading order well via `get_text("text")` mode
  - Is fast (no Java/external binaries)

We deliberately keep this module *just* PDF -> text. Chunking is a separate
concern handled in `chunker.py`, so it's reusable across non-PDF sources later.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF


@dataclass
class Page:
    text: str
    page_number: int  # 1-indexed for human-friendly citations
    source_file: str  # filename only (not absolute path) for portable citations


@dataclass
class Document:
    doc_id: str  # stable hash of file contents — enables idempotent re-ingest
    source_file: str
    title: str | None
    pages: list[Page]

    @property
    def total_pages(self) -> int:
        return len(self.pages)


def _doc_id_for(path: Path) -> str:
    """Hash file contents. If the same PDF is re-ingested, we can skip / replace."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def parse_pdf(path: Path) -> Document:
    """Read a single PDF into a Document."""
    if not path.exists():
        raise FileNotFoundError(path)

    doc_id = _doc_id_for(path)
    pages: list[Page] = []
    title: str | None = None

    with fitz.open(path) as pdf:
        meta = pdf.metadata or {}
        title = (meta.get("title") or "").strip() or None
        for i, page in enumerate(pdf, start=1):
            # "text" mode gives a natural reading order for most papers.
            txt = page.get_text("text")
            # Light cleanup — collapse runs of blank lines, strip page artifacts.
            txt = _clean_page_text(txt)
            if txt.strip():
                pages.append(Page(text=txt, page_number=i, source_file=path.name))

    return Document(doc_id=doc_id, source_file=path.name, title=title, pages=pages)


def _clean_page_text(text: str) -> str:
    lines = [ln.rstrip() for ln in text.splitlines()]
    # Drop fully empty repeats (more than 2 in a row).
    cleaned: list[str] = []
    blank_run = 0
    for ln in lines:
        if not ln.strip():
            blank_run += 1
            if blank_run <= 1:
                cleaned.append("")
        else:
            blank_run = 0
            cleaned.append(ln)
    return "\n".join(cleaned).strip()
