"""
Token-aware text chunking.

Why "token-aware" matters
-------------------------
A character-based splitter ("every 1000 chars") splits unpredictably for
embedding models that count by tokens. Dense embedders typically have an
input cap of 512 or 8192 tokens; if you exceed it, the input is silently
truncated and you lose half the chunk's meaning.

Our strategy (the standard "RecursiveCharacterTextSplitter" pattern):
  1. Try to split on paragraph boundaries.
  2. If a paragraph is too big, split on sentence boundaries.
  3. If a sentence is too big, hard-split on tokens.
  4. Add overlap between consecutive chunks so context is preserved at edges.

Public API
----------
    chunk_pages(pages, target_tokens=500, overlap_tokens=80) -> list[Chunk]
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import tiktoken

from src.ingest.pdf import Page

# `cl100k_base` is the tokenizer used by GPT-4/3.5. Good *approximation* for
# every other model — we only need ballpark counts to stay under input caps.
_ENC = tiktoken.get_encoding("cl100k_base")


@dataclass
class Chunk:
    text: str
    source_file: str
    page_number: int       # page where this chunk STARTS
    chunk_index: int       # 0-based index within the document
    token_count: int


def _tok_count(s: str) -> int:
    return len(_ENC.encode(s))


def _split_paragraphs(text: str) -> list[str]:
    """Split on blank lines; trim each block."""
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")


def _split_sentences(text: str) -> list[str]:
    # Simple but effective for academic English. Skips inside parens by being
    # conservative about what counts as a sentence start.
    parts = _SENT_SPLIT.split(text.strip())
    return [p for p in parts if p.strip()]


def _hard_split_tokens(text: str, target: int) -> list[str]:
    """Last-resort splitter — by raw tokens, ignoring boundaries."""
    ids = _ENC.encode(text)
    out: list[str] = []
    for i in range(0, len(ids), target):
        out.append(_ENC.decode(ids[i : i + target]))
    return out


def _pack(units: list[str], target: int, overlap: int) -> list[str]:
    """
    Greedy packer: combine `units` (paragraphs or sentences) into chunks of
    ~target tokens, then add `overlap` tokens of tail from the previous chunk
    onto the head of the next.
    """
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for u in units:
        u_tokens = _tok_count(u)
        if u_tokens > target:
            # Flush current, then hard-split the oversized unit.
            if current:
                chunks.append("\n\n".join(current))
                current, current_tokens = [], 0
            chunks.extend(_hard_split_tokens(u, target))
            continue

        if current_tokens + u_tokens > target and current:
            chunks.append("\n\n".join(current))
            current, current_tokens = [], 0

        current.append(u)
        current_tokens += u_tokens

    if current:
        chunks.append("\n\n".join(current))

    if overlap <= 0 or len(chunks) <= 1:
        return chunks

    # Add overlap by prepending the tail of chunk[i-1] onto chunk[i].
    overlapped: list[str] = [chunks[0]]
    for i in range(1, len(chunks)):
        prev_ids = _ENC.encode(chunks[i - 1])
        tail = _ENC.decode(prev_ids[-overlap:]) if len(prev_ids) > overlap else chunks[i - 1]
        overlapped.append(tail + "\n\n" + chunks[i])
    return overlapped


def chunk_pages(
    pages: list[Page],
    *,
    target_tokens: int = 500,
    overlap_tokens: int = 80,
) -> list[Chunk]:
    """
    Chunk an ordered list of Pages into ~target_tokens chunks with overlap.

    We chunk per-page so that page citations stay accurate. (A chunk that spans
    pages 4-5 gets credited to page 4 — its starting page.)
    """
    if not pages:
        return []

    all_chunks: list[Chunk] = []
    idx = 0

    for page in pages:
        paragraphs = _split_paragraphs(page.text)
        if not paragraphs:
            continue

        # First try paragraph packing.
        packed = _pack(paragraphs, target=target_tokens, overlap=overlap_tokens)

        # If any chunk is still over target by a lot, drop down to sentences.
        # In practice paragraph packing handles ~95% of academic-paper pages.
        refined: list[str] = []
        for ch in packed:
            if _tok_count(ch) > target_tokens * 1.5:
                refined.extend(_pack(_split_sentences(ch), target=target_tokens, overlap=overlap_tokens))
            else:
                refined.append(ch)

        for ch in refined:
            all_chunks.append(
                Chunk(
                    text=ch,
                    source_file=page.source_file,
                    page_number=page.page_number,
                    chunk_index=idx,
                    token_count=_tok_count(ch),
                )
            )
            idx += 1

    return all_chunks
