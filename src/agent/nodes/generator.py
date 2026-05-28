"""
Generator node — synthesizes the final answer with grounded citations.

How this differs from Phase 1/2 RAG generation
----------------------------------------------
Phase 1/2 had one retrieval -> one prompt -> answer. The generator here
receives a DICT of {sub_question: [chunks]}, so the prompt is structured
by sub-question. That gives the model two important affordances:

  1. It can WRITE per-axis ("On X: ... [S1][S2]. On Y: ... [S3][S4]")
     instead of crushing everything into one paragraph.
  2. If the user's original question was comparative ("how do X and Y
     differ"), the per-sub-Q grouping makes the comparison structure
     obvious — the model doesn't have to infer it from a flat chunk list.

Deduplication
-------------
The same chunk can appear under multiple sub-questions (when retrieval
overlap is high). We assign one stable [S<n>] tag per UNIQUE chunk and
reuse the tag across sub-Qs. Saves prompt tokens and keeps citations
clean.
"""

from __future__ import annotations

import time
from typing import Any

from src.agent.state import AgentState, ContextChunk
from src.config import ModelTier
from src.llm import chat
from src.retrieval import HybridChunk, WebChunk


SYSTEM_PROMPT = """You are a careful research assistant. Answer the user's question \
using ONLY the numbered sources provided below.

Rules:
- Cite every factual claim inline using [S1], [S2], etc.
- If the user's question has multiple parts, structure the answer with clear sections \
  matching the sub-questions you were given evidence for.
- If the sources do not contain the answer, say so plainly. Do NOT invent facts.
- Quote sparingly; paraphrase otherwise.
- Prefer concise, direct prose over bullet lists unless the question itself is a list."""


def _chunk_key(c: ContextChunk) -> tuple[str, int]:
    """
    Stable dedup key across both chunk types.

    HybridChunk -> (source_file, chunk_index)        from local store
    WebChunk    -> (url, -1)                          from Tavily
    """
    # Both expose .source_file and .chunk_index (WebChunk's are property shims).
    return (c.source_file, c.chunk_index)


def _assign_citations(
    chunks_by_subq: dict[str, list[ContextChunk]],
) -> tuple[dict[str, ContextChunk], dict[str, list[str]]]:
    """
    Build a stable [S<n>] map across local + web chunks.

    Returns:
      citation_map: {"S1": chunk1, "S2": chunk2, ...}
      subq_to_tags: {"sub_q_1": ["S1","S3"], "sub_q_2": ["S2","S3"]}

    A chunk that appears under multiple sub-questions gets ONE tag, reused.
    """
    citation_map: dict[str, ContextChunk] = {}
    chunk_to_tag: dict[tuple[str, int], str] = {}
    subq_to_tags: dict[str, list[str]] = {}

    next_n = 1
    for sq, chunks in chunks_by_subq.items():
        tags_for_this_sq: list[str] = []
        for c in chunks:
            key = _chunk_key(c)
            if key not in chunk_to_tag:
                tag = f"S{next_n}"
                chunk_to_tag[key] = tag
                citation_map[tag] = c
                next_n += 1
            tags_for_this_sq.append(chunk_to_tag[key])
        subq_to_tags[sq] = tags_for_this_sq
    return citation_map, subq_to_tags


def _build_context_block(
    chunks_by_subq: dict[str, list[ContextChunk]],
    citation_map: dict[str, ContextChunk],
    subq_to_tags: dict[str, list[str]],
) -> str:
    """
    Group context by sub-question so the model writes structured answers.

    Format:
      ## Sub-question 1: <text>
      [S1] file.pdf p.7 -- title
      <chunk text>
      ---
      [S2] file.pdf p.3
      <chunk text>

      ## Sub-question 2: <text>
      [S2] (same chunk as above, just referenced again — no re-dump)
      ...
    """
    seen_tags: set[str] = set()
    sections: list[str] = []

    for sq, tags in subq_to_tags.items():
        body_parts: list[str] = [f"## Sub-question: {sq}"]
        for tag in tags:
            if tag in seen_tags:
                body_parts.append(f"[{tag}] (see above)")
                continue
            seen_tags.add(tag)
            c = citation_map[tag]
            header = f"[{tag}] {c.citation}"
            if c.doc_title:
                header += f" -- {c.doc_title}"
            body_parts.append(f"{header}\n{c.text.strip()}")
            body_parts.append("---")
        if body_parts[-1] == "---":
            body_parts.pop()
        sections.append("\n\n".join(body_parts))

    return "\n\n".join(sections)


def generate(state: AgentState) -> dict[str, Any]:
    """Synthesize the final answer with [S<n>] citations."""
    question = state["question"]
    chunks_by_subq = state.get("chunks_by_subq") or {}

    if not chunks_by_subq or not any(chunks_by_subq.values()):
        # Should be unreachable because of the NoAnswer edge, but stay safe.
        return {
            "draft_answer": "I don't have enough indexed material to answer this.",
            "citation_map": {},
            "trace": [{"node": "generator", "skipped": "no_chunks"}],
        }

    citation_map, subq_to_tags = _assign_citations(chunks_by_subq)
    context_block = _build_context_block(chunks_by_subq, citation_map, subq_to_tags)

    user_msg = f"""Question: {question}

Evidence (sources are numbered [S<n>] and grouped by sub-question):

{context_block}

Write the answer now. Cite every claim with the [S<n>] tags above."""

    t0 = time.perf_counter()
    answer = chat(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        tier=ModelTier.REASONING,
        temperature=0.1,
        max_tokens=1200,
    )
    dur_ms = int((time.perf_counter() - t0) * 1000)

    return {
        "draft_answer": answer.strip(),
        "citation_map": citation_map,
        "trace": [
            {
                "node": "generator",
                "duration_ms": dur_ms,
                "n_sources": len(citation_map),
                "answer_chars": len(answer),
            }
        ],
    }


def no_answer(state: AgentState) -> dict[str, Any]:
    """
    Terminal node when retrieval surfaced nothing.

    Phase 4 will replace this with a Web-Scraper fallback. For now we
    return a clear "I don't know" — far better than letting the LLM
    hallucinate from no evidence.
    """
    return {
        "draft_answer": (
            "I couldn't find anything relevant in the indexed corpus for this question. "
            "Try rephrasing, or ingest additional documents. "
            "(Phase 4 will add a live web-search fallback for this case.)"
        ),
        "citation_map": {},
        "error": "no_chunks_retrieved",
        "trace": [{"node": "no_answer", "reason": "retrieval empty"}],
    }
