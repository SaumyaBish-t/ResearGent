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

from src.agent.artifacts import ChunkRef, HydratedChunk, hydrate
from src.agent.state import AgentState
from src.config import ModelTier
from src.llm import chat


SYSTEM_PROMPT = """You are a careful research assistant. Answer the user's question \
using ONLY the numbered sources provided below.

Rules:
- Cite every factual claim inline using [S1], [S2], etc.
- If the user's question has multiple parts, structure the answer with clear sections \
  matching the sub-questions you were given evidence for.
- If the sources do not contain the answer, say so plainly. Do NOT invent facts.
- Quote sparingly; paraphrase otherwise.
- Prefer concise, direct prose over bullet lists unless the question itself is a list.

YEAR / RECENCY DISCIPLINE — read carefully.
- The user may ask about "recent", "latest", a specific year ("2026 developments"),
  or "what's new". When they do:
    1. Check the publication year of EACH source you intend to cite (it's in the
       chunk header, e.g. "Title (2024)").
    2. ONLY make a "in <year> X happened" claim when AT LEAST ONE cited source
       is actually from that year and that source supports the claim.
    3. If most sources predate the year the user asked about, say so explicitly:
       "The available evidence is mostly from <year_range>; one 2026 paper [S3]
       discusses ...". Do NOT paper over the gap with vague phrases like
       "advancements published in 2026 include..." when only an older paper is
       cited for that bullet.
- Do NOT repeat the same point under multiple section headers. If two sub-questions
  produce overlapping answers, merge them into one section.

SYNTHESIZE PAPER + WEB — read carefully.

Sources are presented in priority order:
  - PAPER PDFs first (low S<n>):  headers like `arxiv:2308.08155` or a
    semanticscholar URL. These are primary-source PDF excerpts the
    cascade fetched live for THIS query — the paper itself.
  - WEB summaries next:           headers like `https://...` with
    `signal=web:tavily`. These are blog posts, docs sites, and
    third-party explainers paraphrasing the paper.
  - LOCAL notes last:             your own ingested corpus.

Your job is to COMBINE these into one coherent answer using BOTH the
primary papers AND the web summaries together:

1. ANCHOR every paper-content claim on the paper itself.
   When the claim is about WHAT THE PAPER SAYS (definitions, class
   taxonomy, algorithms, equations, benchmarks reported), cite the
   paper:* tags. The paper is the source of record — web summaries
   paraphrasing it are not.

2. CORROBORATE with web where it adds value.
   When a web source CONFIRMS what the paper says with the same fact,
   stack the citations: "AutoGen introduces ConversableAgent as the
   base class [S1][S6]" where [S1] is the paper and [S6] is a web
   explainer. The reader sees both the primary source AND the
   accessible summary in one citation.

3. USE WEB ALONE for ecosystem context.
   When the claim is about adoption, tutorials, the framework's
   reception, or third-party usage patterns, those don't appear in
   the original paper — cite the web sources directly.

4. NEVER cite ONLY the web for a paper-content claim when paper:*
   chunks are present in the evidence pool. If the paper covers the
   claim, the paper goes in the citation either alone or first.

Goal: a reader should be able to verify every claim by checking the
paper, and use the web sources for the accessible/contextual reading."""


def _chunk_key(c: HydratedChunk) -> tuple[str, int]:
    """Stable dedup key. All hydrated chunks expose source_file + chunk_index."""
    return (c.source_file, c.chunk_index)


def _citation_priority(c: HydratedChunk) -> int:
    """
    Order chunks get [S<n>] tags assigned in. Lower priority = lower tag
    number = LLM cites first.

    Why this matters: empirically the generator LLM cites in numerical
    order. When paper:* chunks got high tags (S20+) because they were
    inserted last into chunks_by_subq, the LLM cited the web chunks
    (S1-S3) for the main claims even though it had the actual paper
    text available at S20+. Promoting paper:* to S1-S5 makes the
    generator cite primary sources first when both are present.
    """
    sig = (getattr(c, "signal", "") or "").lower()
    if sig.startswith("paper:"):
        return 0  # primary sources first — the paper the user named
    if sig.startswith("web:"):
        return 1  # web summaries second
    return 2      # local Chroma/BM25 last (these are background context)


def _assign_citations(
    chunks_by_subq: dict[str, list[HydratedChunk]],
    refs_by_subq: dict[str, list[dict[str, str]]],
) -> tuple[dict[str, HydratedChunk], dict[str, dict[str, str]], dict[str, list[str]]]:
    """
    Build a stable [S<n>] map across local + web chunks.

    Returns:
      citation_map: {"S1": chunk1, "S2": chunk2, ...}
      subq_to_tags: {"sub_q_1": ["S1","S3"], "sub_q_2": ["S2","S3"]}

    A chunk that appears under multiple sub-questions gets ONE tag, reused.

    Tag numbering order: paper:* → web:* → local. The generator LLM cites
    in numerical order, so putting primary sources at S1-S5 gives them
    the citation slots the model uses for main claims.
    """
    citation_map: dict[str, HydratedChunk] = {}
    citation_ref_map: dict[str, dict[str, str]] = {}
    chunk_to_tag: dict[tuple[str, int], str] = {}
    subq_to_tags: dict[str, list[str]] = {}

    # PRIORITY PASS: assign tags to all paper:* chunks across all sub-Qs
    # FIRST, then web:*, then local. Within each priority bucket we
    # preserve sub-Q grouping order so the doc structure the LLM sees
    # is still per-sub-Q.
    #
    # We do this in two phases:
    #   1. Walk every sub-Q's chunks once, ORDERED BY PRIORITY, and
    #      assign tags. This determines what S1, S2, S3... map to.
    #   2. Walk every sub-Q's chunks again in ORIGINAL order to build
    #      subq_to_tags. The LLM sees per-sub-Q sections with the
    #      paper:* chunks naturally bearing low tag numbers.

    # Phase 1: tag assignment by priority bucket, preserving sub-Q +
    # within-sub-Q order within each bucket.
    next_n = 1
    for priority in (0, 1, 2):
        for sq, chunks in chunks_by_subq.items():
            refs_for_sq = refs_by_subq.get(sq) or []
            for i, c in enumerate(chunks):
                if _citation_priority(c) != priority:
                    continue
                key = _chunk_key(c)
                if key in chunk_to_tag:
                    continue
                tag = f"S{next_n}"
                chunk_to_tag[key] = tag
                citation_map[tag] = c
                if i < len(refs_for_sq):
                    citation_ref_map[tag] = refs_for_sq[i]
                next_n += 1

    # Phase 2: build subq_to_tags in ORIGINAL chunk order so the
    # per-sub-Q context blocks read naturally.
    for sq, chunks in chunks_by_subq.items():
        tags_for_this_sq: list[str] = []
        for c in chunks:
            key = _chunk_key(c)
            if key in chunk_to_tag:
                tags_for_this_sq.append(chunk_to_tag[key])
        subq_to_tags[sq] = tags_for_this_sq

    return citation_map, citation_ref_map, subq_to_tags


# Per-chunk char cap for the generator's context block.
#
# Why a cap: with the cascade running end-to-end and the
# paper-floor + priority-tag changes landing more chunks at the
# generator (last AutoGen run had 26 sources at full text), the
# unbounded `c.text.strip()` dump pushed the reasoning prompt to
# ~50K chars. The downstream Cerebras/Groq llama-3.3-70b call
# took 139s and returned an empty draft — auto-save then wrote an
# empty note to the vault, which is the bug the user just reported.
#
# 1500 chars per chunk × ~25 chunks ≈ 37K of context + 5K system
# prompt + question = ~45K total → ~11K tokens, comfortably inside
# the model's window and inside Groq's daily TPD budget. The
# critic uses 800 chars/chunk because it grades; the generator
# needs more because it must quote specific terms, so we settled
# in between.
_GEN_CHUNK_CHAR_CAP = 1500


def _build_context_block(
    chunks_by_subq: dict[str, list[HydratedChunk]],
    citation_map: dict[str, HydratedChunk],
    subq_to_tags: dict[str, list[str]],
    *,
    chunk_char_cap: int = _GEN_CHUNK_CHAR_CAP,
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
            body_text = c.text.strip()
            if len(body_text) > chunk_char_cap:
                body_text = body_text[:chunk_char_cap] + " […truncated]"
            body_parts.append(f"{header}\n{body_text}")
            body_parts.append("---")
        if body_parts[-1] == "---":
            body_parts.pop()
        sections.append("\n\n".join(body_parts))

    return "\n\n".join(sections)


def generate(state: AgentState) -> dict[str, Any]:
    """Synthesize the final answer with [S<n>] citations."""
    question = state["question"]
    refs_by_subq = dict(state.get("chunk_refs_by_subq") or {})

    if not refs_by_subq or not any(refs_by_subq.values()):
        # Should be unreachable because of the NoAnswer edge, but stay safe.
        return {
            "draft_answer": "I don't have enough indexed material to answer this.",
            "citation_refs": {},
            "trace": [{"node": "generator", "skipped": "no_chunks"}],
        }

    # Hydrate once for prompt assembly; the citation map keeps refs (not
    # chunks) for state, so the post-generator checkpoint stays lean.
    chunks_by_subq = hydrate(refs_by_subq)
    citation_map, citation_refs, subq_to_tags = _assign_citations(chunks_by_subq, refs_by_subq)

    def _build_prompt(chunk_cap: int) -> tuple[str, int]:
        ctx = _build_context_block(
            chunks_by_subq, citation_map, subq_to_tags, chunk_char_cap=chunk_cap
        )
        msg = (
            f"Question: {question}\n\n"
            f"Evidence (sources are numbered [S<n>] and grouped by sub-question):\n\n"
            f"{ctx}\n\n"
            f"Write the answer now. Cite every claim with the [S<n>] tags above."
        )
        return msg, len(SYSTEM_PROMPT) + len(msg)

    user_msg, prompt_chars = _build_prompt(_GEN_CHUNK_CHAR_CAP)

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

    # EMPTY-DRAFT GUARD + RETRY.
    #
    # Symptom we hit on the AutoGen run: generator returned "" after
    # 139s with 26 sources at full text. The reasoning LLM (Cerebras
    # llama-3.3-70b) silently produces empty output when the prompt
    # exceeds its happy context band — and downstream auto_save then
    # wrote an empty .md to the vault because verdict was high.
    #
    # On empty output, retry once with a tighter per-chunk cap (1500 →
    # 600). 600 chars × 25 chunks ≈ 15K of context — well inside
    # any reasoning model's working range. If THAT also comes back
    # empty, we emit an explicit "couldn't synthesize" answer instead
    # of saving blank.
    if not answer.strip():
        retry_msg, retry_chars = _build_prompt(chunk_cap=600)
        answer = chat(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": retry_msg},
            ],
            tier=ModelTier.REASONING,
            temperature=0.1,
            max_tokens=1200,
        )
        # Track for the trace so it's visible when the shrink happened.
        prompt_chars = retry_chars

    if not answer.strip():
        # Both attempts came back empty. Refuse to save a blank note.
        answer = (
            "Generator returned an empty draft after retry. This usually means "
            "the reasoning LLM choked on prompt size or hit a rate limit. "
            "Try re-running, or lower the per-paper chunk count in the cascade."
        )

    dur_ms = int((time.perf_counter() - t0) * 1000)

    return {
        "draft_answer": answer.strip(),
        "citation_refs": citation_refs,
        "trace": [
            {
                "node": "generator",
                "duration_ms": dur_ms,
                "n_sources": len(citation_refs),
                "answer_chars": len(answer),
                # Surface prompt size so day-budget pressure + retries are
                # observable. If `prompt_chars` is the shrink-retry value,
                # we know the first attempt came back empty.
                "prompt_chars": prompt_chars,
                "est_tokens": prompt_chars // 4,
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
