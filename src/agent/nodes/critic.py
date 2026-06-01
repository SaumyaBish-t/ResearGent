"""
Critic node — grades retrieved chunks for relevance to their sub-question.

This is the heart of Corrective RAG. Instead of trusting whatever retrieval
surfaced, we ask a small fast model: "is this chunk actually relevant to
the question, or is it just topically nearby?"

Why the FAST tier specifically
------------------------------
The Critic runs many times per query (potentially per chunk, per sub-q,
per retry). Using the REASONING tier here would 10x the wall-clock cost
of every research query. The FAST tier (Groq llama-3.1-8b-instant) is:
  - Sub-second per call
  - More than capable of binary relevance classification
  - Designed exactly for this kind of high-volume filtering work

Per-chunk vs batch grading
--------------------------
We grade in a SINGLE call per sub-question, passing all chunks at once.
The model returns a JSON list of grades. This is dramatically cheaper than
N calls (one per chunk) and lets the model use cross-chunk context for
disambiguation ("S3 mentions the same paper as S1, so they're consistent").

Output contract — JSON only:
    {
      "grades": ["relevant" | "partial" | "irrelevant", ...],
      "verdict": "high" | "medium" | "low",
      "reasoning": "one short sentence"
    }

Confidence policy (derived from grades, deterministic)
------------------------------------------------------
We POST-PROCESS grades into the verdict ourselves rather than trusting the
model's `verdict` field. The rule:
    fraction_relevant = relevant_count / total
    - high   if fraction_relevant >= 0.5
    - medium if fraction_relevant >= 0.2
    - low    otherwise (or if no chunks at all)
This makes confidence predictable + tunable without re-prompting the model.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

from src.agent.artifacts import HydratedChunk, hydrate
from src.agent.state import AgentState
from src.config import ModelTier
from src.llm import chat


_SYSTEM = """You are a STRICT, SKEPTICAL chunk-relevance grader for a research assistant.

Your job: look at the user's question and the numbered chunks we found in our local \
database, and grade each chunk. Be conservative — the system uses your grades to decide \
whether to fire a live Semantic Scholar + web search for better evidence. If the local \
data is not absolutely perfect for the question, you should grade lower so we can go fetch \
the bleeding-edge papers.

Per-chunk grade:
  - "relevant"   : the chunk contains SPECIFIC evidence that directly answers the question. \
You can point to a concrete sentence, named entity, formula, metric, dataset, or claim that \
matches the question's specifics. If the question asks about a specific year, version, \
entity, dataset, or event, the chunk must mention THAT exact specifier — not a related one.
  - "partial"    : the chunk is on the right TOPIC but is missing specific details, uses \
older or different data than asked, or only addresses half the question. Useful background \
but not a direct answer.
  - "irrelevant" : different topic, different entity, different time period, different \
method, or different domain than asked. Same general research area does NOT make a chunk \
relevant.

RECENCY / YEAR DISCIPLINE — this is the rule most graders get wrong:
  If the question asks about a SPECIFIC YEAR ("latest 2026 developments", "what's new in \
2025", "recent advances published this year"), the chunk MUST be from that exact year to \
earn "relevant". A 2024 paper on the same topic is "partial" at best, regardless of how \
on-topic it otherwise is. A 2021 paper on the same topic is "irrelevant".
  Chunks carry their year in the header (e.g. "Title (2024)"). USE IT.

DOMAIN-MATCH IS NOT RELEVANCE:
  Sharing the same domain (agentic_ai, quant_finance, time_series) with the question is the \
floor, not the ceiling. A chunk about LangGraph routing does NOT answer a question about \
ReAct loops just because both are agentic-AI topics. Demand a direct match.

CRITICAL — DO NOT BE GENEROUS:
  When in doubt between "relevant" and "partial", choose "partial".
  When in doubt between "partial" and "irrelevant", choose "irrelevant".
  Grading too generously means the agent ships a thin answer instead of escalating to \
live arXiv + Semantic Scholar + web search. False positives here are worse than false \
negatives — the cascade can rescue under-grading, but over-grading silently degrades the \
final answer.

Examples of correct grading:
  Q: "What are the latest 2026 developments in federated learning for time series anomaly detection?"
    chunk: 2026 paper on federated MTSAD          -> relevant
    chunk: 2024 paper on federated MTSAD          -> partial (right topic, old)
    chunk: 2021 paper on autoencoder anomaly      -> irrelevant (old + different method)
    chunk: 2026 paper on federated learning for image classification -> partial (wrong task)

  Q: "How does Self-RAG handle low-confidence retrieval?"
    chunk defining Self-RAG's reflection tokens    -> partial (related mechanism)
    chunk explaining Self-RAG's retrieval gating   -> relevant (direct answer)
    chunk about CRAG's retrieval evaluator         -> irrelevant (different method)

  Q: "Who won the 2026 Nobel Prize in Physics?"
    chunk about RAG architectures                  -> irrelevant
    chunk about 2025 Nobel Chemistry               -> irrelevant (different prize / year)
    chunk that names the 2026 Physics laureates   -> relevant

CRITERIA AMENDMENT: Be constructive and lenient when grading newly discovered \
papers or live web fallbacks from recent dates (e.g., 2026). If a chunk contains \
the explicit core entities, numbers, or agent names requested by the user — even \
if the surrounding text or snippet formatting is partial or noisy — grade it as \
"relevant" or a high-value "partial". Do not penalise fresh, correct information \
for layout or formatting fragments. The chunk header tells you the source: lines \
beginning with `arxiv:`, `https://www.semanticscholar.org/`, or `web:` are fresh \
external discoveries from the Stage-2 cascade — they have already paid the cost \
of being fetched live, so reward content that hits the user's named specifics \
even when sentence structure is broken.

Output ONLY a JSON object, no preamble, no markdown fence:
{
  "grades": ["relevant" | "partial" | "irrelevant", ...],
  "reasoning": "one short sentence summarising what's missing and whether the cascade should fire"
}

`grades` MUST have exactly the same length as the number of chunks provided, \
in the same order. Use lower case strings exactly as shown."""


_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except Exception:
        pass
    m = _JSON_OBJ_RE.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def _grade_one_subq(
    sub_q: str, chunks: list[HydratedChunk]
) -> tuple[list[str], str, int, int]:
    """
    Grade chunks for a single sub-question.

    Returns (grades, reasoning, ms, prompt_chars). prompt_chars surfaces the
    rough payload size in the trace so daily-token-budget pressure is
    visible (Groq free tier: 100K TPD on llama-3.3-70b-versatile).
    """
    if not chunks:
        return [], "no chunks retrieved", 0, 0

    # Surface `doc_title` in the per-chunk header — for PaperChunks it
    # carries the form "Title (YEAR) — Venue", which the year-discipline
    # rule in the system prompt needs to read in order to apply. Without
    # this the model couldn't enforce "2026 question requires 2026 chunk"
    # for cascade-discovered papers even when the year was known upstream.
    def _hdr(c: HydratedChunk) -> str:
        bits = [c.citation]
        title = (c.doc_title or "").strip()
        if title and title not in c.citation:
            bits.append(title)
        return "  ".join(bits)

    # Phase 15.1 fix: bound the per-call payload.
    #
    # Per-chunk char cap dropped from 1500 -> 800. Year, named entities,
    # and "is-this-on-topic" signal all live in the first ~500 chars of
    # any chunk; we lose nothing by truncating the rest. Halves the
    # Critic's input tokens.
    #
    # Total-chunks cap MAX_CHUNKS_PER_CALL prevents paper_discovery's
    # 5-papers × 5-slices expansion from flooding one Critic prompt.
    # When over the cap, chunks past the cap get auto-stamped "partial"
    # downstream — they're NOT actually graded by the LLM. So WHICH 20
    # we send matters: that's exactly the reorder logic below.
    #
    # Bumped 12 -> 20 because the Critic was upgraded from llama-3.1-8b
    # (Groq) to llama-3.3-70b (Cerebras/Groq cascade). The 70B model
    # handles ~25 chunks per call comfortably inside the token budget,
    # and 12 was leaving paper_discovery's fresh PDF slices ungraded.
    MAX_CHUNK_CHARS = 800
    MAX_CHUNKS_PER_CALL = 20

    # Critical ordering rule for the FIRST MAX_CHUNKS_PER_CALL chunks:
    # fresh external discoveries (paper:*, web:*) must beat local chunks
    # for the grading window. paper_discovery.py appends new PaperChunks
    # *after* existing retriever chunks in the merged refs list, so
    # without this reorder the freshly-downloaded AutoGen PDF slices sit
    # at positions 10-25 and never see the LLM — they get the silent
    # "partial" auto-fill on line "Extend with partial ..." below.
    #
    # CRITICAL: the caller `critique()` slices `original_refs[i]` by
    # grade index `i`. So whatever grades-list we RETURN must be aligned
    # with the ORIGINAL chunk order, not the reordered one. We reorder
    # only for the LLM call, then undo the permutation on the way out.
    #
    # The reorder is stable within each bucket so the upstream RRF +
    # paper-rerank ordering is preserved.
    def _bucket(c: HydratedChunk) -> int:
        sig = (getattr(c, "signal", "") or "").lower()
        if sig.startswith("paper:"):
            return 0  # cascade-fetched PDFs first — they paid the cost
        if sig.startswith("web:"):
            return 1  # web fallbacks second
        return 2      # local Chroma/BM25 chunks last

    # Build the LLM-call order with a permutation map back to the input.
    # `perm[reordered_idx] = original_idx`
    perm = sorted(range(len(chunks)), key=lambda i: (_bucket(chunks[i]), i))
    reordered = [chunks[i] for i in perm]

    chunks_to_grade = reordered[:MAX_CHUNKS_PER_CALL]
    numbered = "\n\n".join(
        f"[Chunk {i+1}] {_hdr(c)}\n{c.text.strip()[:MAX_CHUNK_CHARS]}"
        for i, c in enumerate(chunks_to_grade)
    )
    user = f"Question: {sub_q}\n\nChunks to grade:\n{numbered}"
    prompt_chars = len(_SYSTEM) + len(user)

    t0 = time.perf_counter()
    raw = chat(
        messages=[{"role": "system", "content": _SYSTEM}, {"role": "user", "content": user}],
        tier=ModelTier.FAST,
        temperature=0.0,
        max_tokens=400,
    )
    dur_ms = int((time.perf_counter() - t0) * 1000)

    parsed = _extract_json(raw) or {}
    grades_raw = parsed.get("grades") or []
    reasoning = str(parsed.get("reasoning") or "")

    # Normalize + validate length. We graded `chunks_to_grade` (capped at
    # MAX_CHUNKS_PER_CALL). Any chunks past the cap default to "partial"
    # so the caller can still slot a grade for every input chunk and the
    # filter step downstream behaves correctly. "partial" not "relevant"
    # because we genuinely haven't graded them — we don't want to wave
    # through ungraded evidence as if it were vetted.
    valid = {"relevant", "partial", "irrelevant"}
    grades = [g.lower().strip() if isinstance(g, str) else "partial" for g in grades_raw]
    grades = [g if g in valid else "partial" for g in grades]
    if len(grades) != len(chunks_to_grade):
        # LLM returned wrong-length list — fill/truncate to the GRADED slice.
        grades = (grades + ["partial"] * len(chunks_to_grade))[: len(chunks_to_grade)]
        reasoning = f"(length mismatch; padded) {reasoning}"
    if len(reordered) > len(chunks_to_grade):
        # Extend with "partial" for the chunks we deliberately skipped.
        grades = grades + ["partial"] * (len(reordered) - len(chunks_to_grade))

    # Undo the permutation so `grades[i]` aligns with the ORIGINAL chunks[i]
    # — that's the contract the caller `critique()` relies on when it slices
    # `original_refs[i]` to build `kept_refs`. Without this, paper:* chunks
    # the LLM graded "relevant" at the front of the reordered list would be
    # attributed to (and kept as) whatever local chunk happened to sit at
    # the same index in the input — a silent eviction.
    #
    # `perm[reordered_idx] = original_idx`, so:
    #   grades_in_original_order[original_idx] = grades[reordered_idx]
    grades_in_original_order: list[str] = ["partial"] * len(chunks)
    for reordered_idx, original_idx in enumerate(perm):
        if reordered_idx < len(grades):
            grades_in_original_order[original_idx] = grades[reordered_idx]

    return grades_in_original_order, reasoning, dur_ms, prompt_chars


def _has_external_fresh_source(chunks: list[HydratedChunk]) -> bool:
    """
    True iff the pool contains any chunk freshly discovered by the Stage-2
    cascade — `paper_discovery` (arxiv / semantic_scholar) or `web_fallback`
    (tavily / serper / duckduckgo).

    HydratedChunk.signal carries the provenance verbatim:
      - "local"                — Chroma / BM25 hit
      - "paper:arxiv"          — arXiv discovery
      - "paper:semantic_scholar" — S2 discovery
      - "web:tavily" / "web:serper" / "web:duckduckgo"
      - "graph"                — knowledge-graph expansion

    Used by `_derive_verdict` to relax the HIGH threshold (0.85 -> 0.70)
    when fresh external evidence is in play. The cascade paid for these
    chunks (S2 round-trip, PDF fetch+parse, web search) so when they
    contain the user's named specifics they should be allowed to settle
    the verdict at HIGH without needing local-corpus levels of redundancy.
    """
    for c in chunks:
        sig = (getattr(c, "signal", "") or "").lower()
        if sig.startswith("paper:") or sig.startswith("web:"):
            return True
    return False


def _derive_verdict(
    all_grades: list[str], all_chunks: list[HydratedChunk] | None = None
) -> str:
    """
    Deterministic confidence verdict from grades.

    Phase 15.2 weighted-score policy. Per the spec:
      - "partial" is now worth 0.75 (was the implicit 0.5 baked into the
        absolute-count bands). The rationale: a partial grade means "right
        topic, missing some details" — that's most of the way to a
        usable answer, especially when paired with the cascade-generated
        evidence that surrounds it.
      - HIGH threshold relaxes from 0.85 -> 0.70 when the chunk pool
        includes a fresh external discovery (paper_discovery or
        web_fallback). These chunks cost a live API call to surface; we
        should not require them to also clear the strictest local-corpus
        bar before auto-saving the answer.

    Final formula:
        score        = (1.00 * relevant_n + 0.75 * partial_n) / total
        threshold_hi = 0.70 if pool has external-fresh chunks else 0.85

        score >= threshold_hi                  -> high
        relevant_n >= 1 OR partial_n >= 2       -> medium
        otherwise                               -> low

    The MEDIUM and LOW bands are deliberately unchanged: any real signal
    (one relevant hit OR two partials) is worth synthesising into a draft
    the cascade can then strengthen. Empty / all-irrelevant pools still
    force escalation.
    """
    if not all_grades:
        return "low"

    n = len(all_grades)
    relevant_n = sum(1 for g in all_grades if g == "relevant")
    partial_n = sum(1 for g in all_grades if g == "partial")

    # Phase 15.2: partial weight bumped 0.5 -> 0.75 per the leniency spec.
    _W_RELEVANT = 1.00
    _W_PARTIAL = 0.75
    score = (relevant_n * _W_RELEVANT + partial_n * _W_PARTIAL) / n

    # Source-aware HIGH threshold. Empty pool defaults to the strict bar.
    has_external = bool(all_chunks) and _has_external_fresh_source(all_chunks)
    threshold_hi = 0.70 if has_external else 0.85

    if score >= threshold_hi:
        return "high"
    if relevant_n >= 1 or partial_n >= 2:
        return "medium"
    return "low"


def critique(state: AgentState) -> dict[str, Any]:
    """
    Grade every chunk under every sub-question, set confidence + reasoning.

    Side effect on chunks_by_subq: we DROP "irrelevant" chunks before passing
    state forward. The generator (and any web-fallback decision) only sees
    chunks that survived the filter. Combined with confidence routing, this
    is the corrective half of "Corrective RAG".
    """
    # Pointer-state: refs come in as dicts; hydrate them HERE to do grading,
    # then return surviving refs (sliced from the input). The input refs
    # are dict-typed so we can re-emit them by index without round-tripping
    # through `persist_mixed` again — saving a PG write on every critic call.
    refs_by_subq: dict[str, list[dict[str, str]]] = dict(state.get("chunk_refs_by_subq") or {})
    if not refs_by_subq or not any(refs_by_subq.values()):
        return {
            "confidence": "low",
            "critic_reasoning": "no chunks retrieved",
            "trace": [{"node": "critic", "skipped": "empty_input"}],
        }

    # Hydrate every sub-Q together — one ChromaDB call + one PG query total.
    chunks_by_subq = hydrate(refs_by_subq)

    new_refs_by_subq: dict[str, list[dict[str, str]]] = {}
    all_grades: list[str] = []
    reasonings: list[str] = []
    total_ms = 0
    total_chunks_in = 0
    total_chunks_kept = 0
    total_prompt_chars = 0
    # Phase 15.2: collect every chunk we grade so `_derive_verdict` can
    # inspect provenance signals (paper:* / web:*) and pick the right
    # HIGH threshold. We only need the chunk objects, not the refs.
    all_graded_chunks: list[HydratedChunk] = []

    for sq, chunks in chunks_by_subq.items():
        total_chunks_in += len(chunks)
        grades, reasoning, ms, prompt_chars = _grade_one_subq(sq, chunks)
        total_ms += ms
        total_prompt_chars += prompt_chars
        all_grades.extend(grades)
        all_graded_chunks.extend(chunks)
        if reasoning:
            reasonings.append(reasoning)
        # Keep relevant + partial. Drop irrelevant. Refs sliced from the
        # ORIGINAL input refs by index so we don't write new artifacts.
        original_refs = refs_by_subq.get(sq) or []
        kept_refs = [
            original_refs[i]
            for i, g in enumerate(grades)
            if i < len(original_refs) and g != "irrelevant"
        ]
        new_refs_by_subq[sq] = kept_refs
        total_chunks_kept += len(kept_refs)

    verdict = _derive_verdict(all_grades, all_graded_chunks)
    has_external = _has_external_fresh_source(all_graded_chunks)
    threshold_hi = 0.70 if has_external else 0.85
    # Weighted-score snapshot so the trace shows WHY a given verdict landed.
    rel_n = all_grades.count("relevant")
    par_n = all_grades.count("partial")
    weighted_score = (
        (rel_n * 1.00 + par_n * 0.75) / max(len(all_grades), 1)
    )
    summary = "; ".join(reasonings)[:300] if reasonings else ""

    return {
        "chunk_refs_by_subq": new_refs_by_subq,
        "confidence": verdict,
        "critic_reasoning": summary,
        "trace": [
            {
                "node": "critic",
                "duration_ms": total_ms,
                "chunks_in": total_chunks_in,
                "chunks_kept": total_chunks_kept,
                "verdict": verdict,
                # Phase 15.2: surface the new weighted-score math + which
                # threshold applied so a low/medium verdict is explainable
                # without having to reproduce the formula by hand.
                "score": round(weighted_score, 3),
                "threshold_hi": threshold_hi,
                "external_fresh_pool": has_external,
                # Phase 15.1 telemetry: rough payload size across all sub-Q
                # grading calls combined. Divide by ~4 for a ballpark token
                # count. Bubbles into the CLI trace so day-budget pressure
                # (Groq free tier: 100K TPD) is observable per-run.
                "prompt_chars": total_prompt_chars,
                "est_tokens": total_prompt_chars // 4,
                "grades": {
                    "relevant": all_grades.count("relevant"),
                    "partial": all_grades.count("partial"),
                    "irrelevant": all_grades.count("irrelevant"),
                },
            }
        ],
    }
