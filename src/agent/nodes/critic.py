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


def _grade_one_subq(sub_q: str, chunks: list[HydratedChunk]) -> tuple[list[str], str, int]:
    """
    Grade chunks for a single sub-question. Returns (grades, reasoning, ms).

    Falls back to all-"partial" on parse failure — safer default than
    all-"relevant" (would skip CRAG corrections) or all-"irrelevant"
    (would trigger unnecessary rewrites).
    """
    if not chunks:
        return [], "no chunks retrieved", 0

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

    numbered = "\n\n".join(
        f"[Chunk {i+1}] {_hdr(c)}\n{c.text.strip()[:1500]}"
        for i, c in enumerate(chunks)
    )
    user = f"Question: {sub_q}\n\nChunks to grade:\n{numbered}"

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

    # Normalize + validate length
    valid = {"relevant", "partial", "irrelevant"}
    grades = [g.lower().strip() if isinstance(g, str) else "partial" for g in grades_raw]
    grades = [g if g in valid else "partial" for g in grades]
    if len(grades) != len(chunks):
        # Length mismatch — fill or truncate. Conservative default = "partial"
        grades = (grades + ["partial"] * len(chunks))[: len(chunks)]
        reasoning = f"(length mismatch; padded) {reasoning}"

    return grades, reasoning, dur_ms


def _derive_verdict(all_grades: list[str]) -> str:
    """
    Deterministic confidence verdict from grades.

    Tightened policy (Phase 15 v2 — "do NOT be generous"). The user
    rationale: anything below HIGH triggers paper_discovery + web_fallback
    automatically, so we want HIGH reserved for cases where the local corpus
    genuinely answers the question. False positives (HIGH on thin evidence)
    silently degrade the final answer because the cascade never fires;
    false negatives (MEDIUM on slightly-thin evidence) cost a few seconds of
    live S2 latency and produce a strictly better answer.

    New bands:
      high    = strong evidence:
                  relevant_n >= 4 AND fraction_relevant >= 0.6 AND
                  irrelevant_n / total <= 0.25
                ("at least 4 directly-relevant chunks, majority of the pool
                 is relevant, and the pool isn't half-irrelevant noise")
      medium  = any signal worth synthesising:
                  relevant_n >= 1 OR partial_n >= 2
                (one direct hit OR enough on-topic background to draft an
                 answer the cascade can then strengthen)
      low     = nothing meaningful: zero relevant, <2 partial
                (forces escalation through paper_discovery / web_fallback)

    Why no "frac_relevant>=X without absolute floor": small-batch retrieval
    (k=3) used to award HIGH at 2/3 relevant; the cascade never fired and
    thin evidence shipped. The absolute floor of 4 closes that path.
    """
    if not all_grades:
        return "low"
    n = len(all_grades)
    relevant_n = sum(1 for g in all_grades if g == "relevant")
    partial_n = sum(1 for g in all_grades if g == "partial")
    irrelevant_n = n - relevant_n - partial_n
    frac_rel = relevant_n / n
    frac_irrel = irrelevant_n / n

    # HIGH — earned only by a strong, predominantly-relevant pool.
    if relevant_n >= 4 and frac_rel >= 0.6 and frac_irrel <= 0.25:
        return "high"
    # MEDIUM — some real signal. One genuine hit OR two on-topic chunks
    # to anchor the draft; the cascade will fill the rest.
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

    for sq, chunks in chunks_by_subq.items():
        total_chunks_in += len(chunks)
        grades, reasoning, ms = _grade_one_subq(sq, chunks)
        total_ms += ms
        all_grades.extend(grades)
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

    verdict = _derive_verdict(all_grades)
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
                "grades": {
                    "relevant": all_grades.count("relevant"),
                    "partial": all_grades.count("partial"),
                    "irrelevant": all_grades.count("irrelevant"),
                },
            }
        ],
    }
