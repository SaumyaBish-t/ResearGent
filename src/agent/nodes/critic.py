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

from src.agent.state import AgentState, ContextChunk
from src.config import ModelTier
from src.llm import chat


_SYSTEM = """You are a strict chunk-relevance grader for a research assistant.

For each numbered chunk, decide whether it is:
  - "relevant"   : directly answers (or substantially contributes to) the question
  - "partial"    : touches the topic but doesn't answer; could support a thorough answer
  - "irrelevant" : same domain but doesn't address this specific question

Output ONLY a JSON object, no preamble, no markdown fence:
{
  "grades": ["relevant" | "partial" | "irrelevant", ...],
  "reasoning": "one short sentence summarizing the overall retrieval quality"
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


def _grade_one_subq(sub_q: str, chunks: list[ContextChunk]) -> tuple[list[str], str, int]:
    """
    Grade chunks for a single sub-question. Returns (grades, reasoning, ms).

    Falls back to all-"partial" on parse failure — safer default than
    all-"relevant" (would skip CRAG corrections) or all-"irrelevant"
    (would trigger unnecessary rewrites).
    """
    if not chunks:
        return [], "no chunks retrieved", 0

    numbered = "\n\n".join(
        f"[Chunk {i+1}] {c.citation}\n{c.text.strip()[:1500]}"
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
    """Deterministic confidence verdict from grades. See module docstring."""
    if not all_grades:
        return "low"
    relevant_n = sum(1 for g in all_grades if g == "relevant")
    frac = relevant_n / len(all_grades)
    if frac >= 0.5:
        return "high"
    if frac >= 0.2:
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
    chunks_by_subq = state.get("chunks_by_subq") or {}
    if not chunks_by_subq:
        return {
            "confidence": "low",
            "critic_reasoning": "no chunks retrieved",
            "trace": [{"node": "critic", "skipped": "empty_input"}],
        }

    new_chunks_by_subq: dict[str, list[ContextChunk]] = {}
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
        # Keep relevant + partial. Drop irrelevant.
        kept = [c for c, g in zip(chunks, grades) if g != "irrelevant"]
        new_chunks_by_subq[sq] = kept
        total_chunks_kept += len(kept)

    verdict = _derive_verdict(all_grades)
    summary = "; ".join(reasonings)[:300] if reasonings else ""

    return {
        "chunks_by_subq": new_chunks_by_subq,
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
