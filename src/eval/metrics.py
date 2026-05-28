"""
RAGAS-style metrics — computed via our own LLM stack, no extra deps.

Why roll our own instead of `pip install ragas`
-----------------------------------------------
The reference `ragas` library pulls in langchain, multiple embedding
implementations, and ~200MB of transitive deps. For our three core metrics
(faithfulness, answer relevancy, context precision) the actual logic is
~50 lines each. Self-contained means:
  - No version conflicts with our LangGraph install
  - Metrics use the SAME provider cascade as the agent, so a Cerebras
    rate-limit during eval falls through to NVIDIA just like the agent
  - Every eval call lands in the same `data/llm_calls.jsonl` observability
    log alongside agent calls

Metrics
-------
1. FAITHFULNESS (anti-hallucination)
   Extract atomic factual claims from the answer with an LLM, then for each
   claim ask "is this supported by the cited evidence?". Score = supported / total.
   This is THE most important metric — it catches the failure mode that
   makes LLMs untrustworthy.

2. ANSWER RELEVANCY (did we actually answer the question?)
   Ask an LLM: "What question would this answer be a natural response to?"
   Generate 3 candidate questions. Cosine-similarity each to the original
   question. Score = mean similarity. Low score = answer drifted off-topic.

3. CONTEXT PRECISION (was retrieval useful?)
   For each cited chunk, ask: "did this chunk meaningfully contribute to
   answering the question?". Score = useful_chunks / total_chunks. This
   is similar to the agent's Critic, but evaluated AFTER seeing what the
   final answer used the chunks for.

Each metric returns a float in [0, 1]. Higher is better.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.config import ModelTier
from src.llm import chat, embed
from src.retrieval import HybridChunk, WebChunk

ContextChunk = HybridChunk | WebChunk


@dataclass
class EvalScores:
    """All three RAGAS-style scores plus the raw counts that produced them."""

    faithfulness: float = 0.0          # 0..1, higher = fewer unsupported claims
    answer_relevancy: float = 0.0      # 0..1, higher = better answer-question alignment
    context_precision: float = 0.0     # 0..1, higher = fewer useless cited chunks

    # Diagnostic detail — surface what the metric saw so failures are debuggable.
    n_claims: int = 0
    n_supported_claims: int = 0
    n_chunks_evaluated: int = 0
    n_useful_chunks: int = 0
    relevancy_similarities: list[float] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def overall(self) -> float:
        """Equal-weight mean. Tunable later if some metrics matter more."""
        return (self.faithfulness + self.answer_relevancy + self.context_precision) / 3.0


_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_json(text: str) -> dict | None:
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


# ---------------------------------------------------------------------------
# 1. Faithfulness
# ---------------------------------------------------------------------------

_CLAIM_EXTRACT_SYSTEM = """Extract every standalone factual claim from the given answer.

A claim is a single, atomic factual statement that could be true or false on \
its own. Strip citations and hedges. List one per line, no numbering, no preamble.

Example input: "CRAG uses a T5-based retrieval evaluator [S2]. It scores chunks \
from -1 to 1 [S2] and triggers correction when confidence is low."

Example output:
CRAG uses a T5-based retrieval evaluator
CRAG scores chunks from -1 to 1
CRAG triggers correction when retrieval confidence is low

Output the claims now, one per line, no other text."""

_CLAIM_SUPPORT_SYSTEM = """Given a single factual CLAIM and a body of EVIDENCE, decide:
  YES   — the evidence directly supports the claim
  NO    — the evidence contradicts the claim, or doesn't mention it
  PARTIAL — the evidence is related but doesn't fully establish the claim

Respond with ONLY one word: YES, NO, or PARTIAL."""


def _extract_claims(answer: str) -> list[str]:
    """LLM-extract atomic claims from the answer text."""
    if not answer.strip():
        return []
    out = chat(
        messages=[
            {"role": "system", "content": _CLAIM_EXTRACT_SYSTEM},
            {"role": "user", "content": answer},
        ],
        tier=ModelTier.FAST,
        temperature=0.0,
        max_tokens=500,
    )
    claims = [ln.strip(" -*•\t") for ln in out.splitlines() if ln.strip()]
    # Drop empty / too-short / obviously non-claim noise
    return [c for c in claims if len(c) >= 12]


def _faithfulness(answer: str, evidence_text: str) -> tuple[float, int, int]:
    """Returns (score, n_claims, n_supported)."""
    claims = _extract_claims(answer)
    if not claims:
        return 1.0, 0, 0  # vacuously faithful if there's nothing to support
    supported = 0
    for claim in claims:
        verdict = chat(
            messages=[
                {"role": "system", "content": _CLAIM_SUPPORT_SYSTEM},
                {"role": "user", "content": f"CLAIM: {claim}\n\nEVIDENCE:\n{evidence_text}"},
            ],
            tier=ModelTier.FAST,
            temperature=0.0,
            max_tokens=10,
        ).strip().upper()
        # Be generous with PARTIAL (count as half-credit) — strict YES/NO
        # binary punishes hedged claims too harshly.
        if verdict.startswith("YES"):
            supported += 1.0
        elif verdict.startswith("PARTIAL"):
            supported += 0.5
    score = supported / len(claims)
    # Round counts for display; keep the float for math.
    return float(score), len(claims), int(round(supported))


# ---------------------------------------------------------------------------
# 2. Answer Relevancy
# ---------------------------------------------------------------------------

_GEN_QUESTIONS_SYSTEM = """Given an answer, generate 3 different questions that \
this answer would be a natural and complete response to. Each question should \
sound like something a researcher would actually ask.

Output the 3 questions as a JSON array, one per element. Output ONLY the JSON \
array, no preamble, no markdown fence:

["question 1", "question 2", "question 3"]"""


def _generate_candidate_questions(answer: str) -> list[str]:
    if not answer.strip():
        return []
    raw = chat(
        messages=[
            {"role": "system", "content": _GEN_QUESTIONS_SYSTEM},
            {"role": "user", "content": answer},
        ],
        tier=ModelTier.FAST,
        temperature=0.3,
        max_tokens=300,
    )
    parsed = _parse_json(raw)
    if isinstance(parsed, list):
        return [str(q).strip() for q in parsed if str(q).strip()][:3]
    # Fallback: split on newlines and clean
    return [
        re.sub(r"^[\d\-\.\)\s]+", "", ln).strip().strip('"')
        for ln in raw.splitlines() if ln.strip()
    ][:3]


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a) + 1e-12
    nb = np.linalg.norm(b) + 1e-12
    return float(np.dot(a, b) / (na * nb))


def _answer_relevancy(question: str, answer: str) -> tuple[float, list[float]]:
    """Returns (mean_similarity, all_similarities)."""
    candidates = _generate_candidate_questions(answer)
    if not candidates:
        return 0.0, []
    # Embed original + candidates in one batch.
    vectors = embed([question] + candidates, tier=ModelTier.EMBED)
    if not vectors or len(vectors) < 2:
        return 0.0, []
    qv = np.asarray(vectors[0], dtype=np.float32)
    sims = [_cosine(qv, np.asarray(v, dtype=np.float32)) for v in vectors[1:]]
    mean = float(sum(sims) / len(sims))
    return mean, sims


# ---------------------------------------------------------------------------
# 3. Context Precision
# ---------------------------------------------------------------------------

_CHUNK_USEFUL_SYSTEM = """Given a QUESTION and a single retrieved CHUNK, decide whether \
the chunk contributes meaningfully to answering the question.

Respond with ONLY one word:
  YES — the chunk contains specific info that would appear in (or substantially support) the answer
  NO  — the chunk is off-topic or only tangentially related"""


def _context_precision(question: str, chunks: list[ContextChunk]) -> tuple[float, int, int]:
    """Returns (score, n_evaluated, n_useful)."""
    if not chunks:
        return 0.0, 0, 0
    useful = 0
    for c in chunks:
        snippet = c.text.strip()
        if len(snippet) > 1500:
            snippet = snippet[:1500] + " ..."
        verdict = chat(
            messages=[
                {"role": "system", "content": _CHUNK_USEFUL_SYSTEM},
                {"role": "user", "content": f"QUESTION: {question}\n\nCHUNK ({c.citation}):\n{snippet}"},
            ],
            tier=ModelTier.FAST,
            temperature=0.0,
            max_tokens=10,
        ).strip().upper()
        if verdict.startswith("YES"):
            useful += 1
    return useful / len(chunks), len(chunks), useful


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------


def score_run(
    question: str,
    answer: str,
    cited_chunks: list[ContextChunk],
) -> EvalScores:
    """
    Score one agent run. Pure function — no I/O beyond the LLM calls themselves.

    The runner does the I/O (writing results to JSONL, aggregating). This
    separation makes the metric functions individually testable.
    """
    scores = EvalScores()

    # ---- Faithfulness ----
    if cited_chunks:
        evidence_text = "\n\n".join(
            f"[{i+1}] {c.citation}\n{c.text.strip()[:1500]}"
            for i, c in enumerate(cited_chunks)
        )
    else:
        evidence_text = "(no cited evidence)"
    try:
        f_score, n_claims, n_supp = _faithfulness(answer, evidence_text)
        scores.faithfulness = f_score
        scores.n_claims = n_claims
        scores.n_supported_claims = n_supp
    except Exception as e:
        scores.notes.append(f"faithfulness skipped: {type(e).__name__}: {e}")

    # ---- Answer relevancy ----
    try:
        r_score, sims = _answer_relevancy(question, answer)
        scores.answer_relevancy = r_score
        scores.relevancy_similarities = sims
    except Exception as e:
        scores.notes.append(f"relevancy skipped: {type(e).__name__}: {e}")

    # ---- Context precision ----
    try:
        p_score, n_eval, n_useful = _context_precision(question, cited_chunks)
        scores.context_precision = p_score
        scores.n_chunks_evaluated = n_eval
        scores.n_useful_chunks = n_useful
    except Exception as e:
        scores.notes.append(f"context_precision skipped: {type(e).__name__}: {e}")

    return scores
