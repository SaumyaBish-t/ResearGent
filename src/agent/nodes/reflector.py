"""
Reflector node — audits the generator's draft answer.

What this catches that nothing else does
----------------------------------------
The Critic looks at chunks BEFORE generation: it can filter irrelevant
sources but doesn't see what the generator did with the survivors. The
Reflector looks at the FINISHED draft and can catch:

  1. Unsupported claims  — draft asserts "CRAG uses BERT" but no cited
                           source actually says that
  2. Missing sub-topics  — question asked about X and Y, draft only covered X
  3. Glossed contradictions — [S1] says T5, [S3] says BART, draft picks
                              one without acknowledging the disagreement
  4. False hedges        — draft says "the sources don't say X" when [S2]
                           actually addresses X but the generator missed it

When gaps are found, the Reflector generates FOLLOW-UP SUB-QUESTIONS that
get appended to `state["sub_questions"]`. The graph loops back to retriever,
which fetches evidence for the new sub-questions (existing ones are
re-retrieved cheaply from cache), Critic re-grades, generator re-writes
with the expanded evidence set.

Bounded by `settings.reflection_max_iterations` (default 2) to prevent
infinite refinement loops on truly unanswerable questions.

What this DOESN'T do
--------------------
  - Doesn't trigger on stylistic preferences
  - Doesn't trigger on honest "I don't know" when sources truly lack info
  - Doesn't paraphrase or re-judge the answer text itself
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

from src.agent.artifacts import HydratedChunk, hydrate_one
from src.agent.state import AgentState
from src.config import ModelTier, settings
from src.llm import chat


_SYSTEM = """You are a STRICT and CONSERVATIVE research-answer auditor. Given:
  - The user's ORIGINAL question
  - The assistant's DRAFT answer
  - The numbered EVIDENCE sources [S1], [S2], ... used in the draft
  - The list of sub-questions already covered (DO NOT propose follow-ups \
    that are slight rewordings of these)

YOUR DEFAULT IS TO ACCEPT THE DRAFT. Only flag gaps when you have HIGH \
CONFIDENCE that BOTH:
  (a) the missing info is plausibly in the corpus or on the public web, AND
  (b) a SPECIFIC, differently-phrased follow-up would surface it.

What counts as a REAL gap (trigger reflection):
  - The draft makes a specific factual claim that NO cited evidence supports
  - The original question has a clear sub-topic the draft completely ignored
  - Two cited sources directly contradict each other and the draft picks \
    one without noting the disagreement

What does NOT count as a gap (DO NOT trigger reflection):
  - Stylistic preferences (concision, formatting, tone)
  - Topics the user didn't actually ask about
  - HONEST HEDGES like "the sources don't provide X" when the corpus \
    genuinely lacks X. These are CORRECT — accept them.
  - Asking for "more detail" / "specific examples" / "specific conditions" \
    when the draft already cites the most relevant material in the corpus
  - Follow-ups that are slight rewordings of existing sub-questions \
    (adding "recent research studies" / "according to technical reports" / \
     "specific conditions" / "different scenarios" does NOT make a new \
     sub-question — it's the same query with cosmetic decoration)
  - Asking for info that the web fallback already failed to find

When in doubt, ACCEPT. A reasonable answer with one honest gap is far \
better than wasting retrievals chasing info that doesn't exist.

If you flag a gap, propose AT MOST 2 follow-up sub-questions. Each must:
  - Name a SPECIFIC concrete entity / concept / number to look for
  - Be MEANINGFULLY DIFFERENT from every existing sub-question (a different \
    angle, not a synonym shuffle)
  - Be retrieval-friendly (concrete terms, no vague pronouns)

Output ONLY a JSON object, no preamble, no markdown fence:
{
  "gaps_found": true | false,
  "gap_descriptions": ["one-line description of gap 1", ...],
  "follow_up_questions": ["specific retrieval-friendly question 1", ...],
  "reasoning": "one short paragraph; START with 'ACCEPT:' or 'GAP:'"
}

If gaps_found is false, both arrays should be empty."""


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


def _format_evidence(citation_map: dict[str, HydratedChunk]) -> str:
    """Build a numbered evidence block for the reflector to compare draft against."""
    parts: list[str] = []
    for tag, c in sorted(citation_map.items(), key=lambda kv: int(kv[0][1:])):
        snippet = c.text.strip()
        if len(snippet) > 800:
            snippet = snippet[:800] + " ..."
        parts.append(f"[{tag}] {c.citation}\n{snippet}")
    return "\n\n---\n\n".join(parts)


def reflect(state: AgentState) -> dict[str, Any]:
    """
    Audit the draft. Surface gaps + follow-up sub-questions OR accept.

    Critical state mutations on reflection loopback:
      - Append follow-up questions to sub_questions (retriever picks them up)
      - Bump reflection_attempts
      - RESET rewrite_attempts to 0 so the new sub-Qs get the full CRAG
        rewrite budget, not whatever was left from the previous round
    """
    question = state["question"]
    draft = state.get("draft_answer") or ""
    # Pointer-state: rebuild the citation_map from refs ONLY for this node's
    # prompt. The hydrated chunks live in memory; only refs go back to state.
    citation_refs = state.get("citation_refs") or {}
    if citation_refs:
        # Hydrate in tag order so the evidence numbering stays stable.
        ordered_tags = sorted(citation_refs.keys(), key=lambda t: int(t[1:]))
        hydrated = hydrate_one([citation_refs[t] for t in ordered_tags])
        citation_map = dict(zip(ordered_tags, hydrated))
    else:
        citation_map = {}
    existing_sub_qs = state.get("sub_questions") or [question]
    attempts_so_far = int(state.get("reflection_attempts") or 0)

    # Defensive: nothing to reflect on. Accept immediately.
    if not draft.strip():
        return {
            "reflection_attempts": attempts_so_far + 1,
            "reflection_gaps": [],
            "reflection_follow_ups": [],
            "trace": [{"node": "reflector", "skipped": "empty_draft"}],
        }

    evidence_block = _format_evidence(citation_map)
    user_msg = f"""Original question:
{question}

Existing sub-questions already covered:
{chr(10).join(f"  - {s}" for s in existing_sub_qs)}

Draft answer to audit:
---
{draft}
---

Evidence used in the draft:
{evidence_block}

Now audit the draft. Return JSON only."""

    t0 = time.perf_counter()
    raw = chat(
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        tier=ModelTier.REASONING,
        temperature=0.1,
        max_tokens=600,
    )
    dur_ms = int((time.perf_counter() - t0) * 1000)

    parsed = _extract_json(raw) or {}
    gaps_found = bool(parsed.get("gaps_found"))
    gap_descs = parsed.get("gap_descriptions") or []
    follow_ups_raw = parsed.get("follow_up_questions") or []

    # Normalize + dedupe follow-ups. Two passes:
    #  1. Exact case-insensitive match against existing sub-questions
    #  2. Token-overlap heuristic — drop a follow-up if >=70% of its
    #     content words already appear in an existing sub-q (catches
    #     "specific conditions for X" vs "X under specific conditions")
    def _content_tokens(s: str) -> set[str]:
        # Drop function words + punctuation; keep meaningful tokens
        stop = {"what", "how", "the", "is", "are", "of", "for", "in", "to",
                "and", "or", "a", "an", "do", "does", "did", "be", "with",
                "when", "where", "which", "that", "this", "these", "those",
                "by", "on", "at", "as", "into", "from", "their", "its",
                "specific", "according", "different", "various", "such",
                "based", "any", "all", "some"}
        toks = set()
        for raw in re.findall(r"[A-Za-z][A-Za-z0-9_-]+", s.lower()):
            if raw not in stop and len(raw) > 2:
                toks.add(raw)
        return toks

    existing_lower = {s.strip().lower() for s in existing_sub_qs}
    existing_token_sets = [_content_tokens(s) for s in existing_sub_qs]

    follow_ups: list[str] = []
    max_per_loop = settings.reflection_max_follow_ups_per_loop

    for q in follow_ups_raw:
        if not isinstance(q, str):
            continue
        q = q.strip()
        if not q:
            continue
        if q.lower() in existing_lower:
            continue

        # Token-overlap check — catches near-duplicates the LLM produces by
        # adding cosmetic phrases like "specific conditions" or "according
        # to recent studies".
        q_tokens = _content_tokens(q)
        if q_tokens:
            is_near_dup = any(
                (len(q_tokens & ex_tokens) / max(len(q_tokens), 1)) >= 0.7
                for ex_tokens in existing_token_sets
                if ex_tokens
            )
            if is_near_dup:
                continue

        follow_ups.append(q)
        existing_lower.add(q.lower())
        existing_token_sets.append(q_tokens)
        if len(follow_ups) >= max_per_loop:
            break

    # If gaps_found but the model returned NO valid follow-ups, treat as no
    # actionable gaps — looping back without new questions would be pointless.
    if not follow_ups:
        gaps_found = False

    update: dict[str, Any] = {
        "reflection_attempts": attempts_so_far + 1,
        "reflection_gaps": [str(g) for g in gap_descs][:5],
        "reflection_follow_ups": follow_ups,
        "trace": [
            {
                "node": "reflector",
                "duration_ms": dur_ms,
                "attempt": attempts_so_far + 1,
                "gaps_found": gaps_found,
                "follow_up_count": len(follow_ups),
            }
        ],
    }

    if gaps_found:
        # Loopback: extend sub_questions, reset rewrite budget for the new round.
        update["sub_questions"] = list(existing_sub_qs) + follow_ups
        update["rewrite_attempts"] = 0

    return update
