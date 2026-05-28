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

from src.agent.state import AgentState, ContextChunk
from src.config import ModelTier
from src.llm import chat


_SYSTEM = """You are a strict research-answer auditor. Given:
  - The user's ORIGINAL question
  - The assistant's DRAFT answer
  - The numbered EVIDENCE sources [S1], [S2], ... used in the draft

Decide whether the draft has IMPORTANT GAPS that a follow-up retrieval pass \
could fix.

What counts as a GAP (trigger reflection):
  - The draft asserts a specific claim that NO cited evidence supports
  - The original question explicitly asks about a sub-topic the draft \
    doesn't cover at all
  - Two cited sources disagree on a factual point and the draft picks one \
    without acknowledging the contradiction
  - The draft hedges ("the sources don't say X") but X could plausibly be \
    answered by a different retrieval (e.g. rephrasing, different sub-topic)

What does NOT count as a gap (DO NOT trigger reflection):
  - Stylistic preferences (could be more concise, more bullet-list, etc.)
  - Things the user didn't ask about
  - Honest "I don't know" when the corpus genuinely lacks the info AND \
    you cannot think of a more targeted query that would find it
  - The web fallback was already used and still returned thin results

If you flag a gap, propose 1-3 SPECIFIC follow-up sub-questions that \
would fill it. Each follow-up should:
  - Name a specific entity, concept, or claim to look for
  - Be retrieval-friendly (concrete terms, not vague pronouns)
  - Be different from the existing sub-questions

Output ONLY a JSON object, no preamble, no markdown fence:
{
  "gaps_found": true | false,
  "gap_descriptions": ["one-line description of gap 1", ...],
  "follow_up_questions": ["specific retrieval-friendly question 1", ...],
  "reasoning": "one short paragraph summarizing your audit"
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


def _format_evidence(citation_map: dict[str, ContextChunk]) -> str:
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
    citation_map = state.get("citation_map") or {}
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

    # Normalize + dedupe follow-ups against existing sub-questions (case-insensitive).
    existing_lower = {s.strip().lower() for s in existing_sub_qs}
    follow_ups: list[str] = []
    for q in follow_ups_raw:
        if not isinstance(q, str):
            continue
        q = q.strip()
        if q and q.lower() not in existing_lower:
            follow_ups.append(q)
            existing_lower.add(q.lower())

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
