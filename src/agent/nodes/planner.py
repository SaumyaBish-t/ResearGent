"""
Planner node — decomposes a user question into one or more sub-questions.

Why decompose at all?
---------------------
Naive RAG retrieves once for the whole query. That works for simple lookups
("What is CRAG?") but fails badly on:
  - Comparisons:   "How do CRAG and Self-RAG differ on low-confidence retrieval?"
  - Multi-hop:     "Which paper introduced ISREL tokens, and what year?"
  - Conjunctions:  "Explain Self-RAG's training objective AND its inference loop."

In each case the right behavior is to retrieve SEPARATELY for each
conceptual axis, then synthesize. The planner does step one — the
conceptual decomposition — using the REASONING tier.

Output contract
---------------
The planner returns JSON of the form:
    {
      "is_complex": bool,
      "sub_questions": [str, ...],   # length 1 for simple, >1 for complex
      "reasoning": str               # one-sentence explanation
    }

We parse defensively — if the model returns malformed JSON, we fall back
to treating the question as simple (sub_questions = [original]) so the
graph keeps moving. Better to retrieve once on the raw question than to
crash the run.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

from src.agent.state import AgentState
from src.config import ModelTier
from src.llm import chat


_SYSTEM = """You are a research query planner. Given a user question, decide whether \
it can be answered with a single retrieval, or whether it needs to be decomposed \
into multiple sub-questions for separate retrieval.

Decompose when the question:
  - asks to compare or contrast two or more concepts/methods
  - involves multiple steps or hops (e.g. "Which X is Y, and why?")
  - bundles unrelated sub-topics with "and" / "also"

Do NOT decompose when:
  - the question is a single factual lookup
  - the question is a definition or explanation of one concept
  - decomposition would create redundant or trivial sub-questions

Output ONLY a JSON object, no preamble, no markdown fence:
{
  "is_complex": true | false,
  "sub_questions": ["question 1", "question 2", ...],
  "reasoning": "one sentence explaining your decision"
}

For simple questions, sub_questions must be a one-element list containing the \
ORIGINAL question verbatim. For complex questions, write 2-4 specific, \
retrieval-friendly sub-questions that together cover the original."""


_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> dict[str, Any] | None:
    """
    Best-effort JSON extraction.

    Handles three common model behaviors:
      1. Pure JSON object — parse directly.
      2. JSON wrapped in ```json ... ``` fence — strip then parse.
      3. JSON embedded in prose — regex out the first {...} block.
    """
    text = text.strip()
    if text.startswith("```"):
        # strip code fences (with or without "json" tag)
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


def plan(state: AgentState) -> dict[str, Any]:
    """Plan node — populates sub_questions, is_complex, planner_reasoning."""
    question = state["question"]
    t0 = time.perf_counter()

    raw = chat(
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": question},
        ],
        tier=ModelTier.REASONING,
        temperature=0.1,
        max_tokens=400,
    )

    parsed = _extract_json(raw) or {}
    sub_qs_raw = parsed.get("sub_questions") or []
    if not isinstance(sub_qs_raw, list) or not sub_qs_raw:
        # Fallback — planner failed to produce a valid list. Treat as simple.
        sub_questions = [question]
        is_complex = False
        reasoning = f"planner output unparseable ({raw[:80]!r}); treating as simple"
    else:
        sub_questions = [str(q).strip() for q in sub_qs_raw if str(q).strip()]
        is_complex = bool(parsed.get("is_complex")) and len(sub_questions) > 1
        if not sub_questions:
            sub_questions = [question]
            is_complex = False
        reasoning = str(parsed.get("reasoning") or "")

    dur_ms = int((time.perf_counter() - t0) * 1000)

    return {
        "sub_questions": sub_questions,
        "is_complex": is_complex,
        "planner_reasoning": reasoning,
        "trace": [
            {
                "node": "planner",
                "duration_ms": dur_ms,
                "sub_q_count": len(sub_questions),
                "is_complex": is_complex,
            }
        ],
    }
