"""Review Planner node — decomposes a literature review request into sections."""
from __future__ import annotations

import json
import re
import time
from typing import Any

from src.config import ModelTier, settings
from src.llm import chat


_SYSTEM = """You are a literature review PLANNER. Given a research topic or comparison \\
request, produce a structured outline for a literature review.

The outline must have:
1. A title for the review
2. 3-6 sections, each with a heading and 1-2 sentence description
3. For each section, 2-4 specific search queries to find relevant papers

Comparison reviews (e.g. "compare X and Y") should have:
- An introduction section framing the comparison axes
- One section per approach/model being compared
- A comparison/synthesis section
- A conclusion

Output ONLY a JSON object, no preamble:
{
  "title": "Literature Review: ...",
  "sections": [
    {
      "heading": "Introduction",
      "description": "Framing and scope of this review",
      "search_queries": ["query 1", "query 2"]
    },
    {
      "heading": "...",
      "description": "...",
      "search_queries": ["..."]
    }
  ]
}"""

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except Exception:
        pass
    m = _JSON_RE.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def plan_review(request: str) -> dict[str, Any]:
    """Plan a literature review from a user request. Returns title + sections."""
    t0 = time.perf_counter()
    raw = chat(
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": request},
        ],
        tier=ModelTier.REASONING,
        temperature=0.1,
        max_tokens=2048,
    )
    dur_ms = int((time.perf_counter() - t0) * 1000)

    parsed = _extract_json(raw) or {}
    sections = parsed.get("sections") or []
    title = str(parsed.get("title", "Literature Review"))

    return {
        "title": title,
        "sections": sections,
        "duration_ms": dur_ms,
    }