"""
LLM-only reasoning fallback — the ABSOLUTE last resort.

Fires when every retrieval path has failed:
  - Local corpus: nothing relevant
  - Rewriter: budget exhausted, still low confidence
  - Paper discovery: arXiv + Semantic Scholar returned no usable results
  - Web fallback: Tavily/Serper/DDG all returned nothing

At this point we have two honest choices:
  a) Refuse to answer ("no_answer" path — what we did pre-Phase-7)
  b) Reason from the model's training-time priors, with a LOUD caveat

This node implements (b). It's better than (a) for low-stakes questions
where the model genuinely knows the answer from training but we couldn't
find a fresh citation. CRITICAL: every such answer is prefixed with an
explicit disclaimer so users never confuse "LLM said X" with "the
sources say X". Citations are EMPTY — there are no sources to cite.

When NOT to enable this
-----------------------
Some users (medical / legal / regulatory) explicitly want "no_answer"
over "model-says". Set settings.llm_reasoning_fallback_enabled=false to
disable — the graph then routes to no_answer for the same condition.
"""

from __future__ import annotations

import time
from typing import Any

from src.agent.state import AgentState
from src.config import ModelTier
from src.llm import chat


_SYSTEM = """You are a research assistant whose external sources (corpus, \
academic databases, web search) have ALL FAILED to find evidence for the \
user's question. Answer from your training-time prior knowledge ONLY.

CRITICAL RULES:
  - Be HONEST about uncertainty. If you don't know, say so plainly.
  - NEVER make up specific numbers, dates, citations, URLs, or papers.
  - Start your answer with the literal line:
      [LLM reasoning — no retrieved sources]
  - End your answer with the literal line:
      _Verify any specific claims independently — this is unverified prior knowledge._
  - Distinguish between high-confidence general knowledge ("transformers \
    use self-attention") and uncertain specifics ("the 2026 winner was \
    probably X — I cannot verify").
  - If the question is about events more recent than your training cutoff, \
    say so and recommend the user check current sources directly."""


def reason(state: AgentState) -> dict[str, Any]:
    """Produce a transparently-unverified answer from LLM priors."""
    question = state["question"]
    sub_qs = state.get("sub_questions") or [question]

    # Give the model the planner's decomposition if any — helps it structure
    # the prior-knowledge answer the same way the user would have gotten
    # from the corpus path.
    if len(sub_qs) > 1:
        user_msg = (
            f"Question: {question}\n\n"
            f"Sub-questions to consider:\n"
            + "\n".join(f"  - {s}" for s in sub_qs)
            + "\n\nAnswer from prior knowledge, with the required disclaimers."
        )
    else:
        user_msg = f"Question: {question}\n\nAnswer from prior knowledge, with the required disclaimers."

    t0 = time.perf_counter()
    answer = chat(
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        tier=ModelTier.REASONING,
        temperature=0.2,
        max_tokens=1200,
    )
    dur_ms = int((time.perf_counter() - t0) * 1000)

    return {
        "draft_answer": answer.strip(),
        # No citations — we deliberately have none. The disclaimer in the
        # prompt is the user-visible signal.
        "citation_map": {},
        "error": "no_sources_used_llm_priors",
        "trace": [
            {
                "node": "llm_reasoning",
                "duration_ms": dur_ms,
                "answer_chars": len(answer),
            }
        ],
    }
