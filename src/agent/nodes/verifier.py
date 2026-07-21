"""
Verifier node — grades whether each [S#]-tagged claim in the generator's draft
is actually supported by the cited source's text.

This is POST-GENERATION claim verification, distinct from the Critic's
PRE-GENERATION chunk-relevance grading. The Critic asks "is this chunk
topically relevant to the sub-question?" — the Verifier asks "does the
specific sentence the Generator wrote, citing [S3], get supported by S3's
text?"

Strategy: one FAST-tier batch call, pulling every [S#]-tagged sentence
from the draft and pairing it with the cited source's text. The batch
size is bounded by max_citations (configurable, default 12) — the FAST
tier handles this comfortably within its token budget.

Why a strict 3-way verdict (not a 5-point scale)
-----------------------------------------------
The output feeds into the frontend as a traffic-light badge (green ✓ /
red ✗ / amber ?). A fine-grained scale adds UI complexity without
actionable value for the reader. "Contradicts" is the most valuable
signal — it's the "this source says the opposite of what the answer
claims" case that catches actual fabrication.
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


_SYSTEM = """You are a STRICT claim-verification auditor for a research assistant.

Your job: for each [S#]-tagged claim in the draft answer, read the cited
source's text and decide whether the claim is actually supported.

Per-claim verdict:
  - "supports"     : the source text CONTAINS the specific fact, entity,
                     number, or claim the sentence asserts. You can and MUST quote
                     the exact supporting passage verbatim.
  - "contradicts"  : the source says something that CONTRADICTS the claim
                     (opposite conclusion, different numbers, different entity,
                     different time period). Quote the contradicting passage.
  - "unclear"      : the source text is on the same TOPIC but doesn't explicitly
                     confirm or deny the claim — or the source text is too
                     vague/noisy to determine either way.

DO NOT be generous. If you cannot point to a specific passage that confirms
the claim, mark "unclear" or "contradicts". The reader needs to know which
citations they can trust vs. which need independent verification.

CONTRADICTIONS ARE THE MOST IMPORTANT SIGNAL:
  A "contradicts" verdict means the cited source DISAGREES with what the
  answer claims. That's a hallucination or misreading. If the source says
  "X increased by 5%" but the answer claims "X decreased by 7%", that's
  "contradicts" — not "unclear".

Names, numbers, and years are the easiest things to spot-check. If the
claim says "in 2024" and the source is from 2023, that's "unclear" at best
(claim may or may not be correct, but this source can't confirm it).

Output ONLY a JSON object, no preamble:
{
  "verdicts": {
    "S1": "supports",
    "S2": "unclear",
    "S3": "contradicts"
  },
  "evidence": {
    "S1": "exact quote from source confirming the claim, max 120 chars",
    "S2": "",
    "S3": "exact quote from source contradicting the claim, max 120 chars"
  },
  "reasoning": "one-line summary: what held up and what needed scrutiny"
}

`verdicts` MUST have exactly the same keys as the claims provided.
`evidence` is optional — only fill for "supports" and "contradicts" verdicts."""


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


def _extract_tagged_claims(answer: str, tags: list[str]) -> dict[str, str]:
    """
    Pull sentence(s) around each [S#] tag from the answer.

    For each tag, finds the nearest sentence boundary containing that tag
    and returns it as the claim text to verify. When the tag appears inside
    a paragraph with no clear sentence break, returns up to 200 chars
    surrounding the tag.
    """
    claims: dict[str, str] = {}
    for tag in tags:
        idx = answer.find(f"[{tag}]")
        if idx == -1:
            idx = answer.find(f"[{tag}]".replace("[", "[").replace("]", "]"))
        if idx == -1:
            continue
        # Expand outward to find sentence boundaries
        start = max(0, idx - 120)
        end = min(len(answer), idx + 200)
        snippet = answer[start:end].strip()
        # Try to find proper sentence boundaries
        # Look backwards for sentence start (period + space, or start of line)
        for sep in [". ", ".\n", "? ", "!\n", "\n\n"]:
            pos = snippet.rfind(sep, 0, idx - start)
            if pos > 0:
                snippet = snippet[pos + len(sep):].strip()
                break
        # Look forward for sentence end
        for sep in [". ", ".\n", "? ", "!\n", "\n\n"]:
            pos = snippet.find(sep)
            if pos > 20:
                snippet = snippet[:pos].strip() + "."
                break
        # Ensure the tag is still in the trimmed snippet
        if f"[{tag}]" not in snippet:
            # Fall back to a fixed window around the tag
            snippet = answer[max(0, idx - 60):min(len(answer), idx + 180)]
        claims[tag] = snippet[:400]
    return claims


def _verify_claims(
    claims_texts: dict[str, str],
    source_texts: dict[str, str],
) -> tuple[dict[str, str], dict[str, str], str]:
    """
    One FAST-tier batch call: pair each tagged claim with its source text.

    Returns (verdicts, evidence, reasoning_ms, prompt_chars).
    """
    if not claims_texts:
        return {}, "", 0, 0

    # Build the prompt: one numbered claim+source pair per block
    pairs = []
    tags_ordered = list(claims_texts.keys())
    for i, tag in enumerate(tags_ordered):
        claim = claims_texts.get(tag, "")
        source = source_texts.get(tag, "")
        pairs.append(
            f"[Claim {tag}]\n"
            f"  Draft sentence: {claim[:300]}\n"
            f"  Source text: {source[:600]}"
        )

    user = f"Verify each claim against its cited source:\n\n" + "\n\n".join(pairs)
    prompt_chars = len(_SYSTEM) + len(user)

    t0 = time.perf_counter()
    raw = chat(
        messages=[{"role": "system", "content": _SYSTEM}, {"role": "user", "content": user}],
        tier=ModelTier.FAST,
        temperature=0.0,
        max_tokens=600,
    )
    dur_ms = int((time.perf_counter() - t0) * 1000)

    parsed = _extract_json(raw) or {}
    verdicts_raw = parsed.get("verdicts") or {}
    evidence_raw = parsed.get("evidence") or {}
    reasoning = str(parsed.get("reasoning") or "")

    # Normalize against expected tags
    valid = {"supports", "contradicts", "unclear"}
    verdicts: dict[str, str] = {}
    evidence: dict[str, str] = {}
    for tag in tags_ordered:
        v = str(verdicts_raw.get(tag, "")).lower().strip()
        verdicts[tag] = v if v in valid else "unclear"
        evidence[tag] = str(evidence_raw.get(tag, ""))[:120]

    return verdicts, evidence, reasoning, dur_ms, prompt_chars


# Batched into one call — no per-tag loop.
# Cap at 12 citations (configurable via CITATION_VERIFY_MAX_SOURCES);
# beyond that the FAST-tier token budget gets tight and per-citation
# attention degrades.
_DEFAULT_MAX_SOURCES = 12


def verify(state: AgentState) -> dict[str, Any]:
    """
    Post-generation claim verification.

    Reads draft_answer + citation_refs, pairs each tagged claim with its
    source text, and returns citation_verdicts + citation_evidence.
    """
    answer = state.get("draft_answer") or ""
    citation_refs = state.get("citation_refs") or {}

    if not answer or not citation_refs:
        return {
            "citation_verdicts": {},
            "trace": [{"node": "verifier", "skipped": "no_citations_or_answer"}],
        }

    tags = sorted(citation_refs.keys(), key=lambda t: int(t[1:]))
    max_sources = int(
        getattr(settings, "citation_verify_max_sources", None) or _DEFAULT_MAX_SOURCES
    )
    if len(tags) > max_sources:
        tags = tags[:max_sources]

    refs = [citation_refs[t] for t in tags]
    hydrated = hydrate_one(refs)

    claims = _extract_tagged_claims(answer, tags)
    source_texts = {tag: hc.text[:800] for tag, hc in zip(tags, hydrated) if tag in claims}

    # Only verify tags where we found a claim AND have source text
    tags_to_verify = [t for t in tags if t in claims and t in source_texts]
    if not tags_to_verify:
        return {
            "citation_verdicts": {},
            "trace": [{"node": "verifier", "skipped": "no_claim_source_pairs"}],
        }

    verdicts, evidence, reasoning, dur_ms, prompt_chars = _verify_claims(
        {t: claims[t] for t in tags_to_verify},
        {t: source_texts[t] for t in tags_to_verify},
    )

    return {
        "citation_verdicts": verdicts,
        # evidence dict is for trace only — not persisted to state
        "trace": [
            {
                "node": "verifier",
                "verdicts": verdicts,
                "evidence": evidence,
                "reasoning": reasoning[:200],
                "duration_ms": dur_ms,
                "claims_verified": len(tags_to_verify),
            },
        ],
    }