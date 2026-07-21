"""Memory Keeper node — extracts structured knowledge from each research run.

After the generator finishes, this node parses the question + answer + sources
into topics, entities, relationships, and knowledge gaps, and persists them
to the user's personal knowledge graph.

This is what turns ResearGent from a stateless Q&A tool into a research
companion that builds context over time.
"""
from __future__ import annotations

import json
import re
import time
from typing import Any

from src.agent.state import AgentState
from src.config import ModelTier, settings
from src.llm import chat


_SYSTEM = """You are a research MEMORY EXTRACTOR. Given a research question, answer, \\
citation sources, and claim-verdicts, extract structured knowledge that will be \\
stored in a persistent knowledge graph.

Extract:

1. TOPICS — the main research topics/concepts this run covered (3-6). Each needs:
   - label: short noun phrase (e.g. "ReAct Framework", "Transformer Attention")
   - domain: which domain it belongs to (agentic_ai, quant_finance, time_series, or null)

2. ENTITIES — specific named entities mentioned (methods, papers, algorithms,
   datasets, people). Each needs:
   - name: the entity (e.g. "ReAct", "arXiv:2308.08155", "Adam optimizer")
   - type: method | paper | algorithm | dataset | person | concept
   - parent_topic: the label of the topic this entity belongs under

3. RELATIONSHIPS — connections BETWEEN the topics extracted. Each needs:
   - source: one topic label
   - target: another topic label
   - type: compares | extends | contradicts | builds_on | related
   - strength: 0.0 to 1.0

4. GAPS — unanswered questions or knowledge gaps this run exposed or the user
   should investigate next. These are statements like "unresolved: whether X
   scales to Y" or "missing: benchmark results for Z under W conditions".
   Extract 0-3 gaps.

Output ONLY a JSON object, no preamble, no markdown fence:
{
  "topics": [{"label": "...", "domain": "..."}],
  "entities": [{"name": "...", "type": "method|paper|algorithm|dataset|person|concept", "parent_topic": "..."}],
  "relationships": [{"source": "...", "target": "...", "type": "compares|extends|contradicts|builds_on|related", "strength": 0.5}],
  "gaps": ["unresolved: ..."]
}"""

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


def extract_memory(state: AgentState) -> dict[str, Any]:
    """
    Post-answer memory extraction.

    Reads draft_answer + citation_verdicts, calls REASONING-tier LLM to
    extract structured knowledge, and returns a 'memory_payload' the
    streamer can persist.
    """
    question = state.get("question", "")
    answer = state.get("draft_answer") or ""
    verdicts = state.get("citation_verdicts") or {}
    domain_scope = state.get("domain_scope") or []

    if not answer or len(answer.strip()) < 100:
        return {
            "memory_payload": None,
            "trace": [{"node": "memory_keeper", "skipped": "answer_too_short"}],
        }

    # Build a compact source summary for the extraction prompt
    citation_refs = state.get("citation_refs") or {}
    source_summary = ""
    for tag in sorted(citation_refs.keys(), key=lambda t: int(t[1:])):
        src = citation_refs[tag]
        title = src.get("doc_title", src.get("citation", "untitled"))[:80]
        v = verdicts.get(str(tag), "?")
        source_summary += f"  [{tag}] {title} (verdict: {v})\n"

    user_msg = f"""Research question:
{question}

Answer:
{answer[:3000]}

Sources cited:
{source_summary}

Domain context: {domain_scope or "none"}

Extract topics, entities, relationships, and knowledge gaps from this research run."""

    t0 = time.perf_counter()
    raw = chat(
        messages=[{"role": "system", "content": _SYSTEM}, {"role": "user", "content": user_msg}],
        tier=ModelTier.FAST,
        temperature=0.0,
        max_tokens=800,
    )
    dur_ms = int((time.perf_counter() - t0) * 1000)

    parsed = _extract_json(raw) or {}
    topics = parsed.get("topics") or []
    entities = parsed.get("entities") or []
    relationships = parsed.get("relationships") or []
    gaps = parsed.get("gaps") or []

    return {
        "memory_payload": {
            "topics": topics,
            "entities": entities,
            "relationships": relationships,
            "gaps": gaps,
        },
        "trace": [
            {
                "node": "memory_keeper",
                "duration_ms": dur_ms,
                "topics_extracted": len(topics),
                "entities_extracted": len(entities),
                "gaps_extracted": len(gaps),
            }
        ],
    }