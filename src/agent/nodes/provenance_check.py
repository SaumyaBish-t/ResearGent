"""
Provenance Check node — Phase 19.

Runs BETWEEN retriever and critic. Looks up every retrieved chunk's DOI/
arXiv ID against Crossref + the Retraction Watch cache, tags each chunk's
provenance status, and hard-drops any chunk whose paper has been retracted
— before the Critic ever grades it for relevance. This is the single
architectural difference from the nine tools in the retraction-detection
study this project is built against: it runs UNCONDITIONALLY on every
retrieval, not only when the user asks "is this retracted?" directly.

Three outcomes per chunk:
  clean     -> passes through untouched
  flagged   -> passes through, but HydratedChunk.provenance_status="flagged"
               so the Critic can apply provenance_flagged_score_multiplier
               and the generator surfaces a caveat
  retracted -> HARD-DROPPED from chunk_refs_by_subq entirely. Never reaches
               the Critic, never reaches the generator, cannot be cited.
"""
from __future__ import annotations

import asyncio
from typing import Any

from src.agent.artifacts import HydratedChunk, hydrate
from src.agent.state import AgentState
from src.config import settings
from src.provenance.checker import check_many


def _identifiers_for(c: HydratedChunk) -> tuple[str, str]:
    doi = (getattr(c, "doi", "") or "").strip()
    arxiv_id = (getattr(c, "arxiv_id", "") or "").strip()
    return (doi, arxiv_id)


def _ref_key(ref: dict[str, str]) -> str:
    """Stable key for a pointer, safe to carry in lean graph state."""
    return f"{ref.get('kind', '')}:{ref.get('id', '')}"


def provenance_check(state: AgentState) -> dict[str, Any]:
    if not settings.provenance_check_enabled:
        return {"trace": [{"node": "provenance_check", "skipped": "disabled"}]}

    refs_by_subq: dict[str, list[dict[str, str]]] = dict(state.get("chunk_refs_by_subq") or {})
    if not refs_by_subq or not any(refs_by_subq.values()):
        return {"trace": [{"node": "provenance_check", "skipped": "empty_input"}]}

    chunks_by_subq = hydrate(refs_by_subq)

    # Collect unique identifiers across the whole batch — one lookup pass.
    all_chunks: list[HydratedChunk] = [c for bucket in chunks_by_subq.values() for c in bucket]
    identifiers = list({_identifiers_for(c) for c in all_chunks})

    results = asyncio.run(check_many(identifiers))

    kept_refs_by_subq: dict[str, list[dict[str, str]]] = {}
    provenance_by_ref: dict[str, dict[str, str]] = {}
    dropped_count = 0
    flagged_count = 0
    clean_count = 0
    blocked_sources: list[dict[str, str]] = []

    for sq, chunks in chunks_by_subq.items():
        original_refs = refs_by_subq.get(sq) or []
        kept_refs: list[dict[str, str]] = []
        for i, c in enumerate(chunks):
            status_info = results.get(_identifiers_for(c), {"status": "clean"})
            status = status_info.get("status", "clean")
            ref = original_refs[i] if i < len(original_refs) else None
            if status == "retracted":
                dropped_count += 1
                blocked_sources.append(
                    {
                        "citation": c.citation,
                        "doi": getattr(c, "doi", "") or "",
                        "arxiv_id": getattr(c, "arxiv_id", "") or "",
                        "reason": status_info.get("reason", "retracted"),
                    }
                )
                continue  # HARD DROP — never re-emitted into state
            elif status in ("corrected", "concern"):
                flagged_count += 1
            else:
                clean_count += 1
            if ref is not None:
                kept_refs.append(ref)
                if status != "clean":
                    provenance_by_ref[_ref_key(ref)] = {
                        "status": status,
                        "reason": status_info.get("reason", ""),
                        "notice_url": status_info.get("notice_url", ""),
                    }
        kept_refs_by_subq[sq] = kept_refs

    return {
        "chunk_refs_by_subq": kept_refs_by_subq,
        "provenance_dropped_count": dropped_count,
        "provenance_flagged_count": flagged_count,
        "provenance_by_ref": provenance_by_ref,
        "trace": [{
            "node": "provenance_check",
            "clean": clean_count,
            "flagged": flagged_count,
            "retracted_dropped": dropped_count,
            "blocked_sources": blocked_sources[:50],
        }],
    }
