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
import re
from typing import Any

from src.agent.artifacts import HydratedChunk, hydrate
from src.agent.state import AgentState
from src.config import settings
from src.provenance.checker import check_many


_DOI_RE = re.compile(r"\b(10\.\d{4,9}/[-._;()/:A-Z0-9]+)", re.IGNORECASE)
_ARXIV_RE = re.compile(
    r"(?:arxiv(?:\.org/(?:abs|pdf)/)?\s*[:/]?\s*)"
    r"((?:\d{4}\.\d{4,5}|[a-z-]+/\d{7})(?:v\d+)?)",
    re.IGNORECASE,
)


def _identifiers_for(c: HydratedChunk) -> tuple[str, str]:
    metadata_text = " ".join(
        str(value or "")
        for value in (
            getattr(c, "doi", ""),
            getattr(c, "arxiv_id", ""),
            getattr(c, "url", ""),
            getattr(c, "citation", ""),
            getattr(c, "source_file", ""),
            getattr(c, "doc_title", ""),
        )
    )
    doi_match = _DOI_RE.search(metadata_text)
    # Search snippets frequently include the DOI in their text even when the
    # result URL is a publisher landing page. Paper full text is deliberately
    # excluded here because it can contain DOIs for cited references.
    if not doi_match and (getattr(c, "signal", "") or "").startswith("web:"):
        doi_match = _DOI_RE.search((getattr(c, "text", "") or "")[:5000])
    doi = (doi_match.group(1) if doi_match else "").rstrip(".,;:)]}").lower()
    arxiv_id = (getattr(c, "arxiv_id", "") or "").strip()
    if not arxiv_id:
        arxiv_match = _ARXIV_RE.search(metadata_text)
        if not arxiv_match and (getattr(c, "signal", "") or "").startswith("web:"):
            arxiv_match = _ARXIV_RE.search((getattr(c, "text", "") or "")[:5000])
        arxiv_id = arxiv_match.group(1) if arxiv_match else ""
    return (doi, arxiv_id)


def _normalized_title(chunk: HydratedChunk) -> str:
    title = (getattr(chunk, "doc_title", "") or "").strip()
    if not title:
        title = (getattr(chunk, "citation", "") or "").strip()
    # Search providers prefix the same original article with "RETRACTED
    # ARTICLE:". Strip that marker so it matches the paper title recorded by
    # arXiv/Semantic Scholar. Keep "Retraction:" intact: the notice is a
    # distinct, valid source that can explain why the original was retracted.
    title = re.sub(r"^\s*retracted\s+article\s*[:\-–—]?\s*", "", title, flags=re.I)
    title = re.split(r"\s*\|\s*", title, maxsplit=1)[0]
    title = re.sub(r"\s+[—–-]\s+Nature(?:\s+.*)?$", "", title, flags=re.I)
    title = re.sub(r"\s*\(\d{4}\)\s*", " ", title)
    return re.sub(r"[^a-z0-9]+", " ", title.casefold()).strip()


def _previously_blocked(
    chunk: HydratedChunk,
    identifiers: tuple[str, str],
    blocked_sources: list[dict[str, str]],
) -> dict[str, str] | None:
    doi, arxiv_id = identifiers
    title = _normalized_title(chunk)
    for source in blocked_sources:
        if doi and doi == (source.get("doi") or "").lower():
            return source
        if arxiv_id and arxiv_id == source.get("arxiv_id"):
            return source
        if title and title == source.get("normalized_title"):
            return source
    return None


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
    previously_blocked = list(state.get("provenance_blocked_sources") or [])

    # Collect unique identifiers across the whole batch — one lookup pass.
    all_chunks: list[HydratedChunk] = [c for bucket in chunks_by_subq.values() for c in bucket]
    identifiers = list(
        {
            _identifiers_for(c)
            for c in all_chunks
            if _previously_blocked(
                c, _identifiers_for(c), previously_blocked
            ) is None
        }
    )

    results = asyncio.run(check_many(identifiers))

    kept_refs_by_subq: dict[str, list[dict[str, str]]] = {}
    provenance_by_ref: dict[str, dict[str, str]] = {}
    dropped_count = 0
    flagged_count = 0
    clean_count = 0
    unchecked_count = 0
    unchecked_examples: list[dict[str, str]] = []
    blocked_sources: list[dict[str, str]] = []
    cumulative_blocked = list(previously_blocked)

    for sq, chunks in chunks_by_subq.items():
        original_refs = refs_by_subq.get(sq) or []
        kept_refs: list[dict[str, str]] = []
        for i, c in enumerate(chunks):
            identifiers = _identifiers_for(c)
            prior_block = _previously_blocked(c, identifiers, previously_blocked)
            status_info = prior_block or results.get(
                identifiers, {"status": "clean"}
            )
            status = status_info.get("status", "clean")
            ref = original_refs[i] if i < len(original_refs) else None
            if status == "retracted":
                dropped_count += 1
                blocked_entry = {
                    "citation": c.citation,
                    "title": c.doc_title or c.citation,
                    "normalized_title": _normalized_title(c),
                    "doi": identifiers[0],
                    "arxiv_id": identifiers[1],
                    "reason": status_info.get("reason", "retracted"),
                    "notice_url": status_info.get("notice_url", ""),
                    "status": "retracted",
                }
                blocked_sources.append(blocked_entry)
                if _previously_blocked(c, identifiers, cumulative_blocked) is None:
                    cumulative_blocked.append(blocked_entry)
                continue  # HARD DROP — never re-emitted into state
            elif status in ("corrected", "concern"):
                flagged_count += 1
            elif status == "unchecked":
                unchecked_count += 1
                if len(unchecked_examples) < 10:
                    unchecked_examples.append({
                        "title": (c.doc_title or c.citation or "")[:180],
                        "doi": identifiers[0],
                        "arxiv_id": identifiers[1],
                        "reason": status_info.get("reason", "identifier unavailable"),
                    })
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
        "provenance_blocked_sources": cumulative_blocked,
        "trace": [{
            "node": "provenance_check",
            "clean": clean_count,
            "flagged": flagged_count,
            "unchecked": unchecked_count,
            "unchecked_examples": unchecked_examples,
            "retracted_dropped": dropped_count,
            "blocked_sources": blocked_sources[:50],
        }],
    }
