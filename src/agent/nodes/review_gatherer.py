"""Review Gatherer — gathers papers for each section via paper discovery."""
from __future__ import annotations

import time
from typing import Any, Generator, Iterator

from src.agent.nodes.review_planner import plan_review
from src.agent.nodes.review_writer import write_section
from src.agent.nodes.review_formatter import format_review


def run_review(request: str) -> Iterator[dict[str, Any]]:
    """
    Run the full literature review pipeline: plan → gather → write → format.

    Yields progress events as each blocking phase starts, so clients can show live work.
    """
    t0 = time.perf_counter()

    # Phase 1: Plan
    yield _progress("planning", "Building the review outline")
    plan = plan_review(request)
    title = plan["title"]
    sections_spec = plan["sections"]

    yield {
        "type": "review_planned",
        "title": title,
        "sections": [s["heading"] for s in sections_spec],
        "ts": time.time(),
    }

    if not sections_spec:
        yield {
            "type": "review_error",
            "error": "Planner returned no sections",
            "ts": time.time(),
        }
        return

    # Phase 2: Gather + Write (section by section)
    citation_pool: dict[str, dict[str, Any]] = {}
    next_tag = 1
    section_bodies: list[tuple[str, str]] = []

    for i, sec in enumerate(sections_spec):
        heading = sec.get("heading", f"Section {i + 1}")
        desc = sec.get("description", "")
        search_queries = sec.get("search_queries", [request])

        # Keep each section focused: the paper search itself expands each query
        # into variants, so repeating all planner queries caused many serial API calls.
        queries = list(dict.fromkeys(q.strip() for q in search_queries if isinstance(q, str) and q.strip()))[:2]
        if not queries:
            queries = [f"{heading} {desc}".strip() or request]

        yield _progress(
            "searching_papers",
            f"Searching academic papers for section {i + 1}/{len(sections_spec)}: {heading}",
            section_index=i,
            section_count=len(sections_spec),
        )
        evidence_text = yield from _gather_evidence_for_section(
            heading, desc, queries, citation_pool, next_tag,
            section_index=i, section_count=len(sections_spec),
        )
        # Track how many tags this section consumed
        new_tags_count = len(
            [k for k in citation_pool if int(k[1:]) >= next_tag]
        )
        next_tag += new_tags_count

        yield _progress(
            "writing_section",
            f"Writing section {i + 1}/{len(sections_spec)}: {heading}",
            section_index=i,
            section_count=len(sections_spec),
            source_count=new_tags_count,
        )
        body = write_section(heading, desc, evidence_text)
        section_bodies.append((heading, body))

        yield {
            "type": "review_section_done",
            "section_index": i,
            "heading": heading,
            "n_sources": new_tags_count,
            "body_len": len(body),
            "ts": time.time(),
        }

    yield _progress("formatting", "Formatting the review and references")
    markdown = format_review(title, section_bodies, citation_pool)

    dur_ms = int((time.perf_counter() - t0) * 1000)

    yield {
        "type": "review_complete",
        "title": title,
        "markdown": markdown,
        "sections": [h for h, _ in section_bodies],
        "n_sources": len(citation_pool),
        "duration_ms": dur_ms,
        "ts": time.time(),
    }


def _gather_evidence_for_section(
    heading: str,
    desc: str,
    queries: list[str],
    pool: dict[str, dict[str, Any]],
    start_tag: int,
    *, section_index: int, section_count: int,
) -> Generator[dict[str, Any], None, str]:
    """Gather evidence for a section using academic paper discovery + web fallback.

    Returns a formatted evidence block with [S#] tags.
    """
    evidence_parts: list[str] = []
    from src.retrieval.papers import discover_papers
    from src.retrieval.web import web_search

    current_tag = start_tag

    for query_index, query in enumerate(queries):
        yield _progress(
            "searching_papers", f"Searching papers ({query_index + 1}/{len(queries)}): {query[:90]}",
            section_index=section_index, section_count=section_count,
        )
        # 1. Try academic paper discovery
        try:
            papers = discover_papers(
                query, max_results=3, enrich_full_text=False,
                max_search_queries=1, retry_weak=False, auto_ingest=False,
            )
            yield _progress(
                "searching_papers", f"Academic search found {len(papers)} papers",
                section_index=section_index, section_count=section_count,
            )
            for paper in papers:
                tag = f"S{current_tag}"
                current_tag += 1
                source = {
                    "tag": tag,
                    "citation": paper.citation,
                    "doc_title": paper.title,
                    "url": paper.url,
                    "preview": paper.text[:300],
                    "authors": paper.authors,
                    "year": paper.year,
                    "venue": paper.venue,
                    "arxiv_id": paper.arxiv_id,
                    "signal": paper.signal,
                }
                pool[tag] = source
                evidence_parts.append(
                    f"[{tag}] {paper.title}\n{paper.url}\n{paper.text[:600]}"
                )
        except Exception:
            pass

        # 2. Try web search
        try:
            yield _progress(
                "searching_web", f"Checking web sources ({query_index + 1}/{len(queries)})",
                section_index=section_index, section_count=section_count,
            )
            results = web_search(query, max_results=3)
            yield _progress(
                "searching_web", f"Web search found {len(results)} sources",
                section_index=section_index, section_count=section_count,
            )
            for r in results:
                tag = f"S{current_tag}"
                current_tag += 1
                title = r.title or "Untitled"
                url = r.url or ""
                snippet = r.text or ""
                source = {
                    "tag": tag,
                    "citation": title,
                    "doc_title": title,
                    "url": url,
                    "preview": snippet[:300],
                    "authors": [],
                    "year": None,
                    "venue": "",
                    "arxiv_id": "",
                    "signal": r.signal,
                }
                pool[tag] = source
                evidence_parts.append(
                    f"[{tag}] {title}\n{url}\n{snippet[:600]}"
                )
        except Exception:
            pass

    if not evidence_parts:
        evidence_parts.append("(No evidence found for this section)")

    return "\n\n---\n\n".join(evidence_parts)


def _progress(stage: str, message: str, **details: Any) -> dict[str, Any]:
    return {"type": "review_progress", "stage": stage, "message": message, **details, "ts": time.time()}
