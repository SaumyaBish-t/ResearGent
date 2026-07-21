"""
BibTeX export — serialize sources into .bib citation entries.

Sources arrive as a list of dicts (the same shape stored in Turn.sources_json
or emitted by the FinalEvent SSE). Each dict carries the fields that
HydratedChunk would have: tag, citation, doc_title, signal, url, preview,
and optionally authors, year, venue, arxiv_id.

We map this shape into BibTeX @misc entries (the safest generic type —
handles papers, web pages, and local documents identically). arXiv papers
get an @misc with eprint + archivePrefix = {arXiv} so Zotero / JabRef /
Google Scholar auto-recognize the arXiv metadata.
"""

from __future__ import annotations

from typing import Any


def _sanitize_bibtex(text: str) -> str:
    """Escape BibTeX special characters and curly braces."""
    text = text.replace("\\", "\\\\")
    text = text.replace("{", "\\{")
    text = text.replace("}", "\\}")
    text = text.replace("&", "\\&")
    text = text.replace("~", "\\~{}")
    text = text.replace("#", "\\#")
    text = text.replace("$", "\\$")
    text = text.replace("%", "\\%")
    text = text.replace("_", "\\_")
    text = text.replace("^", "\\^{}")
    return text


def _cite_key(source: dict[str, Any], tag: str) -> str:
    """Generate a citation key: FirstAuthorYear or anonYearTag."""
    authors: list[str] = source.get("authors") or []
    year = source.get("year") or ""
    if authors:
        last = authors[0].split()[-1] if authors[0] else "anon"
        return f"{_sanitize_bibtex(last)}{year}{tag.replace('S', '')}"
    return f"anon{year}{tag.replace('S', '')}"


def source_to_bibtex(source: dict[str, Any], tag: str) -> str:
    """Convert one source dict to a BibTeX @misc entry."""
    key = _cite_key(source, tag)
    title = _sanitize_bibtex(source.get("doc_title") or source.get("citation") or "Untitled")
    authors_raw = source.get("authors") or []
    author_str = " and ".join(_sanitize_bibtex(a) for a in authors_raw) if authors_raw else "Anonymous"
    year = source.get("year")
    arxiv_id = source.get("arxiv_id") or ""
    url = source.get("url") or source.get("citation") or ""
    venue = source.get("venue") or ""
    preview = source.get("preview", "")[:200]

    lines = [f"@misc{{{key},"]
    lines.append(f"  title = {{{title}}},")
    lines.append(f"  author = {{{author_str}}},")
    if year:
        lines.append(f"  year = {{{year}}},")
    if venue:
        lines.append(f"  journal = {{{_sanitize_bibtex(venue)}}},")
    if arxiv_id:
        lines.append(f"  eprint = {{{arxiv_id}}},")
        lines.append(f"  archivePrefix = {{arXiv}},")
    if url and not arxiv_id:
        lines.append(f"  url = {{{url}}},")
    lines.append(f"  note = {{{_sanitize_bibtex(preview)[:150]}}}")
    lines.append("}")
    return "\n".join(lines)


def sources_to_bibtex(
    sources: list[dict[str, Any]],
    *,
    turn_index: int = 0,
) -> str:
    """Convert a list of source dicts to a complete .bib file string."""
    if not sources:
        return f"% ResearGent turn {turn_index} — no sources to export\n"

    entries: list[str] = []
    entries.append(f"% ResearGent turn {turn_index} — auto-exported {len(sources)} source(s)")
    entries.append("")

    for i, src in enumerate(sources):
        tag = src.get("tag", f"S{i + 1}")
        entries.append(source_to_bibtex(src, tag))
        entries.append("")

    return "\n".join(entries).strip() + "\n"