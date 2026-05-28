"""
Write an agent run back into an Obsidian vault as a markdown note.

Output shape — what an Obsidian user actually wants
---------------------------------------------------
Filename:  <vault>/<OBSIDIAN_OUTPUT_FOLDER>/<YYYY-MM-DD>/<slug>.md

Frontmatter (so the note is queryable from Dataview / similar plugins):
    ---
    title: <question>
    date: 2026-05-28
    source: ResearGent
    run_id: ...
    confidence: high|medium|low
    rewrites: 0
    web_used: true|false
    papers_used: true|false
    reflections: 1
    tags: [researgent, ai-research]
    ---

Body:
    # <question>

    <answer with [Sn] citations preserved>

    ## Sources
    - [S1] [[Existing Vault Note]]                ← if citation already exists as a note
    - [S2] [filename.pdf p.7](file://...)         ← local PDF, link to disk
    - [S3] [DeepSeek V3 (arxiv:2412.19437)](https://arxiv.org/abs/2412.19437)
    - [S4] [example.com — Real-time MoE](https://...)

    ## Provenance
    <run metadata in a collapsible block>

Why wikilinks for vault-resident sources
----------------------------------------
If [S1] is "[[Mixture of Experts]]" and the user already has a note
called "Mixture of Experts" in their vault, the answer note creates a
LIVE BACKLINK. Obsidian's backlinks panel shows "This note is linked
from your ResearGent answer about X" — turns research into a navigable
graph instead of disconnected dumps.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_INVALID_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]+')


def _slugify(s: str, max_len: int = 60) -> str:
    """Vault-safe filename: keep alnum + spaces + hyphens, truncate."""
    s = (s or "").strip()
    s = _INVALID_FILENAME_CHARS.sub("", s)
    # Collapse internal whitespace; preserve readability (spaces are fine in
    # markdown filenames on Mac/Linux and Windows alike).
    s = re.sub(r"\s+", " ", s).strip()
    return s[:max_len] or "untitled"


def _format_value(v: Any) -> str:
    """YAML-friendly scalar formatting."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, list):
        # Inline list: [a, b, c]
        return "[" + ", ".join(_format_value(x).strip('"') for x in v) + "]"
    if isinstance(v, (int, float)):
        return str(v)
    s = str(v)
    if any(c in s for c in ":#-[]{},") or s.startswith(" "):
        return f'"{s}"'
    return s


def _frontmatter(meta: dict) -> str:
    lines = ["---"]
    for k, v in meta.items():
        lines.append(f"{k}: {_format_value(v)}")
    lines.append("---")
    return "\n".join(lines)


_VAULT_NOTE_CITATION_RE = re.compile(r"^(?P<path>.+?\.md)(?:\s+p\.\d+)?\s*$", re.IGNORECASE)
_PDF_CITATION_RE = re.compile(r"^(?P<file>.+?\.pdf)\s+p\.(?P<page>\d+)\s*$", re.IGNORECASE)


def _format_citation_line(tag: str, source) -> str:
    """
    Render one citation as a markdown line. Smart-link based on type:
      - vault note ("Projects/Foo.md p.0")     -> Obsidian wikilink [[Foo]]
      - web URL                                -> markdown link [title](url)
      - arxiv:<id>                             -> link to arxiv.org abs page
      - local PDF ("file.pdf p.7")             -> plain text (no clickable target)
    """
    cit = getattr(source, "citation", "") or ""
    title = getattr(source, "doc_title", "") or ""

    # Web — most common after Phase 4 web fallback
    if cit.startswith(("http://", "https://")):
        display = title if title else cit
        return f"- **[{tag}]** [{display}]({cit})"

    # arXiv canonical citation
    if cit.startswith("arxiv:"):
        arxiv_id = cit.split(":", 1)[1].strip()
        url = f"https://arxiv.org/abs/{arxiv_id}"
        display = title if title else cit
        return f"- **[{tag}]** [{display}]({url}) — `{cit}`"

    # Vault note — accept both "Foo.md" and "Foo.md p.0" forms.
    # Obsidian's `[[wikilink]]` resolves case-insensitively across the vault.
    m = _VAULT_NOTE_CITATION_RE.match(cit)
    if m:
        rel = m.group("path")
        stem = Path(rel).stem
        return f"- **[{tag}]** [[{stem}]]"

    # Local PDF — keep as plain reference (no clickable target).
    if _PDF_CITATION_RE.match(cit):
        return f"- **[{tag}]** `{cit}`"

    # Fallback — anything else (unknown citation shape)
    return f"- **[{tag}]** {cit}"


def write_run_to_vault(
    vault_path: str | Path,
    output_subfolder: str,
    *,
    question: str,
    answer: str,
    sources: dict,                # tag -> chunk
    sub_questions: list[str],
    is_complex: bool,
    confidence: str,
    rewrite_attempts: int,
    web_used: bool,
    papers_used: bool,
    reflection_attempts: int,
    run_id: str,
    extra_tags: list[str] | None = None,
) -> Path:
    """Materialize one agent run as a markdown note inside the vault."""
    vault = Path(vault_path).resolve()
    if not vault.exists():
        raise FileNotFoundError(f"Vault not found: {vault}")

    ts = datetime.now(timezone.utc)
    date_str = ts.strftime("%Y-%m-%d")
    out_dir = vault / output_subfolder / date_str
    out_dir.mkdir(parents=True, exist_ok=True)

    filename = _slugify(question)
    # Ensure uniqueness if the same question is asked twice in a day.
    target = out_dir / f"{filename}.md"
    counter = 2
    while target.exists():
        target = out_dir / f"{filename} ({counter}).md"
        counter += 1

    # ---- Build frontmatter ----
    tags = list(extra_tags or []) + ["researgent"]
    if confidence == "low":
        tags.append("low-confidence")
    if not sources:
        tags.append("no-sources")
    meta = {
        "title": question,
        "date": date_str,
        "source": "ResearGent",
        "run_id": run_id,
        "confidence": confidence or "unknown",
        "rewrites": rewrite_attempts,
        "web_used": web_used,
        "papers_used": papers_used,
        "reflections": reflection_attempts,
        "n_sources": len(sources or {}),
        "tags": sorted(set(tags)),
    }

    # ---- Build body ----
    parts: list[str] = []
    parts.append(_frontmatter(meta))
    parts.append("")
    parts.append(f"# {question}")
    parts.append("")

    if is_complex and len(sub_questions) > 1:
        parts.append("**Decomposed into sub-questions:**")
        for sq in sub_questions:
            parts.append(f"- {sq}")
        parts.append("")

    parts.append(answer.strip() if answer else "_(no answer produced)_")
    parts.append("")

    # ---- Sources ----
    if sources:
        parts.append("## Sources")
        parts.append("")
        for tag, src in sorted(sources.items(), key=lambda kv: int(kv[0][1:])):
            parts.append(_format_citation_line(tag, src))
        parts.append("")

    # ---- Provenance (collapsible for Obsidian-friendly readability) ----
    parts.append("## Provenance")
    parts.append("")
    parts.append("```yaml")
    parts.append(f"run_id:         {run_id}")
    parts.append(f"timestamp:      {ts.isoformat(timespec='seconds')}")
    parts.append(f"confidence:     {confidence}")
    parts.append(f"rewrites:       {rewrite_attempts}")
    parts.append(f"web_fallback:   {web_used}")
    parts.append(f"paper_discovery:{papers_used}")
    parts.append(f"reflections:    {reflection_attempts}")
    parts.append(f"sources_total:  {len(sources or {})}")
    parts.append("```")

    target.write_text("\n".join(parts) + "\n", encoding="utf-8")
    return target
