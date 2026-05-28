"""
Obsidian vault ingestion.

An Obsidian vault is just a folder of markdown files with conventions:
  - YAML frontmatter at the top (between two `---` lines)
  - `[[wikilinks]]` connecting notes (sometimes `[[Note|Display Text]]`)
  - `#tags` for categorization
  - Folders for hierarchy

We treat the vault as a first-class corpus with TWO important advantages
over PDF ingestion:

  1. **Native heading structure**  — markdown headings let us chunk on
     SEMANTIC boundaries instead of fixed token windows. A chunk is one
     "section" (heading + body until the next same-or-higher heading)
     unless it's too big, in which case we split intelligently.

  2. **Citation graph baked in**   — wikilinks ARE the citation graph.
     We store each chunk's outgoing wikilinks as metadata so Phase 9 can
     do 1-hop graph expansion ("retrieved A — A links to B and C, pull
     those too as context").

What we explicitly DON'T do
---------------------------
  - We don't follow attachment links (images, PDFs) — those are noise for
    text retrieval.
  - We don't try to mirror Obsidian's plugin behaviors (Dataview, Tasks,
    Templater). We see the raw markdown as it's stored on disk.
  - We don't write back into the user's notes during ingestion. Output
    is a separate explicit step (`--save-to-vault`).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import tiktoken

# Reuse the same tokenizer the PDF chunker uses — keeps token budgets
# directly comparable across PDF + vault chunks at the retrieval layer.
_ENC = tiktoken.get_encoding("cl100k_base")


@dataclass
class VaultChunk:
    """One ingestable piece of a vault note. Same shape as PDF Chunk."""

    text: str
    source_file: str          # vault-relative path (e.g. "Projects/Foo.md")
    page_number: int          # 0 — vault notes don't have pages
    chunk_index: int          # 0-based within the note
    token_count: int
    # Vault-specific metadata
    note_title: str           # YAML title OR H1 OR filename
    heading_path: str         # breadcrumb "Section > Subsection > ..."
    tags: list[str] = field(default_factory=list)
    wikilinks: list[str] = field(default_factory=list)


@dataclass
class VaultNote:
    """One parsed markdown note ready to chunk."""

    doc_id: str
    rel_path: str             # vault-relative
    title: str
    body: str                 # frontmatter stripped
    frontmatter: dict
    tags: list[str]
    wikilinks: list[str]


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_WIKILINK_RE = re.compile(r"\[\[([^\[\]\|]+)(?:\|[^\]]+)?\]\]")
# Tags: #name where the name starts with a letter and may contain /-_letters/digits.
# Avoid matching `#header` markdown by requiring no whitespace BEFORE the # and
# at least one letter immediately after.
_TAG_RE = re.compile(r"(?:(?<=\s)|(?<=^))#([A-Za-z][\w\-/]*)")
# Markdown headings — captured to build the breadcrumb during chunking.
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
# Strip fenced code blocks before tag/wikilink extraction so `# python` in a
# code block doesn't become a tag.
_CODE_BLOCK_RE = re.compile(r"```.*?```", re.DOTALL)


def _doc_id_for(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """
    Extract YAML frontmatter if present. We do a TINY hand-roll instead of
    pulling pyyaml just for this — Obsidian frontmatter is overwhelmingly
    flat scalar fields (title: foo, date: 2026-01-01, tags: [a, b]).

    For anything richer, the body is still parsed correctly — we just
    don't surface the structured fields.
    """
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}, text
    raw = m.group(1)
    body = text[m.end():]
    meta: dict = {}
    for line in raw.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if val.startswith("[") and val.endswith("]"):
            # Simple list: tags: [a, b, c]
            inner = val[1:-1]
            meta[key] = [x.strip().strip('"').strip("'") for x in inner.split(",") if x.strip()]
        else:
            meta[key] = val
    return meta, body


def _extract_tags(body: str) -> list[str]:
    # Strip code blocks first so `# todo:` in code doesn't become #todo.
    cleaned = _CODE_BLOCK_RE.sub("", body)
    return sorted({m for m in _TAG_RE.findall(cleaned)})


def _extract_wikilinks(body: str) -> list[str]:
    cleaned = _CODE_BLOCK_RE.sub("", body)
    # Strip optional `#anchor` suffix from each wikilink target
    targets = [m.split("#")[0].strip() for m in _WIKILINK_RE.findall(cleaned)]
    return sorted({t for t in targets if t})


def parse_note(path: Path, vault_root: Path) -> VaultNote:
    """Read one .md file and surface frontmatter + tags + wikilinks."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    fm, body = _parse_frontmatter(text)

    # Title precedence: frontmatter > first H1 > filename
    title = ""
    if isinstance(fm.get("title"), str) and fm["title"].strip():
        title = fm["title"].strip()
    if not title:
        h1 = re.search(r"^#\s+(.+?)\s*$", body, re.MULTILINE)
        if h1:
            title = h1.group(1).strip()
    if not title:
        title = path.stem

    # Frontmatter tags can be a list OR a comma string ("tags: a, b").
    fm_tags_raw = fm.get("tags") or []
    if isinstance(fm_tags_raw, str):
        fm_tags = [t.strip() for t in fm_tags_raw.split(",") if t.strip()]
    elif isinstance(fm_tags_raw, list):
        fm_tags = [str(t).strip() for t in fm_tags_raw if str(t).strip()]
    else:
        fm_tags = []
    body_tags = _extract_tags(body)
    all_tags = sorted(set(fm_tags + body_tags))

    rel = str(path.relative_to(vault_root)).replace("\\", "/")
    return VaultNote(
        doc_id=_doc_id_for(path),
        rel_path=rel,
        title=title,
        body=body.strip(),
        frontmatter=fm,
        tags=all_tags,
        wikilinks=_extract_wikilinks(body),
    )


# ---------------------------------------------------------------------------
# Markdown-aware chunking
# ---------------------------------------------------------------------------


def _tok_count(s: str) -> int:
    return len(_ENC.encode(s))


@dataclass
class _Section:
    heading_path: list[str]   # ["Section", "Subsection"]
    text: str
    start_token: int          # cumulative token offset (for ordering)


def _split_into_sections(body: str) -> list[_Section]:
    """
    Walk the markdown linearly, tracking heading depth, and emit one
    Section per leaf-content block under a heading.

    A "section" is heading-line + body until the next same-or-higher
    heading. We preserve the breadcrumb path so chunks can cite their
    location precisely.
    """
    sections: list[_Section] = []
    lines = body.splitlines()

    heading_stack: list[tuple[int, str]] = []  # [(level, title), ...]
    buf: list[str] = []

    def flush() -> None:
        text = "\n".join(buf).strip()
        if not text:
            buf.clear()
            return
        path = [t for _, t in heading_stack]
        sections.append(_Section(heading_path=path, text=text, start_token=0))
        buf.clear()

    for ln in lines:
        m = _HEADING_RE.match(ln)
        if not m:
            buf.append(ln)
            continue

        # We hit a new heading: flush the previous section's body first.
        flush()

        level = len(m.group(1))
        title = m.group(2).strip()

        # Pop stack down to (level - 1), then push this one
        while heading_stack and heading_stack[-1][0] >= level:
            heading_stack.pop()
        heading_stack.append((level, title))

    flush()
    return sections


def chunk_note(note: VaultNote, *, target_tokens: int = 500, overlap_tokens: int = 80) -> list[VaultChunk]:
    """
    Chunk a parsed note into VaultChunks.

    Strategy:
      1. Split into heading-bounded sections (preserves semantic structure).
      2. For each section, if it fits under target_tokens → one chunk.
      3. If a section is oversized, recursively pack its paragraphs with
         the same token-aware packer used by the PDF chunker.

    Each emitted chunk inherits the section's heading breadcrumb so
    citations look like "Projects/Foo.md > Architecture > Database".
    """
    sections = _split_into_sections(note.body)
    if not sections and note.body.strip():
        # Note has no headings at all — treat the whole body as one section.
        sections = [_Section(heading_path=[], text=note.body.strip(), start_token=0)]

    chunks: list[VaultChunk] = []
    idx = 0

    for sec in sections:
        sec_tokens = _tok_count(sec.text)
        breadcrumb = " > ".join(sec.heading_path) if sec.heading_path else ""

        # Fits in one chunk — easy path.
        if sec_tokens <= target_tokens:
            chunks.append(VaultChunk(
                text=sec.text,
                source_file=note.rel_path,
                page_number=0,
                chunk_index=idx,
                token_count=sec_tokens,
                note_title=note.title,
                heading_path=breadcrumb,
                tags=list(note.tags),
                wikilinks=list(note.wikilinks),
            ))
            idx += 1
            continue

        # Oversize section — pack paragraphs greedily within target_tokens.
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", sec.text) if p.strip()]
        current: list[str] = []
        current_tokens = 0
        for para in paragraphs:
            p_tokens = _tok_count(para)
            if current and current_tokens + p_tokens > target_tokens:
                packed = "\n\n".join(current)
                chunks.append(VaultChunk(
                    text=packed,
                    source_file=note.rel_path,
                    page_number=0,
                    chunk_index=idx,
                    token_count=_tok_count(packed),
                    note_title=note.title,
                    heading_path=breadcrumb,
                    tags=list(note.tags),
                    wikilinks=list(note.wikilinks),
                ))
                idx += 1
                # Carry forward overlap for context continuity.
                if overlap_tokens > 0 and current:
                    tail_ids = _ENC.encode("\n\n".join(current))[-overlap_tokens:]
                    current = [_ENC.decode(tail_ids)]
                    current_tokens = _tok_count(current[0])
                else:
                    current, current_tokens = [], 0
            current.append(para)
            current_tokens += p_tokens

        if current:
            packed = "\n\n".join(current)
            chunks.append(VaultChunk(
                text=packed,
                source_file=note.rel_path,
                page_number=0,
                chunk_index=idx,
                token_count=_tok_count(packed),
                note_title=note.title,
                heading_path=breadcrumb,
                tags=list(note.tags),
                wikilinks=list(note.wikilinks),
            ))
            idx += 1

    return chunks


# ---------------------------------------------------------------------------
# Vault walker
# ---------------------------------------------------------------------------


# Folders Obsidian users commonly want ignored during ingestion.
_DEFAULT_IGNORE = {".obsidian", ".trash", "_archive", "_attachments", "Attachments"}


def iter_vault_notes(vault_root: Path, *, ignore: set[str] | None = None) -> list[Path]:
    """List all .md files in the vault, skipping common cruft folders."""
    if ignore is None:
        ignore = _DEFAULT_IGNORE
    out: list[Path] = []
    for p in vault_root.rglob("*.md"):
        rel = p.relative_to(vault_root)
        if any(part in ignore for part in rel.parts):
            continue
        out.append(p)
    return sorted(out)
