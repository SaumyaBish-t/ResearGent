"""
Knowledge-graph expansion — the "brain" behavior.

Idea
----
Hybrid retrieval finds chunks that are SEMANTICALLY similar to the query.
But a real knowledge graph (like an Obsidian vault) encodes a SECOND
relevance signal: STRUCTURAL connections. If note A is retrieved and A
contains `[[B]]`, then B is also likely relevant — the user EXPLICITLY
linked them because the concepts belong together.

This module walks 1 hop along the wikilink edges of the initial retrieval
results and surfaces additional chunks the embedder might have ranked low.

Why 1 hop (and not 2)
---------------------
Two-hop expansion balloons fast: a typical Obsidian note links to 3-10
others, so 2-hop = 9-100 candidates per seed. Even with capping, the
signal-to-noise ratio degrades quickly past 1 hop. We expose the hop
limit as a setting so power users can experiment.

Mutual-link boost
-----------------
When the target note (B) links BACK to a seed note (A) — i.e. mutual
linking — the relationship is structurally stronger. Mutual targets get
their expansion score boosted by `MUTUAL_LINK_BOOST` so they're more
likely to survive truncation to `max_extra_chunks`.

Performance
-----------
For our scale (<10K chunks per vault) we iterate the full store once per
expansion call. That's sub-100ms on typical hardware and avoids needing
a separate inverted-index for note-name -> chunks. If the project ever
needs 100K+ chunks we'll add a precomputed name->chunks map at ingest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src.config import settings
from src.retrieval.hybrid import HybridChunk
from src.store import get_or_create_papers_collection


# Mutual links indicate a STRONGER conceptual connection than one-way.
# Boost factor is a multiplier on the base expansion score.
MUTUAL_LINK_BOOST = 1.5


@dataclass
class GraphChunk:
    """
    A chunk pulled in via wikilink expansion rather than direct retrieval.

    Mirrors HybridChunk's public interface (text, source_file, page_number,
    chunk_index, citation, signal, doc_title) so the generator/critic don't
    care that it came from graph expansion vs hybrid retrieval.
    """

    text: str
    source_file: str
    page_number: int
    chunk_index: int
    expansion_score: float          # base 1.0; *MUTUAL_LINK_BOOST if mutual
    seed_source_file: str           # which retrieved chunk's note linked here
    hop_distance: int               # 1 for direct link, 2 for transitive (future)
    is_mutual: bool
    doc_title: str = ""
    wikilinks: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    @property
    def citation(self) -> str:
        # Show the structural provenance — readers immediately see this came
        # via the graph from a specific seed note.
        seed_stem = Path(self.seed_source_file).stem if self.seed_source_file else "?"
        return f"{self.source_file} (via [[{seed_stem}]])"

    @property
    def signal(self) -> str:
        return "graph:mutual" if self.is_mutual else "graph:link"

    # For score-display compatibility with HybridChunk in the generator's
    # source-footer rendering (which checks `rrf_score` and `score`).
    @property
    def rrf_score(self) -> float | None:
        return None

    @property
    def score(self) -> float:
        return self.expansion_score


# ---------------------------------------------------------------------------
# Note-name normalization
# ---------------------------------------------------------------------------


def _note_name_from_path(path: str) -> str:
    """Extract the wikilink-comparable note name from a vault-relative path."""
    if not path:
        return ""
    return Path(path).stem


def _norm(s: str) -> str:
    """Wikilinks resolve case-insensitively in Obsidian."""
    return (s or "").strip().lower()


# ---------------------------------------------------------------------------
# Public — graph expansion
# ---------------------------------------------------------------------------


def expand_via_wikilinks(
    seed_chunks: list[HybridChunk],
    *,
    max_extra: int | None = None,
    exclude_keys: set[tuple[str, int]] | None = None,
) -> list[GraphChunk]:
    """
    1-hop wikilink expansion. Returns up to `max_extra` GraphChunks ordered
    by expansion_score.

    `exclude_keys` is an optional set of (source_file, chunk_index) tuples
    to filter out — useful when the caller already has these chunks in
    state and doesn't want duplicates.

    Returns [] when:
      - graph expansion is disabled in settings
      - no seed chunks carry wikilinks (typical for a pure-PDF corpus)
      - no candidate chunks match any wikilink target
    """
    if not settings.graph_expansion_enabled:
        return []
    if not seed_chunks:
        return []

    cap = max_extra if max_extra is not None else settings.graph_expansion_max_extra_chunks
    if cap <= 0:
        return []

    # ---- Collect targets and seed-note metadata ----
    # We need:
    #   targets       : set of wikilink target names (normalized) to find
    #   seed_notes    : set of seed note names (normalized) for mutual-link check
    #   seed_keys     : (source_file, chunk_index) of every seed chunk so we
    #                   skip them in candidates
    targets: set[str] = set()
    seed_notes: set[str] = set()
    seed_keys: set[tuple[str, int]] = set(exclude_keys or set())
    # Reverse map: target_note_name -> set of seed_note_names that linked to it.
    # We use this when emitting GraphChunks to record WHICH seed brought them in.
    targets_to_seeds: dict[str, set[str]] = {}

    for c in seed_chunks:
        seed_keys.add((c.source_file, c.chunk_index))
        # Add BOTH the file stem AND the doc_title to seed_notes — Obsidian
        # users link by title (`[[Mixture of Experts]]`), not by filename
        # stem (`MoE`). Either should match for mutual-link detection.
        stem = _norm(_note_name_from_path(c.source_file))
        if stem:
            seed_notes.add(stem)
        title = _norm(getattr(c, "doc_title", "") or "")
        if title:
            seed_notes.add(title)
        for wl in c.wikilinks or []:
            n = _norm(wl)
            if n and n not in seed_notes:  # never expand back to our own seeds
                targets.add(n)
                targets_to_seeds.setdefault(n, set()).add(c.source_file)

    if not targets:
        return []

    # ---- Scan the store once for candidates ----
    # For each stored chunk: if its note name matches a target AND it's not
    # a seed chunk, it's a candidate. Compute its expansion score (with the
    # mutual-link boost when applicable) and keep top-N.
    col = get_or_create_papers_collection()
    if col.count() == 0:
        return []

    all_data = col.get(include=["documents", "metadatas"])
    ids = all_data.get("ids") or []
    docs = all_data.get("documents") or []
    metas = all_data.get("metadatas") or []

    candidates: list[GraphChunk] = []
    for cid, doc, meta in zip(ids, docs, metas):
        if not isinstance(meta, dict):
            continue
        sf = meta.get("source_file", "") or ""
        ci = int(meta.get("chunk_index", -1))
        key = (sf, ci)
        if key in seed_keys:
            continue
        # Match wikilink target against BOTH stem and doc_title (Obsidian
        # wikilinks are by title, but a note's title may equal its stem).
        stem_norm = _norm(_note_name_from_path(sf))
        title_norm = _norm(meta.get("doc_title", "") or "")
        matched_target = None
        if stem_norm and stem_norm in targets:
            matched_target = stem_norm
        elif title_norm and title_norm in targets:
            matched_target = title_norm
        if matched_target is None:
            continue

        # Parse wikilinks of THIS chunk for the mutual-link check.
        wl_raw = meta.get("wikilinks", "") or ""
        tg_raw = meta.get("tags", "") or ""
        outgoing = [w.strip() for w in wl_raw.split(",") if w.strip()]
        outgoing_norm = {_norm(w) for w in outgoing}

        # Mutual = this target's note links back to ANY of the seed notes.
        is_mutual = bool(outgoing_norm & seed_notes)
        score = 1.0 * (MUTUAL_LINK_BOOST if is_mutual else 1.0)

        # Pick ONE seed source_file as the attribution — the first seed that
        # linked to this target. Stable for display.
        seed_for_this = next(iter(targets_to_seeds.get(matched_target, set())), "")

        candidates.append(
            GraphChunk(
                text=doc,
                source_file=sf,
                page_number=int(meta.get("page_number", 0)),
                chunk_index=ci,
                expansion_score=score,
                seed_source_file=seed_for_this,
                hop_distance=1,
                is_mutual=is_mutual,
                doc_title=meta.get("doc_title", "") or "",
                wikilinks=outgoing,
                tags=[t.strip() for t in tg_raw.split(",") if t.strip()],
            )
        )

    # ---- Rank + truncate ----
    # Sort by expansion_score desc, then by note name for determinism.
    candidates.sort(key=lambda c: (-c.expansion_score, c.source_file, c.chunk_index))
    return candidates[:cap]
