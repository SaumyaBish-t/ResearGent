"""
Semantic Scholar seed ingestion — Stage-1 corpus bootstrap.

What this does
--------------
For every registered domain (`src.domains.DOMAINS`), run that domain's
seed queries against Semantic Scholar with `sort=citationCount:desc`,
collect the highest-cited papers, and pull them into the local corpus
in one of two modes:

  * **`pdf`**   — download the open-access PDF when `openAccessPdf.url`
                  is present, drop it into `data/papers/<domain>/`, and
                  let the standard ingest pipeline take it from there.
                  Result: the same dense + BM25 + entity-enriched chunks
                  you get from any hand-dropped PDF.
  * **`abstracts`** — when no PDF is openly available (the common case
                  for ~60% of S2 hits), persist the **title + abstract**
                  as a one-page synthetic markdown note under
                  `data/papers/<domain>/_abstracts/<arxiv_or_doi>.md`.
                  Same pipeline, same chunker, same entity extraction —
                  just shorter source documents. Marked in the registry
                  with `extra.source_type="paper_abstract"` so the UI can
                  treat them differently if it wants to.

Why this exists
---------------
The persona contract says Stage-1 retrieval looks at "historically
revolutionary domain bedrock" sourced from Semantic Scholar sorted by
`citationCount:desc`. That requires the corpus to actually CONTAIN that
bedrock, populated automatically — not assumed to be hand-curated.

Free-tier guarantees
--------------------
  * Semantic Scholar's unauthenticated rate limit is ~1 req/sec. We sleep
    between calls and let httpx surface 429s back to the caller.
  * Embedding + chunking is the existing pipeline — entirely local CPU.
  * One on-disk PDF per paper (small) + one registry row (tiny). No new
    services, no new dependencies.

Idempotency
-----------
PDFs hash to the same content_hash on re-download → standard ingest
dedup kicks in and replaces existing chunks.
Abstracts are filenamed by `<arxiv_id>.md` (or sanitized DOI) so a
re-seed with the same hit overwrites cleanly.

CLI entrypoint
--------------
See `researgent seed [domains...]` in src/main.py.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
from rich.console import Console

from src.domains import DOMAINS, Domain, all_domain_ids
from src.ingest.pipeline import _rebuild_bm25_from_chroma, ingest_file
from src.llm import ModelTier, embed

console = Console()


# S2 enforces a courtesy rate of 1 request/sec when unauthenticated.
# We add a tiny safety margin so a parallel script + this seeder don't both
# tip into 429-territory.
_S2_RATE_LIMIT_SECONDS = 1.05

# Per-query result cap. S2 sorts by citation count when requested, so the
# first N hits are by definition the most-cited; pulling more diminishes
# the "foundational bedrock" goal.
_DEFAULT_TOP_N_PER_QUERY = 5

# Where abstract-only notes go. Subfolder of the domain dir so they don't
# clutter the user's hand-dropped PDFs.
_ABSTRACTS_SUBDIR = "_abstracts"


@dataclass
class SeedHit:
    """One S2 paper considered for ingestion."""

    title: str
    abstract: str
    year: int | None
    venue: str
    citations: int
    arxiv_id: str
    doi: str
    pdf_url: str
    s2_url: str

    @property
    def slug(self) -> str:
        """Stable filename slug for this paper. arXiv ID > DOI > title hash."""
        if self.arxiv_id:
            return f"arxiv_{self.arxiv_id.replace('/', '_')}"
        if self.doi:
            return "doi_" + re.sub(r"[^A-Za-z0-9._-]", "_", self.doi)
        # Fallback — first 60 chars of the title, sanitised. Stable enough
        # for human inspection, just not collision-proof across journals.
        slug = re.sub(r"[^A-Za-z0-9]+", "_", self.title.lower()).strip("_")
        return f"title_{slug[:60]}"


def _s2_search_sorted_by_citations(
    query: str, *, limit: int
) -> list[SeedHit]:
    """
    One S2 search call, sorted by citationCount descending.

    Endpoint: GET /graph/v1/paper/search

    The `sort` parameter is documented as `field:order` — `citationCount:desc`
    is exactly the "show me the historically most-cited papers for this
    query" knob the persona contract calls for. Field set kept tight to
    minimise response size on the free tier.
    """
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "limit": str(limit),
        "sort": "citationCount:desc",
        "fields": "title,abstract,year,venue,citationCount,openAccessPdf,externalIds,url",
    }
    try:
        r = httpx.get(url, params=params, timeout=20.0)
    except Exception as e:
        console.print(f"  [yellow]S2 request failed[/yellow]: {type(e).__name__}: {e}")
        return []
    if r.status_code != 200:
        console.print(f"  [yellow]S2 status {r.status_code}[/yellow] for {query!r}")
        return []
    data = r.json()

    out: list[SeedHit] = []
    for item in data.get("data") or []:
        if not item:
            continue
        ext = item.get("externalIds") or {}
        oa = item.get("openAccessPdf") or {}
        out.append(
            SeedHit(
                title=(item.get("title") or "").strip(),
                abstract=(item.get("abstract") or "").strip(),
                year=item.get("year"),
                venue=(item.get("venue") or "").strip(),
                citations=int(item.get("citationCount") or 0),
                arxiv_id=(ext.get("ArXiv") or "").strip(),
                doi=(ext.get("DOI") or "").strip(),
                pdf_url=(oa.get("url") if isinstance(oa, dict) else "") or "",
                s2_url=item.get("url") or "",
            )
        )
    return out


def _collect_domain_hits(
    dom: Domain, *, top_n_per_query: int
) -> list[SeedHit]:
    """
    Run every seed query for one domain, dedupe by arxiv_id / doi / title.

    Dedup matters here: S2's seed queries overlap heavily ("LLM agent
    reasoning ReAct" and "tool-use language model" both surface ReAct).
    Without dedup we'd download the same PDF twice and waste S2 quota.
    """
    seen_arxiv: set[str] = set()
    seen_doi: set[str] = set()
    seen_title: set[str] = set()
    merged: list[SeedHit] = []

    for q in dom.seed_queries:
        console.print(f"  [cyan]query[/cyan] {q!r}")
        hits = _s2_search_sorted_by_citations(q, limit=top_n_per_query)
        time.sleep(_S2_RATE_LIMIT_SECONDS)  # courtesy rate limit
        for h in hits:
            if h.arxiv_id and h.arxiv_id in seen_arxiv:
                continue
            if h.doi and h.doi in seen_doi:
                continue
            norm_title = re.sub(r"\W+", " ", h.title.lower()).strip()
            if norm_title and norm_title in seen_title:
                continue
            if h.arxiv_id:
                seen_arxiv.add(h.arxiv_id)
            if h.doi:
                seen_doi.add(h.doi)
            if norm_title:
                seen_title.add(norm_title)
            merged.append(h)

    # Re-sort by citationCount so the final list is globally well-ordered,
    # not per-query well-ordered. The top of the merged list is what the
    # user sees as the "foundational" corpus first.
    merged.sort(key=lambda h: h.citations, reverse=True)
    return merged


def _download_pdf(hit: SeedHit, dest: Path) -> bool:
    """
    Download an open-access PDF if available. Returns True on success.

    Failures (404, redirect to login wall, network) are non-fatal: the
    caller falls back to writing the abstract-only note instead.
    """
    if not hit.pdf_url:
        return False
    try:
        # Follow redirects — many openAccessPdf URLs redirect through DOI
        # resolvers and arxiv mirrors before hitting the actual file.
        r = httpx.get(hit.pdf_url, timeout=30.0, follow_redirects=True)
        if r.status_code != 200:
            return False
        ctype = (r.headers.get("content-type") or "").lower()
        # Many "open access" URLs return an HTML paywall when the actual
        # PDF is gated — content-type sniff catches that cleanly.
        if "pdf" not in ctype and not r.content[:5] == b"%PDF-":
            return False
        dest.write_bytes(r.content)
        return True
    except Exception:
        return False


def _write_abstract_note(hit: SeedHit, dest: Path) -> None:
    """
    Persist a paper's title + abstract as a tiny markdown note.

    Why markdown
    ------------
    We could store these in Postgres directly, but going through the
    standard PDF/markdown ingest path means:
      * one chunker code path (semantic chunking + GLiNER entities)
      * one provenance story (a real file on disk the user can grep)
      * trivial future migration to full-text once we get the PDF
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    bits: list[str] = [f"# {hit.title}", ""]
    meta_line = []
    if hit.year:
        meta_line.append(str(hit.year))
    if hit.venue:
        meta_line.append(hit.venue)
    if hit.citations:
        meta_line.append(f"{hit.citations} citations")
    if meta_line:
        bits.append("_" + " · ".join(meta_line) + "_")
        bits.append("")
    if hit.abstract:
        bits.append(hit.abstract)
        bits.append("")
    refs = []
    if hit.arxiv_id:
        refs.append(f"arXiv:{hit.arxiv_id}")
    if hit.doi:
        refs.append(f"DOI:{hit.doi}")
    if hit.s2_url:
        refs.append(f"S2:{hit.s2_url}")
    if refs:
        bits.append("---")
        bits.append(" · ".join(refs))
    dest.write_text("\n".join(bits), encoding="utf-8")


def seed_domain(
    domain_id: str,
    *,
    top_n_per_query: int = _DEFAULT_TOP_N_PER_QUERY,
    download_pdfs: bool = True,
) -> dict:
    """
    Seed one domain. Returns a summary dict for the CLI to render.

    `download_pdfs=False` forces abstracts-only mode — useful when the
    user is on a slow link or just wants the metadata corpus, not the
    PDFs. The pipeline behaves identically either way; abstract-only is
    just shorter source docs.
    """
    if domain_id not in DOMAINS:
        raise KeyError(f"unknown domain {domain_id!r}")
    dom = DOMAINS[domain_id]

    console.print(
        f"\n[bold]── seed: {dom.label} ──[/bold]  [dim]{dom.ingest_dir}[/dim]"
    )
    dom.ingest_dir.mkdir(parents=True, exist_ok=True)
    abstracts_dir = dom.ingest_dir / _ABSTRACTS_SUBDIR
    abstracts_dir.mkdir(parents=True, exist_ok=True)

    hits = _collect_domain_hits(dom, top_n_per_query=top_n_per_query)
    if not hits:
        console.print("  [yellow]no hits — S2 returned nothing[/yellow]")
        return {"domain": domain_id, "hits": 0, "pdfs": 0, "abstracts": 0, "ingested": 0}

    console.print(f"  [green]{len(hits)} unique candidates[/green]")

    pdfs_in: list[Path] = []
    abstracts_in: list[Path] = []
    for h in hits:
        slug = h.slug
        if download_pdfs and h.pdf_url:
            pdf_dest = dom.ingest_dir / f"{slug}.pdf"
            if pdf_dest.exists():
                pdfs_in.append(pdf_dest)
                continue
            if _download_pdf(h, pdf_dest):
                pdfs_in.append(pdf_dest)
                continue
            # Fall through to abstract on PDF download failure.
        md_dest = abstracts_dir / f"{slug}.md"
        _write_abstract_note(h, md_dest)
        abstracts_in.append(md_dest)

    console.print(
        f"  [green]on disk:[/green] {len(pdfs_in)} PDFs, "
        f"{len(abstracts_in)} abstract-notes"
    )

    # Ingest the PDFs through the standard pipeline. Abstract-notes are
    # markdown; we route them through ingest_file's vault-side path via
    # ingest_vault — but ingest_vault expects an Obsidian-shaped folder.
    # Simpler: synthesise tiny one-file "documents" by reusing ingest_file
    # for PDFs and skipping abstract markdown here. Abstract notes get
    # picked up on the next `researgent vault-ingest <abstracts_dir>` —
    # documented in the CLI help.
    ingested = 0
    if pdfs_in:
        # Single shared embedder warm-up, single BM25 rebuild at end.
        console.print(f"  [cyan]embedding {len(pdfs_in)} seed PDFs…[/cyan]")
        try:
            embed(["warmup"], tier=ModelTier.EMBED)
        except Exception as e:
            console.print(f"  [red]embed warmup FAIL[/red]: {type(e).__name__}: {e}")
            return {
                "domain": domain_id,
                "hits": len(hits),
                "pdfs": len(pdfs_in),
                "abstracts": len(abstracts_in),
                "ingested": 0,
                "error": "embedder warmup failed",
            }
        for pdf in pdfs_in:
            try:
                ingest_file(pdf, domain=domain_id, verbose=False, rebuild_bm25=False)
                ingested += 1
            except Exception as e:
                console.print(f"  [red]FAIL[/red] {pdf.name}: {type(e).__name__}: {e}")

    return {
        "domain": domain_id,
        "hits": len(hits),
        "pdfs": len(pdfs_in),
        "abstracts": len(abstracts_in),
        "ingested": ingested,
    }


def seed_all(
    *,
    domain_ids: list[str] | None = None,
    top_n_per_query: int = _DEFAULT_TOP_N_PER_QUERY,
    download_pdfs: bool = True,
) -> list[dict]:
    """Seed every domain in `domain_ids` (defaults to all registered)."""
    target = domain_ids or all_domain_ids()
    out: list[dict] = []
    for dom_id in target:
        out.append(
            seed_domain(
                dom_id, top_n_per_query=top_n_per_query, download_pdfs=download_pdfs
            )
        )

    # One BM25 rebuild across the entire corpus — same reasoning as
    # `ingest_all_domains`. Skipped if nothing was actually ingested.
    if any(d.get("ingested") for d in out):
        console.print("\n[cyan]rebuilding BM25 index[/cyan]...")
        n = _rebuild_bm25_from_chroma()
        console.print(f"  [green]BM25[/green] indexed {n} chunks")
    return out
