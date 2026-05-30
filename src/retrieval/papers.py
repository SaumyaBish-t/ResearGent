"""
Open-domain paper discovery — arXiv + Semantic Scholar.

When the local corpus comes up short, we go to the academic literature
BEFORE the open web. Two reasons:

  1. Authority — abstracts from peer-reviewed (or pre-print) papers beat
     blog posts and SEO content for technical research questions.
  2. Density — a paper abstract is ~200 tokens that contains the core
     claim. Cheaper than scraping a webpage that takes 2000 tokens to
     say the same thing surrounded by ads and nav.

Why abstracts only (not full PDFs)
----------------------------------
Downloading + parsing + chunking + embedding a 15-page PDF takes 30-60s
inside an interactive query. The marginal answer-quality gain over a
well-written abstract is usually small for "what is X" / "what's new
in X" questions. For deep questions where full-text matters, the user
should `researgent ingest` the paper into their permanent corpus.

Optional `--ingest-top-n` will be a future Phase 7.5 — auto-promote the
most-cited discovered papers into the permanent store.

Provider mix
------------
  - arXiv         CS/ML/physics pre-prints. Free, no key, official API.
                  STRONG for ML / NLP / agents / RAG / LLM topics.
  - Semantic      Cross-discipline coverage, citation counts, openAccessPdf
    Scholar      flag. Free, no key (rate-limited 1 RPS unconditionally).
                  STRONG for biology/medicine/economics/etc.

Both run in parallel-ish (sequential but each is fast). We dedupe by
ArXiv ID where available, otherwise by exact-title match.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass
class PaperChunk:
    """
    A discovered paper, exposed as the same interface as HybridChunk/WebChunk
    so generators/critics can mix all three without type-branching.

    `text` is the paper's TITLE + ABSTRACT — that's what the generator
    sees as evidence. Full-text retrieval is a separate path.
    """

    title: str
    abstract: str
    url: str               # link to paper (arxiv abs/, or DOI/landing)
    authors: list[str]
    year: int | None
    venue: str = ""        # journal/conference or "arXiv"
    source: str = ""       # "arxiv" | "semantic_scholar"
    citations: int | None = None
    arxiv_id: str = ""     # canonical dedup key when present
    pdf_url: str = ""      # when openly available
    score: float = 0.0     # query-relevance, [0..1], filled by ranker

    # ---- Public interface shared with HybridChunk / WebChunk ----
    @property
    def text(self) -> str:
        """What the generator sees as evidence. Title + abstract."""
        if self.abstract:
            return f"{self.title}\n\n{self.abstract}"
        return self.title

    @property
    def source_file(self) -> str:
        return self.url or self.title

    @property
    def page_number(self) -> int:
        return 0

    @property
    def chunk_index(self) -> int:
        return -1

    @property
    def doc_title(self) -> str:
        bits = [self.title]
        if self.year:
            bits.append(f"({self.year})")
        if self.venue:
            bits.append(f"— {self.venue}")
        return " ".join(bits)

    @property
    def citation(self) -> str:
        if self.arxiv_id:
            return f"arxiv:{self.arxiv_id}"
        return self.url or self.title

    @property
    def signal(self) -> str:
        return f"paper:{self.source}" if self.source else "paper"


# ---------------------------------------------------------------------------
# arXiv
# ---------------------------------------------------------------------------


def _arxiv_search(query: str, max_results: int = 5) -> list[PaperChunk]:
    """Use the official arxiv client. Synchronous; ~1s per query typically."""
    import arxiv

    client = arxiv.Client(page_size=max_results, delay_seconds=0.5, num_retries=2)
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,  # arxiv's relevance > date
    )

    out: list[PaperChunk] = []
    try:
        for r in client.results(search):
            # arxiv lib returns IDs like "http://arxiv.org/abs/2401.15884v3"
            arxiv_id = ""
            if r.entry_id:
                m = re.search(r"abs/([\w.\-]+?)(?:v\d+)?$", r.entry_id)
                if m:
                    arxiv_id = m.group(1)

            year = r.published.year if r.published else None
            out.append(
                PaperChunk(
                    title=(r.title or "").strip(),
                    abstract=(r.summary or "").strip(),
                    url=r.entry_id or "",
                    authors=[a.name for a in (r.authors or [])][:6],
                    year=year,
                    venue="arXiv",
                    source="arxiv",
                    arxiv_id=arxiv_id,
                    pdf_url=r.pdf_url or "",
                )
            )
    except Exception:
        # Non-fatal — discovery is a fallback path; return whatever we got.
        pass
    return out


# ---------------------------------------------------------------------------
# Semantic Scholar
# ---------------------------------------------------------------------------


def _semantic_scholar_search(query: str, max_results: int = 5) -> list[PaperChunk]:
    """
    Free unauthenticated API. Documented at "≈1 req/sec" but observed to
    429 under burst load — we pay a 3-second courtesy gap AFTER the call so
    a flurry of low-confidence agent runs (each one firing this on the
    Critic-gated fallback path) can't trip the throttle. One in-flight
    query per process; latency cost is bounded by `max_results`, not by
    the gap.

    Endpoint: /graph/v1/paper/search?query=...&limit=...&fields=...
    """
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "limit": str(max_results),
        "fields": "title,abstract,year,venue,authors,citationCount,openAccessPdf,externalIds,url",
    }
    try:
        r = httpx.get(url, params=params, timeout=15.0)
        # Match the seeder's gap; see src/ingest/s2_seed.py for the
        # rationale on why 1 RPS wasn't enough for free-tier S2 in 2026.
        time.sleep(3.0)
        if r.status_code != 200:
            return []
        data = r.json()
    except Exception:
        # Sleep even on transport failure — same reasoning as in s2_seed.
        time.sleep(3.0)
        return []

    out: list[PaperChunk] = []
    for item in (data.get("data") or []):
        if not item:
            continue
        ext = item.get("externalIds") or {}
        arxiv_id = (ext.get("ArXiv") or "").strip()
        oa = item.get("openAccessPdf") or {}
        pdf_url = (oa.get("url") if isinstance(oa, dict) else "") or ""

        out.append(
            PaperChunk(
                title=(item.get("title") or "").strip(),
                abstract=(item.get("abstract") or "").strip(),
                url=item.get("url") or "",
                authors=[
                    (a.get("name") or "") for a in (item.get("authors") or [])
                ][:6],
                year=item.get("year"),
                venue=(item.get("venue") or "").strip(),
                source="semantic_scholar",
                citations=item.get("citationCount"),
                arxiv_id=arxiv_id,
                pdf_url=pdf_url,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Discovery orchestrator
# ---------------------------------------------------------------------------


def _dedupe(papers: list[PaperChunk]) -> list[PaperChunk]:
    """Dedupe by arxiv_id first, then by normalized title."""
    seen_arxiv: set[str] = set()
    seen_title: set[str] = set()
    out: list[PaperChunk] = []
    for p in papers:
        norm_title = re.sub(r"\W+", " ", (p.title or "").lower()).strip()
        if p.arxiv_id and p.arxiv_id in seen_arxiv:
            continue
        if norm_title and norm_title in seen_title:
            continue
        if p.arxiv_id:
            seen_arxiv.add(p.arxiv_id)
        if norm_title:
            seen_title.add(norm_title)
        out.append(p)
    return out


def _rank_by_relevance(query: str, papers: list[PaperChunk], top_k: int) -> list[PaperChunk]:
    """
    Embedding-based reranking.

    arXiv/SS each have their own ranking, but they're not directly comparable
    and tend to weight recency / citations heavily. For our use case we want
    SEMANTIC relevance to the user's question — cosine on the embedder
    handles that and lets us merge cross-provider results fairly.
    """
    if len(papers) <= top_k:
        # Still score for display, just don't truncate.
        pass

    # Import here to avoid pulling the LLM stack when discovery is used
    # purely for display (e.g. the `discover` CLI command without ingestion).
    import numpy as np
    from src.llm import embed
    from src.config import ModelTier

    texts = [p.text[:2000] for p in papers]  # cap to keep embed batch sane
    try:
        vectors = embed([query] + texts, tier=ModelTier.EMBED)
    except Exception:
        # Embedder unavailable — fall back to source-native order.
        return papers[:top_k]

    if not vectors or len(vectors) < 2:
        return papers[:top_k]

    qv = np.asarray(vectors[0], dtype=np.float32)
    qn = qv / (np.linalg.norm(qv) + 1e-12)

    for p, v in zip(papers, vectors[1:]):
        pv = np.asarray(v, dtype=np.float32)
        pn = pv / (np.linalg.norm(pv) + 1e-12)
        # Clamp to [0, 1] — cosine can go negative for orthogonal vectors
        # but for natural-language embeddings that's vanishingly rare and
        # negative scores confuse downstream display.
        p.score = max(0.0, float(np.dot(qn, pn)))

    papers.sort(key=lambda p: p.score, reverse=True)
    return papers[:top_k]


def discover_papers(query: str, *, max_results: int = 5) -> list[PaperChunk]:
    """
    Search arXiv + Semantic Scholar, dedupe, rerank by query relevance.

    Returns up to `max_results` PaperChunks. Returns [] gracefully if every
    provider fails or returns nothing — caller can route to other fallbacks.
    """
    # Each provider gets a generous pool so dedupe + rerank can pick the best.
    per_provider = max(max_results, 5)

    t0 = time.perf_counter()
    arxiv_hits = _arxiv_search(query, max_results=per_provider)
    ss_hits = _semantic_scholar_search(query, max_results=per_provider)

    merged = _dedupe(arxiv_hits + ss_hits)
    if not merged:
        return []

    ranked = _rank_by_relevance(query, merged, top_k=max_results)
    _ = time.perf_counter() - t0  # caller logs timing into the agent trace
    return ranked
