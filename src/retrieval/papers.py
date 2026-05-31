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

import asyncio
import re
import sys
import time
import traceback
from dataclasses import dataclass
from typing import Any

import httpx


import os
from datetime import datetime
from pathlib import Path

# Sidecar log file — written in addition to stderr so we have ground truth
# even when terminal capture, Rich panels, or Windows stdout buffering hide
# the live output. Path is overridable via `RESEARGENT_PAPER_LOG`.
_PAPER_LOG_PATH = Path(
    os.environ.get("RESEARGENT_PAPER_LOG")
    or (Path.cwd() / "logs" / "paper_cascade.log")
)


def _debug(msg: str) -> None:
    """
    Route discovery/PDF-cascade debug lines to BOTH stderr AND a sidecar
    log file at logs/paper_cascade.log (or $RESEARGENT_PAPER_LOG).

    Why both: on Windows under `uv run`, stdout from inside the LangGraph
    async cascade is line-buffered and frequently never reaches the
    terminal before the Rich `Panel` renders the final result. stderr is
    *usually* unbuffered, but Rich and some agent runtimes still capture
    it. The sidecar log is the ground-truth source — if the cascade ran,
    these lines exist there regardless of what the terminal shows.

    Inspect after a run with (PowerShell):
        Get-Content .\logs\paper_cascade.log -Wait
    """
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    line = f"[{ts}] {msg}"
    # 1) stderr — visible in the terminal when not captured
    print(line, file=sys.stderr, flush=True)
    # 2) sidecar log — always works, regardless of terminal capture
    try:
        _PAPER_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _PAPER_LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except Exception:
        # Logging must NEVER break the cascade. Silently swallow disk errors.
        pass


@dataclass
class PaperChunk:
    """
    A discovered paper, exposed as the same interface as HybridChunk/WebChunk
    so generators/critics can mix all three without type-branching.

    `text` is the paper's evidence body that the generator and critic see.
    Three sources, used in priority order:
      1. `chunk_text` — one semantic slice of the parsed PDF (Phase 15.1).
         When the open-access PDF was fetched + parsed + chunked, the
         original PaperChunk gets cloned into multiple, each holding one
         slice. Each slice cites the same paper but quotes a specific
         passage.
      2. `full_text`  — the full parsed PDF, before semantic chunking.
         Surfaces when the caller wants the raw document (e.g. saving the
         whole PDF body to disk) but normally you should not see this in
         a generator prompt — it'd blow the token budget.
      3. `abstract`   — the title + abstract fallback. Used when the PDF
         was paywalled, 404'd, captcha-walled, or unparseable.
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

    # Phase 15.1: full-text enrichment fields. Both empty by default so all
    # legacy callers keep getting title+abstract via the `.text` fallback.
    full_text: str = ""    # raw concatenated PDF page text, set by fetcher
    chunk_text: str = ""   # one semantic slice; set by the chunker after fetch
    chunk_idx: int = 0     # 0-based index when this PaperChunk is a slice

    # ---- Public interface shared with HybridChunk / WebChunk ----
    @property
    def text(self) -> str:
        """What the generator/critic sees. Slice > full_text > abstract."""
        # `chunk_text` wins when set — that's a bounded passage (~500-800
        # tokens) safe to feed straight into the generator prompt.
        if self.chunk_text:
            return f"{self.title}\n\n{self.chunk_text}"
        # `full_text` fallback. Rarely emitted to the prompt path; mostly
        # useful when a caller wants the whole document for offline use.
        if self.full_text:
            return f"{self.title}\n\n{self.full_text}"
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
        # `chunk_idx` is set by the post-fetch chunker when this PaperChunk
        # is one slice of a parsed PDF (0, 1, 2…). Otherwise it's a single
        # abstract-only chunk and we return -1 for backward compatibility
        # with the dedup keys downstream code computes from chunk_index.
        return self.chunk_idx if self.chunk_text else -1

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


# Connectives / interrogatives / generic CS-lingo that hurt S2 ranking when
# mixed in with named entities. Kept lowercase, compared case-insensitively.
_S2_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "and", "are", "as", "at", "based", "be", "by", "complex",
    "control", "do", "does", "during", "each", "et", "execution", "exactly",
    "explain", "explained", "for", "from", "handle", "handles", "have", "her",
    "here", "his", "how", "i", "in", "into", "introduce", "introduced", "introduces",
    "is", "it", "its", "latest", "method", "methods", "model", "models", "new",
    "novel", "now", "of", "on", "or", "paper", "paradigm", "recent", "study",
    "studies", "system", "systems", "technique", "techniques", "than", "that",
    "the", "their", "them", "then", "this", "those", "through", "to", "two",
    "use", "uses", "using", "via", "way", "what", "when", "where", "which",
    "while", "who", "whom", "why", "with", "work", "works", "would",
    # "et al." artifacts after punctuation strip
    "al",
})


def _shorten_for_s2(query: str, *, max_terms: int = 5) -> str:
    """
    Distill a verbose research query into a short, S2-friendly keyword string.

    S2's /paper/search ranks by token-overlap and severely penalizes long
    keyword piles: a 12-word query like "Wu et al. AutoGen multi-agent
    complex control flow conversational programming paradigm" returns
    near-random results (colitis studies, manufacturing libraries) because
    every extra generic term ("complex", "control", "system", "paradigm")
    dilutes the signal from the actual named entity ("AutoGen").

    Heuristic, in priority order:
      1. Prefer CapitalCase / ALLCAPS / mixed-case tokens (proper nouns,
         acronyms, framework names — these are the real semantic anchors)
      2. Then content words ≥4 chars that survive the stoplist
      3. Cap at `max_terms` tokens, preserving original order so phrase-y
         queries ("conversational programming") stay together

    arXiv handles the verbose query fine, so this is S2-only.
    """
    # Strip common author-citation noise that fragments tokenization
    cleaned = re.sub(r"\bet\s+al\.?", "", query, flags=re.IGNORECASE)
    cleaned = cleaned.replace("'", "").replace('"', "")

    # Token boundaries on whitespace; keep internal hyphens/dots (e.g. "Wu",
    # "AutoGen", "GPT-4", "v2.0")
    raw_tokens = re.findall(r"[A-Za-z][A-Za-z0-9.\-]*", cleaned)

    capitalized: list[str] = []
    content: list[str] = []
    seen: set[str] = set()

    for tok in raw_tokens:
        key = tok.lower()
        if key in seen or key in _S2_STOPWORDS:
            continue
        # ALLCAPS acronyms (LLM, RAG, AutoGen-ish) and TitleCase entities are
        # the highest-signal tokens for S2 keyword matching.
        if tok[0].isupper() and tok.lower() != tok:
            capitalized.append(tok)
            seen.add(key)
        elif len(tok) >= 4:
            content.append(tok)
            seen.add(key)

    # Capitalized tokens first (entities anchor the query), then content
    # tokens to fill the budget. Preserve insertion order within each bucket
    # so multi-word entity phrases stay co-occurring.
    picked = (capitalized + content)[:max_terms]
    if not picked:
        # No usable tokens (degenerate query) — fall back to original so we
        # don't send an empty string to S2.
        return query.strip()
    short = " ".join(picked)
    _debug(f"[S2] query shortened: {query!r} → {short!r}")
    return short


def _semantic_scholar_search(query: str, max_results: int = 5) -> list[PaperChunk]:
    """
    Semantic Scholar paper search.

    Two auth regimes:
      * No key set:  public endpoint, 3.0s courtesy gap after each call.
                     Documented as "≈1 req/sec" but 429s under burst.
      * Key set:     `x-api-key` header per the official tutorial
                     (https://www.semanticscholar.org/product/api/tutorial),
                     1.1s courtesy gap — just under the documented 1 RPS
                     ceiling, ~3× faster than the public path.

    Endpoint: /graph/v1/paper/search?query=...&limit=...&fields=...
    """
    from src.config import settings

    # S2 ranking collapses on long keyword piles — distill to ≤5 high-signal
    # tokens (entities + acronyms first) before hitting the endpoint. Logged
    # via _debug so we can audit the rewrite in paper_cascade.log.
    s2_query = _shorten_for_s2(query)

    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": s2_query,
        "limit": str(max_results),
        "fields": "title,abstract,year,venue,authors,citationCount,openAccessPdf,externalIds,url",
    }
    # Auth + gap chosen at call time so a .env change takes effect on the
    # next agent run without a restart.
    key = settings.semantic_scholar_api_key
    headers = {"x-api-key": key} if key else {}
    gap = 1.1 if key else 3.0
    try:
        r = httpx.get(url, params=params, headers=headers, timeout=15.0)
        # Match s2_seed.py's pattern — sleep AFTER every successful or
        # 4xx-returning call so the next caller in the same process gets
        # the throttle for free.
        time.sleep(gap)
        if r.status_code != 200:
            _debug(
                f"[S2] HTTP {r.status_code} on /paper/search "
                f"q={s2_query!r} body={r.text[:200]!r}"
            )
            return []
        data = r.json()
    except Exception as e:
        # Sleep even on transport failure — a flapping endpoint shouldn't
        # be hammered by the next query in the same agent run.
        _debug(f"[S2] transport error on /paper/search q={s2_query!r}: {type(e).__name__}: {e}")
        traceback.print_exc(file=sys.stderr)
        time.sleep(gap)
        return []

    out: list[PaperChunk] = []
    for item in (data.get("data") or []):
        if not item:
            continue
        ext = item.get("externalIds") or {}
        arxiv_id = (ext.get("ArXiv") or "").strip()
        oa = item.get("openAccessPdf") or {}
        pdf_url = (oa.get("url") if isinstance(oa, dict) else "") or ""
        # Explicit per-paper trace so we can SEE whether S2 returned an
        # openAccessPdf URL for each candidate, vs. silently no-OA.
        _debug(
            f"[S2] Checking for OA PDF: {item.get('title')!r} "
            f"→ {pdf_url if pdf_url else '(none returned by S2)'}"
        )

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


# ---------------------------------------------------------------------------
# Phase 15.1 — async full-text enrichment for open-access PDFs
# ---------------------------------------------------------------------------
#
# When a discovered paper exposes an `openAccessPdf.url`, we fetch the raw PDF
# bytes asynchronously, parse the text with pypdf, semantically chunk it, and
# RANK each chunk by its query relevance. The original abstract-only PaperChunk
# is then replaced with the top-N most relevant slices. Net effect: the Critic
# now grades real passages from the paper rather than blurbs, and the Generator
# can cite specific evidence rather than restate abstracts.
#
# Robustness contract: ANY failure (HTTP 4xx/5xx, captcha redirect, timeout,
# corrupted bytes, pypdf parse error, network drop) falls back to the
# abstract-only path with a warning. The agent's discovery cascade never
# breaks because of a flaky open-access mirror.

# Standard browser-ish User-Agent. Many open-access mirrors 403 on `python-httpx`
# default. We don't lie about being a real browser — just identify ourselves
# clearly enough to pass the trivial bot filters most journal CDNs apply.
_HTTP_HEADERS = {
    "User-Agent": (
        "ResearGent/0.1 (+https://github.com/SaumyaBish-t/ResearGent) "
        "httpx/async pypdf"
    ),
    "Accept": "application/pdf, */*",
}

# Per-PDF timeout for the async GET. 15s is the user-spec value; chosen so a
# slow mirror doesn't gate the agent on one paper while N-1 others succeed.
_PDF_FETCH_TIMEOUT = 15.0

# Max parallel downloads. Open-access mirrors aren't rate-limit-friendly the
# way the S2 search API is; 4 is a good "fast but not abusive" default.
_MAX_PARALLEL_FETCHES = 4

# Max semantic slices kept per paper after chunking. The cascade returns at
# most ~5 papers; expanding each into 3 chunks puts ~15 evidence units in
# front of the Critic — enough to grade specifics, not so many that the
# generator's prompt budget blows.
_MAX_CHUNKS_PER_PAPER = 3

# Hard cap on raw extracted PDF text. Some open-access PDFs are 80+ page
# theses; semantically chunking 200K chars per paper is wasted work since
# we only keep top-N anyway. Truncating to the first ~60K chars covers
# title + abstract + intro + most of methods, where research-question
# evidence almost always lives.
_MAX_FULL_TEXT_CHARS = 60_000


async def _fetch_pdf_bytes(client: "httpx.AsyncClient", url: str) -> bytes | None:
    """
    Download one PDF asynchronously. Returns the raw bytes on success, None
    on any failure (logged but non-fatal).
    """
    try:
        # follow_redirects: openAccessPdf URLs commonly chain through DOI
        # resolvers → publisher landing → CDN before hitting the actual file.
        r = await client.get(url, follow_redirects=True)
    except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.RemoteProtocolError) as e:
        _debug(f"⚠️ PDF Fetch Failed (timeout/protocol) for {url}: {type(e).__name__}: {e}")
        traceback.print_exc(file=sys.stderr)
        return None
    except Exception as e:
        _debug(f"⚠️ PDF Fetch Failed (transport) for {url}: {type(e).__name__}: {e}")
        traceback.print_exc(file=sys.stderr)
        return None

    if r.status_code != 200:
        # 403 = bot wall, 404 = link rot, 451 = legal gate. All resolve to
        # "fall back to abstract", same code path.
        _debug(f"⚠️ PDF Fetch Failed (HTTP {r.status_code}) for {url}")
        return None

    # Content-type sniff — many "open access" pages return an HTML paywall
    # or captcha shim when the actual PDF is gated. The PDF magic number
    # check is the ground truth.
    body = r.content or b""
    ctype = (r.headers.get("content-type") or "").lower()
    if "pdf" not in ctype and not body[:5].startswith(b"%PDF-"):
        _debug(f"⚠️ PDF Fetch Failed (non-PDF response, content-type={ctype!r}) for {url}")
        return None
    return body


def _parse_pdf_bytes(data: bytes) -> str:
    """
    Extract text from PDF bytes using pypdf. Concatenates all pages into one
    document string for downstream semantic chunking.

    Raises on parse failure — callers should wrap in try/except and fall back
    to abstract on any error (matches the spec's "fall back to abstract" rule
    for `pypdf.errors.PdfReadError` and friends).
    """
    import io
    import pypdf  # local import — heavy module, only loaded on the cascade path

    reader = pypdf.PdfReader(io.BytesIO(data))
    pages: list[str] = []
    for page in reader.pages:
        # `extract_text()` itself can raise on pages with weird font tables;
        # swallowing one bad page is better than discarding the rest of the
        # document. The OUTER try in `_enrich_one_paper` catches any failure
        # the per-page swallow couldn't.
        try:
            t = page.extract_text() or ""
        except Exception:
            t = ""
        if t.strip():
            pages.append(t)
    return "\n\n".join(pages)


async def _enrich_one_paper(client: "httpx.AsyncClient", paper: PaperChunk) -> None:
    """
    Fetch + parse one paper's open-access PDF, mutate `paper.full_text`
    in place. Silent no-op when the paper has no `pdf_url` or anything
    fails — the abstract fallback in `PaperChunk.text` then takes over.
    """
    url = (paper.pdf_url or "").strip()
    if not url:
        _debug(f"[enrich] SKIP (no pdf_url) source={paper.source} title={paper.title!r}")
        return

    _debug(f"[enrich] FETCH source={paper.source} url={url}")

    try:
        data = await _fetch_pdf_bytes(client, url)
        if not data:
            return
        text = _parse_pdf_bytes(data)
    except Exception as e:
        # Catches pypdf.errors.PdfReadError, malformed-stream errors,
        # decryption-required errors, and anything else pypdf surfaces.
        # We never want one bad paper to break the whole discovery cascade.
        _debug(
            f"⚠️ PDF Parse Failed for {url} "
            f"(paper: {paper.citation}): {type(e).__name__}: {e}"
        )
        traceback.print_exc(file=sys.stderr)
        return

    _debug(
        f"[enrich] OK url={url} parsed_chars={len(text)} "
        f"(truncated to {_MAX_FULL_TEXT_CHARS})"
    )

    if not text.strip():
        # PDF parsed but yielded no text — usually a scanned/image-only PDF
        # (theses, very old papers). No point setting full_text="".
        return

    paper.full_text = text[:_MAX_FULL_TEXT_CHARS]


async def _enrich_async(papers: list[PaperChunk]) -> None:
    """
    Concurrently fetch + parse all papers that have an open-access PDF URL.
    Mutates each paper's `full_text` in place; no return value.

    Concurrency is bounded by `_MAX_PARALLEL_FETCHES` via a semaphore so
    we don't open 50 sockets when discovery returns a big list.
    """
    pdf_papers = [p for p in papers if p.pdf_url]
    if not pdf_papers:
        return

    sem = asyncio.Semaphore(_MAX_PARALLEL_FETCHES)

    async with httpx.AsyncClient(
        timeout=_PDF_FETCH_TIMEOUT,
        headers=_HTTP_HEADERS,
        follow_redirects=True,
    ) as client:
        async def _bound(p: PaperChunk) -> None:
            async with sem:
                await _enrich_one_paper(client, p)

        await asyncio.gather(*(_bound(p) for p in pdf_papers), return_exceptions=False)


def _enrich_with_full_text(papers: list[PaperChunk]) -> None:
    """
    Synchronous entry point used by `discover_papers`.

    We `asyncio.run` rather than spinning a long-lived loop because
    paper-discovery is a one-shot fan-out: open N sockets, gather, close.
    Inside an existing event loop (the FastAPI streaming path could
    in principle call this), `asyncio.run` would raise — we catch that
    and silently fall back, preserving the abstract path.
    """
    if not papers:
        return
    try:
        asyncio.run(_enrich_async(papers))
    except RuntimeError:
        # Already inside an event loop. Skip enrichment rather than try to
        # nest; the abstract fallback is correct and bounded.
        return
    except Exception as e:
        # Defensive — _enrich_async swallows its own errors per-paper, but
        # an asyncio-level surprise shouldn't kill the cascade.
        print(f"  [paper-enrich warn] async pool failed: {type(e).__name__}: {e}")


def _expand_with_semantic_chunks(
    query: str, papers: list[PaperChunk]
) -> list[PaperChunk]:
    """
    For papers with `full_text` set, run the semantic chunker and replace
    the single PaperChunk with the top-N slices most relevant to `query`.

    Papers without full_text are passed through unchanged (abstract path).
    Order is preserved: original rank-by-relevance ordering at the paper
    level still holds; within an expanded paper, slices come in
    relevance-descending order.

    Why expand here, not in the node
    --------------------------------
    Keeping fetch + parse + chunk all inside `discover_papers` means:
      * The async event loop is opened and closed exactly once per cascade.
      * Downstream nodes (paper_discovery.py, the critic, the generator)
        keep treating every PaperChunk as opaque evidence — no chunking
        logic leaks into the graph layer.
    """
    if not papers:
        return papers

    # Lazy import — the chunker module pulls torch via sentence-transformers,
    # which is expensive cold. Only paid when we actually have full-text PDFs.
    try:
        from src.ingest.chunker import semantic_chunk_text
    except Exception:
        # Chunker unavailable (e.g. sentence-transformers missing). Falls
        # back to "use the full_text as one big chunk" — the generator will
        # take the first ~K tokens and the rest will be wasted, but at
        # least no crash.
        semantic_chunk_text = None  # type: ignore[assignment]

    out: list[PaperChunk] = []
    # For ranking slices within one paper, compute the query embedding once
    # and reuse. Embedding the query per-paper would be wasted work.
    import numpy as np
    qvec = None
    try:
        from src.config import ModelTier
        from src.llm import embed as _embed_fn
        qvec_raw = _embed_fn([query], tier=ModelTier.EMBED)
        if qvec_raw and qvec_raw[0]:
            qv = np.asarray(qvec_raw[0], dtype=np.float32)
            qvec = qv / (np.linalg.norm(qv) + 1e-12)
    except Exception:
        qvec = None  # rank-fallback path below uses original chunk order

    for p in papers:
        if not p.full_text or semantic_chunk_text is None:
            out.append(p)
            continue

        # Semantic chunker yields ~500-800 token chunks aligned on topical
        # boundaries. Same primitive the ingest pipeline uses.
        try:
            slices = semantic_chunk_text(p.full_text)
        except Exception as e:
            print(f"  [paper-chunk warn] {p.citation}: {type(e).__name__}: {e}")
            slices = []

        if not slices:
            out.append(p)
            continue

        # Rank slices by similarity to the query. When embedder is broken,
        # fall back to "first N slices" — usually intro + methods, decent
        # default for research questions.
        scored: list[tuple[float, str]] = []
        if qvec is not None:
            try:
                from src.config import ModelTier
                from src.llm import embed as _embed_fn
                slice_vecs = _embed_fn([s[:2000] for s in slices], tier=ModelTier.EMBED)
                for s, v in zip(slices, slice_vecs):
                    sv = np.asarray(v, dtype=np.float32)
                    sn = sv / (np.linalg.norm(sv) + 1e-12)
                    scored.append((float(np.dot(qvec, sn)), s))
                scored.sort(reverse=True)
            except Exception:
                scored = [(0.0, s) for s in slices]
        else:
            scored = [(0.0, s) for s in slices]

        top_n = scored[:_MAX_CHUNKS_PER_PAPER]
        # Emit one PaperChunk per kept slice. We shallow-copy the original
        # paper's metadata so each slice keeps the same citation / year /
        # url — downstream code dedupes by (source_file, chunk_index) which
        # we keep distinct via chunk_idx.
        for i, (score_i, slice_text) in enumerate(top_n):
            sliced = PaperChunk(
                title=p.title,
                abstract=p.abstract,
                url=p.url,
                authors=list(p.authors),
                year=p.year,
                venue=p.venue,
                source=p.source,
                citations=p.citations,
                arxiv_id=p.arxiv_id,
                pdf_url=p.pdf_url,
                score=score_i if score_i > 0 else p.score,
                full_text="",          # don't carry the 60KB blob on every slice
                chunk_text=slice_text,
                chunk_idx=i,
            )
            out.append(sliced)

    return out


def discover_papers(
    query: str, *, max_results: int = 5, enrich_full_text: bool = True
) -> list[PaperChunk]:
    """
    Search arXiv + Semantic Scholar, dedupe, rerank by query relevance, then
    enrich the open-access papers with full-text PDF parsing (Phase 15.1).

    Returns up to ~max_results × _MAX_CHUNKS_PER_PAPER PaperChunks when
    enrichment is on (each open-access paper expands into multiple slices).
    Returns up to `max_results` abstract-only chunks when off, or when no
    papers have OA PDFs.

    Set `enrich_full_text=False` for the CLI `discover` command (display-only)
    where the fetch+parse latency isn't worth paying for a list view.
    """
    # Each provider gets a generous pool so dedupe + rerank can pick the best.
    per_provider = max(max_results, 5)

    _debug(
        f"=== discover_papers START query={query!r} "
        f"max_results={max_results} enrich={enrich_full_text} ==="
    )

    t0 = time.perf_counter()
    arxiv_hits = _arxiv_search(query, max_results=per_provider)
    _debug(f"[arxiv] returned {len(arxiv_hits)} hits")
    ss_hits = _semantic_scholar_search(query, max_results=per_provider)
    _debug(f"[S2] returned {len(ss_hits)} hits")

    merged = _dedupe(arxiv_hits + ss_hits)
    if not merged:
        return []

    ranked = _rank_by_relevance(query, merged, top_k=max_results)

    if enrich_full_text:
        _debug(
            f"[enrich] ranked={len(ranked)} papers, "
            f"with_pdf_url={sum(1 for p in ranked if p.pdf_url)}"
        )
        # Async fetch + parse mutates each paper's full_text in place. Bounded
        # at _MAX_PARALLEL_FETCHES concurrent sockets so we don't hammer
        # journal mirrors. Total wall-time: ~1-3s for 5 papers on a warm
        # connection, dominated by the slowest single download.
        _enrich_with_full_text(ranked)
        # Expand each PDF-enriched paper into top-N semantic slices ranked
        # by query relevance. Papers without full_text pass through unchanged.
        ranked = _expand_with_semantic_chunks(query, ranked)
        _debug(f"[enrich] DONE after_expand={len(ranked)} chunks")

    dur = time.perf_counter() - t0
    _debug(f"=== discover_papers END total={len(ranked)} chunks in {dur:.1f}s ===")
    return ranked
