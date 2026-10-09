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
import os
import re
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx

# Sidecar log file — written in addition to stderr so we have ground truth
# even when terminal capture, Rich panels, or Windows stdout buffering hide
# the live output. Path is overridable via `RESEARGENT_PAPER_LOG`.
_PAPER_LOG_PATH = Path(
    os.environ.get("RESEARGENT_PAPER_LOG")
    or (Path.cwd() / "logs" / "paper_cascade.log")
)


def _debug(msg: str) -> None:
    r"""
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
    except Exception as e:
        # Previously a silent `pass` — but a flapping arXiv quietly returning
        # 0 hits is exactly the regression we just hit on the AutoGen query
        # (5 hits → 0 hits between runs with no other changes). Log the
        # actual exception + traceback into paper_cascade.log so we can
        # diagnose it instead of guessing.
        _debug(f"⚠️ arXiv search failed q={query!r}: {type(e).__name__}: {e}")
        traceback.print_exc(file=sys.stderr)
    if not out:
        # Empty result is by itself a signal — arXiv normally returns SOMETHING
        # for any well-formed natural-language query. A 0-hit response usually
        # means a 429 rate-limit, transient 5xx, or a malformed query the
        # arxiv client swallowed without raising.
        _debug(f"[arxiv] WARNING: 0 results for q={query!r} — check rate limits / connectivity")
    return out


# ---------------------------------------------------------------------------
# Semantic Scholar
# ---------------------------------------------------------------------------


# Connectives / interrogatives / generic CS-lingo that hurt S2 ranking when
# mixed in with named entities. Kept lowercase, compared case-insensitively.
_S2_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "and", "are", "as", "at", "based", "be", "broad", "by", "class",
    "classes", "complex", "control", "deep", "design", "designed", "do", "does",
    "during", "each", "et", "execution", "exactly", "explain", "explained",
    # "flow" is overloaded — "control flow" intent gets dragged to "flow-shop
    # scheduling" on S2 unless removed
    "flow",
    "for", "from", "general", "handle", "handles", "have", "her", "here",
    "his", "how", "i", "in", "into", "introduce", "introduced", "introduces",
    "is", "it", "its", "large", "latest", "learning", "method", "methods", "model",
    "models", "multi", "network", "new", "novel", "now", "of", "on", "or", "paper",
    "paradigm", "process", "real", "recent", "review", "study", "studies", "survey",
    "system", "systems", "technique", "techniques", "than", "that", "the", "their",
    "them", "then", "this", "those", "through", "to", "two", "use", "uses", "using",
    "via", "way", "what", "when", "where", "which", "while", "who", "whom", "why",
    "with", "work", "works", "would",
    # "et al." artifacts after punctuation strip
    "al",
})


# ---------------------------------------------------------------------------
# Multi-query generation for stronger paper discovery
# ---------------------------------------------------------------------------

# Domain vocabulary expansions — synonyms and related terms that help recall
# when the user's question uses different terminology than the paper.
# Loaded from domains.py routing_keywords + seed_queries for automatic coverage.
_DOMAIN_VOCAB: dict[str, list[str]] = {}


def _load_domain_vocab() -> None:
    """Populate _DOMAIN_VOCAB from the domain registry."""
    global _DOMAIN_VOCAB
    if _DOMAIN_VOCAB:
        return
    try:
        from src.domains import DOMAINS
        for dom_id, dom in DOMAINS.items():
            terms: list[str] = []
            terms.extend(dom.routing_keywords)
            for sq in dom.seed_queries:
                # Extract meaningful tokens from seed queries
                tokens = re.findall(r"[A-Za-z][A-Za-z0-9.\-]*", sq)
                for tok in tokens:
                    if len(tok) >= 3 and tok.lower() not in _S2_STOPWORDS:
                        terms.append(tok)
            # Dedupe, keep order
            seen = set()
            unique = []
            for t in terms:
                key = t.lower()
                if key not in seen:
                    seen.add(key)
                    unique.append(t)
            _DOMAIN_VOCAB[dom_id] = unique
    except Exception:
        _DOMAIN_VOCAB = {}


def _expand_with_domain_vocab(query: str, domain_id: str | None) -> list[str]:
    """
    Generate query variants by injecting domain-relevant terms.
    
    Returns the original query plus up to 2 expanded variants that add
    domain vocabulary terms not already present in the query.
    """
    if not domain_id:
        return [query]
    _load_domain_vocab()
    vocab = _DOMAIN_VOCAB.get(domain_id, [])
    if not vocab:
        return [query]
    
    q_lower = query.lower()
    # Find vocab terms not already in the query
    missing = [t for t in vocab if t.lower() not in q_lower]
    if not missing:
        return [query]
    
    # Create 2 variants: add top 1-2 missing terms each
    variants = [query]
    # Variant 1: add top missing term
    v1 = query + " " + missing[0]
    variants.append(v1)
    # Variant 2: add top 2 missing terms (if available)
    if len(missing) >= 2:
        v2 = query + " " + " ".join(missing[:2])
        variants.append(v2)
    return variants


def _generate_search_queries(
    question: str, 
    domain_id: str | None = None,
    max_queries: int = 5
) -> list[str]:
    """
    Generate multiple diverse search queries from a single research question.
    
    Strategies:
    1. Original — user's verbatim question (arXiv handles this well)
    2. Entity-focused — only named entities + acronyms (S2 loves this)
    3. Technical — LLM-rewritten to academic keywords (current _extract_search_query behavior)
    4. Domain-expanded — inject domain vocabulary for recall
    5. Broad — strip to 1-2 core concepts
    
    Returns up to max_queries unique, non-empty queries.
    """
    from src.agent.nodes.paper_discovery import _extract_named_entities, _extract_search_query
    
    queries: list[str] = []
    seen: set[str] = set()
    
    def add(q: str) -> None:
        q = q.strip()
        if q and q.lower() not in seen:
            seen.add(q.lower())
            queries.append(q)
    
    # 1. Original question (verbatim)
    add(question)

    # Callers that already supply a concise, targeted query can skip the
    # rewrite step and its extra model call entirely.
    if max_queries == 1:
        return queries
    
    # 2. Entity-focused — just the anchors
    anchors = _extract_named_entities(question)
    if anchors:
        add(" ".join(anchors))
    
    # 3. Technical/LLM-rewritten (current behavior)
    technical = _extract_search_query(question)
    add(technical)
    
    # 4. Domain-expanded variants
    for variant in _expand_with_domain_vocab(technical, domain_id):
        add(variant)
    
    # 5. Broad — take first 1-2 anchor/content tokens
    if anchors:
        add(" ".join(anchors[:2]))
    else:
        # Fallback: first few meaningful words from technical query
        tokens = re.findall(r"[A-Za-z][A-Za-z0-9.\-]*", technical)
        content_tokens = [t for t in tokens if len(t) >= 4 and t.lower() not in _S2_STOPWORDS]
        if content_tokens:
            add(" ".join(content_tokens[:2]))
    
    return queries[:max_queries]


def _is_anchor_token(tok: str) -> bool:
    """
    True for entity-anchor tokens — PascalCase / CamelCase compounds
    (AutoGen, ChatGPT, LangChain), or ALLCAPS acronyms ≥3 chars (BERT,
    GLiNER, GPT). These are the strongest signal for S2 keyword search
    because they almost certainly name THE thing the user wants.

    Single-char or 2-char capitalized tokens ("Wu", "Li") are NOT anchors
    even if technically capitalized — they're surname noise.
    """
    if len(tok) < 3:
        return False
    if tok.isupper():
        return True  # ALLCAPS acronym (BERT, GPT, GLiNER-ish written as such)
    # PascalCase / CamelCase: starts upper AND has another upper later
    # (AutoGen, LangChain, ChatGPT). Plain "Conversational" doesn't qualify.
    return tok[0].isupper() and any(c.isupper() for c in tok[1:])


def _shorten_for_s2(query: str, *, max_terms: int = 4) -> str:
    """
    Distill a verbose research query into a short, S2-friendly keyword string.

    S2's /paper/search ranks by token-overlap and severely penalizes long
    keyword piles: a 12-word query like "Wu et al. AutoGen multi-agent
    complex control flow conversational programming paradigm" returns
    near-random results (manufacturing flow-shop, colitis studies) because
    every extra generic term ("complex", "control", "system", "paradigm")
    dilutes the signal from the actual named entity ("AutoGen") and broad
    terms like "flow" actively mis-rank toward the wrong sub-literature.

    Heuristic, in priority order:
      1. ENTITY ANCHORS — PascalCase / CamelCase / ALLCAPS-≥3 tokens
         (AutoGen, ChatGPT, BERT). When at least one anchor is present
         the budget tightens to ≤3 tokens (1-2 anchors + ≤1 supporting
         content word) because S2 ranks much better with a tight query
         centered on the entity.
      2. TitleCase common-words (Conversational, Programming) — kept as
         secondary signal when no PascalCase anchor exists.
      3. Content words ≥4 chars that survive the stoplist — only used
         to round out the query when entity signal is weak.
      4. Single/two-char tokens ("Wu", "Li", "et al." remnants) DROPPED —
         too generic, drag in unrelated papers by other authors.

    arXiv handles the verbose query fine, so this is S2-only.
    """
    # Strip common author-citation noise that fragments tokenization
    cleaned = re.sub(r"\bet\s+al\.?", "", query, flags=re.IGNORECASE)
    cleaned = cleaned.replace("'", "").replace('"', "")

    # Token boundaries on whitespace; keep internal hyphens/dots (e.g.
    # "AutoGen", "GPT-4", "v2.0")
    raw_tokens = re.findall(r"[A-Za-z][A-Za-z0-9.\-]*", cleaned)

    anchors: list[str] = []
    titlecase: list[str] = []
    content: list[str] = []
    seen: set[str] = set()

    for tok in raw_tokens:
        key = tok.lower()
        if key in seen or key in _S2_STOPWORDS:
            continue
        # Drop short surname-like tokens — even capitalized "Wu" / "Li"
        # are net negative for S2 ranking on framework questions.
        if len(tok) <= 2:
            continue
        if _is_anchor_token(tok):
            anchors.append(tok)
        elif tok[0].isupper() and len(tok) >= 4:
            titlecase.append(tok)
        elif len(tok) >= 4:
            content.append(tok)
        else:
            continue
        seen.add(key)

    # Budget allocation:
    #   anchors present  → ≤2 anchors, then fill to 3 total from titlecase → content
    #   no anchors       → ≤2 titlecase + ≤2 content (4 total)
    if anchors:
        picked: list[str] = list(anchors[:2])
        for tok in titlecase + content:
            if len(picked) >= 3:
                break
            picked.append(tok)
    else:
        picked = (titlecase[:2] + content[:2])[:max_terms]

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

        # FALLBACK: if S2 doesn't expose an openAccessPdf URL but the paper
        # HAS an arXiv ID, synthesize the arxiv.org/pdf URL directly.
        #
        # Why this matters: S2's `openAccessPdf` field is sparsely populated
        # — many high-impact papers (including 2308.08155 AutoGen itself
        # with 1700+ citations) have an empty openAccessPdf despite being
        # freely available on arxiv. Without this synthesis, the cascade
        # discovers the paper but can't download it, falls back to abstract,
        # and the answer ends up grounded on Tavily web scrapes instead of
        # the real PDF text.
        if not pdf_url and arxiv_id:
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
            _debug(f"[S2] synthesized arxiv PDF URL for {arxiv_id} → {pdf_url}")

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
    Embedding-based reranking with citation and recency awareness.
    
    Combines:
    - Semantic similarity (cosine) — 80% weight
    - Citation count (log-normalized) — 15% weight
    - Recency bonus (newer papers get slight boost) — 5% weight
    
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
    from datetime import datetime

    import numpy as np

    from src.config import ModelTier
    from src.llm import embed

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

    # Compute max citations for normalization
    max_citations = max((p.citations or 0) for p in papers)
    current_year = datetime.now().year

    for p, v in zip(papers, vectors[1:]):
        pv = np.asarray(v, dtype=np.float32)
        pn = pv / (np.linalg.norm(pv) + 1e-12)
        # Clamp to [0, 1] — cosine can go negative for orthogonal vectors
        # but for natural-language embeddings that's vanishingly rare and
        # negative scores confuse downstream display.
        semantic_score = max(0.0, float(np.dot(qn, pn)))
        
        # Citation score: log(citations + 1) / log(max_citations + 1)
        citations = p.citations or 0
        if max_citations > 0:
            citation_score = np.log(citations + 1) / np.log(max_citations + 1)
        else:
            citation_score = 0.0
        
        # Recency score: linear decay from current year, papers >= 10 years old get 0
        year = p.year or current_year
        age = current_year - year
        recency_score = max(0.0, 1.0 - age / 10.0)
        
        # Combined score
        p.score = 0.8 * semantic_score + 0.15 * citation_score + 0.05 * recency_score

    papers.sort(key=lambda p: p.score, reverse=True)
    return papers[:top_k]


def _domain_boost(papers: list[PaperChunk], domain_id: str | None) -> list[PaperChunk]:
    """
    Boost papers that match the domain's vocabulary.
    
    Adds a small bonus (up to +0.1) to papers whose title/abstract/venue
    contain domain routing keywords or seed query terms.
    """
    if not domain_id:
        return papers
    
    _load_domain_vocab()
    vocab = _DOMAIN_VOCAB.get(domain_id, [])
    if not vocab:
        return papers
    
    vocab_lower = {v.lower() for v in vocab}
    
    for p in papers:
        # Check title, abstract, venue for domain terms
        text = f"{p.title} {p.abstract} {p.venue}".lower()
        matches = sum(1 for v in vocab_lower if v in text)
        if matches > 0:
            # Small boost: +0.02 per match, capped at +0.1
            boost = min(0.02 * matches, 0.1)
            p.score = min(1.0, p.score + boost)
    
    papers.sort(key=lambda p: p.score, reverse=True)
    return papers


def _multi_query_search(
    queries: list[str], 
    max_results: int,
    domain_id: str | None = None
) -> list[PaperChunk]:
    """
    Run multiple search queries against both providers, merge and dedupe results.
    
    Each query runs against arXiv and Semantic Scholar. Results are merged,
    deduped, and ranked by relevance.
    """
    all_papers: list[PaperChunk] = []
    
    for q in queries:
        _debug(f"[multi-query] Searching: {q!r}")
        arxiv_hits = _arxiv_search(q, max_results=max_results)
        _debug(f"[multi-query] arXiv returned {len(arxiv_hits)} hits for {q!r}")
        ss_hits = _semantic_scholar_search(q, max_results=max_results)
        _debug(f"[multi-query] S2 returned {len(ss_hits)} hits for {q!r}")
        all_papers.extend(arxiv_hits)
        all_papers.extend(ss_hits)
    
    # Dedupe across all queries
    merged = _dedupe(all_papers)
    _debug(f"[multi-query] After dedupe: {len(merged)} unique papers")
    
    if not merged:
        return []
    
    # Rank by relevance (using the first query as the primary for semantic scoring)
    primary_query = queries[0] if queries else ""
    ranked = _rank_by_relevance(primary_query, merged, top_k=max_results * 2)  # Get more for domain boost
    
    # Apply domain boost if available
    if domain_id:
        ranked = _domain_boost(ranked, domain_id)
    
    return ranked[:max_results]


def _crag_rewrite_and_retry(
    original_question: str,
    current_papers: list[PaperChunk],
    domain_id: str | None,
    attempt: int,
    max_attempts: int = 2
) -> list[PaperChunk]:
    """
    CRAG-style query rewriting when initial search yields weak results.
    
    If we have fewer than 3 papers above the relevance floor, use the FAST
    LLM to rewrite the query and retry. Up to max_attempts rewrites.
    """
    if attempt >= max_attempts:
        return current_papers
    
    # Count papers above floor
    strong_papers = [p for p in current_papers if p.score >= _PAPER_SCORE_FLOOR]
    
    if len(strong_papers) >= 3:
        return current_papers  # Good enough, no rewrite needed
    
    _debug(f"[CRAG] Weak results ({len(strong_papers)} strong papers), rewriting query (attempt {attempt + 1})")
    
    try:
        import re

        from src.config import ModelTier
        from src.llm import chat
        
        rewrite_prompt = f"""The user asked: "{original_question}"

Our paper search returned only {len(strong_papers)} highly relevant papers (score >= {_PAPER_SCORE_FLOOR}).
The top results were:
{chr(10).join(f'- {p.title[:80]} (score={p.score:.2f}, citations={p.citations})' for p in current_papers[:5])}

Rewrite the search query to find MORE relevant papers. Focus on:
- Different terminology the literature might use
- Broader or narrower concepts
- Related methods/frameworks

Output ONLY the rewritten search query. Use PLAIN TEXT only — NO quotes, NO boolean operators (AND/OR/NOT), NO parentheses, NO special characters. Just a simple keyword query like "AutoGen multi-agent conversation framework"."""
        
        rewritten = chat(
            messages=[
                {"role": "system", "content": "You are a query rewriter for academic paper search. Output only a plain keyword query — no quotes, no boolean operators, no special characters."},
                {"role": "user", "content": rewrite_prompt},
            ],
            tier=ModelTier.FAST,
            temperature=0.3,
            max_tokens=60,
        ).strip().strip('"').strip("'").split('\n')[0].strip()
        
        # Clean up any remaining special characters that break arXiv
        rewritten = re.sub(r'[\"\'\\(\\)\[\]\{\}\+\-\&\|\!\~\*\:]', ' ', rewritten)
        rewritten = re.sub(r'\s+', ' ', rewritten).strip()
        
        if rewritten and rewritten.lower() != original_question.lower():
            _debug(f"[CRAG] Rewritten query: {rewritten!r}")
            # Generate new queries from rewritten question
            new_queries = _generate_search_queries(rewritten, domain_id, max_queries=3)
            new_papers = _multi_query_search(new_queries, max_results=5, domain_id=domain_id)
            
            # Merge with existing, dedupe, re-rank
            all_papers = _dedupe(current_papers + new_papers)
            reranked = _rank_by_relevance(original_question, all_papers, top_k=5)
            if domain_id:
                reranked = _domain_boost(reranked, domain_id)
            
            # Recursive retry
            return _crag_rewrite_and_retry(original_question, reranked, domain_id, attempt + 1, max_attempts)
    except Exception as e:
        _debug(f"[CRAG] Rewrite failed: {type(e).__name__}: {e}")
    
    return current_papers


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
# most ~5 papers; expanding each into 5 chunks puts ~25 evidence units in
# front of the Critic — enough breadth that the intro+methods+results
# sections all have a shot at surviving the per-chunk relevance grading,
# without blowing the generator's prompt budget.
#
# Why 5 not 3: empirically (AutoGen paper, 157K chars), the section that
# *defines the class taxonomy* (§2.1 — ConversableAgent / AssistantAgent /
# UserProxyAgent) didn't rank in the top-3 by cosine to the generic query
# "AutoGen conversational programming" — deeper methodology sections
# scored higher and crowded it out, so the Critic graded all surviving
# slices irrelevant to "what are the two classes of agents". Three slices
# retain both early definitions and one query-matched passage per paper.
_MAX_CHUNKS_PER_PAPER = 3
_PAPER_SCORE_FLOOR = 0.50

# Hard cap on raw extracted PDF text. Some open-access PDFs are 80+ page
# theses; semantically chunking 200K chars per paper is wasted work since
# we only keep top-N anyway. Truncating to the first ~60K chars covers
# title + abstract + intro + most of methods, where research-question
# evidence almost always lives.
_MAX_FULL_TEXT_CHARS = 60_000


async def _fetch_pdf_bytes(client: httpx.AsyncClient, url: str) -> bytes | None:
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
            t = t.replace("\x00", "").replace("\u0000", "")
        except Exception:
            t = ""
        if t.strip():
            pages.append(t)
    return "\n\n".join(pages)


async def _enrich_one_paper(client: httpx.AsyncClient, paper: PaperChunk) -> None:
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
            # Non-fatal: this is an expected degradation path (e.g. the MiniLM
            # encoder can't load under low system memory / a small Windows
            # paging file → OSError 1455). We log a concise one-line warning and
            # fall back to non-semantic chunking — no full traceback, since it
            # adds no actionable detail and just clutters the serve console.
            _debug(
                f"⚠️ semantic_chunk_text failed for {p.citation} — falling back "
                f"to non-semantic chunking: {type(e).__name__}: {e}"
            )
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

        # Pin early slices (intro/abstract + first methods section).
        #
        # Empirical reason for two, not one: in AutoGen (arxiv:2308.08155),
        # the literal text defining the class taxonomy —
        #     "ConversableAgent ... AssistantAgent and UserProxyAgent
        #      are two example built-in agents"
        # — lives in §2.1 "Conversable Agents", which the semantic
        # chunker places at slice INDEX 1 or 2, not slice 0. Pinning
        # only slice 0 (title + abstract framing) means questions like
        # "what are the two broad classes of X" still get answered from
        # web summaries because the definition-bearing slice loses the
        # cosine race to deeper methodology sections.
        #
        # Pinning the first 2 covers both the intro framing AND the
        # first concrete-definitions section, which together handle
        # ~all "what is X / how many X" style questions. The remaining
        # One slot stays query-similarity ranked for tail questions
        # like "how does X handle Y under condition Z".
        #
        # Keep a small per-paper budget so discovery can include more
        # distinct papers without flooding critic/generator context.
        _PIN_EARLY_N = 2
        for early in slices[:_PIN_EARLY_N]:
            if any(s == early for _, s in top_n):
                continue
            if len(top_n) >= _MAX_CHUNKS_PER_PAPER:
                top_n[-1] = (top_n[-1][0], early)  # replace lowest-scored
            else:
                top_n.append((0.0, early))

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
    query: str, *, max_results: int = 5, enrich_full_text: bool = True, domain_id: str | None = None,
    max_search_queries: int = 5, retry_weak: bool = True, auto_ingest: bool = True,
) -> list[PaperChunk]:
    """
    Search arXiv + Semantic Scholar with multi-query strategy, dedupe, rerank by
    query relevance with citation/recency awareness, apply domain boost, then
    enrich the open-access papers with full-text PDF parsing (Phase 15.1).
    Includes CRAG-style query rewriting when initial results are weak.

    Returns up to ~max_results × _MAX_CHUNKS_PER_PAPER PaperChunks when
    enrichment is on (each open-access paper expands into multiple slices).
    Returns up to `max_results` abstract-only chunks when off, or when no
    papers have OA PDFs.

    Set `enrich_full_text=False` for the CLI `discover` command (display-only)
    where the fetch+parse latency isn't worth paying for a list view.
    """
    # Auto-detect domain if not provided
    if domain_id is None:
        from src.domains import infer_domains_from_query
        detected = infer_domains_from_query(query, min_hits=1)
        if detected:
            domain_id = detected[0]
            _debug(f"[discover] Auto-detected domain: {domain_id}")

    # Generate multiple diverse search queries
    queries = _generate_search_queries(query, domain_id, max_queries=max(1, max_search_queries))
    _debug(f"[discover] Generated {len(queries)} search queries: {queries}")

    # Multi-query search across both providers
    ranked = _multi_query_search(queries, max_results=max_results, domain_id=domain_id)

    # CRAG-style query rewriting on weak results
    if retry_weak:
        ranked = _crag_rewrite_and_retry(query, ranked, domain_id, attempt=0, max_attempts=2)

    if not ranked:
        return []

    t0 = time.perf_counter()

    # Keep candidates above a permissive search-score floor; the Critic makes the final evidence decision.
    if ranked:
        kept = [p for p in ranked if p.score >= _PAPER_SCORE_FLOOR]
        if not kept:
            kept = ranked[:1]  # fall back to single best paper
        dropped = len(ranked) - len(kept)
        _debug(
            f"[filter] score_floor={_PAPER_SCORE_FLOOR} "
            f"kept={len(kept)} dropped={dropped} "
            f"(top_score={ranked[0].score:.3f}, "
            f"kept_scores={[round(p.score, 3) for p in kept]})"
        )
        ranked = kept

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

    # Auto-promote discovered papers into the permanent central vector store
    if ranked and auto_ingest:
        _auto_ingest_discovered_chunks(query, ranked)

    dur = time.perf_counter() - t0
    _debug(f"=== discover_papers END total={len(ranked)} chunks in {dur:.1f}s ===")
    return ranked


def _auto_ingest_discovered_chunks(query: str, chunks: list[PaperChunk]) -> None:
    """
    Auto-promote top discovered paper chunks into the permanent vector store (data/store/*.pkl).
    This ensures papers discovered by one user are indexed permanently and
    available for all future queries across all users without re-downloading.
    """
    if not chunks:
        return

    try:
        from datetime import datetime

        from src.llm.provider import embed
        from src.store import get_or_create_papers_collection

        col = get_or_create_papers_collection()

        new_ids: list[str] = []
        new_texts: list[str] = []
        new_metadatas: list[dict] = []

        for c in chunks:
            raw_key = c.arxiv_id or re.sub(r"[^a-zA-Z0-9]+", "_", c.title.lower())[:50]
            chunk_id = f"discovery::{raw_key}::{c.chunk_index}"

            # Skip if chunk is already present in store
            existing = col.get_by_ids([chunk_id])
            if existing and existing.get("ids"):
                continue

            # Truncate to a safe token budget (~500 tokens / 2000 chars) for local embedders (Ollama/NVIDIA)
            body = c.chunk_text or c.abstract or c.title
            text_to_store = f"{c.title}\n\n{body}"[:2000]

            if not text_to_store.strip():
                continue

            meta = {
                "source_file": c.source_file,
                "page_number": c.page_number,
                "chunk_index": c.chunk_index,
                "citation": c.source_file,
                "title": c.title,
                "url": c.url or "",
                "authors": ", ".join(c.authors) if c.authors else "",
                "year": c.year or 0,
                "venue": c.venue or "arXiv",
                "arxiv_id": c.arxiv_id or "",
                "pdf_url": c.pdf_url or "",
                "discovered_via_query": query[:100],
                "ingested_at": datetime.now(UTC).isoformat(),
            }

            new_ids.append(chunk_id)
            new_texts.append(text_to_store)
            new_metadatas.append(meta)

        if new_ids:
            vectors = embed(new_texts)
            col.add(ids=new_ids, embeddings=vectors, documents=new_texts, metadatas=new_metadatas)
            _debug(f"[auto-ingest] Persisted {len(new_ids)} new paper chunk(s) to permanent vector store.")
    except Exception as e:
        _debug(f"[auto-ingest warning] Failed to persist discovered papers: {type(e).__name__}: {e}")

