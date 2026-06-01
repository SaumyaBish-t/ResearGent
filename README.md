# ResearGent

An **Agentic Research Engine** with Corrective RAG, Self-Reflection, hybrid retrieval, semantic chunking with local entity extraction, domain-aware corpora, a Semantic Scholar Stage-1 seeder, an arXiv + S2 Stage-2 deep-dive on low-confidence retrieval, a web fallback cascade, and end-to-end evaluation.

> Built phase-by-phase — each phase ships a system that works end-to-end before the next layer is added.

**Architecture in one line:** five LLM providers behind a tier router → semantic chunker (`all-MiniLM-L6-v2`) + GLiNER entity extraction → domain-tagged hybrid retrieval (dense + BM25 + RRF) → LangGraph DAG with Critic-gated rewriter, **arXiv/Semantic Scholar paper discovery**, web cascade (Tavily → Serper → DDG), Reflector with bounded loop-back → Postgres-backed pointer-state checkpoints (~3 KB/snapshot) — 100% free-tier, CPU-only for everything that isn't an LLM call.

---

## Build Roadmap

| Phase | Capability | Status |
|---|---|---|
| **0** | Provider abstraction (5 providers) + 4 tiers + cascade fallback + observability | ✅ done |
| **1** | Naive RAG — PDF ingest, embed, retrieve, generate with citations | ✅ done |
| **2** | Hybrid retrieval — dense + BM25 + reciprocal rank fusion | ✅ done |
| **3** | LangGraph agent — Planner → Retriever → Generator as stateful graph | ✅ done |
| **4** | Corrective RAG — Critic grades chunks, rewrites, **web cascade** (Tavily → Serper → DDG) | ✅ done |
| **5** | Self-Reflection — Reflector loops back with follow-up sub-questions (bounded, deduped) | ✅ done |
| **6** | RAGAS-style eval + FastAPI streaming + single-file web UI | ✅ done |
| **12** | **PostgreSQL persistence** — shared `psycopg_pool`, `PostgresSaver` checkpoints, `documents_registry` table, raw bytes on disk, TTL pruner, `doc_ids` retriever filter | ✅ done |
| **13** | **Pointer-based state** — `ChunkRef` pointers in checkpoints, hydration at node entry, `agent_artifacts` JSONB store → ~3KB/snapshot instead of ~200KB (60× smaller) | ✅ done |
| **14** | **Semantic chunking + local NER** — `all-MiniLM-L6-v2` topic-shift chunker, GLiNER entity extraction, entity-augmented embeds & BM25 (free-tier, CPU-only) | ✅ done |
| **15** | **Domain-aware corpus + S2 seed ingestion** — three isolated per-domain ingest roots, domain-stamped Chroma metadata + registry, keyword auto-router, Semantic Scholar `citationCount:desc` Stage-1 seeder | ✅ done |
| **15.1** | **Async PDF enrichment for Stage-2** — `httpx.AsyncClient` + `pypdf` fetch/parse for open-access papers, ranked semantic slices replace abstract-only stubs, graceful fallback on 404/captcha/corrupted bytes | ✅ done |
| **15.2** | **Critic strictness & telemetry** — weighted score (`partial=0.75`), source-aware HIGH threshold (`0.70` external / `0.85` local), per-call payload bounds + `prompt_chars`/`est_tokens` trace, year-discipline rule | ✅ done |

---

## Phase 0 — Setup

### 1. Prerequisites

- **Python 3.11 or 3.12** (3.13 may have compatibility issues with later-phase libs)
- **[uv](https://docs.astral.sh/uv/)** for dependency management
- At least one of these providers (all OpenAI-compatible, all have free tiers):

  | Provider | Free tier | Best for | Get a key |
  |---|---|---|---|
  | **Cerebras** | 1M tokens/day, 30 RPM | **Fastest** inference (~1000 TPS) — agent loops | [cloud.cerebras.ai](https://cloud.cerebras.ai) |
  | **NVIDIA NIM** | Generous credits | 80+ models incl. MiniMax M2.7, Kimi, DeepSeek; **only top-tier embedder** | [build.nvidia.com](https://build.nvidia.com) |
  | **Groq** | High RPM | 315 TPS Llama 70B — fast tier (Critic) | [console.groq.com/keys](https://console.groq.com/keys) |
  | **OpenRouter** | 50 req/day (1000 with $10 deposit) | 200+ models via one endpoint (DeepSeek R1, Step 3.5, Gemini 2.5) | [openrouter.ai/keys](https://openrouter.ai/keys) |
  | **Ollama** | Unlimited (local) | Privacy + offline | [ollama.com](https://ollama.com) |

### 2. Install

```bash
git clone https://github.com/SaumyaBish-t/ResearGent.git
cd ResearGent

# uv handles venv + deps in one go
uv sync
```

### 3. Configure providers

```bash
cp .env.example .env
# Edit .env — paste at least one API key, or leave Ollama defaults if running locally
```

If using **Ollama**, pull the default models first:

```bash
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

### 4. Verify

```bash
# Show providers, tier routing, and cascade fallback chains
uv run researgent status

# Send a real prompt to each chat tier (reasoning / fast / tool)
uv run researgent smoke

# One-off question — no retrieval yet, just direct LLM call
uv run researgent ask "What is corrective RAG in one paragraph?"

# View per-call observability log (latency, tokens, cascade usage)
uv run researgent stats
```

### Model tiers (the heart of the model strategy)

| Tier | Used by | Default model picks (when keys are available) |
|---|---|---|
| **REASONING** | Planner, Reflector, Report Generator | Cerebras Qwen3-235B → NVIDIA Llama 3.3 70B → ... |
| **FAST** | Critic, Grader, Query rewriter | Groq llama-3.1-8b-instant (315 TPS) → ... |
| **TOOL** | Any agent doing tool / function calls | Groq GPT-OSS-120B (best free tool-caller) → ... |
| **EMBED** | Ingestion + retrieval | NVIDIA nv-embed-v1 → ... |

Each tier resolves to a **cascade chain** — if the primary provider returns a transient error (429 rate limit, 5xx, timeout), the system automatically retries on the next configured provider. View the full chain with `researgent status`.

---

## Phase 1 — Naive RAG usage

```powershell
# 1. Drop one or more PDFs into data/papers/
#    (Try a couple of arXiv papers to test.)

# 2. Ingest — parses pages, chunks at ~500 tokens with 80-token overlap,
#    embeds via the active EMBED tier, stores in ChromaDB
uv run researgent ingest

# 3. Inspect what's in the store
uv run researgent store info

# 4. Retrieve raw chunks (no LLM call — useful for debugging retrieval quality)
uv run researgent retrieve "what is corrective RAG?" --k 5

# 5. Ask a question — top-k retrieval + cited generation
uv run researgent rag-ask "How does CRAG decide when to call web search?"

# Drop the index and start over (e.g. when switching embedding models)
uv run researgent store reset
```

**Pipeline:** `PDF → PyMuPDF parse → token-aware chunker → embed (tier=EMBED) → ChromaDB persistent → cosine top-k → LLM with [S1]..[Sk] citations`.

One collection exists per `(embed-provider, embed-model)` combination, so switching providers in `.env` creates a fresh collection rather than mixing incompatible embedding dimensions.

---

## Phase 2 — Hybrid retrieval

```powershell
# Hybrid is now the default for both retrieve and rag-ask
uv run researgent retrieve "What is RAFT fine-tuning?" --k 5
uv run researgent rag-ask "What is RAFT fine-tuning?"

# Force a single strategy
uv run researgent retrieve "FlashAttention-2" --mode bm25      # lexical-only
uv run researgent retrieve "How does CRAG decide?" --mode naive  # dense-only

# Side-by-side benchmark — shows which chunks each strategy surfaces uniquely
uv run researgent bench "What's the formula for RRF in the Cormack paper?"
```

**How it works:** ingest now builds two parallel indexes — Chroma (dense embeddings) and a persisted BM25Okapi pickle. At query time:

1. Both indexes return their top-N (default 4×k).
2. Reciprocal Rank Fusion combines them: `score(d) = Σ 1/(60 + rank_i(d))`.
3. Top-k by RRF score is returned. Each chunk records which retriever(s) ranked it ("BOTH" / "dense" / "bm25"), making retrieval debuggable.

**Why both:** dense alone misses exact terms (acronyms, product names, code identifiers); BM25 alone misses paraphrases. RRF combines them parameter-free.

---

## Phase 3-5 — Agent (Plan → Retrieve → Critique → Generate → Reflect)

```powershell
# Full agentic research — decomposes complex questions, hybrid-retrieves
# per sub-question, grades chunks, rewrites/web-fallbacks on low confidence,
# generates structured cited answer, reflects on the draft and (optionally)
# loops back with follow-up sub-questions.
uv run researgent research "Compare CRAG and Self-RAG"

# Adjust the chunk budget shared across sub-questions
uv run researgent research "<question>" --k 12

# Replay a past run by checkpoint id
uv run researgent research "<same question>" --run-id <previous-run-id>
```

The CRAG status line above each answer shows what the agent did:
`_CRAG: conf=high  rewrites=1  web_fallback=YES  reflections=1_`

---

## Phase 6 — Evaluation, API, Web UI

```powershell
# Run a YAML test suite, compute faithfulness / relevancy / context-precision
uv run researgent eval eval_suites/sample.yaml

# Launch the FastAPI server + live web UI
uv run researgent serve
# -> open http://localhost:8000
```

The UI streams every agent node live via SSE — you see the planner decompose,
the critic grade, the rewriter retry, the web fallback fire, the generator
produce text, the reflector audit, all as it happens. The final answer
appears with clickable `[Sn]` citations mapped to local PDFs **or** web URLs.

Eval suite YAML format:
```yaml
name: my-suite
queries:
  - id: q1
    question: "What is X?"
    tags: [definition]
```
Results persist to `data/eval/runs.jsonl`, one flat row per query, ready
for jq/pandas analysis.

---

## Phase 12 — PostgreSQL persistence layer

Moved checkpointing and document tracking off SQLite/MemorySaver onto Postgres so runs survive restarts, can be inspected by other tools, and scale beyond a single process.

```powershell
# Set DATABASE_URL in .env (any Postgres 14+; Neon/Supabase free tiers work)
uv run researgent db init       # create tables (idempotent)
uv run researgent db status     # show row counts + recent checkpoints
uv run researgent db prune      # drop checkpoints + artifacts older than the TTL
```

- **Checkpoints**: `PostgresSaver` from `langgraph-checkpoint-postgres`, fed by a shared `psycopg_pool.ConnectionPool` configured with the three settings the saver silently requires (`autocommit=True`, `row_factory=dict_row`, `prepare_threshold=0`). Falls back to `MemorySaver` when `DATABASE_URL` is empty — local-dev still works with zero setup.
- **`documents_registry` (SQLAlchemy)**: one row per ingested PDF/note — `doc_id` (UUID, also stamped into every Chroma chunk's metadata), `content_hash`, filename, title, source type, storage URL, file size, chunk count.
- **Raw bytes**: copied to `data/storage/<content_hash>.<ext>` on ingest; column type is a `file://` URL today, swap for S3 by editing `_persist_raw()` only.
- **TTL pruner**: synthesizes a UUIDv6 cutoff to bulk-delete old checkpoints (UUIDv6 sorts by timestamp), and clears matching `agent_artifacts` rows in lockstep.
- **`Retriever.doc_ids`** filter wired through Chroma's `where={"$in": [...]}` — scope a query to a specific subset of registered documents.

---

## Phase 13 — Pointer-based state management

Even with a 4 KB cap on `chunks_by_subq`, ~10 checkpoints/run × ~200 KB/checkpoint filled 500 MB in ~250 runs. Phase 13 keeps only pointers in checkpoint state.

```
Before:    state.chunks_by_subq = { sq1: [Chunk{text=…, 1.5KB}, …] }       ← in checkpoint
After:     state.refs_by_subq  = { sq1: [ChunkRef{store, id, ~80B}, …] }   ← in checkpoint
           text lives in Chroma (local) or agent_artifacts (web/paper/graph)
```

- **`src/agent/artifacts.py`** — `ChunkRef` pointers, `HydratedChunk` unified view, `agent_artifacts` JSONB table.
- Every graph node was refactored to: read refs → hydrate at entry → operate → return refs at exit.
- **Result**: per-snapshot size dropped from ~200 KB to ~3 KB → the same 500 MB now buys ~15,000 runs instead of ~250 (**~60× improvement**).
- The Phase 12 TTL pruner was extended to clear `agent_artifacts` in lockstep with checkpoints.

---

## Phase 14 — Semantic chunking + local entity extraction

Naive token-bucket chunking ignored *meaning* — a chunk frequently straddled a topical boundary because the boundary fell mid-budget. Phase 14 replaces it with a fully-local, free-tier-only semantic pipeline.

### What changed

1. **Semantic chunker** (`src/ingest/chunker.py`)
   - Sentences → embeddings via `sentence-transformers/all-MiniLM-L6-v2` (22M params, ~50 ms / page on CPU).
   - Cosine *distance* between every adjacent sentence pair.
   - Percentile-based threshold (default 90th) on those distances → adaptive per-page topic-shift detector.
   - Greedy pack with `target_tokens=500`, `max_tokens=800`, `min_chunk_tokens` guard against 1-sentence slivers. Hard-split as last resort.
   - **No more fixed overlap** — topical boundaries replace it.

2. **Local entity extraction** (GLiNER)
   - `urchade/gliner_small-v2.1` (166M, ~150 MB), CPU-friendly zero-shot NER.
   - Labels: `["Algorithm", "Framework", "Scientific Concept", "Organization", "Person", "Metric", "Dataset"]`.
   - Threshold 0.5, case-insensitive de-dup, capped at 25 entities/chunk, fails-soft to `[]` on any error.
   - Both models are `@lru_cache(maxsize=1)` singletons → load cost paid once per process.

3. **Metadata-enriched RAG** (`src/ingest/pipeline.py`)
   - Each chunk's extracted entities are appended to the chunk text **before** embedding:
     ```
     [chunk body…]

     [Extracted Entities: Corrective RAG, Reciprocal Rank Fusion, LangGraph]
     ```
   - Net effect: **both** the dense vector and the BM25 token stream pick up the technical terms — GraphRAG-style recall without a graph DB.
   - Same enriched text is stored as the Chroma document → BM25 (which rebuilds from Chroma docs) tokenizes the entity line automatically.
   - Entities also stored in Chroma metadata as a comma-joined string (Chroma metadata is scalar-only; matches the existing `tags`/`wikilinks` convention). Greppable via `where_document={"$contains": "<entity>"}`.

4. **Vault parity** (`src/ingest/obsidian.py`)
   - `VaultChunk.entities` field added; `chunk_note()` runs GLiNER over each emitted chunk so Obsidian ingest gets the same enrichment.

### Strict free-tier guarantee

Everything in the chunking + extraction path runs **locally on CPU** — no LLM API calls, no managed NER service, no graph database. First import downloads ~230 MB of model checkpoints to the HuggingFace cache; later runs start in ~1-2 s.

### New dependencies

```toml
"sentence-transformers>=3.0.0"
"gliner>=0.2.13"
```

Run `uv sync` after pulling.

---

## Phase 15 — Domain-aware corpus + Semantic Scholar seed ingestion

The persona contract pins ResearGent to three explicit corpora — **Agentic AI**, **Quantitative Finance**, **ML for Time-Series Forecasting** — and a Stage-1 / Stage-2 retrieval protocol where Stage 1 hits historically foundational, citation-weighted bedrock. Phase 15 makes that contract real.

### 1. Domain registry — `src/domains.py`

Single source of truth for the three corpora. Each `Domain` carries:

- `id` — short slug used as the subfolder name AND the Chroma `domain` metadata value (`agentic_ai`, `quant_finance`, `time_series`).
- `ingest_dir` — `data/papers/<id>/` — where PDFs for that domain live.
- `seed_queries` — broad-coverage search strings the S2 seeder runs sorted by `citationCount:desc`.
- `routing_keywords` — tokens the auto-router uses to map user questions to the right domain bucket.

Add a fourth domain by editing one dict, nothing else.

### 2. Domain-tagged ingest

The pipeline now stamps `domain` onto every chunk in **two** places, in lockstep:

- **Chroma metadata** — `{"domain": "<id>"}` per chunk, so retrieval filters via `where={"domain": {"$in": [...]}}`.
- **Postgres `documents_registry.extra->>'domain'`** — so SQL-level queries (`"how many quant_finance docs ingested last week?"`) work without scanning Chroma.

Three ways to set it:

```powershell
# 1. Auto-detect from path — drop PDFs under data/papers/<domain>/ and just run:
uv run researgent ingest

# 2. Explicit override — tag PDFs that live outside the standard tree:
uv run researgent ingest /some/other/path --domain agentic_ai

# 3. All-in-one — walks every data/papers/<domain>/ subdir, ingests with domain tag,
#    ONE embedder warm-up + ONE BM25 rebuild across the whole corpus.
uv run researgent ingest-domains
uv run researgent ingest-domains --only agentic_ai,time_series
```

### 3. Domain-scoped retrieval

`hybrid_retrieve(query, domains=[...])` plumbs the filter through:

- **Dense (Chroma)** — combined with existing `doc_ids` filter via `$and`.
- **BM25** — post-filter on `metadata["domain"]` (same approach as `doc_ids`).
- **RRF fusion** — operates on the already-filtered pools so cross-domain noise can't leak into the top-k.

Two ways to set the scope on a query:

```powershell
# Explicit (skips the auto-router):
uv run researgent research "PatchTST vs Informer for long-horizon forecasting" --domain time_series
uv run researgent research "..." --domain agentic_ai,quant_finance

# Implicit — the planner's keyword auto-router fires when --domain is omitted.
# Substring matches against each domain's routing_keywords; sets domain_scope
# only when the signal is strong. Ambiguous queries fall back to "search every domain".
uv run researgent research "How does LangGraph route between planner and critic?"
# -> auto-routes to agentic_ai (no LLM call — deterministic, free)
```

**Why a keyword router and not an LLM classifier:** free-tier budget. A FAST-tier classification on every query would consume ~50% of a typical Cerebras free quota for zero recall gain on the common case. Substring matching hits ~80% of queries deterministically; the rest fall through to searching everywhere (which is correct behaviour for ambiguous questions).

### 4. Semantic Scholar seed ingestion — `src/ingest/s2_seed.py`

The Stage-1 seeder. For every registered domain, runs that domain's `seed_queries` against the public S2 search endpoint with `sort=citationCount:desc`, dedupes hits by arXiv ID / DOI / normalised title, and pulls them into the corpus:

```powershell
uv run researgent seed                                    # seed every domain
uv run researgent seed --only agentic_ai                  # one domain
uv run researgent seed --top-n 10                         # 10 papers / seed query
uv run researgent seed --abstracts-only                   # skip PDF downloads
```

What lands on disk:

- **Open-access PDFs** → `data/papers/<domain>/arxiv_<id>.pdf`, then run through the standard semantic chunker + GLiNER + entity-enriched embed path. Tagged `domain=<id>` automatically.
- **No open-access PDF** → `data/papers/<domain>/_abstracts/<slug>.md` — title + abstract + citation count + arXiv/DOI/S2 URL footer. Ingest these into the same domain bucket with:
  ```powershell
  uv run researgent vault-ingest data/papers/agentic_ai/_abstracts
  ```

Free-tier guarantees:

- **No S2 API key required** — works against the public unauthenticated endpoint out of the box, with a 3.0s courtesy gap between calls (empirically clears the 429s observed on free-tier bursts of ~18 queries).
- **Optional `SEMANTIC_SCHOLAR_API_KEY`** — email `s2-api@allenai.org` for a personal key. When set in `.env`, every S2 call sends it as the `x-api-key` header (per the [official tutorial](https://www.semanticscholar.org/product/api/tutorial)) and the courtesy gap drops to 1.1s — just under their documented 1 RPS cumulative ceiling, ~3× faster than the public path. Both the seeder AND the Stage-2 paper-discovery fallback pick the key up automatically.
- **Idempotent** — re-seeding the same paper re-uses the existing content hash, replacing chunks rather than duplicating.

### 5. Stage-1 / Stage-2 protocol — already wired

The persona spec calls for Stage-2 (S2 deep-dive) on a Critic Low-Confidence verdict. This was already shipped as part of Phase 7 — confirmed during this session, no rewire needed:

```
critic (medium/low + retries exhausted) ──► paper_discovery ──► critic (re-grade) ──► generator
                                                  │
                                       arXiv + Semantic Scholar
                                       (live, query-dependent)
```

Phase 15's domain tagging strengthens Stage 1 (the seeded local corpus is now bedrock-weighted by citation count and domain-bucketed for clean retrieval), Phase 7's paper_discovery node handles Stage 2 (just-in-time live S2 queries when local retrieval underperforms). The Critic's `low/medium + budget exhausted` decision is the single trigger.

### 6. CLI surface — Phase 15 commands

```powershell
uv run researgent domains              # show the three registered domains
uv run researgent ingest-domains       # ingest every data/papers/<domain>/ subdir
uv run researgent seed                 # seed all domains from S2 (citationCount:desc)
uv run researgent research "..." --domain quant_finance    # scope retrieval explicitly
```

---

## Phase 15.1 — Async PDF enrichment for Stage-2 paper discovery

Before 15.1, `paper_discovery` fed only title+abstract into the Critic and Generator — 5 abstract stubs total per cascade fire. For research questions that need specific evidence (formulae, metrics, dataset names, year-anchored claims), abstracts are too thin and the generator paraphrases around the gaps.

### What changed (`src/retrieval/papers.py`)

1. **Async fetch.** `httpx.AsyncClient(timeout=15.0)` + ResearGent User-Agent header. `asyncio.Semaphore(4)` bounds parallel downloads so journal mirrors aren't hammered. `follow_redirects=True` for DOI → publisher → CDN chains.
2. **pypdf parse.** `pypdf.PdfReader(io.BytesIO(...))`, iterate `reader.pages`, concat extracted text. Per-page extract is itself wrapped so one bad page doesn't lose the document. 60 KB hard cap on raw text per paper.
3. **Semantic chunking.** Parsed full text runs through the Phase 14 `semantic_chunk_text` (`all-MiniLM-L6-v2`). Slices ranked by cosine similarity to the original query using the EMBED tier; top **3** per paper kept.
4. **Per-slice PaperChunks.** Each kept slice becomes its own `PaperChunk` with `chunk_idx` set; all citation metadata (title, year, url, arxiv_id) preserved so generator citations remain coherent across slices of the same paper.
5. **Graceful fallback contract.** Three layered guards — HTTP non-200, content-type/magic-number check rejecting HTML paywalls, `try/except` around the entire fetch + parse path. Any failure logs one concise warning and falls back to abstract via the `PaperChunk.text` priority chain (`chunk_text > full_text > abstract > title`). The cascade never breaks on a flaky mirror.

### Verified end-to-end

Real arXiv PDF (Self-RAG, 2310.11511): fetch+parse 60 000 chars → semantic chunker emits 3 slices → query-relevance scores `[0.65, 0.63, 0.62]` → generator sees bounded 1–3 KB passages from the actual paper body.

Failure modes (404, HTML masquerade, DNS-fail): all three cleanly fell back to abstract-only with one warning each. No exception escaped.

### New dependency

```toml
"pypdf>=5.0"
```

---

## Phase 15.2 — Critic strictness, leniency for fresh sources, and load bounds

Three rounds of Critic improvements landed in quick succession, all in `src/agent/nodes/critic.py`. Net effect: better verdict quality, better cost control, and the cascade's fresh discoveries actually count toward the auto-save gate.

### 1. Strict year-aware grading (commit `e8a8969`)

The Critic gates the entire Stage-2 escalation cascade. Over-grading thin local evidence as HIGH silently degrades answers because the cascade never fires. Tightened rubric:

- **Recency discipline.** If the question asks about a specific year (`"latest 2026 developments"`), the chunk must be from that exact year for `relevant`. A 2024 paper on the same topic is `partial` at best. A 2021 paper is `irrelevant`.
- **Domain-match-is-not-relevance.** Same domain bucket (`agentic_ai`, `quant_finance`, `time_series`) is the floor, not the ceiling. A LangGraph chunk doesn't answer a ReAct question just because both are agentic-AI.
- **`doc_title` surfaced in chunk headers** so the model can actually see the year (`PaperChunk.doc_title` carries `"Title (YEAR) — Venue"`).

### 2. Source-aware leniency for cascade discoveries (commit `50ecfe0`)

Replaced absolute-count bands with a weighted score and a source-aware threshold:

```
score = (relevant_n × 1.00 + partial_n × 0.75) / total

threshold_hi = 0.70  if pool has external-fresh chunks (signal startswith "paper:" or "web:")
threshold_hi = 0.85  otherwise (all-local pool)

score ≥ threshold_hi              → high
relevant_n ≥ 1 OR partial_n ≥ 2   → medium
otherwise                         → low
```

Cascade-discovered chunks (arxiv / semantic_scholar / tavily / serper / duckduckgo) get the lenient 0.70 bar — they already paid a live API call to surface, so a single Tavily hit or a Stage-2 S2 paper full of partials can settle the verdict at HIGH and auto-save to your vault. Local-only pools keep the strict 0.85 bar.

The Critic prompt also carries a **CRITERIA AMENDMENT** instructing the model to be constructive on fresh discoveries: "if a chunk contains the explicit core entities, numbers, or agent names requested by the user — even if the surrounding text or snippet formatting is partial or noisy — grade it as `relevant` or a high-value `partial`. Do not penalise fresh, correct information for layout or formatting fragments."

### 3. Per-call payload bounds + token telemetry (commit `b60e75b`)

Phase 15.1's PDF enrichment ballooned Critic input ~9× (5 abstracts → up to 15 PDF slices), and the stricter year-aware rules ~2-3×'d the call count. Combined → daily Groq TPD exhausted by run 4–5.

Bounds added:

- Per-chunk char truncation **1500 → 800**. Year, named entities, and on-topic signal all sit in the first ~500 chars; the rest was filler.
- **`MAX_CHUNKS_PER_CALL = 12`**. paper_discovery floods get sliced (excess defaults to `partial` so downstream filter still emits a grade per input chunk).
- Critic trace now surfaces `score`, `threshold_hi`, `external_fresh_pool`, `prompt_chars`, `est_tokens` — every verdict is fully explainable from the CLI trace.

Combined: ~85% reduction in per-Critic-call payload. The same run that ate ~100K tokens at Groq now uses ~15K.

### Resilient FAST cascade

Companion `.env` change (not committed — `.env` is gitignored):

```
FAST_CASCADE=cerebras,groq,openrouter,nvidia,ollama
```

Five fallback rungs. When Cerebras (5 RPM) and Groq (100K TPD) both exhaust, the Critic seamlessly hops to OpenRouter (`deepseek-v4-flash:free`), then NVIDIA (`Llama-3.1-8B`), then offline Ollama. No more terminal 429s.

---

## Architecture (current shipped state — Phases 0–15.2)

```
                       ┌──────────────────┐
        user query ──► │     Planner      │  decompose into sub-questions
                       │  + auto-router   │  + keyword-infer domain_scope
                       └────────┬─────────┘     (free, deterministic)
                                │
                       ┌────────▼─────────┐
                       │    Retriever     │  Stage 1: hybrid (dense + BM25 + RRF)
                       │  domain-filtered │  scoped by domain_scope + doc_id_scope
                       │  + KG expansion  │  + 1-hop wikilink expansion (vault)
                       └────────┬─────────┘
                                │
                       ┌────────▼─────────┐
                       │      Critic      │  grades each sub-q's chunks {high|med|low}
                       └─┬──┬────┬────┬───┘
              high       │  │    │    │     budget exhausted + low/medium
           (sufficient)  │  │    │    │
                         │  │    │    └────────┐
                         │  │    │             │
                  ┌──────┘  │    │       ┌─────▼───────────┐
                  │         │    │       │ paper_discovery │  Stage 2: arXiv + S2 live
                  │         │    │       │  (Critic-gated) │  query rewritten to
                  │         │    │       └─────┬───────────┘  keywords; ranked by
                  │         │    │             │              relevance + citations
                  │         │    │             └────► Critic re-grade (loop ≤1)
                  │         │    │
                  │   medium/low │   papers tried but still weak
                  │   + budget   │
                  │   left       │   ┌─────────────────┐
                  │         │    └──►│  web_fallback   │  Tavily → Serper → DDG
                  │         │        │   (cascade)     │  cascade on transient errors
                  │         │        └────────┬────────┘
                  │         │                 │
                  │         │                 └────► Critic re-grade (loop ≤1)
                  │         │
                  │  ┌──────▼─────┐
                  │  │  rewriter  │  Critic-driven sub-q rephrase + retry
                  │  └──────┬─────┘  (≤ crag_max_rewrites)
                  │         │
                  │         └────► Critic re-grade
                  │
                  ▼
           ┌────────────┐                  no usable evidence at all
           │ Generator  │ ◄──────────┐     ┌─────────────────┐
           │   cited    │            └─────┤  llm_reasoning  │  LAST-RESORT priors
           └─────┬──────┘                  │  ("no sources") │  with loud disclaimer
                 │                         └─────────────────┘
                 ▼
           ┌────────────┐  audits draft for gaps; if any AND
           │ Reflector  │  budget left → appends follow-up
           └─────┬──────┘  sub-questions, loops back to retriever
                 │
       gaps + budget left  │  no gaps OR budget done
                 │         │
                 │         └──► END (+ optional auto-save to notes folder)
                 │
                 └──► retriever  (Phase 5 reflection loop, ≤ N iters)


  ──────  state lives as ChunkRef pointers in Postgres checkpoints (~3 KB/snapshot, Phase 13)
  ──────  chunk text + entity manifests live in Chroma (Phase 14) / agent_artifacts JSONB
  ──────  per-domain corpora seeded from S2 citationCount:desc (Phase 15)
```

Key flow notes:

- **Stage 1** is the seeded local corpus (Phase 15 S2 seeder → domain-bucketed Chroma).
- **Stage 2** is `paper_discovery` — fires when Critic is unhappy AND rewriter budget is exhausted. arXiv + S2 live, abstract-only, ranked by query relevance, fed BACK through the Critic for re-grading rather than straight to the generator.
- **Domain scope** is set either by `--domain` (explicit) or by the planner's keyword auto-router (implicit). The Retriever passes it into every hybrid call as a Chroma `where` filter + BM25 post-filter.
- **Web fallback** runs only AFTER paper_discovery has been tried — peer-reviewed abstracts beat web snippets for research questions.

---

## Project Layout

```
researgent/
├── src/
│   ├── config.py             # typed settings via pydantic-settings (+ FAST/REASONING/TOOL_CASCADE overrides)
│   ├── domains.py            # Phase 15: registered corpora + consolidated data/{domain}/{papers,abstract_notes,research_data}/ layout
│   ├── llm/
│   │   └── provider.py       # unified chat() / embed() over NVIDIA/Groq/Ollama/Cerebras/OpenRouter
│   ├── ingest/
│   │   ├── pdf.py            # PyMuPDF -> Page records
│   │   ├── chunker.py        # Phase 14: semantic chunker (MiniLM) + GLiNER entity extraction
│   │   ├── obsidian.py       # vault parser + heading-aware chunker (entity-enriched)
│   │   ├── s2_seed.py        # Phase 15: Semantic Scholar Stage-1 seeder (citationCount:desc, x-api-key authed)
│   │   └── pipeline.py       # chunks -> entity-augmented embeds -> Chroma + BM25 + registry (domain-tagged on both PDF + vault paths)
│   ├── retrieval/
│   │   ├── naive.py          # dense top-k from Chroma (baseline) + domain filter
│   │   ├── bm25.py           # persisted BM25Okapi + RRF fusion + domain post-filter
│   │   └── papers.py         # Phase 15.1: async httpx + pypdf fetch/parse for OA papers -> ranked semantic slices
│   ├── rag/
│   │   └── naive.py          # retrieve -> stuff -> generate (cited)
│   ├── agent/
│   │   ├── graph.py          # LangGraph DAG (planner/retriever/critic/paper_discovery/web/gen/reflector); empty-retrieval escalation
│   │   ├── nodes/
│   │   │   └── critic.py     # Phase 15.2: weighted score, source-aware HIGH threshold, year-discipline prompt, payload bounds
│   │   └── artifacts.py      # Phase 13: ChunkRef pointers + agent_artifacts JSONB store
│   ├── registry.py           # Phase 12: documents_registry (SQLAlchemy) + TTL pruner
│   ├── store.py              # ChromaDB client + collection management
│   └── main.py               # Typer CLI (domains, ingest-domains, seed, retag-domain, research --domain)
├── data/
│   ├── agentic_ai/           # Phase 15 (consolidated): one folder per domain
│   │   ├── papers/           #   PDF ingest input
│   │   ├── abstract_notes/   #   S2 abstract-only cards (formerly _abstracts/)
│   │   └── research_data/    #   auto-saved research run notes (YYYY-MM-DD/)
│   ├── quant_finance/
│   ├── time_series/
│   ├── storage/              # Phase 12: raw bytes by content_hash (gitignored)
│   └── chroma_db/            # vector store (gitignored)
├── .env.example
├── pyproject.toml
└── README.md
```

---

## License

MIT
