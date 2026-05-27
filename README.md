# ResearGent

An **Agentic Research Engine** with Corrective RAG, Self-Reflection, hybrid retrieval, web fallback, and end-to-end evaluation.

> Built phase-by-phase — each phase ships a system that works end-to-end before the next layer is added.

---

## Build Roadmap

| Phase | Capability | Status |
|---|---|---|
| **0** | Provider abstraction (5 providers) + 4 tiers + cascade fallback + observability | ✅ done |
| **1** | Naive RAG — PDF ingest, embed, retrieve, generate with citations | ✅ done |
| **2** | Hybrid retrieval — dense + BM25 + reciprocal rank fusion | ✅ done |
| **3** | LangGraph agent — Planner → Retriever → Generator as stateful graph | ⏳ |
| **4** | Corrective RAG — Critic grades chunks, rewrites queries, web fallback | ⏳ |
| **5** | Self-Reflection — Reflector critiques drafts, re-enters the graph | ⏳ |
| **6** | RAGAS evaluation + FastAPI streaming + React frontend | ⏳ |

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

## Architecture (target — final state)

```
                  ┌─────────────┐
   user query ─►  │   Planner   │  decomposes into sub-questions
                  └──────┬──────┘
                         │
                  ┌──────▼──────┐
                  │  Retriever  │  hybrid: dense + BM25 + RRF
                  └──────┬──────┘
                         │
                  ┌──────▼──────┐
                  │   Critic    │  grades chunks for relevance
                  └──┬───────┬──┘
       low confidence│       │ enough evidence
                     │       │
              ┌──────▼──┐    │
              │  Web    │    │
              │ Scraper │    │
              └────┬────┘    │
                   │         │
                   └────┬────┘
                        │
                  ┌─────▼─────┐
                  │ Generator │  drafts answer with citations
                  └─────┬─────┘
                        │
                  ┌─────▼─────┐
                  │ Reflector │  critiques draft, finds gaps
                  └─────┬─────┘
                        │
            (loop back if needed, ≤ N iterations)
                        │
                  ┌─────▼──────┐
                  │   Report   │  final markdown with eval scores
                  └────────────┘
```

---

## Project Layout

```
researgent/
├── src/
│   ├── config.py             # typed settings via pydantic-settings
│   ├── llm/
│   │   └── provider.py       # unified chat() / embed() over NVIDIA/Groq/Ollama
│   ├── ingest/
│   │   ├── pdf.py            # PyMuPDF -> Page records
│   │   ├── chunker.py        # token-aware chunking with overlap
│   │   └── pipeline.py       # PDFs -> chunks -> embeddings -> Chroma
│   ├── retrieval/
│   │   └── naive.py          # dense top-k from Chroma (baseline)
│   ├── rag/
│   │   └── naive.py          # retrieve -> stuff -> generate (cited)
│   ├── store.py              # ChromaDB client + collection management
│   └── main.py               # Typer CLI
├── data/
│   ├── papers/               # drop your PDFs here
│   └── chroma_db/            # vector store (gitignored)
├── .env.example
├── pyproject.toml
└── README.md
```

---

## License

MIT
