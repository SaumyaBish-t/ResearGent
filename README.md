# ResearGent

An **Agentic Research Engine** with Corrective RAG, Self-Reflection, hybrid retrieval, web fallback, and end-to-end evaluation.

> Built phase-by-phase — each phase ships a system that works end-to-end before the next layer is added.

---

## Build Roadmap

| Phase | Capability | Status |
|---|---|---|
| **0** | Provider abstraction (NVIDIA NIM / Groq / Ollama) + CLI skeleton | ✅ done |
| **1** | Naive RAG — PDF ingest, embed, retrieve, generate with citations | ✅ done |
| **2** | Hybrid retrieval — dense + BM25 + reciprocal rank fusion | ⏳ |
| **3** | LangGraph agent — Planner → Retriever → Generator as stateful graph | ⏳ |
| **4** | Corrective RAG — Critic grades chunks, rewrites queries, web fallback | ⏳ |
| **5** | Self-Reflection — Reflector critiques drafts, re-enters the graph | ⏳ |
| **6** | RAGAS evaluation + FastAPI streaming + React frontend | ⏳ |

---

## Phase 0 — Setup

### 1. Prerequisites

- **Python 3.11 or 3.12** (3.13 may have compatibility issues with later-phase libs)
- **[uv](https://docs.astral.sh/uv/)** for dependency management
- At least one of:
  - [NVIDIA NIM API key](https://build.nvidia.com) (free credits)
  - [Groq API key](https://console.groq.com/keys) (free tier)
  - [Ollama](https://ollama.com) installed locally (free, needs a GPU for usable speeds)

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
# Show which providers are configured and how tiers are routed
uv run researgent status

# Send a real prompt to each tier — confirms wiring + credentials
uv run researgent smoke

# One-off question (Phase 0 has no retrieval — just direct LLM call)
uv run researgent ask "What is corrective RAG in one paragraph?"
```

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
