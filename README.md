# ResearGent

An **Agentic Research Engine** with Corrective RAG, Self-Reflection, hybrid retrieval, web fallback, and end-to-end evaluation.

> Built phase-by-phase — each phase ships a system that works end-to-end before the next layer is added.

---

## Build Roadmap

| Phase | Capability | Status |
|---|---|---|
| **0** | Provider abstraction (NVIDIA NIM / Groq / Ollama) + CLI skeleton | ✅ in progress |
| **1** | Naive RAG — PDF ingest, embed, retrieve, generate with citations | ⏳ |
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
│   ├── config.py          # typed settings via pydantic-settings
│   ├── llm/
│   │   ├── provider.py    # unified chat() / embed() over NVIDIA/Groq/Ollama
│   │   └── __init__.py
│   └── main.py            # Typer CLI
├── data/
│   └── papers/            # drop your PDFs here (Phase 1+)
├── .env.example
├── pyproject.toml
└── README.md
```

---

## License

MIT
