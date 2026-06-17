# 🌌 ResearGent

**A hallucination-resistant, multi-agent research engine with a live 3D dashboard.**

[![Live](https://img.shields.io/badge/Live-resear--gent.vercel.app-22d3ee)](https://resear-gent.vercel.app)
[![API](https://img.shields.io/badge/API-researgent.onrender.com-34d399)](https://researgent.onrender.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Backend: LangGraph](https://img.shields.io/badge/Backend-LangGraph-orange)](https://langchain-ai.github.io/langgraph/)
[![Frontend: Next.js + R3F](https://img.shields.io/badge/Frontend-Next.js%20%2B%20R3F-black)](https://docs.pmnd.rs/react-three-fiber)
[![Bring Your Own Model](https://img.shields.io/badge/LLM-Bring%20Your%20Own-success)](#-bring-your-own-model-byom)

> 🌐 **Live demo:** **<https://resear-gent.vercel.app>**
> Sign in with Google → 3 free researches / month + 3 follow-ups per thread.
> Lifetime unlock for ₹499 (one-time) via Razorpay. Best in Chrome / Firefox
> (Brave users: disable Shields for the site so the cross-site session cookie
> can travel between Vercel ↔ Render).

ResearGent answers research questions by **grounding every claim in evidence it can cite** — and refusing to bluff when it can't. Instead of trusting a single vector-search pass, it runs an adversarial **Corrective-RAG + self-reflection** loop: a Critic grades the retrieved context, a Rewriter retries weak queries, and when local knowledge runs out the agent **cascades to academic APIs (arXiv / Semantic Scholar) and live web search** before writing a cited answer. High-confidence results are auto-saved to a local Markdown knowledge base that grows over time.

The whole agent graph streams to a **Next.js + React Three Fiber dashboard** that visualizes the pipeline in real time over Server-Sent Events.

> <!-- Add a screenshot or GIF of the 3D dashboard here, e.g. docs/demo.gif -->
> *(Run `researgent serve` + the frontend to see the live 3D agent network — or skip setup and try it at the live link above.)*

---

## ⚡ Why ResearGent

- **It proves before it answers.** A strict Critic node grades retrieved chunks (relevant / partial / irrelevant) and assigns a confidence score. Weak evidence triggers correction, not confident-sounding fabrication.
- **It cascades instead of giving up.** Local retrieval → query rewriting → academic paper discovery (arXiv + Semantic Scholar, full-text PDF parsing) → live web fallback → last-resort priors. Each stage only fires when the previous one falls short.
- **It self-reflects.** After drafting, a Reflector node looks for gaps and can loop back for another targeted retrieval pass within a bounded budget.
- **It only saves what it trusts.** Answers that clear a configurable confidence gate are written back to your notes folder as cited Markdown. Pure-guess answers (no sources) are never saved.
- **It's model-agnostic.** Every LLM call is routed by *capability tier*, so you bring your own provider and decide which model does the heavy thinking vs. the fast inner-loop work. See [BYOM](#-bring-your-own-model-byom).
- **It's observable.** A cinematic 3D dashboard shows each node activating, data flowing along edges, and the final cited answer — all live.

---

## 🏗️ Architecture

A browser dashboard talks to a FastAPI backend over Server-Sent Events. The backend runs the LangGraph agent, which routes every LLM call through a capability-tier router (your models) and pulls evidence from local retrieval, academic APIs, and the web — saving trusted answers back to your knowledge base.

```mermaid
flowchart TB
    User(["👤 User"])

    subgraph FE["Frontend · Next.js + React Three Fiber"]
        DASH["Live 3D agent dashboard"]
    end

    subgraph BE["Backend · FastAPI + LangGraph"]
        SSE["SSE endpoint · /api/research"]
        AGENT["Agent graph<br/>Planner → Critic → … → Vault"]
    end

    subgraph ROUTER["LLM tier router · Bring Your Own Model"]
        TIERS["Reasoning · Fast · Tool · Embed<br/>auto-route + cascade fallback"]
    end

    PROVIDERS["OpenAI-compatible providers<br/>NVIDIA · Groq · Cerebras · OpenRouter · Ollama"]

    subgraph KB["Knowledge & retrieval"]
        STORE[("Vector store + BM25")]
        PAPERS["arXiv + Semantic Scholar"]
        WEB["Tavily → Serper → DuckDuckGo"]
        VAULT[("Markdown knowledge base")]
    end

    User -->|asks a question| DASH
    DASH <-->|"live node events (SSE)"| SSE
    SSE --> AGENT
    AGENT -->|every LLM call| TIERS
    TIERS --> PROVIDERS
    AGENT -->|retrieve| STORE
    AGENT -->|fallback| PAPERS
    AGENT -->|fallback| WEB
    AGENT -->|"cited answer (high confidence)"| VAULT
```

---

## 🧠 How it works

The agent is a state machine: retrieve, **grade**, and only ship when the evidence holds up — otherwise correct course (rewrite → papers → web) before answering.

```mermaid
flowchart TD
    Q(["Query"]) --> P["Planner"]
    P --> R["Local Retriever<br/>dense + BM25 + RRF"]
    R --> C{"Critic<br/>grade + confidence"}
    C -->|high| G["Generator<br/>cited answer"]
    C -->|"low / medium · retries left"| RW["Rewriter"]
    RW --> C
    C -->|"budget exhausted"| PD["Paper Discovery<br/>arXiv + Semantic Scholar"]
    PD --> C
    PD -.->|"still weak"| WF["Web Fallback<br/>Tavily → Serper → DDG"]
    WF --> C
    G --> RF{"Reflector<br/>gap audit"}
    RF -->|"gaps found · budget left"| R
    RF -->|accept| V[("Vault Gate<br/>auto-save if confident")]
    V --> A(["Cited Answer"])
```

**The nodes:**

1. **Planner** — decomposes complex queries into structured, atomic sub-questions.
2. **Local Retriever** — hybrid retrieval (dense vectors + BM25, fused with Reciprocal Rank Fusion) over your ingested corpus, with optional knowledge-graph expansion along note links.
3. **The Critic** — grades retrieved context and assigns a confidence verdict (`high` / `medium` / `low`). The gatekeeper that decides whether to ship or correct.
4. **Rewriter** — re-engineers the query to bridge semantic gaps, then re-retrieves (bounded retry budget).
5. **Paper Discovery** — when local evidence is insufficient, searches arXiv + Semantic Scholar and parses open-access PDFs on the fly.
6. **Web Fallback** — live web search (Tavily → Serper → DuckDuckGo cascade) as a resilient last external resort.
7. **Generator** — synthesizes a single answer with inline `[S#]` citations tied to the evidence.
8. **Reflector** — audits the draft for gaps and can trigger one more retrieval loop.
9. **Vault Gate** — writes high-confidence, cited answers to your local Markdown knowledge base.

> The retrieval cascade is **corrective**: each fallback stage only runs when the Critic isn't satisfied, so cheap local answers stay cheap and only hard questions pay for paper/web lookups.

---

## 🔌 Bring Your Own Model (BYOM)

ResearGent never hardcodes a model. Every LLM call is tagged with a **capability tier**, and you map each tier to whatever provider/model you prefer. This lets you put a strong model where reasoning matters and a cheap, fast model in the tight inner loops.

| Tier | Used by | What it needs |
|------|---------|---------------|
| **Reasoning** | Planner, Generator, Reflector | Strong synthesis & decomposition. Put your best model here. |
| **Fast** | Critic, Rewriter | Ultra-low latency / high rate limits — these run many times per query. A small, cheap model is ideal. |
| **Tool** | Tool / function-calling paths | A model that's reliable at structured output. |
| **Embed** | Ingestion & retrieval | An embedding model. Required for local retrieval. |

**Configuring a tier** takes three values — point it at any OpenAI-compatible endpoint:

```dotenv
REASONING_API_KEY=...      REASONING_BASE_URL=https://.../v1      REASONING_MODEL=...
FAST_API_KEY=...           FAST_BASE_URL=https://.../v1           FAST_MODEL=...
TOOL_API_KEY=...           TOOL_BASE_URL=https://.../v1           TOOL_MODEL=...
EMBED_API_KEY=...          EMBED_BASE_URL=http://localhost:11434/v1   EMBED_MODEL=...
```

That works with **OpenAI, OpenRouter, Groq, NVIDIA NIM, Cerebras, Together, Ollama, vLLM, LM Studio** — anything exposing an OpenAI-style `/v1`. Use the same endpoint for every tier, or mix (a big model for reasoning, a cheap fast one for the Critic).

**Rate-limit fallbacks (optional).** Free tiers hit 429s. Add one or more provider slots (`<PROVIDER>_API_KEY` + `<PROVIDER>_MODEL_<TIER>` for `cerebras`/`nvidia`/`groq`/`openrouter`/`ollama`) and they're automatically appended as fallbacks after your tier endpoint. On a transient failure (429 / 5xx / timeout) a tier rolls to the next option — and you can pin an exact order with `FAST_CASCADE=groq,cerebras,openrouter`, etc. See [`.env.example`](.env.example) for the full annotated template.

Check exactly how your config resolves at any time:

```bash
researgent status     # shows configured providers, per-tier routing, and cascade chains
researgent smoke      # pings each chat tier with one prompt to confirm credentials
researgent doctor     # verifies the embedding tier (and Ollama, if used) is reachable
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.11 or 3.12**
- **Node.js 18.17+** (for the 3D dashboard)
- **[uv](https://github.com/astral-sh/uv)** — fast Python package manager
- An **embedding model**. The simplest free/local option is [Ollama](https://ollama.com) with an embedding model pulled (`ollama pull nomic-embed-text`); or use a hosted embed-capable provider.
- *(Optional)* API keys for whichever LLM provider(s) and web-search providers you want to use.

### 1. Clone & install the backend

```bash
git clone https://github.com/SaumyaBish-t/ResearGent.git
cd ResearGent

# Create the venv and install everything from the lockfile
uv sync
```

> All backend commands below can be run as `uv run researgent <command>`, or activate the venv first
> (`source .venv/bin/activate`, or `.venv\Scripts\activate` on Windows) and just call `researgent <command>`.

### 2. Configure your models (`.env`)

Create a `.env` file in the project root. **You bring your own model for each tier** — a key + endpoint + model. A copyable starting point (`cp .env.example .env` for the fully annotated version):

```dotenv
# ── LLM tiers — point each at any OpenAI-compatible endpoint ──────────────────
# Use the same provider for all of them, or mix per tier.
REASONING_API_KEY=<your key>
REASONING_BASE_URL=<https://your-provider/v1>
REASONING_MODEL=<your strong reasoning model>

FAST_API_KEY=<your key>
FAST_BASE_URL=<https://your-provider/v1>
FAST_MODEL=<your fast, low-latency model>

TOOL_API_KEY=<your key>
TOOL_BASE_URL=<https://your-provider/v1>
TOOL_MODEL=<your tool / function-calling model>

# ── Embeddings (required for local retrieval) ────────────────────────────────
# Local example with Ollama (free): `ollama pull nomic-embed-text`
EMBED_API_KEY=ollama
EMBED_BASE_URL=http://localhost:11434/v1
EMBED_MODEL=nomic-embed-text

# ── Optional: external fallbacks ─────────────────────────────────────────────
TAVILY_API_KEY=               # live web search (optional; falls back to DuckDuckGo)
SEMANTIC_SCHOLAR_API_KEY=     # optional — higher Semantic Scholar rate limits

# ── Knowledge base + behavior ────────────────────────────────────────────────
NOTES_FOLDER_PATH=./notes         # where cited answers are auto-saved (any Markdown folder)
AUTO_SAVE_MIN_CONFIDENCE=medium   # high | medium | low | always
```

> **Rate-limit fallbacks:** add provider slots (`<PROVIDER>_API_KEY` + `<PROVIDER>_MODEL_<TIER>`) and they become automatic fallbacks for the matching tier — see [`.env.example`](.env.example).
> **Fully local / offline:** point every tier at `http://localhost:11434/v1` (Ollama) with any API-key value.

Verify your wiring before ingesting anything:

```bash
uv run researgent status     # confirm routing looks right
uv run researgent doctor     # confirm embeddings work
```

### 3. Add a knowledge base

ResearGent answers from a corpus you give it. Ingest any of:

```bash
# Ingest a PDF or a folder of PDFs
uv run researgent ingest ./path/to/papers/

# Ingest a folder of Markdown notes (Obsidian-style [[wikilinks]] become graph edges)
uv run researgent vault-ingest ./path/to/notes/

# Or seed a topic straight from Semantic Scholar abstracts
uv run researgent seed "your research topic"
```

(No corpus yet? The agent will still cascade to paper discovery and web search.)

### 4. Run the backend

```bash
uv run researgent serve            # http://localhost:8000  (API + a built-in lightweight UI)
# API docs at:                     # http://localhost:8000/docs
```

You can also run a research query straight from the CLI, no frontend needed:

```bash
uv run researgent research "How does the ReAct framework combine reasoning and acting in LLM agents?"
```

### 5. Run the 3D dashboard (frontend)

In a second terminal:

```bash
cd frontend
cp .env.local.example .env.local   # points NEXT_PUBLIC_API_BASE at http://localhost:8000
npm install
npm run dev                        # http://localhost:3000
```

Open **http://localhost:3000**, type a question, and watch the agent network light up node-by-node, fire data particles along its edges, and present the final cited answer.

---

## 🗂️ Configuration reference

| Variable | Purpose |
|----------|---------|
| `REASONING_API_KEY` / `_BASE_URL` / `_MODEL` | The model for the **reasoning** tier (Planner, Generator, Reflector). |
| `FAST_API_KEY` / `_BASE_URL` / `_MODEL` | The model for the **fast** tier (Critic, Rewriter). |
| `TOOL_API_KEY` / `_BASE_URL` / `_MODEL` | The model for the **tool** tier. |
| `EMBED_API_KEY` / `_BASE_URL` / `_MODEL` | The **embedding** model (required for local retrieval). |
| `<PROVIDER>_API_KEY` + `<PROVIDER>_MODEL_<TIER>` | *(Optional)* Provider slots (`cerebras`/`nvidia`/`groq`/`openrouter`/`ollama`) used as rate-limit fallbacks. |
| `FAST_CASCADE` / `REASONING_CASCADE` / `TOOL_CASCADE` | *(Optional)* Comma-separated explicit fallback order across provider slots. |
| `CASCADE_FALLBACK_ENABLED` | Toggle automatic fallback on transient errors (default `true`). |
| `TAVILY_API_KEY` / `SERPER_API_KEY` | Web-search providers (DuckDuckGo is the keyless final fallback). |
| `SEMANTIC_SCHOLAR_API_KEY` | Optional — higher rate limits for paper discovery. |
| `NOTES_FOLDER_PATH` | Folder where cited answers are auto-saved (plain `.md`). |
| `AUTO_SAVE_MIN_CONFIDENCE` | `high` / `medium` / `low` / `always` gate for auto-saving. |
| `CORS_ALLOW_ORIGINS` | Origins allowed to call the API from a browser. |
| `DATABASE_URL` | *(Optional)* Postgres for durable LangGraph checkpointing; in-memory if unset. |

Run `researgent status` for a live view of how these resolve.

---

## 📁 Project structure

```
ResearGent/
├── src/                     # Python backend
│   ├── agent/               # LangGraph state machine (nodes, graph, streaming)
│   ├── api/                 # FastAPI app + SSE endpoint
│   ├── llm/                 # Provider-agnostic LLM routing + cascade
│   ├── retrieval/           # Hybrid retrieval, paper discovery, web fallback
│   ├── ingest/              # PDF / Markdown chunking + embedding pipeline
│   └── main.py              # `researgent` CLI
├── frontend/                # Next.js + React Three Fiber 3D dashboard
│   ├── app/                 # routes + global styles
│   ├── components/          # Scene, AgentNode, Edges, Overlay, SearchBar, ...
│   └── lib/                 # Zustand store (SSE bridge), graph config, types
├── data/                    # Vector store + ingested corpora (gitignored)
├── notes/                   # Default knowledge-base output folder
├── pyproject.toml           # Backend deps (managed by uv)
└── .env                     # Your configuration (create this)
```

---

## 🛠️ Useful commands

```bash
researgent status              # provider routing + cascade chains
researgent smoke               # ping each LLM tier (credential check)
researgent doctor              # embedding / Ollama health check
researgent ingest <path>       # ingest PDF(s)
researgent vault-ingest <dir>  # ingest a Markdown notes folder
researgent research "<query>"  # full agentic run in the terminal
researgent serve               # launch the API + SSE stream
researgent store info          # inspect the vector store
```

---

## 🚀 Live Deployment

The hosted app at **<https://resear-gent.vercel.app>** runs on this stack:

| Layer | Provider | Notes |
|---|---|---|
| Frontend | **Vercel** (Hobby) | Next.js + R3F, env var `NEXT_PUBLIC_API_BASE` points at the Render API |
| Backend  | **Render** (Free) | FastAPI + Uvicorn; spins down after 15 min idle (~30s cold start) |
| Postgres | **Neon** (Free) | Users, threads, turns, subscriptions, LangGraph checkpoints |
| LLMs     | **Ollama Cloud** | `qwen3-coder:480b` across all tiers; Cerebras + NVIDIA + Groq + OpenRouter as cascade fallbacks |
| Auth     | **Google OAuth** | HttpOnly JWT session cookie (`SameSite=None; Secure` for cross-site Vercel↔Render) |
| Billing  | **Razorpay** | One-time ₹499 lifetime unlock (no webhook needed — HMAC-verified on `/billing/verify`) |

Production-only kill-switches in env (`ENABLE_LOCAL_RETRIEVAL=false`,
`AUTO_SAVE_TO_NOTES=false`, `COOKIE_SECURE=true`) skip the local-vault
retriever and notes auto-save since there's no shared filesystem in
serverless deployment — the graph runs web + papers + LLM priors only.

To deploy your own: `render.yaml` provisions the API service on Render
(Python runtime, `pip install -r requirements.txt`, `uvicorn` factory
start). Push the frontend folder to Vercel with `NEXT_PUBLIC_API_BASE`
set to your backend URL.

---

## 🤝 Contributing

Contributions are welcome. Good first areas:

- Additional academic providers (OpenAlex, PubMed, CORE)
- Smarter chunking heuristics for dense PDFs
- A paper-aware Critic rubric (grading full-text PDF chunks vs. clean web prose)
- Frontend polish and accessibility

Please open an issue to discuss larger changes, then submit a PR.

---

## 📄 License

MIT — see [`LICENSE`](LICENSE).
