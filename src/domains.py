"""
Domain registry — the three corpora ResearGent's persona contract pins to.

Why a registry, not strings sprinkled across the codebase
--------------------------------------------------------
The persona spec names three explicit corpora — agentic AI, quantitative
finance, ML for time-series forecasting — each with its own ingest root,
its own Stage-1 seed-search query set, and its own keyword fingerprint for
auto-routing queries. Putting all three in one module gives us:

  * One place to add a fourth domain (e.g. "computer_vision") without
    touching the pipeline, the retriever, or the CLI.
  * One place to tune Semantic Scholar seed queries when we notice the
    seed corpus drifting (e.g. add "PatchTST" once the paper is famous).
  * A single source of truth that the ingest pipeline, the retrieval
    filter, the planner, and the README can ALL import from.

What's intentionally NOT here
-----------------------------
  * No per-domain *retrieval logic*. All three domains share the same
    semantic chunker, GLiNER labels, hybrid retriever, RRF parameters.
    The domain only acts as a metadata filter on a unified store.
  * No per-domain *embedder*. One embedding space across all domains
    lets cross-domain queries ("HMM regime detection for LOB execution
    using TimesFM") fuse evidence from multiple corpora.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


# All ingest roots sit under one parent so `researgent ingest-domains` can
# walk the parent and dispatch by subfolder name. Keeping the parent name
# stable means an op can grep `data/papers/` and see the whole brain.
PAPERS_ROOT = Path("data") / "papers"


@dataclass(frozen=True)
class Domain:
    """One named corpus with a local subdir and Stage-1 seed queries."""

    # Canonical short id. Used as the subfolder name AND the metadata value
    # stamped into Chroma + the registry. MUST be lowercase + underscored so
    # it survives CLI args and filesystem paths on any OS.
    id: str

    # Human-friendly label for status panels and the README. No code path
    # branches on this — purely cosmetic.
    label: str

    # One-line summary surfaced by `researgent domains` and the planner's
    # auto-router. Keep short — it's read into the routing prompt.
    description: str

    # Search queries fed to Semantic Scholar at seed time, sorted by
    # `citationCount:desc`. The point is to surface the FOUNDATIONAL papers
    # of each subfield — anchors a fresh corpus on bedrock work rather than
    # bleeding-edge noise. Pick 3-6 broad terms, not 20 narrow ones.
    seed_queries: tuple[str, ...]

    # Keywords that, when present in a user query, strongly signal this
    # domain. Used by the auto-router to set domain_scope when the user
    # didn't pass --domain explicitly. Case-insensitive substring match.
    routing_keywords: tuple[str, ...] = ()

    @property
    def ingest_dir(self) -> Path:
        """Where PDFs for this domain live on disk."""
        return PAPERS_ROOT / self.id

    @property
    def notes_dir(self) -> Path:
        """
        Where ResearGent auto-saved research notes for this domain live.

        Kept distinct from `ingest_dir` so PDFs (input) and generated
        research notes (output) don't share a folder — a future ingest
        pass over `data/papers/{domain}/` should not pick up the notes
        we wrote about that same corpus.

        Layout:   data/notes/{domain_id}/YYYY-MM-DD/<note>.md

        Returned path may not yet exist; the vault writer creates it.
        """
        return Path("data") / "notes" / self.id


DOMAINS: dict[str, Domain] = {
    # ---- Agentic & multi-agent systems --------------------------------------
    # The seed queries lean into the bedrock — LangGraph, ReAct, MCP, A2A —
    # not the trendy paper of the week. citationCount:desc already biases
    # toward established work; we just have to point it at the right neighborhood.
    "agentic_ai": Domain(
        id="agentic_ai",
        label="Agentic & Multi-Agent Systems",
        description=(
            "LangGraph topologies, ReAct loops, multi-agent routing, "
            "Model Context Protocol, Agent-to-Agent networks, reflection/critic patterns."
        ),
        seed_queries=(
            "LLM agent reasoning ReAct",
            "multi-agent LLM orchestration",
            "LangGraph stateful agent",
            "tool-use language model",
            "self-reflection LLM critique",
            "retrieval augmented generation corrective",
        ),
        routing_keywords=(
            "agent", "agentic", "langgraph", "react", "tool-use",
            "multi-agent", "mcp", "a2a", "critic", "reflector",
            "planner", "rag", "crag", "self-rag", "hitl", "human-in-the-loop",
        ),
    ),
    # ---- Quantitative finance & market microstructure -----------------------
    # SEBI/NSE-specific seed terms are deliberately omitted from the S2 set
    # — that literature is sparse on S2 and better seeded by hand-drop PDFs.
    # The S2 queries focus on academic finance which is dense and well-cited.
    "quant_finance": Domain(
        id="quant_finance",
        label="Quantitative Finance & Market Microstructure",
        description=(
            "Alpha generation, factor investing, regime detection (HMM, changepoint), "
            "volatility surfaces, RL execution, limit order book dynamics, VaR/CVaR."
        ),
        seed_queries=(
            "limit order book microstructure",
            "regime detection hidden markov financial",
            "reinforcement learning execution optimal trading",
            "factor investing equity premium",
            "value at risk conditional VaR",
            "high frequency trading market making",
        ),
        routing_keywords=(
            "alpha", "factor", "regime", "hmm", "volatility", "greeks",
            "var", "cvar", "limit order", "lob", "microstructure",
            "execution", "market making", "sebi", "nse", "bse",
            "trading", "portfolio", "options", "futures", "derivative",
        ),
    ),
    # ---- ML for time-series forecasting -------------------------------------
    # Seeded toward the transformer + linear + zero-shot foundation-model
    # axes the persona spec calls out. PatchTST/DLinear/Informer are explicit
    # because they're the named-entity hits the GLiNER extractor will trigger
    # on most often in this domain.
    "time_series": Domain(
        id="time_series",
        label="ML for Time-Series Forecasting",
        description=(
            "PatchTST, DLinear, Informer, Mixture of Linear Experts, "
            "zero-shot foundation models (TimesFM, Chronos, lag-llama), MSE/MAE/DTW."
        ),
        seed_queries=(
            "transformer time series forecasting",
            "PatchTST long horizon forecasting",
            "DLinear decomposition time series",
            "foundation model time series zero shot",
            "Informer probsparse attention",
            "dynamic time warping similarity",
        ),
        routing_keywords=(
            "time series", "time-series", "forecast", "forecasting",
            "patchtst", "dlinear", "informer", "mole", "timesfm",
            "chronos", "lag-llama", "mse", "mae", "dtw",
            "seasonality", "stationarity", "horizon",
        ),
    ),
}


def all_domain_ids() -> list[str]:
    """Stable-ordered list of registered domain ids — used by CLI defaults."""
    return list(DOMAINS.keys())


def get_domain(domain_id: str) -> Domain:
    """Strict lookup. Raises with a helpful list when the id is unknown."""
    if domain_id not in DOMAINS:
        raise KeyError(
            f"unknown domain {domain_id!r}; known: {', '.join(DOMAINS.keys())}"
        )
    return DOMAINS[domain_id]


def infer_domains_from_query(query: str, *, min_hits: int = 1) -> list[str]:
    """
    Cheap, deterministic auto-router.

    Goal: when the user runs `researgent research "..."` without --domain,
    decide whether the query strongly maps to one (or two) of the registered
    domains. If it does, we scope retrieval to just those Chroma metadata
    buckets — recall improves because BM25 stops competing with off-domain
    duplicates and dense top-k stops being diluted.

    Why NOT an LLM call here
    ------------------------
    Free-tier budget. Doing a FAST-tier classification on EVERY research
    query would burn ~50% of a typical Cerebras free quota. Substring
    matching with the routing keyword set hits the ~80% case at zero cost
    — and when it returns no match the caller falls back to "search
    everything", which is exactly the right behaviour.

    Returns
    -------
    A list of domain_ids the query likely belongs to (0, 1, or 2 entries).
    Empty list means "no strong signal — search across all corpora".
    """
    q = query.lower()
    hits: list[tuple[str, int]] = []
    for dom in DOMAINS.values():
        n = sum(1 for kw in dom.routing_keywords if kw in q)
        if n >= min_hits:
            hits.append((dom.id, n))

    # Sort by strongest signal first, keep at most 2 domains. Past that
    # we're effectively saying "search everywhere" anyway.
    hits.sort(key=lambda x: x[1], reverse=True)
    return [d for d, _ in hits[:2]]
