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
# Each domain owns one folder under DATA_ROOT with three subfolders:
#   data/{domain}/papers/          ← PDF ingest (input)
#   data/{domain}/abstract_notes/  ← abstract-only paper cards from S2 seed
#   data/{domain}/research_data/   ← auto-saved research run notes (output)
#
# This consolidation replaces the old split layout where PDFs lived under
# `data/papers/{domain}/` and notes under `data/notes/{domain}/`. One root
# per domain makes the brain easier to inspect, backup, and migrate.
DATA_ROOT = Path("data")


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

    # ---- New consolidated layout (data/{domain}/{papers,abstract_notes,research_data}) ----

    @property
    def root_dir(self) -> Path:
        """The one folder this domain owns: data/{id}/"""
        return DATA_ROOT / self.id

    @property
    def papers_dir(self) -> Path:
        """PDF ingest input — data/{id}/papers/"""
        return self.root_dir / "papers"

    @property
    def abstract_notes_dir(self) -> Path:
        """
        Abstract-only paper cards from the S2 seed pass —
        data/{id}/abstract_notes/

        Used when a paper has no open-access PDF: we store a markdown
        card with title + abstract + citation count + URL so the
        ingest pipeline can still chunk + embed it as evidence.
        Previously known as `_abstracts` under `papers_dir`.
        """
        return self.root_dir / "abstract_notes"

    @property
    def research_data_dir(self) -> Path:
        """
        Auto-saved research run notes — data/{id}/research_data/YYYY-MM-DD/

        Output side of the brain. Kept separate from `papers_dir` so a
        future ingest pass over the PDFs doesn't pick up the notes we
        wrote ABOUT that corpus and create a feedback loop.
        """
        return self.root_dir / "research_data"

    # ---- Backwards-compat aliases ----
    #
    # Kept so callers that haven't migrated to the new property names
    # still work. New code should use papers_dir / abstract_notes_dir /
    # research_data_dir directly.

    @property
    def ingest_dir(self) -> Path:
        """DEPRECATED alias for `papers_dir`."""
        return self.papers_dir

    @property
    def notes_dir(self) -> Path:
        """DEPRECATED alias for `research_data_dir`."""
        return self.research_data_dir


# Legacy constant kept so older callers that imported PAPERS_ROOT still
# resolve. Points at the OLD location used by the legacy-layout migrator.
# New code MUST use Domain.papers_dir instead — there is no longer a single
# "papers root" since papers now live under each domain's own folder.
PAPERS_ROOT = DATA_ROOT / "papers"


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


def migrate_legacy_layout(*, verbose: bool = False) -> list[str]:
    """
    Migrate the old split layout to the consolidated per-domain layout.

    OLD:
      data/papers/{domain}/*.pdf
      data/papers/{domain}/_abstracts/*.md
      data/notes/{domain}/YYYY-MM-DD/*.md

    NEW:
      data/{domain}/papers/*.pdf
      data/{domain}/abstract_notes/*.md
      data/{domain}/research_data/YYYY-MM-DD/*.md

    Idempotent: no-op if nothing legacy exists. Returns a list of human-
    readable lines describing what was moved, so the CLI can surface it.
    Safe to call on every startup — costs a few `os.path.exists` checks
    in the steady state.
    """
    import shutil

    moves: list[str] = []

    for dom in DOMAINS.values():
        legacy_papers = PAPERS_ROOT / dom.id
        legacy_abstracts = legacy_papers / "_abstracts"
        legacy_notes = DATA_ROOT / "notes" / dom.id

        # 1. PDFs:  data/papers/{id}/*.pdf  →  data/{id}/papers/
        if legacy_papers.exists() and legacy_papers.is_dir():
            pdfs = [p for p in legacy_papers.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"]
            if pdfs:
                dom.papers_dir.mkdir(parents=True, exist_ok=True)
                for pdf in pdfs:
                    target = dom.papers_dir / pdf.name
                    if not target.exists():
                        shutil.move(str(pdf), str(target))
                        moves.append(f"  {pdf}  →  {target}")

        # 2. abstract cards:  data/papers/{id}/_abstracts/*  →  data/{id}/abstract_notes/
        if legacy_abstracts.exists() and legacy_abstracts.is_dir():
            cards = [p for p in legacy_abstracts.iterdir() if p.is_file()]
            if cards:
                dom.abstract_notes_dir.mkdir(parents=True, exist_ok=True)
                for c in cards:
                    target = dom.abstract_notes_dir / c.name
                    if not target.exists():
                        shutil.move(str(c), str(target))
                        moves.append(f"  {c}  →  {target}")
            # Remove the now-empty _abstracts folder so the old layout fully
            # collapses. shutil.rmtree is safe here because we already moved
            # every file out.
            try:
                if not any(legacy_abstracts.iterdir()):
                    legacy_abstracts.rmdir()
            except OSError:
                pass

        # 3. research notes:  data/notes/{id}/  →  data/{id}/research_data/
        if legacy_notes.exists() and legacy_notes.is_dir():
            for child in legacy_notes.iterdir():
                target = dom.research_data_dir / child.name
                if not target.exists():
                    dom.research_data_dir.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(child), str(target))
                    moves.append(f"  {child}  →  {target}")
            try:
                if not any(legacy_notes.iterdir()):
                    legacy_notes.rmdir()
            except OSError:
                pass

        # 4. clean up an empty data/papers/{id}/ shell if everything moved.
        try:
            if legacy_papers.exists() and not any(legacy_papers.iterdir()):
                legacy_papers.rmdir()
        except OSError:
            pass

    # Try to remove the now-empty parent legacy roots so the layout fully
    # collapses to the new shape. Skipped silently if other untracked stuff
    # still lives there.
    for legacy_root in (PAPERS_ROOT, DATA_ROOT / "notes"):
        try:
            if legacy_root.exists() and not any(legacy_root.iterdir()):
                legacy_root.rmdir()
        except OSError:
            pass

    if verbose and moves:
        print(f"[migrate] moved {len(moves)} item(s) to new per-domain layout:")
        for m in moves:
            print(m)

    return moves


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
