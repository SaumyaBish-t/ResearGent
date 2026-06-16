"""
Centralized configuration.

Loads from .env (via pydantic-settings) and exposes typed settings the rest of
the codebase consumes. Single source of truth — no os.environ reads scattered
around the codebase.

Why pydantic-settings?
  - Type-checked at load time (catches a typo'd env var name immediately).
  - Easy to add validation rules later (e.g. "API key must start with 'nvapi-'").
  - Trivially mockable in tests.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelTier(str, Enum):
    """
    Agents in this system have different cost/latency/quality needs.

    REASONING — heavy thinking. Used by Planner, Reflector, Report Generator.
                Wants the strongest model available.
    FAST      — quick classifications. Used by Critic/Grader, query rewriter.
                Wants the FASTEST cheap model — these run many times per query.
    TOOL      — function-calling / tool-use. Different from REASONING because
                tool-use is a distinct skill where small differences in model
                training matter a lot (GPT-OSS-120B and Qwen3 are exceptional;
                some otherwise-strong reasoning models do tool use badly).
    EMBED     — embedding model for retrieval. Used by ingestion + retriever.
    """

    REASONING = "reasoning"
    FAST = "fast"
    TOOL = "tool"
    EMBED = "embed"


# All supported providers. Each maps to an OpenAI-compatible HTTP endpoint —
# that's what makes adding new ones cheap (no new SDK to learn).
ProviderName = Literal["cerebras", "nvidia", "groq", "openrouter", "ollama"]

# Priority order for AUTO-selection when no override is set. Also the order
# in which the CASCADE FALLBACK chain is built for each tier — primary tries
# first, then we walk down on transient failure (rate limit / 5xx / timeout).
#   - cerebras first  -> 1000+ TPS, huge models (Qwen3-235B)
#   - nvidia next     -> widest model catalog, has embeddings
#   - groq next       -> 315 TPS for the fast/tool tiers
#   - openrouter      -> variety / experimentation
#   - ollama last     -> local fallback when nothing else is configured
_AUTO_PRIORITY: list[ProviderName] = ["cerebras", "nvidia", "groq", "openrouter", "ollama"]

# Providers that host embedding models. Keeps tier resolution honest.
_EMBED_CAPABLE: set[ProviderName] = {"nvidia", "openrouter", "ollama"}


class Settings(BaseSettings):
    """All app config in one typed object. Imported as `settings` everywhere."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---- Web frontend (Next.js / R3F) --------------------------------------
    # Comma-separated list of origins allowed to hit the API from a browser.
    # The Next.js dev server runs on :3000; production builds typically serve
    # from a single origin. Kept as a string and split at use-site so it round-
    # trips cleanly through a flat .env value (CORS_ALLOW_ORIGINS=...).
    cors_allow_origins: str = "http://localhost:3000,http://127.0.0.1:3000"

    @property
    def cors_origins_list(self) -> list[str]:
        """Parsed, de-whitespaced origins. `*` short-circuits to allow-all."""
        raw = (self.cors_allow_origins or "").strip()
        if raw == "*":
            return ["*"]
        return [o.strip() for o in raw.split(",") if o.strip()]

    # ---- Auth + billing (Phase 16) ----------------------------------------
    # Google OAuth — used to identify users for quotas + billing.
    google_client_id: str | None = None
    google_client_secret: str | None = None
    google_redirect_uri: str = "http://localhost:8000/auth/callback"

    # Session JWT signing key. Generate via: python -c "import secrets;print(secrets.token_urlsafe(48))"
    session_secret: str = "CHANGE_ME_BEFORE_PRODUCTION_USE_A_REAL_SECRET"
    session_cookie_name: str = "researgent_session"
    session_ttl_hours: int = 24 * 14  # 2 weeks
    cookie_secure: bool = False        # true in prod (HTTPS)

    # Comma-separated list of emails granted unlimited quota.
    # Each admin email is matched case-insensitively against Google's verified
    # email. Set via .env: ADMIN_EMAILS=you@example.com,coworker@example.com
    admin_emails: str = "sambisht123k@gmail.com"

    @property
    def admin_email_set(self) -> set[str]:
        return {e.strip().lower() for e in (self.admin_emails or "").split(",") if e.strip()}

    # Free-tier quota knobs (subscribers + admins bypass).
    free_threads_per_month: int = 3
    free_turns_per_thread: int = 3     # original Q + up to 2 follow-ups

    # Razorpay subscription billing.
    razorpay_key_id: str | None = None
    razorpay_key_secret: str | None = None
    razorpay_webhook_secret: str | None = None
    razorpay_plan_id: str | None = None
    razorpay_price_inr: int = 499      # display only; canonical price is on the plan

    # Where the user lands after a successful Razorpay checkout.
    billing_success_url: str = "http://localhost:3000/?billing=success"
    billing_cancel_url: str = "http://localhost:3000/?billing=cancelled"

    # ---- NVIDIA NIM ---------------------------------------------------------
    nvidia_api_key: str | None = None
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    # Models come from .env — NVIDIA_MODEL_{REASONING,FAST,TOOL,EMBED}.
    nvidia_model_reasoning: str | None = None
    nvidia_model_fast: str | None = None
    nvidia_model_tool: str | None = None
    nvidia_model_embed: str | None = None

    # ---- Groq ---------------------------------------------------------------
    groq_api_key: str | None = None
    groq_base_url: str = "https://api.groq.com/openai/v1"
    # Models come from .env — GROQ_MODEL_{REASONING,FAST,TOOL}.
    groq_model_reasoning: str | None = None
    groq_model_fast: str | None = None
    groq_model_tool: str | None = None

    # ---- Cerebras -----------------------------------------------------------
    cerebras_api_key: str | None = None
    cerebras_base_url: str = "https://api.cerebras.ai/v1"
    # Models come from .env — CEREBRAS_MODEL_{REASONING,FAST,TOOL}.
    cerebras_model_reasoning: str | None = None
    cerebras_model_fast: str | None = None
    cerebras_model_tool: str | None = None

    # ---- OpenRouter ---------------------------------------------------------
    openrouter_api_key: str | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    # Models come from .env — OPENROUTER_MODEL_{REASONING,FAST,TOOL,EMBED}.
    openrouter_model_reasoning: str | None = None
    openrouter_model_fast: str | None = None
    openrouter_model_tool: str | None = None
    openrouter_model_embed: str | None = None
    openrouter_app_url: str = "https://github.com/SaumyaBish-t/ResearGent"
    openrouter_app_name: str = "ResearGent"

    # ---- Ollama (local) -----------------------------------------------------
    ollama_base_url: str = "http://localhost:11434/v1"
    # Models come from .env — OLLAMA_MODEL_{REASONING,FAST,TOOL,EMBED}.
    ollama_model_reasoning: str | None = None
    ollama_model_fast: str | None = None
    ollama_model_tool: str | None = None
    ollama_model_embed: str | None = None

    # ---- Routing ------------------------------------------------------------
    primary_provider: ProviderName | None = Field(
        default=None,
        description="Force a provider for ALL tiers. Overrides auto-detection.",
    )
    reasoning_provider: ProviderName | None = None
    fast_provider: ProviderName | None = None

    # Per-tier cascade override (Phase 15+).
    #
    # By default `resolve_cascade()` walks `_AUTO_PRIORITY` and adds every
    # configured provider in that order — fine for most setups, but not when
    # you want, say, "Cerebras Llama-3.3 primary, fall straight through to
    # Groq Llama-3.3 on a 429, skip NVIDIA's 8B in between."
    #
    # When set, this list REPLACES the auto-derived chain for the FAST tier.
    # Comma-separated env var: FAST_CASCADE=cerebras,groq
    #
    # Unknown / unconfigured providers in the list are silently dropped at
    # resolve time, so a bad value degrades to "use whichever subset works"
    # rather than crashing the run.
    fast_cascade: list[str] | None = None
    reasoning_cascade: list[str] | None = None
    tool_cascade: list[str] | None = None
    tool_provider: ProviderName | None = None
    embed_provider: ProviderName | None = None

    # ---- Tier-direct "Bring Your Own Model" (simplest path) ----------------
    # Point a capability tier straight at ANY OpenAI-compatible endpoint by
    # giving it three things: an API key, a base URL, and a model. When all
    # three are set for a tier, that tier uses them directly and takes
    # precedence over the provider slots above (which remain available as an
    # optional cascade fallback). This is the recommended BYOM setup:
    # one endpoint + key + model per tier (reasoning / fast / tool / embed).
    reasoning_api_key: str | None = None
    reasoning_base_url: str | None = None
    reasoning_model: str | None = None
    fast_api_key: str | None = None
    fast_base_url: str | None = None
    fast_model: str | None = None
    tool_api_key: str | None = None
    tool_base_url: str | None = None
    tool_model: str | None = None
    embed_api_key: str | None = None
    embed_base_url: str | None = None
    embed_model: str | None = None

    # ---- Cascade fallback ---------------------------------------------------
    # When True, transient failures (429 / 5xx / timeout) on the primary
    # provider for a tier automatically retry on the next configured provider.
    # Set False to surface raw errors (useful for debugging).
    cascade_fallback_enabled: bool = True

    # ---- Web fallback (Phase 4) --------------------------------------------
    # Cascade of web search providers — same pattern as LLM cascade. Tried in
    # order, falls through on quota exhausted / failure / empty results.
    #
    # 1. Tavily — primary. Agent-tuned snippets, 1000/mo free.
    #    Get a key: https://tavily.com
    tavily_api_key: str | None = None

    # 2. Serper.dev — secondary. Real Google SERP, 2500 free signup credits,
    #    then $1/1k. Highest raw quality, no attribution requirement.
    #    Get a key: https://serper.dev
    serper_api_key: str | None = None

    # 3. DuckDuckGo — last resort. Free forever, no API key, no signup.
    #    Uses HTML scrape via the `ddgs` library. Rate-limited (~1 req/sec).
    #    Always considered "configured" — guaranteed working fallback.

    # ---- Semantic Scholar API key (Phase 15+) ------------------------------
    # OPTIONAL. The public endpoint works without a key but 429s under any
    # real burst (the Phase 15 seed run hit this on the back half of an
    # 18-query sweep). Email s2-api@allenai.org with a short blurb about
    # your usage and they issue a personal key, also rate-limited at "≈1
    # request per second cumulative across all endpoints" but reliably so.
    #
    # When set, every Semantic Scholar HTTP call (the seeder + the agent's
    # Stage-2 paper-discovery fallback) sends the key as the `x-api-key`
    # header — the official auth scheme documented at
    #   https://www.semanticscholar.org/product/api/tutorial
    # and the in-process courtesy gap drops from 3s (public) to ~1.1s
    # (just under the documented 1 RPS ceiling).
    semantic_scholar_api_key: str | None = None

    # Order of attempts when the agent calls web_fallback.
    web_search_cascade: list[str] = ["tavily", "serper", "duckduckgo"]
    # Max bounded retries when Critic flags retrieval as low quality. Each
    # rewrite costs one FAST-tier call + one full retrieve. 2 is the sweet
    # spot — diminishing returns past that, and we have web fallback after.
    crag_max_rewrites: int = 2
    # Confidence thresholds. Critic returns one of {high, medium, low}.
    #   - high   -> straight to generator
    #   - medium -> rewrite & retry (if budget left) else generator
    #   - low    -> rewrite & retry (if budget left) else web fallback
    # No env-tunable for the thresholds themselves — those are baked into the
    # Critic's prompt + parsing. The action POLICY above is the knob.

    # ---- Knowledge-graph expansion (Phase 10) -----------------------------
    # When True, retrieval results trigger a 1-hop walk along stored
    # wikilink edges to surface STRUCTURALLY-related chunks the embedder
    # might have ranked low. The "AI brain" behavior — your notes' link
    # graph becomes part of retrieval, not just a UI feature in Obsidian.
    graph_expansion_enabled: bool = True
    # Max extra chunks added to context per query via graph expansion.
    # Bounded to keep prompts manageable — too high and the generator
    # drowns in tangentially-related context.
    graph_expansion_max_extra_chunks: int = 6

    # Master kill-switch for the local vault retriever. False = the retrieve
    # node returns zero local chunks (per sub-question), the Critic falls to
    # low confidence, and the graph routes through web/papers + LLM priors
    # only. Set ENABLE_LOCAL_RETRIEVAL=false in production where there's no
    # shared filesystem of notes to query.
    enable_local_retrieval: bool = True

    # ---- Markdown knowledge-base integration (Phase 8 + 11) ----------------
    # Path to a FOLDER OF MARKDOWN NOTES that serves as your knowledge base.
    # The folder is just plain `.md` files — works with VS Code, Obsidian,
    # Logseq, Foam, vim, or anything else that edits text. We use the same
    # `[[wikilink]]` and `#tag` conventions Obsidian made popular, but the
    # app itself is NOT required.
    #
    # Resolution order at runtime:
    #   1. NOTES_FOLDER_PATH env var (this setting)        ← preferred
    #   2. OBSIDIAN_VAULT_PATH env var (legacy alias)      ← backward compat
    #   3. The project-local ./notes folder                ← convenience default
    notes_folder_path: str | None = None
    # Legacy alias — same effect as notes_folder_path. Kept so existing
    # configs don't break.
    obsidian_vault_path: str | None = None
    # Subfolder inside the notes folder where ResearGent writes generated
    # answers. Defaults to ResearGent/ so machine-generated notes stay
    # cleanly separated from notes you wrote by hand.
    obsidian_output_folder: str = "ResearGent"

    # ---- Auto-save to knowledge base ---------------------------------------
    # When True, every research run that meets the confidence gate is
    # written back as a note in the notes folder — the brain grows
    # automatically without you clicking "save" each time.
    auto_save_to_notes: bool = True
    # Minimum Critic verdict required to auto-save.
    #   high     — only save when Critic was satisfied with retrieval
    #   medium   — save unless retrieval was clearly poor (NEW DEFAULT)
    #   low      — save anything that produced cited sources
    #   always   — save even thin/disagreement-flagged answers
    # Pure-LLM-priors answers (no sources at all) are NEVER auto-saved
    # regardless of this setting — they'd inject hallucinations.
    auto_save_min_confidence: str = "medium"

    # When the verdict is exactly `medium`, ALSO require the Critic's
    # weighted score to be at least this floor before auto-saving.
    # The `medium` band covers everything from "barely any signal"
    # (score ~0.15, two partials and lots of irrelevants) up to
    # "almost high" (score ~0.69). Saving the bottom of that range
    # injects half-grounded answers into the brain; saving the top
    # captures genuinely useful runs that just barely missed the
    # external-fresh HIGH threshold of 0.70.
    #
    # A `high` verdict bypasses this floor (it already cleared a
    # stricter score gate upstream); a `low` verdict bypasses it the
    # other way (already rejected). So this is purely a within-medium
    # tightener.
    auto_save_min_score: float = 0.50

    def resolve_notes_folder(self) -> str | None:
        """Pick the active notes folder per the resolution order above."""
        from pathlib import Path as _P
        if self.notes_folder_path:
            return self.notes_folder_path
        if self.obsidian_vault_path:
            return self.obsidian_vault_path
        default = _P("notes").resolve()
        if default.exists() and default.is_dir():
            return str(default)
        return None

    # ---- Open-domain paper discovery (Phase 7) -----------------------------
    # When True: after the rewrite budget is exhausted with low/medium
    # confidence, the agent searches arXiv + Semantic Scholar for the
    # original question BEFORE falling through to web search. False keeps
    # the Phase 4-6 behavior (skip straight to web_fallback).
    paper_discovery_enabled: bool = True

    # LLM-only reasoning is the ABSOLUTE LAST RESORT when every retrieval
    # path (corpus, rewriter, papers, web) has produced no usable evidence.
    # When True the agent answers from training-time priors with a loud
    # "no sources" disclaimer. When False the agent routes to no_answer
    # ("I don't know") — appropriate for medical/legal/regulatory use.
    llm_reasoning_fallback_enabled: bool = True

    # ---- Self-reflection (Phase 5) -----------------------------------------
    # Max TOTAL Reflector audit calls (= 1 initial audit + (N-1) loopbacks).
    # Default 2 means: generator runs, reflector audits, ONE loopback allowed,
    # generator runs again, reflector audits again, accept. Total reflector
    # calls = 2, total generator calls = 2. Strictly bounded.
    reflection_max_iterations: int = 2

    # Hard ceiling on total sub-questions to prevent runaway decomposition.
    # Without this, each reflection loop adds K follow-ups and the retriever
    # + critic costs grow super-linearly. 8 is comfortable for academic
    # research questions (4-axis comparison + a couple of follow-ups).
    reflection_max_subq_total: int = 8

    # Max follow-up questions the Reflector can add per loopback. Tighter
    # than what the prompt asks for ("1-3") so a misbehaving model can't
    # blow out the sub-question count in one shot.
    reflection_max_follow_ups_per_loop: int = 2

    # ---- Observability ------------------------------------------------------
    # Logs every chat()/embed() call to a JSONL file. Surface via `researgent stats`.
    observability_enabled: bool = True
    observability_log_path: str = "data/llm_calls.jsonl"

    # ---- PostgreSQL persistence (Phase 12) ---------------------------------
    # The relational brain. Holds:
    #   - LangGraph agent checkpoints (PostgresSaver)
    #   - documents_registry: PDF/note metadata, file hashes, storage URLs
    #   - (future) eval runs, user prefs
    # ChromaDB still owns vectors; raw files still live on disk/S3.
    #
    # We target the 500 MB free tier (Supabase / Neon / Render). That forces
    # discipline elsewhere — see state.py's lean-state contract for the agent
    # (no raw HTML/PDF in checkpoints) and the 7-day TTL pruner.
    #
    # Either set DATABASE_URL directly (preferred — what most managed PG
    # providers hand you) or set the discrete fields below.
    database_url: str | None = None
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "researgent"
    postgres_user: str = "postgres"
    postgres_password: str | None = None
    postgres_sslmode: str = "prefer"  # "require" for Supabase/Neon

    # Connection pool sizing. Free-tier Postgres usually caps at ~20-60 conns,
    # and the agent is sync + single-process — small pool is correct.
    postgres_pool_min_size: int = 1
    postgres_pool_max_size: int = 8

    # Checkpoint TTL in days. The pruner (see `researgent db prune`) deletes
    # checkpoints + checkpoint_writes older than this. 7 days is enough to
    # replay a recent run for debugging without blowing the 500 MB budget.
    checkpoint_ttl_days: int = 7

    def resolve_database_url(self) -> str | None:
        """
        Return a libpq-compatible URL for psycopg/SQLAlchemy.

        Precedence: explicit DATABASE_URL > discrete fields (only if a
        password is set — we refuse to silently build a passwordless URL).
        Returns None when Postgres isn't configured at all, so callers can
        fall back to MemorySaver during local dev.
        """
        if self.database_url:
            return self.database_url
        if not self.postgres_password:
            return None
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
            f"?sslmode={self.postgres_sslmode}"
        )

    # ---- Validators ---------------------------------------------------------
    @field_validator(
        "primary_provider",
        "reasoning_provider",
        "fast_provider",
        "tool_provider",
        "embed_provider",
        mode="before",
    )
    @classmethod
    def _blank_to_none(cls, v):
        # Empty .env values (`PRIMARY_PROVIDER=`) should mean "unset", not "".
        if isinstance(v, str) and not v.strip():
            return None
        return v

    @field_validator(
        "fast_cascade",
        "reasoning_cascade",
        "tool_cascade",
        mode="before",
    )
    @classmethod
    def _parse_csv_cascade(cls, v):
        """
        Let users write FAST_CASCADE=cerebras,groq in .env instead of the
        JSON form pydantic-settings expects by default for list[str] fields.
        Tolerant of whitespace and case ("Cerebras, Groq" → ["cerebras","groq"]).
        Empty string → None (treated as "no override").
        """
        if v is None:
            return None
        if isinstance(v, list):
            return [str(x).strip().lower() for x in v if str(x).strip()]
        if isinstance(v, str):
            s = v.strip()
            if not s:
                return None
            return [tok.strip().lower() for tok in s.split(",") if tok.strip()]
        return v

    # ---- Helpers ------------------------------------------------------------
    def tier_direct(self, tier: ModelTier) -> tuple[str, str, str] | None:
        """
        Tier-direct BYOM config as (api_key, base_url, model), or None.

        Returns a value only when ALL THREE of the tier's *_API_KEY / *_BASE_URL
        / *_MODEL are set. This takes precedence over the provider slots and is
        the simplest way to bring your own model for a capability tier.
        """
        trio = {
            ModelTier.REASONING: (self.reasoning_api_key, self.reasoning_base_url, self.reasoning_model),
            ModelTier.FAST: (self.fast_api_key, self.fast_base_url, self.fast_model),
            ModelTier.TOOL: (self.tool_api_key, self.tool_base_url, self.tool_model),
            ModelTier.EMBED: (self.embed_api_key, self.embed_base_url, self.embed_model),
        }[tier]
        key, base, model = trio
        if key and base and model:
            return (key, base, model)
        return None

    def configured_providers(self) -> list[ProviderName]:
        """Which providers have credentials? Ollama is 'always available'."""
        out: list[ProviderName] = []
        if self.cerebras_api_key:
            out.append("cerebras")
        if self.nvidia_api_key:
            out.append("nvidia")
        if self.groq_api_key:
            out.append("groq")
        if self.openrouter_api_key:
            out.append("openrouter")
        out.append("ollama")  # local — always available
        return out

    def resolve_provider(self, tier: ModelTier) -> ProviderName:
        """
        Pick the PRIMARY provider for a given tier.

        Resolution order:
          1. Per-tier override (REASONING_PROVIDER, FAST_PROVIDER, TOOL_PROVIDER, EMBED_PROVIDER)
          2. Global override (PRIMARY_PROVIDER) — skipped for EMBED if provider can't embed
          3. Auto: first configured provider in `_AUTO_PRIORITY` that supports the tier
        """
        tier_override = {
            ModelTier.REASONING: self.reasoning_provider,
            ModelTier.FAST: self.fast_provider,
            ModelTier.TOOL: self.tool_provider,
            ModelTier.EMBED: self.embed_provider,
        }[tier]

        if tier_override:
            return tier_override

        if self.primary_provider:
            if tier == ModelTier.EMBED and self.primary_provider not in _EMBED_CAPABLE:
                pass
            else:
                return self.primary_provider

        priority = list(_AUTO_PRIORITY)
        if tier == ModelTier.EMBED:
            priority = [p for p in priority if p in _EMBED_CAPABLE]

        configured = set(self.configured_providers())
        for p in priority:
            if p in configured:
                return p

        raise RuntimeError("No providers available. Configure at least one in .env")

    def resolve_cascade(self, tier: ModelTier) -> list[ProviderName]:
        """
        Build the ordered FALLBACK chain for a tier.

        Primary first (from `resolve_provider`), then any other configured
        providers that can serve this tier, in `_AUTO_PRIORITY` order.

        When `cascade_fallback_enabled=False`, returns just the primary.

        Per-tier override (Phase 15+):
        When `<tier>_cascade` is set (e.g. FAST_CASCADE=cerebras,groq), it
        REPLACES the auto-derived chain — useful when you want a specific
        primary→fallback pairing (e.g. Cerebras Llama-3.3 → Groq Llama-3.3)
        and don't want some other configured provider sneaking into the
        middle of the chain just because it appears earlier in
        `_AUTO_PRIORITY`. Unknown / unconfigured providers in the override
        are silently dropped so a typo degrades gracefully.
        """
        primary = self.resolve_provider(tier)
        if not self.cascade_fallback_enabled:
            return [primary]

        # Per-tier override path. Honoured even when the user's primary
        # provider doesn't appear in the override list — we still prepend
        # the resolved primary so `<tier>_provider` and `<tier>_cascade`
        # don't silently fight each other.
        override = {
            ModelTier.FAST: self.fast_cascade,
            ModelTier.REASONING: self.reasoning_cascade,
            ModelTier.TOOL: self.tool_cascade,
        }.get(tier)
        if override:
            configured = set(self.configured_providers())
            seen: set[str] = set()
            chain_o: list[ProviderName] = []
            for p_raw in [primary, *override]:
                p = str(p_raw).strip().lower()
                if p in seen or p not in configured:
                    continue
                # Filter against the typed ProviderName universe (mypy-friendly,
                # also drops obvious garbage values from a typo'd env var).
                if p not in _AUTO_PRIORITY:
                    continue
                seen.add(p)
                chain_o.append(p)  # type: ignore[arg-type]
            if chain_o:
                return chain_o
            # Override produced nothing usable — fall through to auto-derive.

        chain: list[ProviderName] = [primary]
        candidates = list(_AUTO_PRIORITY)
        if tier == ModelTier.EMBED:
            candidates = [p for p in candidates if p in _EMBED_CAPABLE]

        configured = set(self.configured_providers())
        for p in candidates:
            if p == primary or p not in configured:
                continue
            chain.append(p)
        return chain


settings = Settings()
