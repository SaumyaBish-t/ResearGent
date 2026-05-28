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

    # ---- NVIDIA NIM ---------------------------------------------------------
    nvidia_api_key: str | None = None
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    nvidia_model_reasoning: str = "meta/llama-3.3-70b-instruct"
    nvidia_model_fast: str = "meta/llama-3.1-8b-instruct"
    nvidia_model_tool: str = "meta/llama-3.3-70b-instruct"
    nvidia_model_embed: str = "nvidia/nv-embed-v1"

    # ---- Groq ---------------------------------------------------------------
    # GPT-OSS-120B on Groq is the best free-tier tool-caller available right now.
    groq_api_key: str | None = None
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_model_reasoning: str = "llama-3.3-70b-versatile"
    groq_model_fast: str = "llama-3.1-8b-instant"
    groq_model_tool: str = "openai/gpt-oss-120b"

    # ---- Cerebras -----------------------------------------------------------
    cerebras_api_key: str | None = None
    cerebras_base_url: str = "https://api.cerebras.ai/v1"
    # Cerebras free-tier roster (May 2026): gpt-oss-120b + zai-glm-4.7.
    # Both at 5 RPM — tight for inner-loop agents, fine for burst / fallback.
    #   - zai-glm-4.7        ZhipuAI GLM-4.7, strong reasoning, different
    #                        architecture family from rest of stack (useful
    #                        for cascade diversity). Preview tier.
    #   - gpt-oss-120b       OpenAI open weights MoE. Production tier.
    #                        Excellent tool use. Also available on Groq.
    cerebras_model_reasoning: str = "zai-glm-4.7"
    cerebras_model_fast: str = "gpt-oss-120b"
    cerebras_model_tool: str = "gpt-oss-120b"

    # ---- OpenRouter ---------------------------------------------------------
    openrouter_api_key: str | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model_reasoning: str = "deepseek/deepseek-r1:free"
    openrouter_model_fast: str = "meta-llama/llama-3.1-8b-instruct:free"
    openrouter_model_tool: str = "qwen/qwen3-235b-a22b:free"
    openrouter_model_embed: str = "nvidia/nv-embed-v1"
    openrouter_app_url: str = "https://github.com/SaumyaBish-t/ResearGent"
    openrouter_app_name: str = "ResearGent"

    # ---- Ollama (local) -----------------------------------------------------
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_model_reasoning: str = "llama3.1:8b"
    ollama_model_fast: str = "llama3.1:8b"
    ollama_model_tool: str = "llama3.1:8b"
    ollama_model_embed: str = "nomic-embed-text"

    # ---- Routing ------------------------------------------------------------
    primary_provider: ProviderName | None = Field(
        default=None,
        description="Force a provider for ALL tiers. Overrides auto-detection.",
    )
    reasoning_provider: ProviderName | None = None
    fast_provider: ProviderName | None = None
    tool_provider: ProviderName | None = None
    embed_provider: ProviderName | None = None

    # ---- Cascade fallback ---------------------------------------------------
    # When True, transient failures (429 / 5xx / timeout) on the primary
    # provider for a tier automatically retry on the next configured provider.
    # Set False to surface raw errors (useful for debugging).
    cascade_fallback_enabled: bool = True

    # ---- Observability ------------------------------------------------------
    # Logs every chat()/embed() call to a JSONL file. Surface via `researgent stats`.
    observability_enabled: bool = True
    observability_log_path: str = "data/llm_calls.jsonl"

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

    # ---- Helpers ------------------------------------------------------------
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
        """
        primary = self.resolve_provider(tier)
        if not self.cascade_fallback_enabled:
            return [primary]

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
