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
    EMBED     — embedding model for retrieval. Used by ingestion + retriever.
    """

    REASONING = "reasoning"
    FAST = "fast"
    EMBED = "embed"


# All supported providers. Each maps to an OpenAI-compatible HTTP endpoint —
# that's what makes adding new ones cheap (no new SDK to learn).
ProviderName = Literal["cerebras", "nvidia", "groq", "openrouter", "ollama"]

# Priority order for AUTO-selection when no override is set. Engineered so the
# default user experience is "as fast and as smart as the keys allow":
#   - cerebras first  -> 1000+ TPS, huge models (Qwen3-235B)
#   - nvidia next     -> widest model catalog, has embeddings
#   - groq next       -> 315 TPS for the fast tier
#   - openrouter      -> variety / experimentation
#   - ollama last     -> local fallback when nothing else is configured
_AUTO_PRIORITY: list[ProviderName] = ["cerebras", "nvidia", "groq", "openrouter", "ollama"]

# Providers that host embedding models. Keeps `resolve_provider` honest for the
# EMBED tier so we don't try to embed with Groq/Cerebras (neither hosts embedders).
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
    nvidia_model_embed: str = "nvidia/nv-embed-v1"

    # ---- Groq ---------------------------------------------------------------
    groq_api_key: str | None = None
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_model_reasoning: str = "llama-3.3-70b-versatile"
    groq_model_fast: str = "llama-3.1-8b-instant"

    # ---- Cerebras -----------------------------------------------------------
    # Cerebras serves models on wafer-scale chips at 1000+ TPS — the fastest
    # commercial inference available. Perfect for agent loops where many
    # sequential calls compound into long wall-clock times.
    cerebras_api_key: str | None = None
    cerebras_base_url: str = "https://api.cerebras.ai/v1"
    cerebras_model_reasoning: str = "qwen-3-235b-a22b-instruct-2507"
    cerebras_model_fast: str = "llama3.1-8b"

    # ---- OpenRouter ---------------------------------------------------------
    # Single OpenAI-compatible endpoint fronting 200+ models. Free-tier models
    # carry the ":free" suffix. Sends optional HTTP-Referer + X-Title headers
    # so you appear on the OpenRouter dashboard (set them in .env if you want).
    openrouter_api_key: str | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model_reasoning: str = "deepseek/deepseek-r1:free"
    openrouter_model_fast: str = "meta-llama/llama-3.1-8b-instruct:free"
    openrouter_model_embed: str = "nvidia/nv-embed-v1"  # OpenRouter proxies NVIDIA's embedder
    openrouter_app_url: str = "https://github.com/SaumyaBish-t/ResearGent"
    openrouter_app_name: str = "ResearGent"

    # ---- Ollama (local) -----------------------------------------------------
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_model_reasoning: str = "llama3.1:8b"
    ollama_model_fast: str = "llama3.1:8b"
    ollama_model_embed: str = "nomic-embed-text"

    # ---- Routing ------------------------------------------------------------
    primary_provider: ProviderName | None = Field(
        default=None,
        description="Force a provider for ALL tiers. Overrides auto-detection.",
    )
    reasoning_provider: ProviderName | None = None
    fast_provider: ProviderName | None = None
    embed_provider: ProviderName | None = None

    # ---- Validators ---------------------------------------------------------
    # .env files commonly leave keys blank (`PRIMARY_PROVIDER=`). Pydantic's
    # strict Literal validator rejects "" — coerce blanks to None so blank
    # entries behave as "unset".
    @field_validator(
        "primary_provider",
        "reasoning_provider",
        "fast_provider",
        "embed_provider",
        mode="before",
    )
    @classmethod
    def _blank_to_none(cls, v):
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
        # Ollama is considered available — we'll check the server at call time.
        out.append("ollama")
        return out

    def resolve_provider(self, tier: ModelTier) -> ProviderName:
        """
        Pick the provider for a given tier.

        Resolution order:
          1. Per-tier override (REASONING_PROVIDER, FAST_PROVIDER, EMBED_PROVIDER)
          2. Global override (PRIMARY_PROVIDER) — skipped for EMBED if the
             chosen provider can't embed
          3. Auto: first configured provider in `_AUTO_PRIORITY` that supports
             the tier (EMBED filters to `_EMBED_CAPABLE`)
        """
        tier_override = {
            ModelTier.REASONING: self.reasoning_provider,
            ModelTier.FAST: self.fast_provider,
            ModelTier.EMBED: self.embed_provider,
        }[tier]

        if tier_override:
            return tier_override

        if self.primary_provider:
            if tier == ModelTier.EMBED and self.primary_provider not in _EMBED_CAPABLE:
                pass  # fall through to auto-pick a real embedder
            else:
                return self.primary_provider

        priority = list(_AUTO_PRIORITY)
        if tier == ModelTier.EMBED:
            priority = [p for p in priority if p in _EMBED_CAPABLE]

        configured = set(self.configured_providers())
        for p in priority:
            if p in configured:
                return p

        # Should be unreachable — Ollama is always in configured_providers.
        raise RuntimeError("No providers available. Configure at least one in .env")


settings = Settings()
