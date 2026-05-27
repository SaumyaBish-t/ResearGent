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


ProviderName = Literal["nvidia", "groq", "ollama"]


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
        if self.nvidia_api_key:
            out.append("nvidia")
        if self.groq_api_key:
            out.append("groq")
        # Ollama is considered available — we'll check the server in the smoke test.
        out.append("ollama")
        return out

    def resolve_provider(self, tier: ModelTier) -> ProviderName:
        """
        Pick the provider for a given tier.

        Resolution order:
          1. Per-tier override (REASONING_PROVIDER, FAST_PROVIDER, EMBED_PROVIDER)
          2. Global override (PRIMARY_PROVIDER)
          3. Auto: first configured provider in priority order [nvidia, groq, ollama]

        Special case for EMBED: Groq doesn't host embeddings — skip it.
        """
        tier_override = {
            ModelTier.REASONING: self.reasoning_provider,
            ModelTier.FAST: self.fast_provider,
            ModelTier.EMBED: self.embed_provider,
        }[tier]

        if tier_override:
            return tier_override
        if self.primary_provider:
            if tier == ModelTier.EMBED and self.primary_provider == "groq":
                # Groq can't embed — fall through to auto-pick.
                pass
            else:
                return self.primary_provider

        priority: list[ProviderName] = ["nvidia", "groq", "ollama"]
        if tier == ModelTier.EMBED:
            priority = ["nvidia", "ollama"]  # skip groq

        configured = set(self.configured_providers())
        for p in priority:
            if p in configured:
                return p

        # Should be unreachable — Ollama is always in configured_providers.
        raise RuntimeError("No providers available. Configure at least one in .env")


settings = Settings()
