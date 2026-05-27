"""
Unified LLM provider interface.

Design rationale
----------------
NVIDIA NIM, Groq, and Ollama all expose **OpenAI-compatible** REST APIs.
That means a single `openai` client works for all three — we only swap
`base_url` + `api_key` per provider. No per-provider client classes needed.

Public API
----------
    chat(messages, tier=ModelTier.REASONING)   -> str
    embed(texts, tier=ModelTier.EMBED)         -> list[list[float]]
    get_client(provider)                       -> OpenAI client (escape hatch)
    list_status()                              -> dict for the smoke test

Every later phase calls `chat()` / `embed()` and never touches provider details.
That's what makes "swap from cloud to local" a 0-line code change.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

from openai import OpenAI

from src.config import ModelTier, ProviderName, settings


# ---------------------------------------------------------------------------
# Provider descriptors — model name + endpoint per (provider, tier).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProviderConfig:
    name: ProviderName
    base_url: str
    api_key: str  # "ollama" is fine as a dummy — local server ignores it
    models: dict[ModelTier, str | None]


def _provider_configs() -> dict[ProviderName, ProviderConfig]:
    s = settings
    return {
        "cerebras": ProviderConfig(
            name="cerebras",
            base_url=s.cerebras_base_url,
            api_key=s.cerebras_api_key or "",
            models={
                ModelTier.REASONING: s.cerebras_model_reasoning,
                ModelTier.FAST: s.cerebras_model_fast,
                ModelTier.EMBED: None,  # Cerebras does not host embeddings.
            },
        ),
        "nvidia": ProviderConfig(
            name="nvidia",
            base_url=s.nvidia_base_url,
            api_key=s.nvidia_api_key or "",
            models={
                ModelTier.REASONING: s.nvidia_model_reasoning,
                ModelTier.FAST: s.nvidia_model_fast,
                ModelTier.EMBED: s.nvidia_model_embed,
            },
        ),
        "groq": ProviderConfig(
            name="groq",
            base_url=s.groq_base_url,
            api_key=s.groq_api_key or "",
            models={
                ModelTier.REASONING: s.groq_model_reasoning,
                ModelTier.FAST: s.groq_model_fast,
                ModelTier.EMBED: None,  # Groq does not host embeddings.
            },
        ),
        "openrouter": ProviderConfig(
            name="openrouter",
            base_url=s.openrouter_base_url,
            api_key=s.openrouter_api_key or "",
            models={
                ModelTier.REASONING: s.openrouter_model_reasoning,
                ModelTier.FAST: s.openrouter_model_fast,
                ModelTier.EMBED: s.openrouter_model_embed,
            },
        ),
        "ollama": ProviderConfig(
            name="ollama",
            base_url=s.ollama_base_url,
            api_key="ollama",  # placeholder; Ollama ignores auth headers
            models={
                ModelTier.REASONING: s.ollama_model_reasoning,
                ModelTier.FAST: s.ollama_model_fast,
                ModelTier.EMBED: s.ollama_model_embed,
            },
        ),
    }


@lru_cache(maxsize=8)
def get_client(provider: ProviderName) -> OpenAI:
    """Cached OpenAI client per provider. Cheap to call repeatedly."""
    cfg = _provider_configs()[provider]
    if not cfg.api_key:
        raise RuntimeError(
            f"Provider '{provider}' is not configured. "
            f"Set the corresponding API key in .env"
        )

    # OpenRouter recommends two headers for app attribution on its dashboard.
    # They're optional — only sent if the user set them in .env.
    default_headers = None
    if provider == "openrouter":
        default_headers = {
            "HTTP-Referer": settings.openrouter_app_url,
            "X-Title": settings.openrouter_app_name,
        }

    return OpenAI(
        base_url=cfg.base_url,
        api_key=cfg.api_key,
        default_headers=default_headers,
    )


def _resolve(tier: ModelTier) -> tuple[ProviderName, str]:
    """Return (provider, model_id) for a given tier."""
    provider = settings.resolve_provider(tier)
    model = _provider_configs()[provider].models[tier]
    if model is None:
        raise RuntimeError(
            f"Provider '{provider}' has no model assigned for tier '{tier.value}'."
        )
    return provider, model


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def chat(
    messages: list[dict[str, str]],
    *,
    tier: ModelTier = ModelTier.REASONING,
    temperature: float = 0.2,
    max_tokens: int | None = 1024,
    **kwargs,
) -> str:
    """
    Send a chat request to whichever provider+model serves this tier.

    Returns the assistant message content as a plain string. For streaming or
    raw responses, drop down to `get_client(provider).chat.completions.create(...)`.
    """
    provider, model = _resolve(tier)
    client = get_client(provider)
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    )
    return resp.choices[0].message.content or ""


def embed(
    texts: Iterable[str],
    *,
    tier: ModelTier = ModelTier.EMBED,
) -> list[list[float]]:
    """
    Embed a batch of texts. Used heavily from Phase 1 onward.

    Note: NVIDIA NIM embedding endpoints require an `input_type` extra field
    ("query" vs "passage"). We default to "passage" — the retrieval module
    will override per-call when querying.
    """
    provider, model = _resolve(tier)
    client = get_client(provider)
    texts = list(texts)

    extra_body = {}
    if provider == "nvidia":
        extra_body = {"input_type": "passage", "truncate": "END"}

    resp = client.embeddings.create(
        model=model,
        input=texts,
        extra_body=extra_body or None,
    )
    return [d.embedding for d in resp.data]


def list_status() -> dict[str, dict]:
    """
    Diagnostic snapshot for the smoke test.

    Tells you for each tier: which provider was picked, which model, and whether
    its credentials are configured. Does NOT make network calls.
    """
    cfgs = _provider_configs()
    configured = set(settings.configured_providers())

    out: dict[str, dict] = {"providers": {}, "routing": {}}

    for name, cfg in cfgs.items():
        out["providers"][name] = {
            "configured": name in configured and bool(cfg.api_key),
            "base_url": cfg.base_url,
            "models": {t.value: m for t, m in cfg.models.items()},
        }

    for tier in ModelTier:
        try:
            provider, model = _resolve(tier)
            out["routing"][tier.value] = {"provider": provider, "model": model}
        except Exception as e:
            out["routing"][tier.value] = {"error": str(e)}

    return out
