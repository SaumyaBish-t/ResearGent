"""
Unified LLM provider interface — with cascade fallback + observability.

Design rationale
----------------
NVIDIA NIM, Groq, Cerebras, OpenRouter, and Ollama all expose **OpenAI-compatible**
REST APIs. A single `openai` client works for all five — we only swap
`base_url` + `api_key` per provider.

Cascade fallback
----------------
For each tier we build an ordered chain of (provider, model) pairs from
`settings.resolve_cascade(tier)`. `chat()` walks the chain on TRANSIENT
failures (rate limits, 5xx, timeouts) — never on logical errors (400 bad
request, schema mismatches). This makes the system genuinely robust to:
  - free-tier RPM ceilings being hit mid-loop
  - provider partial outages
  - one provider's model returning empty / garbage output

Observability
-------------
Every call is wrapped in `observability.track()` so the JSONL log captures
duration, tokens, success, cascade depth. The `researgent stats` command
reads this back.

Public API
----------
    chat(messages, tier=...)             -> str
    embed(texts, tier=...)               -> list[list[float]]
    get_client(provider)                 -> OpenAI client (escape hatch)
    list_status()                        -> dict for the smoke test
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

import openai
from openai import OpenAI

from src.config import ModelTier, ProviderName, settings
from src.llm import observability as obs


# Errors that mean "transient — try the next provider in the cascade".
# We DELIBERATELY do not retry BadRequestError / AuthenticationError — those
# are bugs in our code or stale keys, not problems the next provider can fix.
_TRANSIENT_ERRORS = (
    openai.RateLimitError,
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.InternalServerError,
)


# ---------------------------------------------------------------------------
# Provider descriptors
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProviderConfig:
    name: ProviderName
    base_url: str
    api_key: str
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
                ModelTier.TOOL: s.cerebras_model_tool,
                ModelTier.EMBED: None,
            },
        ),
        "nvidia": ProviderConfig(
            name="nvidia",
            base_url=s.nvidia_base_url,
            api_key=s.nvidia_api_key or "",
            models={
                ModelTier.REASONING: s.nvidia_model_reasoning,
                ModelTier.FAST: s.nvidia_model_fast,
                ModelTier.TOOL: s.nvidia_model_tool,
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
                ModelTier.TOOL: s.groq_model_tool,
                ModelTier.EMBED: None,
            },
        ),
        "openrouter": ProviderConfig(
            name="openrouter",
            base_url=s.openrouter_base_url,
            api_key=s.openrouter_api_key or "",
            models={
                ModelTier.REASONING: s.openrouter_model_reasoning,
                ModelTier.FAST: s.openrouter_model_fast,
                ModelTier.TOOL: s.openrouter_model_tool,
                ModelTier.EMBED: s.openrouter_model_embed,
            },
        ),
        "ollama": ProviderConfig(
            name="ollama",
            base_url=s.ollama_base_url,
            api_key="ollama",
            models={
                ModelTier.REASONING: s.ollama_model_reasoning,
                ModelTier.FAST: s.ollama_model_fast,
                ModelTier.TOOL: s.ollama_model_tool,
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
            f"Provider '{provider}' is not configured. Set the API key in .env"
        )
    default_headers = None
    if provider == "openrouter":
        default_headers = {
            "HTTP-Referer": settings.openrouter_app_url,
            "X-Title": settings.openrouter_app_name,
        }
    # 45s timeout: long enough for slow embedders / 70B models on busy free
    # tiers, short enough to fail fast instead of hanging indefinitely.
    # Cascade fallback catches the resulting APITimeoutError and rolls to
    # the next provider in the chain.
    return OpenAI(
        base_url=cfg.base_url,
        api_key=cfg.api_key,
        default_headers=default_headers,
        timeout=45.0,
    )


# ---------------------------------------------------------------------------
# Cascade resolution
# ---------------------------------------------------------------------------


def _cascade_for(tier: ModelTier) -> list[tuple[ProviderName, str]]:
    """
    Build the [(provider, model), ...] chain for a tier.

    Filters out providers that don't have a model defined for this tier
    (e.g. Groq lacks an EMBED model) or aren't configured (no API key).
    """
    cfgs = _provider_configs()
    configured = set(settings.configured_providers())
    out: list[tuple[ProviderName, str]] = []
    for provider in settings.resolve_cascade(tier):
        if provider not in configured:
            continue
        model = cfgs[provider].models.get(tier)
        if not model:
            continue
        out.append((provider, model))
    if not out:
        raise RuntimeError(
            f"No usable providers for tier '{tier.value}'. Check .env keys + model assignments."
        )
    return out


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
    Send a chat request. Walks the cascade chain on transient failures.

    Raises the LAST encountered exception only if every provider in the chain
    fails — partial failures are silently retried (and logged via observability).
    """
    chain = _cascade_for(tier)
    last_exc: Exception | None = None

    for step, (provider, model) in enumerate(chain):
        try:
            with obs.track("chat", tier, provider, model, cascade_step=step) as ctx:
                client = get_client(provider)
                resp = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs,
                )
                ctx["usage"] = resp.usage
                return resp.choices[0].message.content or ""
        except _TRANSIENT_ERRORS as e:
            last_exc = e
            continue  # try next provider
        except Exception:
            # Non-transient (BadRequest, Auth, config error) — fail fast.
            raise

    assert last_exc is not None
    raise last_exc


def _embed_ollama_native(model: str, texts: list[str]) -> list[list[float]]:
    """
    Embed via Ollama's NATIVE API.

    Ollama exposes two endpoints depending on version:
      - /api/embed         (since v0.4, Sep 2024) — batched, `input: [...]`
      - /api/embeddings    (older, singular)      — one string at a time, `prompt: "..."`

    The /v1/embeddings OpenAI-compat shim has known quirks (silent hangs,
    empty responses on first request). Going native is more reliable.

    Strategy: try /api/embed first (batched, fast). On 404/405 fall back to
    /api/embeddings looped per input. On other errors, surface them with the
    response body so the failure is debuggable.
    """
    import httpx
    base = settings.ollama_base_url.rstrip("/v1").rstrip("/")

    # ---- Try batched /api/embed first ----
    try:
        r = httpx.post(
            f"{base}/api/embed",
            json={"model": model, "input": texts},
            timeout=120.0,
        )
        if r.status_code in (404, 405):
            raise _BatchedEndpointMissing()
        if r.status_code >= 400:
            # Bubble up the body so the user can see what Ollama actually said.
            raise RuntimeError(
                f"Ollama /api/embed returned {r.status_code}: {r.text[:300]}"
            )
        data = r.json()
        emb = data.get("embeddings") or ([data["embedding"]] if "embedding" in data else None)
        if not emb:
            raise RuntimeError(f"Ollama /api/embed returned no embeddings: {data}")
        return emb
    except _BatchedEndpointMissing:
        pass  # fall through to legacy endpoint

    # ---- Fallback: legacy /api/embeddings, one call per input ----
    out: list[list[float]] = []
    for text in texts:
        r = httpx.post(
            f"{base}/api/embeddings",
            json={"model": model, "prompt": text},
            timeout=120.0,
        )
        if r.status_code >= 400:
            raise RuntimeError(
                f"Ollama /api/embeddings returned {r.status_code}: {r.text[:300]}"
            )
        data = r.json()
        vec = data.get("embedding")
        if not vec:
            raise RuntimeError(f"Ollama /api/embeddings returned no embedding: {data}")
        out.append(vec)
    return out


class _BatchedEndpointMissing(Exception):
    """Signal that /api/embed is unavailable so we fall back to /api/embeddings."""


def embed(
    texts: Iterable[str],
    *,
    tier: ModelTier = ModelTier.EMBED,
) -> list[list[float]]:
    """Embed a batch with cascade fallback. NVIDIA needs `input_type` extra_body."""
    chain = _cascade_for(tier)
    texts_list = list(texts)
    last_exc: Exception | None = None

    for step, (provider, model) in enumerate(chain):
        try:
            with obs.track("embed", tier, provider, model, cascade_step=step) as ctx:
                if provider == "ollama":
                    # Native endpoint — more reliable than the OpenAI-compat shim.
                    # 120s timeout covers cold model loads on first request.
                    return _embed_ollama_native(model, texts_list)

                client = get_client(provider)
                extra_body = (
                    {"input_type": "passage", "truncate": "END"}
                    if provider == "nvidia"
                    else None
                )
                resp = client.embeddings.create(
                    model=model, input=texts_list, extra_body=extra_body
                )
                ctx["usage"] = getattr(resp, "usage", None)
                return [d.embedding for d in resp.data]
        except _TRANSIENT_ERRORS as e:
            last_exc = e
            continue
        except Exception:
            raise

    assert last_exc is not None
    raise last_exc


def list_status() -> dict[str, dict]:
    """
    Diagnostic snapshot for the smoke test.

    Tells you for each tier: the resolved primary, the full cascade chain,
    and per-provider configuration. Does NOT make network calls.
    """
    cfgs = _provider_configs()
    configured = set(settings.configured_providers())

    out: dict[str, dict] = {"providers": {}, "routing": {}, "cascade": {}}

    for name, cfg in cfgs.items():
        out["providers"][name] = {
            "configured": name in configured and bool(cfg.api_key),
            "base_url": cfg.base_url,
            "models": {t.value: m for t, m in cfg.models.items()},
        }

    for tier in ModelTier:
        try:
            chain = _cascade_for(tier)
            out["routing"][tier.value] = {"provider": chain[0][0], "model": chain[0][1]}
            out["cascade"][tier.value] = [f"{p}:{m}" for p, m in chain]
        except Exception as e:
            out["routing"][tier.value] = {"error": str(e)}
            out["cascade"][tier.value] = []

    return out
