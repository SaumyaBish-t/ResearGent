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


# Errors that mean "this provider can't serve this request — try the next".
#
# Includes more than just classic transient errors (429/5xx/timeout) because
# the cascade is also a resilience layer for *configuration* mismatches:
#   - NotFoundError (404)     -> wrong model ID for THIS provider; next one
#                                might have the model under a different name
#   - PermissionDeniedError   -> account doesn't have access to model on this
#                                provider; next provider might
#
# We DELIBERATELY keep BadRequestError + AuthenticationError as fatal — those
# are bugs in our code or fully-broken keys that no fallback can fix.
_TRANSIENT_ERRORS = (
    openai.RateLimitError,
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.InternalServerError,
    openai.NotFoundError,
    openai.PermissionDeniedError,
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


@lru_cache(maxsize=32)
def _client(
    base_url: str,
    api_key: str,
    referer: str | None = None,
    title: str | None = None,
) -> OpenAI:
    """
    Build (and cache) an OpenAI client for ANY OpenAI-compatible endpoint.

    timeout=45s + max_retries=0: each attempt fails fast so OUR resolver
    (not the SDK's hidden 2 in-client retries) owns fallback. Without
    max_retries=0 a hung/rate-limited endpoint is retried 3× (~135s) before
    the error surfaces — turning a multi-step chain into a multi-minute hang.
    """
    headers: dict[str, str] | None = None
    if referer or title:
        headers = {}
        if referer:
            headers["HTTP-Referer"] = referer
        if title:
            headers["X-Title"] = title
    return OpenAI(
        base_url=base_url,
        api_key=api_key,
        default_headers=headers,
        timeout=45.0,
        max_retries=0,
    )


def get_client(provider: ProviderName) -> OpenAI:
    """
    OpenAI client for a configured provider slot. Escape hatch used by the
    naive retriever; tier resolution uses `_resolve_steps` instead.
    """
    cfg = _provider_configs()[provider]
    if not cfg.api_key:
        raise RuntimeError(
            f"Provider '{provider}' is not configured. Set the API key in .env"
        )
    if provider == "openrouter":
        return _client(
            cfg.base_url, cfg.api_key, settings.openrouter_app_url, settings.openrouter_app_name
        )
    return _client(cfg.base_url, cfg.api_key)


# ---------------------------------------------------------------------------
# Cascade resolution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Step:
    """One attempt in a tier's resolution chain."""
    label: str       # observability/status label (e.g. "fast(direct)" or "groq")
    base_url: str
    api_key: str
    model: str
    kind: str        # "openai" | "ollama" | "nvidia"  (embed-path specialisation)


def _kind_for(base_url: str, provider: ProviderName | None = None) -> str:
    b = (base_url or "").lower()
    if provider == "ollama" or "11434" in b or "ollama" in b:
        return "ollama"
    if provider == "nvidia" or "nvidia" in b:
        return "nvidia"
    return "openai"


def _resolve_steps(tier: ModelTier) -> list[_Step]:
    """
    Ordered attempts for a tier:
      1) tier-direct BYOM endpoint  — if REASONING_/FAST_/TOOL_/EMBED_ *
         (API_KEY + BASE_URL + MODEL) are set. Takes precedence.
      2) configured provider slots  — cascade fallback, each contributing its
         model for this tier (skipped if it has none / isn't configured).
    """
    steps: list[_Step] = []

    direct = settings.tier_direct(tier)
    if direct:
        key, base, model = direct
        steps.append(_Step(f"{tier.value}(direct)", base, key, model, _kind_for(base)))

    cfgs = _provider_configs()
    configured = set(settings.configured_providers())
    for provider in settings.resolve_cascade(tier):
        if provider not in configured:
            continue
        model = cfgs[provider].models.get(tier)
        if not model:
            continue
        cfg = cfgs[provider]
        steps.append(
            _Step(provider, cfg.base_url, cfg.api_key or "ollama", model, _kind_for(cfg.base_url, provider))
        )

    if not steps:
        T = tier.value.upper()
        raise RuntimeError(
            f"No model configured for the '{tier.value}' tier. Set {T}_API_KEY, "
            f"{T}_BASE_URL and {T}_MODEL in .env (or configure a provider slot "
            f"with a {T} model)."
        )
    return steps


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
    steps = _resolve_steps(tier)
    last_exc: Exception | None = None

    for i, step in enumerate(steps):
        try:
            with obs.track("chat", tier, step.label, step.model, cascade_step=i) as ctx:
                referer = settings.openrouter_app_url if "openrouter" in step.base_url else None
                title = settings.openrouter_app_name if "openrouter" in step.base_url else None
                client = _client(step.base_url, step.api_key, referer, title)
                resp = client.chat.completions.create(
                    model=step.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs,
                )
                ctx["usage"] = resp.usage
                return resp.choices[0].message.content or ""
        except _TRANSIENT_ERRORS as e:
            last_exc = e
            continue  # try next step in the chain
        except Exception:
            # Non-transient (BadRequest, Auth, config error) — fail fast.
            raise

    assert last_exc is not None
    raise last_exc


def _embed_ollama_native(model: str, texts: list[str], base_url: str | None = None) -> list[list[float]]:
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
    base = (base_url or settings.ollama_base_url).rstrip("/v1").rstrip("/")

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
    """Embed a batch with fallback. Ollama uses its native endpoint; NVIDIA
    needs the `input_type` extra_body."""
    steps = _resolve_steps(tier)
    texts_list = list(texts)
    last_exc: Exception | None = None

    for i, step in enumerate(steps):
        try:
            with obs.track("embed", tier, step.label, step.model, cascade_step=i) as ctx:
                if step.kind == "ollama":
                    # Native endpoint — more reliable than the OpenAI-compat shim.
                    # 120s timeout covers cold model loads on first request.
                    return _embed_ollama_native(step.model, texts_list, step.base_url)

                client = _client(step.base_url, step.api_key)
                extra_body = (
                    {"input_type": "passage", "truncate": "END"}
                    if step.kind == "nvidia"
                    else None
                )
                resp = client.embeddings.create(
                    model=step.model, input=texts_list, extra_body=extra_body
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
            steps = _resolve_steps(tier)
            out["routing"][tier.value] = {"provider": steps[0].label, "model": steps[0].model}
            out["cascade"][tier.value] = [f"{s.label}:{s.model}" for s in steps]
        except Exception as e:
            out["routing"][tier.value] = {"error": str(e)}
            out["cascade"][tier.value] = []

    return out
