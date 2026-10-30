"""
RED phase — cascade fallback must survive a dead local provider.

Ollama is the configured EMBED_PROVIDER but the daemon is not running.
`_embed_ollama_native` calls `httpx.post` directly, so a connection refusal
raises `httpx.ConnectError` — which is NOT in `src.llm.provider._TRANSIENT_ERRORS`.
The `except Exception: raise` arm therefore kills the whole call instead of
walking the cascade to the next configured provider (NVIDIA / OpenRouter).

Expected behaviour: a dead local endpoint is a *transient* condition and the
cascade should continue to the next provider.
"""

from __future__ import annotations

import httpx
import openai

from src.llm import provider


def test_httpx_connection_errors_are_treated_as_transient():
    """A refused connection to one provider must not abort the cascade."""
    assert issubclass(httpx.ConnectError, provider._TRANSIENT_ERRORS), (
        "httpx.ConnectError must be classified as transient so the cascade "
        "falls through to the next configured provider"
    )


def test_httpx_timeout_is_treated_as_transient():
    assert issubclass(httpx.ReadTimeout, provider._TRANSIENT_ERRORS)


def test_openai_connection_error_is_treated_as_transient():
    assert issubclass(openai.APIConnectionError, provider._TRANSIENT_ERRORS)


def test_bad_request_stays_fatal():
    """A 400 is a code bug, not a provider outage — must NOT cascade."""
    assert not issubclass(openai.BadRequestError, provider._TRANSIENT_ERRORS)


def test_auth_error_stays_fatal():
    assert not issubclass(openai.AuthenticationError, provider._TRANSIENT_ERRORS)
