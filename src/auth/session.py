"""
Session cookie = signed JWT.

Why JWT and not server-side sessions?
  - Stateless — no extra DB round-trip per request to look up a session row.
  - The only thing we put in the cookie is the user_id (UUID) + an exp claim.
  - HS256 with `settings.session_secret` so the same backend reads its own
    cookie. Asymmetric keys would be overkill for a single-service app.

Security
  - HttpOnly  — JS can't read it (XSS-resistant).
  - SameSite=Lax — browser sends it on top-level GETs from any origin (so the
    Google OAuth redirect back to us works), but not on cross-site POSTs.
  - Secure   — flipped on in production via `COOKIE_SECURE=true`.
  - Short-ish TTL (default 14d) — rotated on every login.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

import jwt
from fastapi import Request, Response

from src.config import settings


def _now() -> datetime:
    return datetime.now(timezone.utc)


def make_token(user_id: str) -> str:
    """Sign a session JWT for `user_id`. Returns the encoded string."""
    payload = {
        "sub": str(user_id),
        "iat": int(_now().timestamp()),
        "exp": int((_now() + timedelta(hours=settings.session_ttl_hours)).timestamp()),
    }
    return jwt.encode(payload, settings.session_secret, algorithm="HS256")


def decode_token(token: str) -> Optional[str]:
    """
    Return the user_id from a valid token, or None on any failure.

    We swallow every exception (expired, malformed, bad signature) and return
    None so the caller treats them uniformly as "not logged in".
    """
    try:
        payload = jwt.decode(token, settings.session_secret, algorithms=["HS256"])
        sub = payload.get("sub")
        return str(sub) if sub else None
    except Exception:
        return None


def set_session_cookie(response: Response, user_id: str) -> None:
    """Attach the session cookie to `response`."""
    response.set_cookie(
        key=settings.session_cookie_name,
        value=make_token(user_id),
        max_age=settings.session_ttl_hours * 3600,
        httponly=True,
        secure=settings.cookie_secure,
        samesite="lax",
        path="/",
    )


def clear_session_cookie(response: Response) -> None:
    """Logout — overwrite with an empty, immediately-expired cookie."""
    response.delete_cookie(
        key=settings.session_cookie_name,
        path="/",
        httponly=True,
        secure=settings.cookie_secure,
        samesite="lax",
    )


def read_session(request: Request) -> Optional[str]:
    """Pull the user_id out of the request's session cookie, or None."""
    token = request.cookies.get(settings.session_cookie_name)
    if not token:
        return None
    return decode_token(token)
