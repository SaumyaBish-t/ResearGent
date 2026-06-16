"""
Google OAuth + session management routes.

Mounted at `/auth/*`. Uses authlib's Starlette integration, which stores the
OAuth state + nonce in the Starlette session (a separate, short-lived cookie
from our own application session JWT).

Endpoints
---------
  GET  /auth/google     — start OAuth: 302 to Google's consent screen
  GET  /auth/callback   — Google redirects here after consent; we exchange the
                          code, verify the id_token, upsert the user, set our
                          session cookie, then 302 back to the frontend
  GET  /auth/me         — JSON snapshot of the signed-in user, or 401
  POST /auth/logout     — clear the session cookie
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import JSONResponse, RedirectResponse

from src.auth.deps import current_user
from src.auth.session import clear_session_cookie, set_session_cookie
from src.auth.users import User, upsert_from_google
from src.config import settings


router = APIRouter(prefix="/auth", tags=["auth"])


def _build_oauth():
    """
    Lazily construct authlib's OAuth registry. Done at call time (not import)
    so the app boots cleanly when Google credentials aren't configured yet —
    the rest of ResearGent should still work for the local CLI.
    """
    from authlib.integrations.starlette_client import OAuth  # type: ignore

    if not settings.google_client_id or not settings.google_client_secret:
        return None

    oauth = OAuth()
    oauth.register(
        name="google",
        client_id=settings.google_client_id,
        client_secret=settings.google_client_secret,
        # OpenID Connect discovery document — saves wiring up token + userinfo
        # URLs by hand. authlib reads jwks_uri from here for id_token verify.
        server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
        client_kwargs={"scope": "openid email profile"},
    )
    return oauth


@router.get("/google")
async def login_google(request: Request):
    """Begin the OAuth dance — 302 to Google's consent screen."""
    oauth = _build_oauth()
    if not oauth:
        raise HTTPException(
            500,
            "Google OAuth not configured. Set GOOGLE_CLIENT_ID + GOOGLE_CLIENT_SECRET in .env.",
        )
    return await oauth.google.authorize_redirect(request, settings.google_redirect_uri)


@router.get("/callback")
async def callback(request: Request):
    """
    Google redirects here. Exchange the code, verify the id_token, upsert
    the user, set our session cookie, then 302 back to the frontend root.
    """
    oauth = _build_oauth()
    if not oauth:
        raise HTTPException(500, "Google OAuth not configured.")

    try:
        token = await oauth.google.authorize_access_token(request)
    except Exception as e:
        # Common causes: state mismatch, expired code, mis-set redirect URI.
        raise HTTPException(400, f"OAuth callback failed: {type(e).__name__}: {e}")

    # `userinfo` is the verified id_token claims. Trust these because authlib
    # already validated the signature against Google's published JWKS.
    info = token.get("userinfo") or {}
    sub = info.get("sub")
    email = info.get("email")
    if not sub or not email:
        raise HTTPException(400, "Google id_token missing sub/email — refusing to sign in.")
    if not info.get("email_verified", True):
        raise HTTPException(400, "Google email not verified.")

    user = upsert_from_google(
        google_sub=sub,
        email=email,
        name=info.get("name"),
        picture=info.get("picture"),
    )

    # Send the user back to the frontend root with our session cookie attached.
    #
    # Pick the CORS origin whose hostname matches THIS request's hostname so
    # the cookie travels (browsers treat `localhost` and `127.0.0.1` as
    # different sites — SameSite=Lax blocks the cookie across that boundary).
    # Falls back to the first configured origin if no match.
    from urllib.parse import urlsplit

    req_host = (request.url.hostname or "").lower()
    origins = [o for o in settings.cors_origins_list if o and o != "*"]
    frontend = next(
        (o for o in origins if urlsplit(o).hostname == req_host),
        origins[0] if origins else "http://127.0.0.1:3000",
    )
    resp = RedirectResponse(url=frontend, status_code=302)
    set_session_cookie(resp, user.id)
    return resp


@router.get("/me")
async def me(user: User = Depends(current_user)):
    """Return the signed-in user as JSON. 401 if no session."""
    return {
        "id": user.id,
        "email": user.email,
        "name": user.name,
        "picture": user.picture,
        "is_admin": user.is_admin,
    }


@router.post("/logout")
async def logout(response: Response):
    """Clear the session cookie. Body is empty `{}`."""
    clear_session_cookie(response)
    return JSONResponse({"ok": True})
