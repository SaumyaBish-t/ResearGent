"""
FastAPI dependencies for protecting routes.

Usage:
    @app.get("/api/protected")
    def view(user: User = Depends(current_user)):  # 401 if not signed in
        return {"hello": user.email}

    @app.get("/api/maybe")
    def view(user: User | None = Depends(current_user_optional)):  # never raises
        ...
"""

from __future__ import annotations

from typing import Optional

from fastapi import Depends, HTTPException, Request, status

from src.auth.session import read_session
from src.auth.users import User, get_by_id


def current_user_optional(request: Request) -> Optional[User]:
    """Return the signed-in user, or None — never raises."""
    user_id = read_session(request)
    if not user_id:
        return None
    return get_by_id(user_id)


def current_user(
    user: Optional[User] = Depends(current_user_optional),
) -> User:
    """Return the signed-in user, or raise 401."""
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Sign in required.",
        )
    return user
