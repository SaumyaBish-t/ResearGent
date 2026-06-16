"""
User CRUD — backed by Postgres `users` table.

`upsert_from_google()` is the one entrypoint OAuth calls after verifying the
Google id_token. It's keyed on `google_sub` (Google's stable user id) — using
the email instead would silently merge accounts if a user changed their
primary Google email.

Admin flag flips on every sign-in if the email matches `settings.admin_email_set`,
so adding a new admin in .env takes effect on the next login (no manual SQL).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.config import settings
from src.db import connection


@dataclass
class User:
    id: str
    email: str
    name: Optional[str]
    picture: Optional[str]
    is_admin: bool


def _row_to_user(row: dict) -> User:
    return User(
        id=str(row["id"]),
        email=row["email"],
        name=row.get("name"),
        picture=row.get("picture"),
        is_admin=bool(row.get("is_admin", False)),
    )


def get_by_id(user_id: str) -> Optional[User]:
    """Fetch user by primary key, or None."""
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT id, email, name, picture, is_admin FROM users WHERE id = %s",
            (user_id,),
        )
        row = cur.fetchone()
    return _row_to_user(row) if row else None


def upsert_from_google(
    *,
    google_sub: str,
    email: str,
    name: Optional[str],
    picture: Optional[str],
) -> User:
    """
    Insert-or-update a user from a verified Google id_token.

    On conflict (existing google_sub):
      - refreshes email/name/picture from Google (they may have changed)
      - re-evaluates is_admin against settings.admin_email_set
    """
    is_admin = email.lower() in settings.admin_email_set

    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO users (email, name, picture, google_sub, is_admin)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (google_sub) DO UPDATE SET
                email    = EXCLUDED.email,
                name     = EXCLUDED.name,
                picture  = EXCLUDED.picture,
                is_admin = EXCLUDED.is_admin
            RETURNING id, email, name, picture, is_admin
            """,
            (email, name, picture, google_sub, is_admin),
        )
        row = cur.fetchone()
    return _row_to_user(row)
