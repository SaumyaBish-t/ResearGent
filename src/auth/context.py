from __future__ import annotations

import contextvars
from typing import Optional

# Thread-local / task-local context variable to hold the active user ID.
# Automatically propagates through async task loops and executor pools.
current_user_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("current_user_id", default=None)

_cli_user_id_cache: Optional[str] = None


def get_cli_user_id() -> str:
    """
    Resolve a default user ID for CLI runs and local tests.
    Uses the first user in the database, or inserts a default local CLI user.
    """
    global _cli_user_id_cache
    if _cli_user_id_cache is not None:
        return _cli_user_id_cache

    from src.db import connection
    try:
        with connection() as conn:
            with conn.cursor() as cur:
                # Find the first existing user
                cur.execute("SELECT id FROM users ORDER BY created_at LIMIT 1;")
                row = cur.fetchone()
                if row:
                    _cli_user_id_cache = str(row["id"])
                    return _cli_user_id_cache

                # No user exists — create a default CLI system user
                cur.execute(
                    """
                    INSERT INTO users (email, name, is_admin)
                    VALUES ('cli@local', 'CLI System User', true)
                    ON CONFLICT (email) DO UPDATE SET email = EXCLUDED.email
                    RETURNING id;
                    """
                )
                row = cur.fetchone()
                if row:
                    _cli_user_id_cache = str(row["id"])
                    return _cli_user_id_cache
    except Exception:
        pass

    # Safe fallback UUID if the database isn't fully set up or reachable yet
    return "00000000-0000-0000-0000-000000000000"


def get_current_user_id() -> str:
    """
    Get the user ID for the current context.
    Falls back to the default CLI user ID when run outside a web request context.
    """
    uid = current_user_id.get()
    if uid:
        return uid
    return get_cli_user_id()
