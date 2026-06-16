"""
Application schema migrations (auth + billing + research history).

Run via `researgent db migrate`. Idempotent — every statement is guarded with
`IF NOT EXISTS` so re-running is safe. Distinct from `researgent db init`,
which sets up LangGraph's checkpoint tables via `PostgresSaver.setup()`.

Schema overview
---------------
  users                — one row per signed-in Google account
  subscriptions        — Razorpay subscription state per user
  research_threads     — one row per "research" (original Q + follow-ups)
  research_turns       — one row per Q+A turn inside a thread

Quota math (enforced at the API layer, not in SQL):
  Free tier:
    - 3 threads created per calendar month
    - 3 turns per thread (turn_index 0..2)  -> original + 2 follow-ups
  Subscribed users + admins: unlimited.
"""

from __future__ import annotations

from src.db import connection


# pgcrypto provides gen_random_uuid(); built into PG ≥13 but only via this ext.
_ENABLE_PGCRYPTO = "CREATE EXTENSION IF NOT EXISTS pgcrypto;"


_DDL: list[str] = [
    # ---- users -------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS users (
        id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        email       TEXT UNIQUE NOT NULL,
        name        TEXT,
        picture     TEXT,
        google_sub  TEXT UNIQUE,
        is_admin    BOOLEAN NOT NULL DEFAULT false,
        created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
    );
    """,
    "CREATE INDEX IF NOT EXISTS users_email_idx ON users(email);",

    # ---- subscriptions -----------------------------------------------------
    # One canonical "current sub" per user. Webhook updates `status` +
    # `current_period_end`. `is_active()` helper checks both.
    """
    CREATE TABLE IF NOT EXISTS subscriptions (
        id                        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        user_id                   UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        razorpay_subscription_id  TEXT UNIQUE,
        razorpay_customer_id      TEXT,
        status                    TEXT NOT NULL,
        current_period_end        TIMESTAMPTZ,
        created_at                TIMESTAMPTZ NOT NULL DEFAULT now(),
        updated_at                TIMESTAMPTZ NOT NULL DEFAULT now()
    );
    """,
    "CREATE INDEX IF NOT EXISTS subscriptions_user_idx ON subscriptions(user_id);",
    "CREATE INDEX IF NOT EXISTS subscriptions_status_idx ON subscriptions(status);",

    # ---- research_threads --------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS research_threads (
        id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        user_id     UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        title       TEXT NOT NULL,
        created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
    );
    """,
    """
    CREATE INDEX IF NOT EXISTS research_threads_user_created_idx
        ON research_threads(user_id, created_at DESC);
    """,

    # ---- research_turns ----------------------------------------------------
    # turn_index is 0-based and unique per thread — enforces "3 turns max" at
    # the DB level too (the app should reject before insert anyway).
    """
    CREATE TABLE IF NOT EXISTS research_turns (
        id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        thread_id     UUID NOT NULL REFERENCES research_threads(id) ON DELETE CASCADE,
        turn_index    INT  NOT NULL,
        question      TEXT NOT NULL,
        answer        TEXT,
        confidence    TEXT,
        score         REAL,
        sources_json  JSONB,
        run_id        TEXT,
        created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
    );
    """,
    """
    CREATE UNIQUE INDEX IF NOT EXISTS research_turns_thread_idx_uniq
        ON research_turns(thread_id, turn_index);
    """,
    """
    CREATE INDEX IF NOT EXISTS research_turns_thread_created_idx
        ON research_turns(thread_id, created_at);
    """,
]


def run_migrations() -> list[str]:
    """
    Apply every DDL statement in order. Returns the list of statements that
    were executed (for the CLI to print). Safe to re-run.
    """
    applied: list[str] = []
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute(_ENABLE_PGCRYPTO)
            applied.append("pgcrypto extension")
            for stmt in _DDL:
                cur.execute(stmt)
                first_line = stmt.strip().splitlines()[0]
                applied.append(first_line)
    return applied
