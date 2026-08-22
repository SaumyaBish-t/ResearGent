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

        # ---- research_memory (Persistent Knowledge Graph — Phase 18) ------------
        # Tracks topics, entities, and knowledge gaps per user across all research.
        # Built automatically after each run; queried on planner startup.
        """
        CREATE TABLE IF NOT EXISTS research_memory_topics (
            id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            user_id       UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            label         TEXT NOT NULL,
            domain        TEXT,
            count         INT  NOT NULL DEFAULT 1,
            last_seen     TIMESTAMPTZ NOT NULL DEFAULT now(),
            created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE(user_id, label)
        );
        """,
        """
        CREATE INDEX IF NOT EXISTS memory_topics_user_idx
            ON research_memory_topics(user_id, last_seen DESC);
        """,
        """
        CREATE TABLE IF NOT EXISTS research_memory_entities (
            id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            user_id       UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            topic_id      UUID REFERENCES research_memory_topics(id) ON DELETE CASCADE,
            entity_name   TEXT NOT NULL,
            entity_type   TEXT DEFAULT 'concept',
            count         INT  NOT NULL DEFAULT 1,
            last_seen     TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        """,
        """
        CREATE INDEX IF NOT EXISTS memory_entities_topic_idx
            ON research_memory_entities(topic_id);
        """,
        """
        CREATE TABLE IF NOT EXISTS research_memory_relationships (
            id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            user_id           UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            source_topic_id   UUID NOT NULL REFERENCES research_memory_topics(id) ON DELETE CASCADE,
            target_topic_id   UUID NOT NULL REFERENCES research_memory_topics(id) ON DELETE CASCADE,
            relationship_type TEXT NOT NULL DEFAULT 'related',
            strength          REAL NOT NULL DEFAULT 0.5,
            last_seen         TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE(user_id, source_topic_id, target_topic_id, relationship_type)
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS research_memory_gaps (
            id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            user_id         UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            topic_id        UUID REFERENCES research_memory_topics(id) ON DELETE CASCADE,
            gap_description TEXT NOT NULL,
            status          TEXT NOT NULL DEFAULT 'open',
            created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
            resolved_at     TIMESTAMPTZ
        );
        """,
        """
        CREATE INDEX IF NOT EXISTS memory_gaps_open_idx
            ON research_memory_gaps(user_id, status)
            WHERE status = 'open';
        """,
        # ---- literature_reviews --------------------------------------------
        """
        CREATE TABLE IF NOT EXISTS literature_reviews (
            id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            user_id     UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            query       TEXT NOT NULL,
            title       TEXT NOT NULL,
            markdown    TEXT,
            duration_ms INT,
            created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        """,
        """
        CREATE INDEX IF NOT EXISTS literature_reviews_user_created_idx
            ON literature_reviews(user_id, created_at DESC);
        """,
        # ---- pgvector chunks (Phase 19) -------------------------------------
        """
        CREATE TABLE IF NOT EXISTS chunks (
            id          TEXT PRIMARY KEY,
            user_id     UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            collection  TEXT NOT NULL,
            embedding   VECTOR,
            document    TEXT NOT NULL,
            metadata    JSONB NOT NULL,
            created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        """,
        "CREATE INDEX IF NOT EXISTS chunks_user_collection_idx ON chunks (user_id, collection);",

        # ---- vault notes (Phase 19) ----------------------------------------
        """
        CREATE TABLE IF NOT EXISTS notes (
            id          TEXT PRIMARY KEY,
            user_id     UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            path        TEXT UNIQUE NOT NULL,
            frontmatter JSONB NOT NULL,
            content     TEXT NOT NULL,
            updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        """,
        "CREATE INDEX IF NOT EXISTS notes_user_idx ON notes (user_id);",

        # ---- retraction_status_cache (Phase 19: Provenance Check) --------------
        # One row per known DOI/arXiv-ID we've ever looked up or bulk-imported.
        # `status` values: "clean" | "retracted" | "corrected" | "concern" (EOC)
        # `reason` is Retraction Watch's free-text reason field when available
        # (e.g. "Data Fabrication", "Duplicate Publication") — used by the
        # severity-weighted confidence penalty (see 19.6 "future work" note).
        """
        CREATE TABLE IF NOT EXISTS retraction_status_cache (
            id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            doi             TEXT,
            arxiv_id        TEXT,
            status          TEXT NOT NULL DEFAULT 'clean',
            reason          TEXT,
            source          TEXT NOT NULL DEFAULT 'crossref',
            notice_url      TEXT,
            checked_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE(doi),
            UNIQUE(arxiv_id)
        );
        """,
        "CREATE INDEX IF NOT EXISTS retraction_status_doi_idx ON retraction_status_cache(doi);",
        "CREATE INDEX IF NOT EXISTS retraction_status_arxiv_idx ON retraction_status_cache(arxiv_id);",
        "CREATE INDEX IF NOT EXISTS retraction_status_checked_idx ON retraction_status_cache(checked_at);",

        # ---- cited_sources (Phase 23: Background Sweep target list) ------------
        # Tracks every (doi|arxiv_id) that has EVER been cited in a saved run,
        # across both the vault (markdown notes) and research_turns (Postgres).
        # The sweep job iterates this table, not the notes folder, so re-flagging
        # doesn't require re-parsing markdown frontmatter.
        """
        CREATE TABLE IF NOT EXISTS cited_sources (
            id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            doi             TEXT,
            arxiv_id        TEXT,
            first_cited_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
            last_cited_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
            cite_locations  JSONB NOT NULL DEFAULT '[]',
            UNIQUE(doi),
            UNIQUE(arxiv_id)
        );
        """,
        "CREATE INDEX IF NOT EXISTS cited_sources_doi_idx ON cited_sources(doi);",

        # ---- provenance_reflags (Phase 23: Background Sweep notifications) -----
        """
        CREATE TABLE IF NOT EXISTS provenance_reflags (
            id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            doi           TEXT,
            arxiv_id      TEXT,
            old_status    TEXT,
            new_status    TEXT NOT NULL,
            detected_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
            notified      BOOLEAN NOT NULL DEFAULT false
        );
        """,
        "CREATE INDEX IF NOT EXISTS provenance_reflags_notified_idx ON provenance_reflags(notified) WHERE notified = false;",
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
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            applied.append("vector extension")
            for stmt in _DDL:
                cur.execute(stmt)
                first_line = stmt.strip().splitlines()[0]
                applied.append(first_line)
    return applied
