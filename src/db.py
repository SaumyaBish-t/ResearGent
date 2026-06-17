"""
PostgreSQL connection pool + LangGraph PostgresSaver factory.

Single shared `psycopg_pool.ConnectionPool` per process. Everything that
needs Postgres — the agent checkpointer, the documents_registry writer, the
TTL pruner — pulls from this pool. Centralising it here means:

  * one set of credentials, one place to tune pool size
  * `checkpointer.setup()` runs exactly once per process via _setup_done
  * tests can monkeypatch `get_pool()` to swap in a throwaway DB

Why `autocommit=True` + `row_factory=dict_row`?
  PostgresSaver.setup() issues DDL in its own transaction. If the pool
  hands it a connection inside an outer transaction, the DDL silently
  no-ops on rollback and the next run can't find the tables. autocommit
  fixes that. dict_row is what the saver expects for its row parsing —
  passing tuples back makes it crash with a KeyError on first read.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from src.config import settings

_pool = None        # psycopg_pool.ConnectionPool — created lazily
_setup_done = False # PostgresSaver.setup() is idempotent but not free


def _build_pool():
    """Construct the shared pool on first use. Raises if PG isn't configured."""
    from psycopg_pool import ConnectionPool
    from psycopg.rows import dict_row

    url = settings.resolve_database_url()
    if not url:
        raise RuntimeError(
            "Postgres is not configured. Set DATABASE_URL in .env, or set "
            "POSTGRES_HOST/POSTGRES_DB/POSTGRES_USER/POSTGRES_PASSWORD."
        )

    # `kwargs` are applied to every connection the pool hands out. This is
    # the documented way to set autocommit + row_factory for PostgresSaver.
    #
    # `check` runs a cheap SELECT 1 before returning a pooled connection so
    # we recycle anything Neon (or any managed PG) silently dropped during
    # an idle-suspend. Without this, the first query after a scale-to-zero
    # crashes with `psycopg.errors.AdminShutdown` and the user sees a 500
    # on what should have been a successful sign-in.
    #
    # `max_idle=240` closes idle conns at 4 min — under Neon free tier's
    # 5-min idle suspend — so the pool sheds connections proactively
    # instead of waiting for the server to slam them.
    return ConnectionPool(
        conninfo=url,
        min_size=settings.postgres_pool_min_size,
        max_size=settings.postgres_pool_max_size,
        kwargs={"autocommit": True, "row_factory": dict_row, "prepare_threshold": 0},
        check=ConnectionPool.check_connection,
        max_idle=240,
        open=True,
    )


def get_pool():
    """Return the process-wide pool, creating it on first call."""
    global _pool
    if _pool is None:
        _pool = _build_pool()
    return _pool


@contextmanager
def connection() -> Iterator:
    """`with connection() as conn:` — yields a pooled psycopg Connection."""
    pool = get_pool()
    with pool.connection() as conn:
        yield conn


def get_checkpointer(*, setup: bool = False):
    """
    Build a LangGraph PostgresSaver bound to the shared pool.

    Pass `setup=True` exactly once per environment (the `db init` command
    does this) to create the checkpoints / checkpoint_writes / checkpoint_blobs
    tables. The call is safe to repeat — PostgresSaver.setup() uses
    `CREATE TABLE IF NOT EXISTS` and a migration version row — but we still
    gate it with `_setup_done` so normal runs don't pay the round trips.
    """
    from langgraph.checkpoint.postgres import PostgresSaver

    global _setup_done
    saver = PostgresSaver(get_pool())
    if setup and not _setup_done:
        saver.setup()
        _setup_done = True
    return saver


def close_pool() -> None:
    """Tear the pool down. Call on shutdown; tests use it between cases."""
    global _pool, _setup_done
    if _pool is not None:
        _pool.close()
        _pool = None
        _setup_done = False
