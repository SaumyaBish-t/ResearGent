"""
Tests for the Provenance Background Sweep + retraction cache (Phase 23/24).

Why these tests exist
---------------------
`src/db.py` builds the shared pool with `row_factory=dict_row`, so EVERY
row that comes back from `src.db.connection()` is a DICT. But the sweep
and the retraction cache index those rows as TUPLES (`row[0]`, `r[3]`).
On a dict, `row[0]` raises KeyError — so the sweep crashed on its very
first row, and `get_cached()` crashed whenever a fresh cache row existed.

These tests pin the row-access contract and the sweep's cache-bypass
behaviour, using a FAKE cursor so they run without a database.

Run:
    PYTHONPATH= .venv/Scripts/python.exe -m pytest tests/test_provenance_sweep.py -v
"""

from __future__ import annotations

import asyncio
import sys
from contextlib import contextmanager

sys.path.insert(0, ".")

from src.db import row_get
from src.provenance import cache, checker, sweep


# ---------------------------------------------------------------------------
# Fake DB — routes rows by a substring of the SQL.
# ---------------------------------------------------------------------------


class FakeCursor:
    def __init__(self, db: "FakeDB") -> None:
        self.db = db
        self._rows: list[dict] = []

    def execute(self, sql: str, params=None) -> None:
        self.db.queries.append(sql)
        self._rows = []
        for key, rows in self.db.results.items():
            if key in sql:
                self._rows = list(rows)
                break

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self) -> list[dict]:
        return list(self._rows)

    def __enter__(self) -> "FakeCursor":
        return self

    def __exit__(self, *exc) -> None:
        return None


class FakeConn:
    def __init__(self, db: "FakeDB") -> None:
        self.db = db

    def cursor(self) -> FakeCursor:
        return FakeCursor(self.db)


class FakeDB:
    """A stand-in for `src.db.connection()` that yields dict rows."""

    def __init__(self, results: dict[str, list[dict]] | None = None) -> None:
        self.results = results or {}
        self.queries: list[str] = []

    @contextmanager
    def connection(self):
        yield FakeConn(self)


# ---------------------------------------------------------------------------
# row_get — the shared, row-factory-agnostic accessor.
# ---------------------------------------------------------------------------


def test_row_get_reads_dict_row():
    assert row_get({"status": "clean"}, "status") == "clean"


def test_row_get_reads_tuple_row():
    assert row_get(("clean", "", "", "crossref"), "status", 0) == "clean"


def test_row_get_returns_none_for_missing_key():
    assert row_get({"a": 1}, "b") is None


def test_row_get_returns_none_for_none_row():
    assert row_get(None, "status") is None


def test_row_get_on_dict_never_raises_keyerror():
    """The exact bug: `row[0]` on a dict_row raised KeyError."""
    row = {"doi": "10.1/x", "status": "retracted"}
    assert row_get(row, "doi", 0) == "10.1/x"


# ---------------------------------------------------------------------------
# Sweep row access — the crash site.
# ---------------------------------------------------------------------------


def test_load_cited_sources_reads_dict_rows(monkeypatch):
    db = FakeDB(
        {
            "FROM cited_sources": [
                {
                    "doi": "10.1234/a",
                    "arxiv_id": None,
                    "last_cited_at": None,
                    "cite_locations": [{"note_path": "n.md", "run_id": "r1"}],
                }
            ],
            "FROM retraction_status_cache": [{"status": "clean"}],
        }
    )
    monkeypatch.setattr(sweep, "connection", db.connection)

    rows = sweep._load_cited_sources()

    assert len(rows) == 1
    assert rows[0]["doi"] == "10.1234/a"
    assert rows[0]["last_known_status"] == "clean"
    assert rows[0]["cite_locations"] == [{"note_path": "n.md", "run_id": "r1"}]


def test_get_cached_status_reads_dict_rows(monkeypatch):
    db = FakeDB(
        {
            "FROM retraction_status_cache": [
                {"status": "retracted", "reason": "fraud", "notice_url": "u"}
            ]
        }
    )
    monkeypatch.setattr(sweep, "connection", db.connection)

    assert sweep._get_cached_status("10.1234/a", "") == "retracted"


# ---------------------------------------------------------------------------
# Retraction cache row access.
# ---------------------------------------------------------------------------


def test_get_cached_reads_dict_rows(monkeypatch):
    db = FakeDB(
        {
            "FROM retraction_status_cache": [
                {
                    "status": "clean",
                    "reason": "",
                    "notice_url": "",
                    "source": "crossref",
                    "checked_at": None,
                }
            ]
        }
    )
    monkeypatch.setattr(cache, "connection", db.connection)

    out = cache.get_cached(doi="10.1234/a")

    assert out is not None
    assert out["status"] == "clean"
    assert out["source"] == "crossref"


# ---------------------------------------------------------------------------
# Sweep must BYPASS the cache — otherwise it can never detect a change.
# ---------------------------------------------------------------------------


def test_check_many_uses_cache_by_default(monkeypatch):
    """Default behaviour stays cache-first (cheap per-run provenance)."""
    lookups: list[str] = []

    monkeypatch.setattr(
        checker,
        "get_cached",
        lambda **kw: {"status": "clean", "reason": "", "notice_url": ""},
    )

    async def fake_lookup_doi(doi: str):
        lookups.append(doi)
        return {"status": "retracted", "reason": "x", "notice_url": "u"}

    monkeypatch.setattr(checker, "lookup_doi", fake_lookup_doi)
    monkeypatch.setattr(checker, "upsert_status", lambda **kw: None)

    out = asyncio.run(checker.check_many([("10.1234/a", "")]))

    assert lookups == []
    assert out[("10.1234/a", "")]["status"] == "clean"


def test_check_many_force_refresh_bypasses_cache(monkeypatch):
    """`force_refresh=True` must re-query live, ignoring the cached status.

    Without this, the Background Sweep is a no-op: it reads the same
    cached row the per-run check wrote and can never observe a status
    flip — the exact failure the sweep exists to catch.
    """
    lookups: list[str] = []

    monkeypatch.setattr(
        checker,
        "get_cached",
        lambda **kw: {"status": "clean", "reason": "", "notice_url": ""},
    )

    async def fake_lookup_doi(doi: str):
        lookups.append(doi)
        return {"status": "retracted", "reason": "data fabrication", "notice_url": "u"}

    monkeypatch.setattr(checker, "lookup_doi", fake_lookup_doi)
    monkeypatch.setattr(checker, "upsert_status", lambda **kw: None)

    out = asyncio.run(checker.check_many([("10.1234/a", "")], force_refresh=True))

    assert lookups == ["10.1234/a"]
    assert out[("10.1234/a", "")]["status"] == "retracted"


def test_sweep_forces_a_live_recheck(monkeypatch):
    """run_sweep must ask for a live re-check, not a cached one."""
    seen: dict[str, bool] = {}

    async def fake_check_many(identifiers, force_refresh: bool = False):
        seen["force_refresh"] = force_refresh
        return {}

    monkeypatch.setattr(sweep, "check_many", fake_check_many)
    monkeypatch.setattr(sweep, "_load_cited_sources", lambda since_days=None: [])

    asyncio.run(sweep.run_sweep())

    assert seen.get("force_refresh") is True
