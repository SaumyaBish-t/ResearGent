"""
Background Sweep — Phase 23.

Re-checks every previously-cited source's retraction status. Anything that
flipped from clean -> retracted/corrected/concern since it was cited gets:
  1. Its retraction_status_cache row updated
  2. A one-line addendum appended to every vault note that cited it
     (never rewrites the note body — append-only, timestamped)
  3. A row in a new `provenance_reflags` table so the API/frontend can
     show "N sources changed status since you cited them" as a dashboard
     notification instead of the user having to open every old note.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.provenance.checker import check_many
from src.db import connection


async def run_sweep(since_days: int | None = None, dry_run: bool = False) -> dict:
    sources = _load_cited_sources(since_days)
    identifiers = [(s["doi"], s["arxiv_id"]) for s in sources]
    results = await check_many(identifiers)

    changed: list[dict] = []
    for s in sources:
        key = (s["doi"], s["arxiv_id"])
        new_status = results.get(key, {}).get("status", "clean")
        if new_status != s.get("last_known_status", "clean"):
            changed.append({**s, "new_status": new_status})
            if not dry_run:
                _append_addendum_to_notes(s["cite_locations"], new_status)
                _record_reflag(s, new_status)

    return {"checked": len(sources), "changed": len(changed), "details": changed}


def _load_cited_sources(since_days: int | None = None) -> list[dict]:
    """Load all rows from cited_sources, optionally filtered by last_cited_at."""
    with connection() as conn:
        with conn.cursor() as cur:
            if since_days:
                cutoff = datetime.now(timezone.utc) - timedelta(days=since_days)
                cur.execute(
                    """
                    SELECT doi, arxiv_id, last_cited_at, cite_locations
                    FROM cited_sources
                    WHERE last_cited_at >= %s
                    """,
                    (cutoff,),
                )
            else:
                cur.execute(
                    """
                    SELECT doi, arxiv_id, last_cited_at, cite_locations
                    FROM cited_sources
                    """
                )
            rows = cur.fetchall()
            return [
                {
                    "doi": r[0],
                    "arxiv_id": r[1],
                    "last_cited_at": r[2],
                    "last_known_status": _get_cached_status(r[0], r[1]),
                    "cite_locations": r[3] or [],
                }
                for r in rows
            ]


def _get_cached_status(doi: str, arxiv_id: str) -> str:
    """Get current status from retraction_status_cache."""
    with connection() as conn:
        with conn.cursor() as cur:
            if doi:
                cur.execute("SELECT status FROM retraction_status_cache WHERE doi = %s", (doi,))
                row = cur.fetchone()
                if row:
                    return row[0]
            if arxiv_id:
                cur.execute("SELECT status FROM retraction_status_cache WHERE arxiv_id = %s", (arxiv_id,))
                row = cur.fetchone()
                if row:
                    return row[0]
    return "clean"


def _append_addendum_to_notes(cite_locations: list[dict], new_status: str) -> None:
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    addendum = (
        f"\n\n---\n**⚠ Provenance update ({ts}):** a source cited in this "
        f"note has changed status to **{new_status}** since this note was "
        f"saved. Verify before relying on this answer.\n"
    )
    for loc in cite_locations:
        path = Path(loc["note_path"])
        if path.exists():
            with open(path, "a", encoding="utf-8") as f:
                f.write(addendum)


def _record_reflag(source: dict, new_status: str) -> None:
    """Insert a row into provenance_reflags for the frontend notification."""
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO provenance_reflags (doi, arxiv_id, old_status, new_status, detected_at)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (source["doi"], source["arxiv_id"], source["last_known_status"], new_status, datetime.now(timezone.utc)),
            )