from __future__ import annotations

from datetime import datetime, timedelta, timezone
from src.db import connection
from src.config import settings


def upsert_status(
    *, doi: str = "", arxiv_id: str = "", status: str,
    reason: str = "", notice_url: str = "", source: str
) -> None:
    """INSERT ... ON CONFLICT (doi) DO UPDATE, mirror the UNIQUE constraints
    from the migration. Handle doi="" and arxiv_id="" cases separately
    since both columns are UNIQUE-but-nullable."""
    if not doi and not arxiv_id:
        return

    now = datetime.now(timezone.utc)
    with connection() as conn:
        with conn.cursor() as cur:
            if doi:
                cur.execute(
                    """
                    INSERT INTO retraction_status_cache (doi, status, reason, notice_url, source, checked_at)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (doi) DO UPDATE SET
                        status = EXCLUDED.status,
                        reason = EXCLUDED.reason,
                        notice_url = EXCLUDED.notice_url,
                        source = EXCLUDED.source,
                        checked_at = EXCLUDED.checked_at;
                    """,
                    (doi, status, reason, notice_url, source, now),
                )
            if arxiv_id:
                cur.execute(
                    """
                    INSERT INTO retraction_status_cache (arxiv_id, status, reason, notice_url, source, checked_at)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (arxiv_id) DO UPDATE SET
                        status = EXCLUDED.status,
                        reason = EXCLUDED.reason,
                        notice_url = EXCLUDED.notice_url,
                        source = EXCLUDED.source,
                        checked_at = EXCLUDED.checked_at;
                    """,
                    (arxiv_id, status, reason, notice_url, source, now),
                )


def get_cached(*, doi: str = "", arxiv_id: str = "") -> dict | None:
    """SELECT ... WHERE doi = %s OR arxiv_id = %s, respect
    settings.retraction_cache_ttl_hours — return None if stale"""
    if not doi and not arxiv_id:
        return None

    ttl_hours = settings.retraction_cache_ttl_hours
    cutoff = datetime.now(timezone.utc) - timedelta(hours=ttl_hours)

    with connection() as conn:
        with conn.cursor() as cur:
            if doi:
                cur.execute(
                    """
                    SELECT status, reason, notice_url, source, checked_at
                    FROM retraction_status_cache
                    WHERE doi = %s AND checked_at >= %s
                    """,
                    (doi, cutoff),
                )
                row = cur.fetchone()
                if row:
                    return {"status": row[0], "reason": row[1], "notice_url": row[2], "source": row[3]}

            if arxiv_id:
                cur.execute(
                    """
                    SELECT status, reason, notice_url, source, checked_at
                    FROM retraction_status_cache
                    WHERE arxiv_id = %s AND checked_at >= %s
                    """,
                    (arxiv_id, cutoff),
                )
                row = cur.fetchone()
                if row:
                    return {"status": row[0], "reason": row[1], "notice_url": row[2], "source": row[3]}

    return None