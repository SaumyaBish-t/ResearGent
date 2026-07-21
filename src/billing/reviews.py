"""
Literature review database persistence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

from src.db import connection


@dataclass
class LiteratureReview:
    id: str
    user_id: str
    query: str
    title: str
    markdown: Optional[str]
    duration_ms: Optional[int]
    created_at: datetime


def _row_to_review(row: dict) -> LiteratureReview:
    return LiteratureReview(
        id=str(row["id"]),
        user_id=str(row["user_id"]),
        query=row["query"],
        title=row["title"],
        markdown=row.get("markdown"),
        duration_ms=row.get("duration_ms"),
        created_at=row["created_at"],
    )


def create_review(
    *,
    user_id: str,
    query: str,
    title: str,
    markdown: Optional[str] = None,
    duration_ms: Optional[int] = None,
) -> LiteratureReview:
    """Create and persist a literature review."""
    title = (title or "").strip().replace("\n", " ")[:160] or "untitled"
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO literature_reviews (user_id, query, title, markdown, duration_ms)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id, user_id, query, title, markdown, duration_ms, created_at
            """,
            (user_id, query, title, markdown, duration_ms),
        )
        row = cur.fetchone()
    return _row_to_review(row)


def get_review(*, review_id: str, user_id: str) -> Optional[LiteratureReview]:
    """Fetch a review by id, scoped to its owner."""
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, user_id, query, title, markdown, duration_ms, created_at
            FROM literature_reviews
            WHERE id = %s AND user_id = %s
            """,
            (review_id, user_id),
        )
        row = cur.fetchone()
    return _row_to_review(row) if row else None


def list_reviews(*, user_id: str, limit: int = 50) -> list[LiteratureReview]:
    """User's literature reviews, newest first."""
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, user_id, query, title, markdown, duration_ms, created_at
            FROM literature_reviews
            WHERE user_id = %s
            ORDER BY created_at DESC
            LIMIT %s
            """,
            (user_id, limit),
        )
        rows = cur.fetchall()
    return [_row_to_review(r) for r in rows]


def count_reviews_by_user(*, user_id: str) -> int:
    """How many literature reviews a user has generated."""
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) AS n FROM literature_reviews WHERE user_id = %s",
            (user_id,),
        )
        return int(cur.fetchone()["n"])
