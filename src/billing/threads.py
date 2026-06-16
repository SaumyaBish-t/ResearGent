"""
Research thread + turn persistence.

A "thread" is one research session: original question + up to N follow-ups.
A "turn" is one Q+A inside a thread (turn_index 0 = original).

`create_thread()` returns a fresh thread_id; `add_turn()` writes the Q+A after
the agent run completes. List/get helpers power the history sidebar.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

from src.db import connection


@dataclass
class Thread:
    id: str
    user_id: str
    title: str
    created_at: datetime


@dataclass
class Turn:
    id: str
    thread_id: str
    turn_index: int
    question: str
    answer: Optional[str]
    confidence: Optional[str]
    score: Optional[float]
    sources: list[dict]
    run_id: Optional[str]
    created_at: datetime


def _row_to_thread(row: dict) -> Thread:
    return Thread(
        id=str(row["id"]),
        user_id=str(row["user_id"]),
        title=row["title"],
        created_at=row["created_at"],
    )


def _row_to_turn(row: dict) -> Turn:
    raw_sources = row.get("sources_json") or []
    if isinstance(raw_sources, str):
        # psycopg sometimes returns JSONB as already-decoded dicts and sometimes
        # as raw strings depending on adapter setup. Normalise.
        raw_sources = json.loads(raw_sources)
    return Turn(
        id=str(row["id"]),
        thread_id=str(row["thread_id"]),
        turn_index=int(row["turn_index"]),
        question=row["question"],
        answer=row.get("answer"),
        confidence=row.get("confidence"),
        score=row.get("score"),
        sources=list(raw_sources),
        run_id=row.get("run_id"),
        created_at=row["created_at"],
    )


def create_thread(*, user_id: str, title: str) -> Thread:
    """Create a new thread for `user_id`. `title` is a short summary of the original Q."""
    # Trim title — DB col is TEXT but the UI shows it as a single line.
    title = (title or "").strip().replace("\n", " ")[:160] or "untitled"
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO research_threads (user_id, title)
            VALUES (%s, %s)
            RETURNING id, user_id, title, created_at
            """,
            (user_id, title),
        )
        row = cur.fetchone()
    return _row_to_thread(row)


def get_thread(*, thread_id: str, user_id: str) -> Optional[Thread]:
    """Fetch a thread by id, scoped to its owner (returns None if not theirs)."""
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT id, user_id, title, created_at FROM research_threads "
            "WHERE id = %s AND user_id = %s",
            (thread_id, user_id),
        )
        row = cur.fetchone()
    return _row_to_thread(row) if row else None


def list_threads(*, user_id: str, limit: int = 50) -> list[Thread]:
    """User's threads, newest first."""
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, user_id, title, created_at
            FROM research_threads
            WHERE user_id = %s
            ORDER BY created_at DESC
            LIMIT %s
            """,
            (user_id, limit),
        )
        rows = cur.fetchall()
    return [_row_to_thread(r) for r in rows]


def list_turns(*, thread_id: str) -> list[Turn]:
    """All turns in a thread, ordered by turn_index."""
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, thread_id, turn_index, question, answer, confidence,
                   score, sources_json, run_id, created_at
            FROM research_turns
            WHERE thread_id = %s
            ORDER BY turn_index ASC
            """,
            (thread_id,),
        )
        rows = cur.fetchall()
    return [_row_to_turn(r) for r in rows]


def count_threads_this_month(*, user_id: str) -> int:
    """How many threads `user_id` created since the start of the current calendar month."""
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT count(*) AS n FROM research_threads
            WHERE user_id = %s
              AND created_at >= date_trunc('month', now())
            """,
            (user_id,),
        )
        return int(cur.fetchone()["n"])


def count_turns(*, thread_id: str) -> int:
    """Number of turns persisted in a thread."""
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) AS n FROM research_turns WHERE thread_id = %s",
            (thread_id,),
        )
        return int(cur.fetchone()["n"])


def next_turn_index(*, thread_id: str) -> int:
    """Returns the index the NEXT turn should claim (0-based)."""
    return count_turns(thread_id=thread_id)


def add_turn(
    *,
    thread_id: str,
    turn_index: int,
    question: str,
    answer: Optional[str],
    confidence: Optional[str],
    score: Optional[float],
    sources: list[dict],
    run_id: Optional[str],
) -> Turn:
    """
    Persist a turn after the agent run completes. `turn_index` must equal
    `next_turn_index(thread_id)` at call time — the UNIQUE index on
    (thread_id, turn_index) will surface concurrent-insert races as IntegrityErrors.
    """
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO research_turns (
                thread_id, turn_index, question, answer,
                confidence, score, sources_json, run_id
            ) VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s)
            RETURNING id, thread_id, turn_index, question, answer,
                      confidence, score, sources_json, run_id, created_at
            """,
            (
                thread_id,
                turn_index,
                question,
                answer,
                confidence,
                score,
                json.dumps(sources or []),
                run_id,
            ),
        )
        row = cur.fetchone()
    return _row_to_turn(row)


def build_context_prefix(turns: list[Turn], max_chars: int = 4000) -> str:
    """
    Render prior turns as a context preamble the planner can read inline.

    Kept simple on purpose: a single human-readable block prepended to the
    new question, no changes to the agent state schema. The Critic still
    grades only the NEW question's retrievals against the new question.

    `max_chars` budget — older / longer answers are truncated tail-first
    so the most recent context survives.
    """
    if not turns:
        return ""
    parts: list[str] = []
    for t in turns:
        ans = (t.answer or "").strip()
        # Cap each prior answer; full text lives in the DB if anyone needs it.
        if len(ans) > 800:
            ans = ans[:800].rsplit(" ", 1)[0] + " …"
        parts.append(f"[Prior Q{t.turn_index + 1}]\n{t.question}\n[Prior A{t.turn_index + 1}]\n{ans}")
    block = "\n\n".join(parts)
    if len(block) > max_chars:
        block = block[-max_chars:]
    return block
