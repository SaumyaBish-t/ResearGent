"""CRUD operations on the research memory tables (Postgres)."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Optional

from src.db import connection
from src.memory.models import MemoryEntity, MemoryRelationship, MemoryTopic, KnowledgeGap


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ── Topics ──────────────────────────────────────────────────────────────────────


def upsert_topic(*, user_id: str, label: str, domain: str | None = None) -> str:
    """Increment count + update last_seen for a topic, or insert it. Returns id."""
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO research_memory_topics (user_id, label, domain, count, last_seen)
            VALUES (%s, %s, %s, 1, now())
            ON CONFLICT (user_id, label)
            DO UPDATE SET count = research_memory_topics.count + 1,
                          last_seen = now(),
                          domain = COALESCE(NULLIF(%s, ''), research_memory_topics.domain)
            RETURNING id
            """,
            (user_id, label, domain, domain),
        )
        return str(cur.fetchone()["id"])


def get_user_topics(*, user_id: str, limit: int = 50) -> list[MemoryTopic]:
    """User's topics, most recently researched first."""
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT id, user_id, label, domain, count, last_seen, created_at "
            "FROM research_memory_topics WHERE user_id = %s "
            "ORDER BY last_seen DESC LIMIT %s",
            (user_id, limit),
        )
        return [_row_to_topic(r) for r in cur.fetchall()]


def get_topic_by_label(*, user_id: str, label: str) -> MemoryTopic | None:
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT id, user_id, label, domain, count, last_seen, created_at "
            "FROM research_memory_topics WHERE user_id = %s AND label = %s",
            (user_id, label),
        )
        row = cur.fetchone()
        return _row_to_topic(row) if row else None


def _row_to_topic(row: dict) -> MemoryTopic:
    return MemoryTopic(
        id=str(row["id"]),
        user_id=str(row["user_id"]),
        label=row["label"],
        domain=row.get("domain"),
        count=int(row["count"]),
        last_seen=row["last_seen"],
        created_at=row["created_at"],
    )


# ── Entities ────────────────────────────────────────────────────────────────────


def upsert_entity(
    *, user_id: str, topic_id: str | None, entity_name: str, entity_type: str = "concept"
) -> str:
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO research_memory_entities (user_id, topic_id, entity_name, entity_type, count, last_seen)
            VALUES (%s, %s, %s, %s, 1, now())
            ON CONFLICT (user_id, entity_name, entity_type)
            DO UPDATE SET count = research_memory_entities.count + 1,
                          last_seen = now(),
                          topic_id = COALESCE(NULLIF(%s::uuid::text, ''), research_memory_entities.topic_id::text)::uuid
            RETURNING id
            """,
            (user_id, topic_id, entity_name, entity_type, topic_id),
        )
        return str(cur.fetchone()["id"])


def get_topic_entities(*, topic_id: str) -> list[MemoryEntity]:
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT id, user_id, topic_id, entity_name, entity_type, count, last_seen "
            "FROM research_memory_entities WHERE topic_id = %s ORDER BY count DESC",
            (topic_id,),
        )
        return [_row_to_entity(r) for r in cur.fetchall()]


def _row_to_entity(row: dict) -> MemoryEntity:
    return MemoryEntity(
        id=str(row["id"]),
        user_id=str(row["user_id"]),
        topic_id=str(row["topic_id"]) if row.get("topic_id") else None,
        entity_name=row["entity_name"],
        entity_type=row["entity_type"],
        count=int(row["count"]),
        last_seen=row["last_seen"],
    )


# ── Relationships ───────────────────────────────────────────────────────────────


def upsert_relationship(
    *,
    user_id: str,
    source_topic_id: str,
    target_topic_id: str,
    relationship_type: str = "related",
    strength: float = 0.5,
) -> str:
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO research_memory_relationships
                (user_id, source_topic_id, target_topic_id, relationship_type, strength, last_seen)
            VALUES (%s, %s, %s, %s, %s, now())
            ON CONFLICT (user_id, source_topic_id, target_topic_id, relationship_type)
            DO UPDATE SET strength = GREATEST(research_memory_relationships.strength, %s),
                          last_seen = now()
            RETURNING id
            """,
            (user_id, source_topic_id, target_topic_id, relationship_type, strength, strength),
        )
        return str(cur.fetchone()["id"])


def get_topic_relationships(*, topic_id: str) -> list[MemoryRelationship]:
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, user_id, source_topic_id, target_topic_id,
                   relationship_type, strength, last_seen
            FROM research_memory_relationships
            WHERE source_topic_id = %s OR target_topic_id = %s
            ORDER BY strength DESC LIMIT 30
            """,
            (topic_id, topic_id),
        )
        return [_row_to_rel(r) for r in cur.fetchall()]


def _row_to_rel(row: dict) -> MemoryRelationship:
    return MemoryRelationship(
        id=str(row["id"]),
        user_id=str(row["user_id"]),
        source_topic_id=str(row["source_topic_id"]),
        target_topic_id=str(row["target_topic_id"]),
        relationship_type=row["relationship_type"],
        strength=float(row["strength"]),
        last_seen=row["last_seen"],
    )


# ── Gaps ────────────────────────────────────────────────────────────────────────


def add_gap(*, user_id: str, topic_id: str | None, gap_description: str) -> str:
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO research_memory_gaps (user_id, topic_id, gap_description) "
            "VALUES (%s, %s, %s) RETURNING id",
            (user_id, topic_id, gap_description),
        )
        return str(cur.fetchone()["id"])


def get_open_gaps(*, user_id: str, limit: int = 20) -> list[KnowledgeGap]:
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT id, user_id, topic_id, gap_description, status, created_at, resolved_at "
            "FROM research_memory_gaps WHERE user_id = %s AND status = 'open' "
            "ORDER BY created_at DESC LIMIT %s",
            (user_id, limit),
        )
        return [_row_to_gap(r) for r in cur.fetchall()]


def resolve_gap(*, gap_id: str) -> None:
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE research_memory_gaps SET status = 'resolved', resolved_at = now() WHERE id = %s",
            (gap_id,),
        )


def _row_to_gap(row: dict) -> KnowledgeGap:
    return KnowledgeGap(
        id=str(row["id"]),
        user_id=str(row["user_id"]),
        topic_id=str(row["topic_id"]) if row.get("topic_id") else None,
        gap_description=row["gap_description"],
        status=row["status"],
        created_at=row["created_at"],
        resolved_at=row.get("resolved_at"),
    )


# ── Context builder ─────────────────────────────────────────────────────────────


def build_memory_context(*, user_id: str, domain_scope: list[str] | None = None) -> str:
    """
    Build a human-readable context block the planner can read, showing:
      - Topics the user has researched recently
      - Unresolved knowledge gaps
      - Cross-topic connections

    Returns an empty string if there's no memory for this user.
    """
    topics = get_user_topics(user_id=user_id, limit=8)
    if not topics:
        return ""

    parts: list[str] = ["## Your Research Memory"]
    parts.append("Topics you've researched before (most recent first):")
    for t in topics:
        parts.append(f"  - {t.label} (researched {t.count}x, last {t.last_seen.date()})")

    gaps = get_open_gaps(user_id=user_id, limit=5)
    if gaps:
        parts.append("")
        parts.append("Unresolved knowledge gaps you might want to follow up on:")
        for g in gaps:
            label = next((t.label for t in topics if t.id == g.topic_id), "")
            prefix = f"[{label}] " if label else ""
            parts.append(f"  - {prefix}{g.gap_description}")

    return "\n".join(parts)


# ── Batch insert from memory_keeper output ─────────────────────────────────────


def ingest_extracted_memory(
    *,
    user_id: str,
    topics: list[dict[str, Any]],
    entities: list[dict[str, Any]],
    relationships: list[dict[str, Any]],
    gaps: list[str],
) -> dict[str, int]:
    """
    Persist a single pass of memory extraction.

    Returns counts of what was added.
    """
    topic_ids: dict[str, str] = {}
    n_topics = n_entities = n_rels = n_gaps = 0

    for t in topics:
        tid = upsert_topic(
            user_id=user_id,
            label=t.get("label", "untitled"),
            domain=t.get("domain"),
        )
        topic_ids[t.get("label", "untitled")] = tid
        n_topics += 1

    for e in entities:
        primary_topic = e.get("parent_topic") or (list(topic_ids.values()) or [None])[0]
        upsert_entity(
            user_id=user_id,
            topic_id=topic_ids.get(primary_topic, primary_topic),
            entity_name=e.get("name", "unknown"),
            entity_type=e.get("type", "concept"),
        )
        n_entities += 1

    for r in relationships:
        src_id = topic_ids.get(r.get("source", ""))
        tgt_id = topic_ids.get(r.get("target", ""))
        if src_id and tgt_id:
            upsert_relationship(
                user_id=user_id,
                source_topic_id=src_id,
                target_topic_id=tgt_id,
                relationship_type=r.get("type", "related"),
                strength=float(r.get("strength", 0.5)),
            )
            n_rels += 1

    for g in gaps:
        add_gap(user_id=user_id, topic_id=None, gap_description=g)
        n_gaps += 1

    return {"topics": n_topics, "entities": n_entities, "relationships": n_rels, "gaps": n_gaps}