"""Memory dataclasses — typed structures for the Persistent Knowledge Graph."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class MemoryTopic:
    id: str
    user_id: str
    label: str
    domain: Optional[str]
    count: int
    last_seen: datetime
    created_at: datetime


@dataclass
class MemoryEntity:
    id: str
    user_id: str
    topic_id: Optional[str]
    entity_name: str
    entity_type: str
    count: int
    last_seen: datetime


@dataclass
class MemoryRelationship:
    id: str
    user_id: str
    source_topic_id: str
    target_topic_id: str
    relationship_type: str
    strength: float
    last_seen: datetime


@dataclass
class KnowledgeGap:
    id: str
    user_id: str
    topic_id: Optional[str]
    gap_description: str
    status: str
    created_at: datetime
    resolved_at: Optional[datetime]