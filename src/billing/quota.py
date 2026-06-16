"""
Quota enforcement.

Two limits on the free tier (configurable via `settings.free_threads_per_month`
and `settings.free_turns_per_thread`):

  1. Per-month thread cap   — only blocks NEW threads
  2. Per-thread turn cap    — only blocks FOLLOW-UPS

Admins and active subscribers bypass both.

The check returns a (`allowed`, `reason`, `meta`) triple so the route can emit a
structured 402-style payload the frontend uses to drive the paywall modal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.auth.users import User
from src.billing import subscription, threads
from src.config import settings


@dataclass
class QuotaDecision:
    allowed: bool
    reason: Optional[str]   # "thread_cap" | "turn_cap" | None
    used: int
    limit: int
    is_subscribed: bool
    is_admin: bool

    def to_dict(self) -> dict:
        return {
            "allowed": self.allowed,
            "reason": self.reason,
            "used": self.used,
            "limit": self.limit,
            "is_subscribed": self.is_subscribed,
            "is_admin": self.is_admin,
        }


def _entitled(user: User) -> bool:
    """Admin or active subscriber — quota does not apply."""
    return user.is_admin or subscription.is_active(user.id)


def check_can_create_thread(user: User) -> QuotaDecision:
    """Can `user` start a new research thread RIGHT NOW?"""
    if _entitled(user):
        return QuotaDecision(
            allowed=True,
            reason=None,
            used=0,
            limit=0,
            is_subscribed=not user.is_admin,
            is_admin=user.is_admin,
        )

    used = threads.count_threads_this_month(user_id=user.id)
    limit = settings.free_threads_per_month
    allowed = used < limit
    return QuotaDecision(
        allowed=allowed,
        reason=None if allowed else "thread_cap",
        used=used,
        limit=limit,
        is_subscribed=False,
        is_admin=False,
    )


def check_can_add_turn(*, user: User, thread_id: str) -> QuotaDecision:
    """Can `user` add ANOTHER turn (follow-up) to `thread_id` RIGHT NOW?"""
    if _entitled(user):
        return QuotaDecision(
            allowed=True,
            reason=None,
            used=0,
            limit=0,
            is_subscribed=not user.is_admin,
            is_admin=user.is_admin,
        )

    used = threads.count_turns(thread_id=thread_id)
    limit = settings.free_turns_per_thread
    allowed = used < limit
    return QuotaDecision(
        allowed=allowed,
        reason=None if allowed else "turn_cap",
        used=used,
        limit=limit,
        is_subscribed=False,
        is_admin=False,
    )


def usage_snapshot(user: User) -> dict:
    """
    Compact usage report for `/api/usage` — what the frontend renders next to
    the user's avatar. Subscribers/admins get a sentinel limit of -1 (unlimited).
    """
    if _entitled(user):
        return {
            "is_subscribed": not user.is_admin,
            "is_admin": user.is_admin,
            "threads_used_this_month": 0,
            "threads_limit": -1,
            "turns_limit_per_thread": -1,
        }
    return {
        "is_subscribed": False,
        "is_admin": False,
        "threads_used_this_month": threads.count_threads_this_month(user_id=user.id),
        "threads_limit": settings.free_threads_per_month,
        "turns_limit_per_thread": settings.free_turns_per_thread,
    }
